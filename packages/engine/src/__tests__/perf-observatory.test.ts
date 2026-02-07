import { describe, it, expect } from 'vitest'

import { RingBuffer } from '../perf-observatory/ring-buffer'
import {
  PerformanceCollector,
  DEFAULT_WINDOW_SIZE,
  DROPPED_FRAME_THRESHOLD,
  LEAK_THRESHOLD_BYTES,
} from '../perf-observatory/collector'
import { ScopedTimer, FrameTimer } from '../perf-observatory/timer'
import type {
  FrameSample,
  MemorySample,
  NetworkSample,
  SolverSample,
  IncrementalSample,
} from '../perf-observatory/types'

// ─── RingBuffer ──────────────────────────────────────────────────────────────

describe('RingBuffer', () => {
  it('starts empty', () => {
    const buf = new RingBuffer<number>(5)
    expect(buf.size).toBe(0)
    expect(buf.full).toBe(false)
    expect(buf.latest()).toBeUndefined()
  })

  it('pushes and reads items', () => {
    const buf = new RingBuffer<number>(5)
    buf.push(10)
    buf.push(20)
    buf.push(30)
    expect(buf.size).toBe(3)
    expect(buf.at(0)).toBe(10)
    expect(buf.at(1)).toBe(20)
    expect(buf.at(2)).toBe(30)
    expect(buf.latest()).toBe(30)
  })

  it('wraps around when full', () => {
    const buf = new RingBuffer<number>(3)
    buf.push(1)
    buf.push(2)
    buf.push(3)
    expect(buf.full).toBe(true)

    buf.push(4) // overwrites 1
    expect(buf.size).toBe(3)
    expect(buf.at(0)).toBe(2) // oldest is now 2
    expect(buf.at(1)).toBe(3)
    expect(buf.at(2)).toBe(4) // newest
    expect(buf.latest()).toBe(4)
  })

  it('out-of-range returns undefined', () => {
    const buf = new RingBuffer<number>(5)
    buf.push(1)
    expect(buf.at(-1)).toBeUndefined()
    expect(buf.at(5)).toBeUndefined()
  })

  it('iterates oldest to newest', () => {
    const buf = new RingBuffer<number>(3)
    buf.push(1)
    buf.push(2)
    buf.push(3)
    buf.push(4) // wraps
    expect(buf.toArray()).toEqual([2, 3, 4])
  })

  it('clears buffer', () => {
    const buf = new RingBuffer<number>(5)
    buf.push(1)
    buf.push(2)
    buf.clear()
    expect(buf.size).toBe(0)
    expect(buf.latest()).toBeUndefined()
  })
})

// ─── ScopedTimer ─────────────────────────────────────────────────────────────

describe('ScopedTimer', () => {
  it('measures phase durations', () => {
    let time = 0
    const timer = new ScopedTimer(() => time)

    timer.begin('render')
    time = 5
    timer.end()

    timer.begin('physics')
    time = 8
    timer.end()

    expect(timer.get('render')).toBe(5)
    expect(timer.get('physics')).toBe(3)
    expect(timer.get('unknown')).toBe(0)
  })

  it('auto-ends previous phase on begin', () => {
    let time = 0
    const timer = new ScopedTimer(() => time)

    timer.begin('a')
    time = 3
    timer.begin('b')  // auto-ends 'a'
    time = 7
    timer.end()

    expect(timer.get('a')).toBe(3)
    expect(timer.get('b')).toBe(4)
  })

  it('accumulates multiple calls to same phase', () => {
    let time = 0
    const timer = new ScopedTimer(() => time)

    timer.begin('render')
    time = 2
    timer.end()

    timer.begin('render')
    time = 5
    timer.end()

    expect(timer.get('render')).toBe(5) // 2 + 3
  })

  it('resets all phases', () => {
    let time = 0
    const timer = new ScopedTimer(() => time)
    timer.begin('a')
    time = 5
    timer.end()
    timer.reset()
    expect(timer.get('a')).toBe(0)
  })
})

// ─── FrameTimer ──────────────────────────────────────────────────────────────

describe('FrameTimer', () => {
  it('produces a FrameSample', () => {
    let time = 0
    const ft = new FrameTimer(() => time)

    ft.beginFrame()
    ft.beginPhase('render')
    time = 5
    ft.endPhase()
    ft.beginPhase('physics')
    time = 8
    ft.endPhase()
    ft.beginPhase('crdt')
    time = 9
    ft.endPhase()
    ft.beginPhase('ui')
    time = 11
    ft.endPhase()
    time = 16
    const sample = ft.endFrame()

    expect(sample.totalMs).toBe(16)
    expect(sample.renderMs).toBe(5)
    expect(sample.physicsMs).toBe(3)
    expect(sample.crdtSyncMs).toBe(1)
    expect(sample.uiMs).toBe(2)
    expect(sample.idleMs).toBe(5)
    expect(sample.timestamp).toBe(0)
  })

  it('idle is at least 0', () => {
    let time = 0
    const ft = new FrameTimer(() => time)
    ft.beginFrame()
    ft.beginPhase('render')
    time = 16
    ft.endPhase()
    time = 16 // frame ends exactly at render end
    const sample = ft.endFrame()
    expect(sample.idleMs).toBe(0)
  })
})

// ─── PerformanceCollector ────────────────────────────────────────────────────

describe('PerformanceCollector', () => {
  function makeFrame(totalMs: number, renderMs = 0): FrameSample {
    return {
      timestamp: Date.now(),
      totalMs,
      renderMs,
      physicsMs: 0,
      crdtSyncMs: 0,
      uiMs: 0,
      idleMs: totalMs - renderMs,
    }
  }

  function makeMemory(jsHeapUsed: number, gpuMemory = 0): MemorySample {
    return {
      timestamp: Date.now(),
      jsHeapUsed,
      jsHeapTotal: jsHeapUsed * 2,
      gpuMemory,
      geometryCount: 10,
      textureCount: 5,
    }
  }

  function makeNetwork(latencyMs: number): NetworkSample {
    return {
      timestamp: Date.now(),
      opsSentPerSec: 100,
      opsRecvPerSec: 100,
      bytesSentPerSec: 5000,
      bytesRecvPerSec: 5000,
      compressionRatio: 4.8,
      latencyMs,
    }
  }

  it('creates with default window size', () => {
    const pc = new PerformanceCollector()
    expect(pc.frames.capacity).toBe(DEFAULT_WINDOW_SIZE)
  })

  it('collects and reports frame stats', () => {
    const pc = new PerformanceCollector(10)
    for (let i = 0; i < 10; i++) {
      pc.pushFrame(makeFrame(16, 8))
    }

    const stats = pc.getFrameStats()
    expect(stats.avgFps).toBeCloseTo(62.5, 0)
    expect(stats.avgRenderMs).toBe(8)
    expect(stats.droppedFrames).toBe(0)
  })

  it('counts dropped frames', () => {
    const pc = new PerformanceCollector(5)
    pc.pushFrame(makeFrame(16))
    pc.pushFrame(makeFrame(25)) // dropped
    pc.pushFrame(makeFrame(16))
    pc.pushFrame(makeFrame(30)) // dropped
    pc.pushFrame(makeFrame(16))

    const stats = pc.getFrameStats()
    expect(stats.droppedFrames).toBe(2)
  })

  it('returns empty stats when no samples', () => {
    const pc = new PerformanceCollector()
    const stats = pc.getFrameStats()
    expect(stats.avgFps).toBe(0)
    expect(stats.droppedFrames).toBe(0)
  })

  it('collects and reports memory stats', () => {
    const pc = new PerformanceCollector(10)
    pc.pushMemory(makeMemory(100 * 1024 * 1024))  // 100 MB
    pc.pushMemory(makeMemory(120 * 1024 * 1024))  // 120 MB

    const stats = pc.getMemoryStats()
    expect(stats.currentJsHeapMB).toBe(120)
    expect(stats.peakJsHeapMB).toBe(120)
  })

  it('detects memory leak', () => {
    const pc = new PerformanceCollector(90)
    // First 30 samples: low memory
    for (let i = 0; i < 30; i++) {
      pc.pushMemory(makeMemory(50 * 1024 * 1024))
    }
    // Middle 30: moderate
    for (let i = 0; i < 30; i++) {
      pc.pushMemory(makeMemory(80 * 1024 * 1024))
    }
    // Last 30: high (>50 MB above first third)
    for (let i = 0; i < 30; i++) {
      pc.pushMemory(makeMemory(120 * 1024 * 1024))
    }

    const stats = pc.getMemoryStats()
    expect(stats.leakSuspected).toBe(true)
  })

  it('no leak when memory stable', () => {
    const pc = new PerformanceCollector(90)
    for (let i = 0; i < 90; i++) {
      pc.pushMemory(makeMemory(100 * 1024 * 1024))
    }
    const stats = pc.getMemoryStats()
    expect(stats.leakSuspected).toBe(false)
  })

  it('collects and reports network stats', () => {
    const pc = new PerformanceCollector(10)
    for (let i = 0; i < 10; i++) {
      pc.pushNetwork(makeNetwork(20 + i))
    }

    const stats = pc.getNetworkStats()
    expect(stats.avgOpsSentPerSec).toBe(100)
    expect(stats.avgLatencyMs).toBeCloseTo(24.5, 1)
    expect(stats.p99LatencyMs).toBeGreaterThanOrEqual(29)
  })

  it('records solver sample', () => {
    const pc = new PerformanceCollector()
    const sample: SolverSample = {
      timestamp: Date.now(),
      solveTimeMs: 150,
      placementAttempts: 200,
      annealingIterations: 500,
      qualityScore: 0.85,
      violations: 0,
      itemsPlaced: 40,
      itemsRequested: 40,
    }
    pc.pushSolver(sample)
    expect(pc.getLatestSolver()).toEqual(sample)
  })

  it('records incremental sample', () => {
    const pc = new PerformanceCollector()
    const sample: IncrementalSample = {
      timestamp: Date.now(),
      nodesStabilized: 15,
      totalNodes: 50,
      cutoffRatio: 0.7,
      maxDepth: 3,
      stabilizeMs: 0.5,
    }
    pc.pushIncremental(sample)
    expect(pc.getLatestIncremental()).toEqual(sample)
  })

  it('respects enabled flag', () => {
    const pc = new PerformanceCollector()
    pc.enabled = false
    pc.pushFrame(makeFrame(16))
    expect(pc.frames.size).toBe(0)

    pc.enabled = true
    pc.pushFrame(makeFrame(16))
    expect(pc.frames.size).toBe(1)
  })

  it('produces full snapshot', () => {
    const pc = new PerformanceCollector(10)
    pc.pushFrame(makeFrame(16, 8))
    pc.pushMemory(makeMemory(100 * 1024 * 1024, 50 * 1024 * 1024))
    pc.pushNetwork(makeNetwork(20))

    const snapshot = pc.snapshot()
    expect(snapshot.capturedAt).toBeDefined()
    expect(snapshot.windowSize).toBe(10)
    expect(snapshot.frame.avgFps).toBeGreaterThan(0)
    expect(snapshot.memory.currentJsHeapMB).toBe(100)
    expect(snapshot.network.avgLatencyMs).toBe(20)
    expect(snapshot.solver).toBeNull()
    expect(snapshot.incremental).toBeNull()
  })

  it('exports valid JSON', () => {
    const pc = new PerformanceCollector(5)
    pc.pushFrame(makeFrame(16))
    const json = pc.exportJSON()
    const parsed = JSON.parse(json)
    expect(parsed.frame).toBeDefined()
    expect(parsed.memory).toBeDefined()
    expect(parsed.network).toBeDefined()
  })

  it('resets all data', () => {
    const pc = new PerformanceCollector(10)
    pc.pushFrame(makeFrame(16))
    pc.pushMemory(makeMemory(100))
    pc.pushNetwork(makeNetwork(20))
    pc.pushSolver({ timestamp: 0, solveTimeMs: 100, placementAttempts: 10, annealingIterations: 50, qualityScore: 0.8, violations: 0, itemsPlaced: 5, itemsRequested: 5 })
    pc.pushIncremental({ timestamp: 0, nodesStabilized: 5, totalNodes: 10, cutoffRatio: 0.5, maxDepth: 2, stabilizeMs: 1 })

    pc.reset()
    expect(pc.frames.size).toBe(0)
    expect(pc.memory.size).toBe(0)
    expect(pc.network.size).toBe(0)
    expect(pc.getLatestSolver()).toBeNull()
    expect(pc.getLatestIncremental()).toBeNull()
  })

  it('p99 frame time is accurate', () => {
    const pc = new PerformanceCollector(100)
    // 99 fast frames + 1 slow frame
    for (let i = 0; i < 99; i++) {
      pc.pushFrame(makeFrame(10))
    }
    pc.pushFrame(makeFrame(50)) // spike

    const stats = pc.getFrameStats()
    expect(stats.p99FrameMs).toBe(50)
  })
})
