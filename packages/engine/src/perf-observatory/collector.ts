/**
 * Performance Observatory: real-time metrics collection and analysis.
 *
 * Collects frame budget, memory, network, solver, and incremental graph metrics
 * into ring buffers. Computes rolling statistics and exports JSON snapshots.
 */

import { RingBuffer } from './ring-buffer'
import type {
  FrameSample,
  MemorySample,
  NetworkSample,
  SolverSample,
  IncrementalSample,
  FrameStats,
  MemoryStats,
  NetworkStats,
  PerformanceSnapshot,
} from './types'

// ─── Constants ───────────────────────────────────────────────────────────────

/** Default rolling window size (frames) */
export const DEFAULT_WINDOW_SIZE = 120

/** Frame budget at 60fps (ms) */
export const FRAME_BUDGET_60FPS = 16.67

/** Frames over budget threshold (ms) to count as dropped */
export const DROPPED_FRAME_THRESHOLD = 20

/** Memory leak detection: if heap grows by this much over the window (bytes) */
export const LEAK_THRESHOLD_BYTES = 50 * 1024 * 1024 // 50 MB

// ─── Collector ───────────────────────────────────────────────────────────────

export class PerformanceCollector {
  readonly frames: RingBuffer<FrameSample>
  readonly memory: RingBuffer<MemorySample>
  readonly network: RingBuffer<NetworkSample>
  private _latestSolver: SolverSample | null = null
  private _latestIncremental: IncrementalSample | null = null
  private _enabled = true

  constructor(windowSize = DEFAULT_WINDOW_SIZE) {
    this.frames = new RingBuffer(windowSize)
    this.memory = new RingBuffer(windowSize)
    this.network = new RingBuffer(windowSize)
  }

  /** Enable/disable collection. When disabled, push calls are no-ops. */
  get enabled(): boolean { return this._enabled }
  set enabled(v: boolean) { this._enabled = v }

  // ── Push Methods ──

  pushFrame(sample: FrameSample): void {
    if (!this._enabled) return
    this.frames.push(sample)
  }

  pushMemory(sample: MemorySample): void {
    if (!this._enabled) return
    this.memory.push(sample)
  }

  pushNetwork(sample: NetworkSample): void {
    if (!this._enabled) return
    this.network.push(sample)
  }

  pushSolver(sample: SolverSample): void {
    if (!this._enabled) return
    this._latestSolver = sample
  }

  pushIncremental(sample: IncrementalSample): void {
    if (!this._enabled) return
    this._latestIncremental = sample
  }

  // ── Frame Budget Analysis ──

  getFrameStats(): FrameStats {
    const samples = this.frames.toArray()
    if (samples.length === 0) {
      return {
        avgFps: 0, minFps: 0, maxFps: 0, p99FrameMs: 0,
        avgRenderMs: 0, avgPhysicsMs: 0, avgCrdtMs: 0,
        avgUiMs: 0, avgIdleMs: 0, droppedFrames: 0,
      }
    }

    const frameTimes = samples.map(s => s.totalMs)
    const sorted = [...frameTimes].sort((a, b) => a - b)

    const avg = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length
    const p99Index = Math.min(Math.floor(frameTimes.length * 0.99), frameTimes.length - 1)

    const avgFps = avg > 0 ? 1000 / avg : 0
    const minFps = sorted[sorted.length - 1]! > 0 ? 1000 / sorted[sorted.length - 1]! : 0
    const maxFps = sorted[0]! > 0 ? 1000 / sorted[0]! : 0

    return {
      avgFps,
      minFps,
      maxFps,
      p99FrameMs: sorted[p99Index]!,
      avgRenderMs: samples.reduce((a, s) => a + s.renderMs, 0) / samples.length,
      avgPhysicsMs: samples.reduce((a, s) => a + s.physicsMs, 0) / samples.length,
      avgCrdtMs: samples.reduce((a, s) => a + s.crdtSyncMs, 0) / samples.length,
      avgUiMs: samples.reduce((a, s) => a + s.uiMs, 0) / samples.length,
      avgIdleMs: samples.reduce((a, s) => a + s.idleMs, 0) / samples.length,
      droppedFrames: frameTimes.filter(t => t > DROPPED_FRAME_THRESHOLD).length,
    }
  }

  // ── Memory Analysis ──

  getMemoryStats(): MemoryStats {
    const samples = this.memory.toArray()
    if (samples.length === 0) {
      return {
        currentJsHeapMB: 0, peakJsHeapMB: 0, currentGpuMemoryMB: 0,
        geometryCount: 0, textureCount: 0, leakSuspected: false,
      }
    }

    const latest = samples[samples.length - 1]!
    const peakHeap = Math.max(...samples.map(s => s.jsHeapUsed))

    // Leak detection: check if first third vs last third shows monotonic increase
    let leakSuspected = false
    if (samples.length >= 30) {
      const third = Math.floor(samples.length / 3)
      const firstThirdAvg = samples.slice(0, third).reduce((a, s) => a + s.jsHeapUsed, 0) / third
      const lastThirdAvg = samples.slice(-third).reduce((a, s) => a + s.jsHeapUsed, 0) / third
      leakSuspected = (lastThirdAvg - firstThirdAvg) > LEAK_THRESHOLD_BYTES
    }

    return {
      currentJsHeapMB: latest.jsHeapUsed / (1024 * 1024),
      peakJsHeapMB: peakHeap / (1024 * 1024),
      currentGpuMemoryMB: latest.gpuMemory / (1024 * 1024),
      geometryCount: latest.geometryCount,
      textureCount: latest.textureCount,
      leakSuspected,
    }
  }

  // ── Network Analysis ──

  getNetworkStats(): NetworkStats {
    const samples = this.network.toArray()
    if (samples.length === 0) {
      return {
        avgOpsSentPerSec: 0, avgOpsRecvPerSec: 0,
        avgBytesSentPerSec: 0, avgBytesRecvPerSec: 0,
        avgCompressionRatio: 0, avgLatencyMs: 0, p99LatencyMs: 0,
      }
    }

    const latencies = samples.map(s => s.latencyMs)
    const sortedLatencies = [...latencies].sort((a, b) => a - b)
    const p99Index = Math.min(Math.floor(latencies.length * 0.99), latencies.length - 1)

    return {
      avgOpsSentPerSec: samples.reduce((a, s) => a + s.opsSentPerSec, 0) / samples.length,
      avgOpsRecvPerSec: samples.reduce((a, s) => a + s.opsRecvPerSec, 0) / samples.length,
      avgBytesSentPerSec: samples.reduce((a, s) => a + s.bytesSentPerSec, 0) / samples.length,
      avgBytesRecvPerSec: samples.reduce((a, s) => a + s.bytesRecvPerSec, 0) / samples.length,
      avgCompressionRatio: samples.reduce((a, s) => a + s.compressionRatio, 0) / samples.length,
      avgLatencyMs: latencies.reduce((a, b) => a + b, 0) / latencies.length,
      p99LatencyMs: sortedLatencies[p99Index]!,
    }
  }

  // ── Solver Metrics ──

  getLatestSolver(): SolverSample | null {
    return this._latestSolver
  }

  // ── Incremental Metrics ──

  getLatestIncremental(): IncrementalSample | null {
    return this._latestIncremental
  }

  // ── Full Snapshot ──

  /** Capture a complete performance snapshot for JSON export. */
  snapshot(): PerformanceSnapshot {
    return {
      capturedAt: new Date().toISOString(),
      windowSize: this.frames.capacity,
      frame: this.getFrameStats(),
      memory: this.getMemoryStats(),
      network: this.getNetworkStats(),
      solver: this._latestSolver,
      incremental: this._latestIncremental,
    }
  }

  /** Export snapshot as JSON string. */
  exportJSON(): string {
    return JSON.stringify(this.snapshot(), null, 2)
  }

  /** Reset all buffers and metrics. */
  reset(): void {
    this.frames.clear()
    this.memory.clear()
    this.network.clear()
    this._latestSolver = null
    this._latestIncremental = null
  }
}
