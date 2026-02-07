/**
 * High-resolution timing utilities for frame budget tracking.
 *
 * Provides a ScopedTimer for measuring named phases within a frame,
 * and a FrameTimer for composing per-frame FrameSample data.
 */

import type { FrameSample } from './types'

// ─── ScopedTimer ─────────────────────────────────────────────────────────────

/**
 * Measures elapsed time for named phases.
 * Uses performance.now()-style timestamps passed externally for testability.
 */
export class ScopedTimer {
  private readonly phases = new Map<string, number>()
  private currentPhase: string | null = null
  private phaseStart = 0

  constructor(private readonly now: () => number = () => performance.now()) {}

  /** Begin timing a named phase. Ends any currently active phase. */
  begin(phase: string): void {
    if (this.currentPhase !== null) {
      this.end()
    }
    this.currentPhase = phase
    this.phaseStart = this.now()
  }

  /** End the current phase. */
  end(): void {
    if (this.currentPhase === null) return
    const elapsed = this.now() - this.phaseStart
    const existing = this.phases.get(this.currentPhase) ?? 0
    this.phases.set(this.currentPhase, existing + elapsed)
    this.currentPhase = null
  }

  /** Get elapsed ms for a phase. */
  get(phase: string): number {
    return this.phases.get(phase) ?? 0
  }

  /** Reset all phases. */
  reset(): void {
    this.phases.clear()
    this.currentPhase = null
  }
}

// ─── FrameTimer ──────────────────────────────────────────────────────────────

/**
 * Tracks a single frame's timing breakdown and produces a FrameSample.
 */
export class FrameTimer {
  private readonly timer: ScopedTimer
  private frameStart = 0

  constructor(private readonly now: () => number = () => performance.now()) {
    this.timer = new ScopedTimer(now)
  }

  /** Call at the start of each frame. */
  beginFrame(): void {
    this.timer.reset()
    this.frameStart = this.now()
  }

  /** Begin a named phase within the frame. */
  beginPhase(phase: string): void {
    this.timer.begin(phase)
  }

  /** End the current phase. */
  endPhase(): void {
    this.timer.end()
  }

  /** Call at the end of each frame. Returns a FrameSample. */
  endFrame(): FrameSample {
    this.timer.end()
    const totalMs = this.now() - this.frameStart

    const renderMs = this.timer.get('render')
    const physicsMs = this.timer.get('physics')
    const crdtSyncMs = this.timer.get('crdt')
    const uiMs = this.timer.get('ui')
    const accounted = renderMs + physicsMs + crdtSyncMs + uiMs
    const idleMs = Math.max(0, totalMs - accounted)

    return {
      timestamp: this.frameStart,
      totalMs,
      renderMs,
      physicsMs,
      crdtSyncMs,
      uiMs,
      idleMs,
    }
  }
}
