/**
 * Hybrid Logical Clock (HLC) for causal ordering of operations.
 *
 * Combines physical wall clock with a logical counter to provide:
 * - Monotonically increasing timestamps even with clock skew
 * - Causal ordering: if A happens-before B, then hlc(A) < hlc(B)
 * - Compact encoding: 48-bit wall clock + 16-bit counter = 8 bytes
 *
 * Based on: "Logical Physical Clocks and Consistent Snapshots in Globally
 * Distributed Databases" (Kulkarni et al., 2014)
 */

import type { HlcTimestamp } from './types'

export class HybridLogicalClock {
  private wallMs: number
  private counter: number

  constructor(
    private readonly now: () => number = () => Date.now(),
  ) {
    this.wallMs = 0
    this.counter = 0
  }

  /** Generate a new timestamp for a local event. */
  tick(): HlcTimestamp {
    const physicalNow = this.now()

    if (physicalNow > this.wallMs) {
      // Physical clock advanced — reset counter
      this.wallMs = physicalNow
      this.counter = 0
    } else {
      // Same or earlier physical time — increment counter
      this.counter++
      if (this.counter > 0xFFFF) {
        // Counter overflow — force wall clock forward
        this.wallMs++
        this.counter = 0
      }
    }

    return { wallMs: this.wallMs, counter: this.counter }
  }

  /** Receive a remote timestamp and merge with local clock. */
  receive(remote: HlcTimestamp): HlcTimestamp {
    const physicalNow = this.now()

    if (physicalNow > this.wallMs && physicalNow > remote.wallMs) {
      // Physical clock is ahead of both — use it
      this.wallMs = physicalNow
      this.counter = 0
    } else if (remote.wallMs > this.wallMs) {
      // Remote is ahead — adopt remote wall time
      this.wallMs = remote.wallMs
      this.counter = remote.counter + 1
    } else if (this.wallMs > remote.wallMs) {
      // Local is ahead — keep local wall time
      this.counter++
    } else {
      // Same wall time — take max counter + 1
      this.counter = Math.max(this.counter, remote.counter) + 1
    }

    if (this.counter > 0xFFFF) {
      this.wallMs++
      this.counter = 0
    }

    return { wallMs: this.wallMs, counter: this.counter }
  }

  /** Current clock state (read-only). */
  current(): HlcTimestamp {
    return { wallMs: this.wallMs, counter: this.counter }
  }
}

// ─── HLC comparison ─────────────────────────────────────────────────────────

/** Compare two HLC timestamps. Returns negative if a < b, 0 if equal, positive if a > b. */
export function hlcCompare(a: HlcTimestamp, b: HlcTimestamp): number {
  if (a.wallMs !== b.wallMs) return a.wallMs - b.wallMs
  return a.counter - b.counter
}

// ─── HLC encoding ───────────────────────────────────────────────────────────

/** Pack HLC into 8 bytes: (wallMs << 16) | counter. */
export function hlcToUint64(hlc: HlcTimestamp): bigint {
  return (BigInt(hlc.wallMs) << 16n) | BigInt(hlc.counter & 0xFFFF)
}

/** Unpack 8 bytes into HLC. */
export function uint64ToHlc(packed: bigint): HlcTimestamp {
  return {
    wallMs: Number(packed >> 16n),
    counter: Number(packed & 0xFFFFn),
  }
}
