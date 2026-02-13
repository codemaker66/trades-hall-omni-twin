// ---------------------------------------------------------------------------
// HPC-7: CRDT — Last-Writer-Wins Register
// ---------------------------------------------------------------------------
// A convergent replicated register where the value with the highest timestamp
// wins. Ties are broken by lexicographic comparison of peer IDs (higher wins).
// ---------------------------------------------------------------------------

import type { LWWRegister } from '../types.js';

/**
 * Create a new LWW register with an initial value.
 * If no timestamp is provided, `Date.now()` is used.
 */
export function createLWWRegister<T>(
  value: T,
  peerId: string,
  timestamp?: number,
): LWWRegister<T> {
  return {
    value,
    timestamp: timestamp ?? Date.now(),
    peerId,
  };
}

/**
 * Attempt to set a new value on the register.
 * The update is applied only if the new timestamp is strictly greater,
 * or equal with a lexicographically higher peerId (tiebreak).
 * Returns a new register (or the original if the update is rejected).
 */
export function lwwSet<T>(
  register: LWWRegister<T>,
  value: T,
  peerId: string,
  timestamp?: number,
): LWWRegister<T> {
  const ts = timestamp ?? Date.now();

  if (
    ts > register.timestamp ||
    (ts === register.timestamp && peerId > register.peerId)
  ) {
    return { value, timestamp: ts, peerId };
  }

  return register;
}

/**
 * Get the current value from the register.
 */
export function lwwGet<T>(register: LWWRegister<T>): T {
  return register.value;
}

/**
 * Merge two registers. The register with the higher timestamp wins.
 * On equal timestamps the higher peerId wins.
 */
export function lwwMerge<T>(
  a: LWWRegister<T>,
  b: LWWRegister<T>,
): LWWRegister<T> {
  if (a.timestamp > b.timestamp) return a;
  if (b.timestamp > a.timestamp) return b;
  // Equal timestamps — higher peerId wins
  return a.peerId >= b.peerId ? a : b;
}
