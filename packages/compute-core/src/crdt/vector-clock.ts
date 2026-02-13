// ---------------------------------------------------------------------------
// HPC-7: CRDT — Vector Clock for causal ordering
// ---------------------------------------------------------------------------
// A vector clock maps peer IDs to logical timestamps. Two events can be
// compared for causal ordering: before, after, concurrent, or equal.
// ---------------------------------------------------------------------------

import type { MutableVectorClock } from '../types.js';

/**
 * Create an empty vector clock.
 */
export function createVectorClock(): MutableVectorClock {
  return { entries: new Map<string, number>() };
}

/**
 * Increment the logical timestamp for the given peer.
 */
export function vcIncrement(clock: MutableVectorClock, peerId: string): void {
  const current = clock.entries.get(peerId) ?? 0;
  clock.entries.set(peerId, current + 1);
}

/**
 * Element-wise maximum merge of two vector clocks.
 * Returns a new clock whose entries are the pointwise max of a and b.
 */
export function vcMerge(
  a: MutableVectorClock,
  b: MutableVectorClock,
): MutableVectorClock {
  const result: MutableVectorClock = { entries: new Map<string, number>() };

  // Copy all entries from a
  for (const [peer, count] of a.entries) {
    result.entries.set(peer, count);
  }

  // Merge entries from b (take max)
  for (const [peer, countB] of b.entries) {
    const countA = result.entries.get(peer) ?? 0;
    result.entries.set(peer, Math.max(countA, countB));
  }

  return result;
}

/**
 * Compare two vector clocks for causal ordering.
 *
 * - `'before'`: a causally precedes b (a < b)
 * - `'after'`: a causally follows b (a > b)
 * - `'concurrent'`: neither dominates the other
 * - `'equal'`: identical clocks
 */
export function vcCompare(
  a: MutableVectorClock,
  b: MutableVectorClock,
): 'before' | 'after' | 'concurrent' | 'equal' {
  // Gather all peer IDs from both clocks
  const allPeers = new Set<string>();
  for (const peer of a.entries.keys()) allPeers.add(peer);
  for (const peer of b.entries.keys()) allPeers.add(peer);

  let aLess = false;
  let bLess = false;

  for (const peer of allPeers) {
    const va = a.entries.get(peer) ?? 0;
    const vb = b.entries.get(peer) ?? 0;

    if (va < vb) {
      aLess = true;
    } else if (va > vb) {
      bLess = true;
    }

    // Early out: if both directions are less, it's concurrent
    if (aLess && bLess) {
      return 'concurrent';
    }
  }

  if (aLess && !bLess) return 'before';
  if (bLess && !aLess) return 'after';
  return 'equal';
}

/**
 * Returns true if clock `a` causally precedes clock `b` (a happens-before b).
 * This means all entries in a are <= corresponding entries in b,
 * and at least one is strictly less.
 */
export function vcHappensBefore(
  a: MutableVectorClock,
  b: MutableVectorClock,
): boolean {
  return vcCompare(a, b) === 'before';
}

/**
 * Get the logical timestamp for a given peer (0 if absent).
 */
export function vcGet(clock: MutableVectorClock, peerId: string): number {
  return clock.entries.get(peerId) ?? 0;
}

/**
 * Deep-copy a vector clock.
 */
export function vcClone(clock: MutableVectorClock): MutableVectorClock {
  return { entries: new Map(clock.entries) };
}
