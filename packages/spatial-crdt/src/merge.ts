/**
 * CRDT merge function.
 *
 * The state of a SpatialDocument is a set of operations (keyed by opId).
 * Merge = set union. This is trivially:
 * - Commutative: A ∪ B = B ∪ A
 * - Associative: (A ∪ B) ∪ C = A ∪ (B ∪ C)
 * - Idempotent: A ∪ A = A
 *
 * After merging op sets, the derived state is recomputed deterministically.
 */

import type { SpatialOp } from './types'

/**
 * Merge two operation maps. Returns a new map containing the union.
 * Does not mutate either input.
 */
export function mergeOpSets(
  a: ReadonlyMap<string, SpatialOp>,
  b: ReadonlyMap<string, SpatialOp>,
): Map<string, SpatialOp> {
  const result = new Map(a)
  for (const [opId, op] of b) {
    if (!result.has(opId)) {
      result.set(opId, op)
    }
  }
  return result
}

/**
 * Compute the set of operations in `b` that are not in `a`.
 * Useful for delta sync: "what ops does the peer have that I don't?"
 */
export function opSetDifference(
  a: ReadonlyMap<string, SpatialOp>,
  b: ReadonlyMap<string, SpatialOp>,
): SpatialOp[] {
  const missing: SpatialOp[] = []
  for (const [opId, op] of b) {
    if (!a.has(opId)) {
      missing.push(op)
    }
  }
  return missing
}
