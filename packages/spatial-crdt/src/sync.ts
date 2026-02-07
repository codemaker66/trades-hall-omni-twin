/**
 * Delta sync protocol for efficient state exchange between replicas.
 *
 * Protocol flow:
 * 1. Peer A sends its state vector to Peer B
 * 2. Peer B computes the ops that A is missing
 * 3. Peer B sends the missing ops to Peer A
 * 4. Peer A applies the ops (merge)
 * 5. Repeat in reverse direction
 *
 * This is the standard state-vector exchange pattern used by Yjs, Automerge, etc.
 */

import type { SpatialOp, StateVector } from './types'
import type { SpatialDocument } from './document'

/** Encode a state vector as a compact array of [replicaId, counter] pairs. */
export function encodeStateVector(sv: StateVector): Array<[string, number]> {
  return [...sv.entries()]
}

/** Decode a state vector from the compact array format. */
export function decodeStateVector(pairs: Array<[string, number]>): StateVector {
  return new Map(pairs)
}

/**
 * Compute a sync message: the ops that the peer needs.
 * Given our document and the peer's state vector, return the missing ops.
 */
export function computeSyncMessage(
  doc: SpatialDocument,
  peerStateVector: StateVector,
): SpatialOp[] {
  return doc.getMissingOps(peerStateVector)
}

/**
 * Apply a sync message to our document.
 * Returns the number of new ops applied.
 */
export function applySyncMessage(
  doc: SpatialDocument,
  ops: SpatialOp[],
): number {
  return doc.applyBatch(ops)
}

/**
 * Full bidirectional sync between two documents (for testing/simulation).
 * Returns the number of ops exchanged in each direction.
 */
export function fullSync(
  docA: SpatialDocument,
  docB: SpatialDocument,
): { aToB: number; bToA: number } {
  const svA = docA.stateVector()
  const svB = docB.stateVector()

  const missingInB = computeSyncMessage(docA, svB)
  const missingInA = computeSyncMessage(docB, svA)

  const aToB = applySyncMessage(docB, missingInB)
  const bToA = applySyncMessage(docA, missingInA)

  return { aToB, bToA }
}
