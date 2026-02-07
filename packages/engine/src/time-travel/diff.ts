/**
 * Visual diff algorithm: compare two projected venue states.
 *
 * Classifies each item as: added, removed, moved, modified, or unchanged.
 * For moved items, computes displacement vectors.
 */

import type { ProjectedVenueState, ProjectedItem } from '../projector'
import type { ItemDiff, StateDiff, DiffStatus } from './types'

// ─── Position / Rotation Comparison ──────────────────────────────────────────

const POSITION_EPSILON = 1e-6

function positionsEqual(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
): boolean {
  return (
    Math.abs(a[0] - b[0]) < POSITION_EPSILON &&
    Math.abs(a[1] - b[1]) < POSITION_EPSILON &&
    Math.abs(a[2] - b[2]) < POSITION_EPSILON
  )
}

function itemsEqual(a: ProjectedItem, b: ProjectedItem): boolean {
  return (
    a.furnitureType === b.furnitureType &&
    positionsEqual(a.position, b.position) &&
    positionsEqual(a.rotation, b.rotation) &&
    positionsEqual(a.scale, b.scale) &&
    a.groupId === b.groupId
  )
}

function onlyPositionChanged(a: ProjectedItem, b: ProjectedItem): boolean {
  return (
    a.furnitureType === b.furnitureType &&
    !positionsEqual(a.position, b.position) &&
    positionsEqual(a.rotation, b.rotation) &&
    positionsEqual(a.scale, b.scale) &&
    a.groupId === b.groupId
  )
}

// ─── Diff ────────────────────────────────────────────────────────────────────

/**
 * Compute the visual diff between two venue states.
 *
 * @param before - The earlier state ("left")
 * @param after  - The later state ("right")
 * @returns StateDiff with per-item classification and displacement vectors
 */
export function computeDiff(
  before: ProjectedVenueState,
  after: ProjectedVenueState,
): StateDiff {
  const diffs: ItemDiff[] = []
  let added = 0
  let removed = 0
  let moved = 0
  let modified = 0
  let unchanged = 0

  // Check items in `after` vs `before`
  for (const [id, afterItem] of after.items) {
    const beforeItem = before.items.get(id)

    if (!beforeItem) {
      // Item exists in after but not before → added
      diffs.push({ itemId: id, status: 'added', after: afterItem })
      added++
    } else if (itemsEqual(beforeItem, afterItem)) {
      // Identical → unchanged
      diffs.push({ itemId: id, status: 'unchanged', before: beforeItem, after: afterItem })
      unchanged++
    } else if (onlyPositionChanged(beforeItem, afterItem)) {
      // Only position differs → moved
      const displacement = [
        afterItem.position[0] - beforeItem.position[0],
        afterItem.position[1] - beforeItem.position[1],
        afterItem.position[2] - beforeItem.position[2],
      ] as const
      diffs.push({
        itemId: id,
        status: 'moved',
        before: beforeItem,
        after: afterItem,
        displacement,
      })
      moved++
    } else {
      // Other differences → modified
      diffs.push({ itemId: id, status: 'modified', before: beforeItem, after: afterItem })
      modified++
    }
  }

  // Check items in `before` not in `after` → removed
  for (const [id, beforeItem] of before.items) {
    if (!after.items.has(id)) {
      diffs.push({ itemId: id, status: 'removed', before: beforeItem })
      removed++
    }
  }

  return { diffs, added, removed, moved, modified, unchanged }
}

/**
 * Filter diffs to only changed items (exclude unchanged).
 */
export function changedOnly(diff: StateDiff): ItemDiff[] {
  return diff.diffs.filter(d => d.status !== 'unchanged')
}

/**
 * Filter diffs by status.
 */
export function filterByStatus(diff: StateDiff, status: DiffStatus): ItemDiff[] {
  return diff.diffs.filter(d => d.status === status)
}
