/**
 * Three-way branch merge for the time-travel debugger.
 *
 * Given two branches with a common ancestor:
 * 1. Compute diffs: base→branchA, base→branchB
 * 2. Auto-merge non-conflicting changes
 * 3. Report conflicts for manual resolution
 *
 * Uses displacement-based merge for concurrent moves (spatial CRDT semantics):
 * if both branches moved the same item, sum the displacement vectors.
 */

import type { ProjectedVenueState, ProjectedItem } from '../projector'
import type { MergeResult, MergeConflict, MergeConflictKind, ConflictResolution } from './types'
import type { ItemDiff } from './types'
import { computeDiff } from './diff'

// ─── Three-Way Merge ─────────────────────────────────────────────────────────

/**
 * Merge two branches (A and B) given their common ancestor (base) state.
 *
 * Strategy:
 * - If only A changed an item → use A's version
 * - If only B changed an item → use B's version
 * - If both moved the same item → sum displacement vectors (CRDT merge)
 * - If both modified the same item (non-move) → conflict
 * - If one moved/modified and the other removed → conflict
 * - Items unchanged in both → keep base version
 */
export function threeWayMerge(
  base: ProjectedVenueState,
  branchA: ProjectedVenueState,
  branchB: ProjectedVenueState,
): MergeResult {
  const diffA = computeDiff(base, branchA)
  const diffB = computeDiff(base, branchB)

  // Index diffs by item ID for fast lookup
  const aDiffs = new Map<string, ItemDiff>()
  const bDiffs = new Map<string, ItemDiff>()
  for (const d of diffA.diffs) aDiffs.set(d.itemId, d)
  for (const d of diffB.diffs) bDiffs.set(d.itemId, d)

  // Collect all item IDs from all three states
  const allIds = new Set<string>()
  for (const id of base.items.keys()) allIds.add(id)
  for (const id of branchA.items.keys()) allIds.add(id)
  for (const id of branchB.items.keys()) allIds.add(id)

  const mergedItems = new Map<string, ProjectedItem>()
  const autoMerged: string[] = []
  const conflicts: MergeConflict[] = []

  for (const itemId of allIds) {
    const aChange = aDiffs.get(itemId)
    const bChange = bDiffs.get(itemId)
    const aStatus = aChange?.status ?? 'unchanged'
    const bStatus = bChange?.status ?? 'unchanged'

    // Neither changed → keep base
    if (aStatus === 'unchanged' && bStatus === 'unchanged') {
      const item = base.items.get(itemId)
      if (item) mergedItems.set(itemId, item)
      continue
    }

    // Only A changed
    if (aStatus !== 'unchanged' && bStatus === 'unchanged') {
      if (aStatus === 'removed') {
        // A removed → omit from merged
      } else if (aChange?.after) {
        mergedItems.set(itemId, aChange.after)
      }
      autoMerged.push(itemId)
      continue
    }

    // Only B changed
    if (aStatus === 'unchanged' && bStatus !== 'unchanged') {
      if (bStatus === 'removed') {
        // B removed → omit from merged
      } else if (bChange?.after) {
        mergedItems.set(itemId, bChange.after)
      }
      autoMerged.push(itemId)
      continue
    }

    // Both changed — need conflict resolution
    // Special case: both moved → sum displacements (CRDT semantics)
    if (aStatus === 'moved' && bStatus === 'moved' && aChange?.displacement && bChange?.displacement) {
      const baseItem = base.items.get(itemId)
      if (baseItem) {
        const merged: ProjectedItem = {
          ...baseItem,
          position: [
            baseItem.position[0] + aChange.displacement[0] + bChange.displacement[0],
            baseItem.position[1] + aChange.displacement[1] + bChange.displacement[1],
            baseItem.position[2] + aChange.displacement[2] + bChange.displacement[2],
          ],
        }
        mergedItems.set(itemId, merged)
        autoMerged.push(itemId)
        continue
      }
    }

    // Both added the same ID — use A's version (LWW)
    if (aStatus === 'added' && bStatus === 'added') {
      if (aChange?.after) {
        mergedItems.set(itemId, aChange.after)
        autoMerged.push(itemId)
      }
      continue
    }

    // One removed, one modified/moved → conflict
    if ((aStatus === 'removed' && (bStatus === 'moved' || bStatus === 'modified')) ||
        ((aStatus === 'moved' || aStatus === 'modified') && bStatus === 'removed')) {
      const kind: MergeConflictKind = aStatus === 'removed' || bStatus === 'removed'
        ? (aStatus === 'moved' || bStatus === 'moved' ? 'move-remove' : 'modify-remove')
        : 'both-modified'
      conflicts.push({
        itemId,
        kind,
        branchAValue: aChange?.after,
        branchBValue: bChange?.after,
        baseValue: base.items.get(itemId),
      })
      // Keep base in merged for now (conflict needs resolution)
      const baseItem = base.items.get(itemId)
      if (baseItem) mergedItems.set(itemId, baseItem)
      continue
    }

    // Both modified (non-move) → conflict
    const kind: MergeConflictKind =
      aStatus === 'moved' || bStatus === 'moved' ? 'both-moved' : 'both-modified'
    conflicts.push({
      itemId,
      kind,
      branchAValue: aChange?.after,
      branchBValue: bChange?.after,
      baseValue: base.items.get(itemId),
    })
    const baseItem = base.items.get(itemId)
    if (baseItem) mergedItems.set(itemId, baseItem)
  }

  // Build merged state
  const mergedState: ProjectedVenueState = {
    venueId: base.venueId,
    name: branchA.name !== base.name ? branchA.name : branchB.name,
    archived: branchA.archived || branchB.archived,
    items: mergedItems,
    groups: new Set([...branchA.groups, ...branchB.groups]),
    scenarios: new Map([...branchA.scenarios, ...branchB.scenarios]),
    version: Math.max(branchA.version, branchB.version),
  }

  return { mergedState, autoMerged, conflicts }
}

// ─── Conflict Resolution ─────────────────────────────────────────────────────

/**
 * Apply a resolution to a single conflict within a merge result.
 * Returns an updated merge result with the conflict resolved.
 */
export function resolveConflict(
  result: MergeResult,
  itemId: string,
  resolution: ConflictResolution,
): MergeResult {
  const conflict = result.conflicts.find(c => c.itemId === itemId)
  if (!conflict) return result

  const items = new Map(result.mergedState.items)

  switch (resolution) {
    case 'use-a':
      if (conflict.branchAValue) {
        items.set(itemId, conflict.branchAValue)
      } else {
        items.delete(itemId)
      }
      break
    case 'use-b':
      if (conflict.branchBValue) {
        items.set(itemId, conflict.branchBValue)
      } else {
        items.delete(itemId)
      }
      break
    case 'use-base':
      if (conflict.baseValue) {
        items.set(itemId, conflict.baseValue)
      } else {
        items.delete(itemId)
      }
      break
    case 'merge-displacements':
      // Sum displacement vectors from base
      if (conflict.baseValue && conflict.branchAValue && conflict.branchBValue) {
        const aDisp = [
          conflict.branchAValue.position[0] - conflict.baseValue.position[0],
          conflict.branchAValue.position[1] - conflict.baseValue.position[1],
          conflict.branchAValue.position[2] - conflict.baseValue.position[2],
        ]
        const bDisp = [
          conflict.branchBValue.position[0] - conflict.baseValue.position[0],
          conflict.branchBValue.position[1] - conflict.baseValue.position[1],
          conflict.branchBValue.position[2] - conflict.baseValue.position[2],
        ]
        items.set(itemId, {
          ...conflict.baseValue,
          position: [
            conflict.baseValue.position[0] + aDisp[0]! + bDisp[0]!,
            conflict.baseValue.position[1] + aDisp[1]! + bDisp[1]!,
            conflict.baseValue.position[2] + aDisp[2]! + bDisp[2]!,
          ],
        })
      }
      break
  }

  return {
    mergedState: { ...result.mergedState, items },
    autoMerged: [...result.autoMerged, itemId],
    conflicts: result.conflicts.filter(c => c.itemId !== itemId),
  }
}
