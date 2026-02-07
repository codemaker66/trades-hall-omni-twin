/**
 * Types for the time-travel debugger with branching timelines.
 *
 * Supports: scrubbing, event markers, branches, visual diffs, three-way merge.
 */

import type { DomainEvent } from '@omni-twin/types'
import type { ProjectedVenueState, ProjectedItem } from '../projector'

// ─── Timeline ────────────────────────────────────────────────────────────────

export interface TimelineSnapshot {
  /** Event index (0-based) this snapshot corresponds to */
  readonly atIndex: number
  /** Version number of the last event applied */
  readonly version: number
  /** Serializable snapshot of projected state */
  readonly state: ProjectedVenueState
}

export interface Branch {
  /** Unique branch ID */
  readonly id: string
  /** Human-readable name */
  readonly name: string
  /** Parent branch ID (null for root) */
  readonly parentId: string | null
  /** Index in parent's event array where this branch forked */
  readonly forkIndex: number
  /** Events specific to this branch (after fork point) */
  readonly events: DomainEvent[]
  /** Periodic snapshots for fast reconstruction */
  readonly snapshots: TimelineSnapshot[]
}

export interface Timeline {
  /** The root (trunk) branch */
  readonly rootBranchId: string
  /** All branches by ID */
  readonly branches: Map<string, Branch>
  /** Currently active branch */
  readonly activeBranchId: string
  /** Current position (event index) within the active branch */
  readonly cursor: number
}

// ─── Event Markers ───────────────────────────────────────────────────────────

export type EventCategory = 'move' | 'add' | 'remove' | 'rotate' | 'scale' | 'group' | 'other'

export interface EventMarker {
  /** Index within the branch's event array */
  readonly index: number
  /** Category for coloring */
  readonly category: EventCategory
  /** Underlying event */
  readonly event: DomainEvent
}

// ─── Visual Diff ─────────────────────────────────────────────────────────────

export type DiffStatus = 'added' | 'removed' | 'moved' | 'modified' | 'unchanged'

export interface ItemDiff {
  readonly itemId: string
  readonly status: DiffStatus
  /** Old state (undefined if added) */
  readonly before?: ProjectedItem
  /** New state (undefined if removed) */
  readonly after?: ProjectedItem
  /** Displacement vector for moved items */
  readonly displacement?: readonly [number, number, number]
}

export interface StateDiff {
  readonly diffs: ItemDiff[]
  readonly added: number
  readonly removed: number
  readonly moved: number
  readonly modified: number
  readonly unchanged: number
}

// ─── Merge ───────────────────────────────────────────────────────────────────

export type MergeConflictKind = 'both-moved' | 'both-modified' | 'move-remove' | 'modify-remove'

export interface MergeConflict {
  readonly itemId: string
  readonly kind: MergeConflictKind
  readonly branchAValue?: ProjectedItem
  readonly branchBValue?: ProjectedItem
  readonly baseValue?: ProjectedItem
}

export type ConflictResolution = 'use-a' | 'use-b' | 'use-base' | 'merge-displacements'

export interface MergeResult {
  /** Merged state (with auto-resolved items) */
  readonly mergedState: ProjectedVenueState
  /** Items that were auto-merged successfully */
  readonly autoMerged: string[]
  /** Conflicts requiring manual resolution */
  readonly conflicts: MergeConflict[]
}
