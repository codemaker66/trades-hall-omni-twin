// Time-travel debugger with branching timelines

export type {
  TimelineSnapshot,
  Branch,
  Timeline,
  EventCategory,
  EventMarker,
  DiffStatus,
  ItemDiff,
  StateDiff,
  MergeConflictKind,
  MergeConflict,
  ConflictResolution,
  MergeResult,
} from './types'

export {
  SNAPSHOT_INTERVAL,
  classifyEvent,
  resetBranchCounter,
  createTimeline,
  findNearestSnapshot,
  reconstructAt,
  reconstructCurrent,
  appendEvent,
  appendEvents,
  seekTo,
  createBranch,
  switchBranch,
  getEventMarkers,
  getActiveBranchLength,
  listBranches,
} from './timeline'

export {
  computeDiff,
  changedOnly,
  filterByStatus,
} from './diff'

export {
  threeWayMerge,
  resolveConflict,
} from './merge'
