// Spatial engine: event sourcing, command handling, ECS, and WASM bridge.

export const ENGINE_VERSION = '0.1.0'

export {
  validateCommand,
  emptyProjectedState,
  type ValidatorItem,
  type ValidatorVenueState,
} from './command-validator'

export {
  applyEvent,
  projectState,
  emptyVenueState,
  type ProjectedItem,
  type ProjectedScenario,
  type ProjectedVenueState,
} from './projector'

export { handleCommand } from './command-handler'

export {
  UndoManager,
  type UndoEntry,
  type ItemSnapshot,
} from './undo-manager'

export {
  createEventSourcedStore,
  placeItemCommand,
  moveItemCommand,
  removeItemCommand,
  moveItemsBatchCommand,
  groupItemsCommand,
  type EventSourcedStoreState,
  type EventSourcedStoreActions,
} from './store-adapter'

export {
  registerMigration,
  clearMigrations,
  migrateEvent,
  migrateEvents,
  needsMigration,
  toVersionedEvent,
  type EventMigration,
  type VersionedEvent,
} from './event-migrator'

// ECS
export {
  Position,
  Rotation,
  Scale,
  BoundingBox,
  FurnitureTag,
  GroupMember,
  Selectable,
  Draggable,
  furnitureTypeToIndex,
  indexToFurnitureType,
  createEcsWorld,
  resetEcsWorld,
  createFurnitureEntity,
  removeFurnitureEntity,
  isFurnitureEntity,
  getAllEntityIds,
  EntityIdMap,
  type EcsWorld,
  type FurnitureEntityInput,
  // Systems
  SpatialHash,
  type AABB,
  getEntityAABB,
  aabbOverlap,
  detectCollisions,
  checkCollision,
  checkPlacementCollision,
  type CollisionPair,
  rectSelect,
  pointSelect,
  snapToGrid,
  snapToHeightGrid,
  findNearest,
  findKNearest,
  snapToNearest,
  type NearestResult,
  // Bridge
  EcsBridge,
  type BridgeItem,
} from './ecs'

// Incremental computation
export {
  IncrementalGraph,
  type InputNode,
  type DerivedNode,
  type ObserverNode,
} from './incremental'

// Constraint solver
export {
  solve,
  validate,
  score,
  validateLayout,
  checkNoOverlap,
  checkBounds,
  checkObstacles,
  checkExitClearance,
  checkAisleWidth,
  scoreLayout,
  scoreCapacity,
  scoreSpaceUtilization,
  scoreSightlines,
  scoreSymmetry,
  scoreExitAccess,
  DEFAULT_WEIGHTS,
  LayoutGrid,
  type RoomConfig,
  type FurnitureSpec,
  type Placement,
  type LayoutRequest,
  type LayoutResult,
  type LayoutScores,
  type ObjectiveWeights,
  type SolverOptions,
  type ValidationResult,
  type Violation,
  type Exit,
  type Point2D,
  type Rect,
  type SolverFurnitureType,
} from './constraint-solver'

// GPU compute spatial analysis (import from '@omni-twin/engine/gpu-compute' for full API)
export {
  detectCollisionsCPU as gpuDetectCollisionsCPU,
  analyzeSightlinesCPU as gpuAnalyzeSightlinesCPU,
  simulateCrowdFlowCPU as gpuSimulateCrowdFlowCPU,
  COLLISION_SHADER,
  SIGHTLINE_SHADER,
  CROWD_FLOW_SHADER,
  type AABB2D,
  type AnalysisItem,
  type RoomGeometry as GpuRoomGeometry,
  type CollisionResult as GpuCollisionResult,
  type SightlineResult,
  type CrowdFlowResult,
  type CrowdFlowParams,
  type GpuCapabilities,
} from './gpu-compute'

// Time-travel debugger
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
  computeDiff,
  changedOnly,
  filterByStatus,
  threeWayMerge,
  resolveConflict,
  type TimelineSnapshot,
  type Branch,
  type Timeline,
  type EventCategory,
  type EventMarker,
  type DiffStatus,
  type ItemDiff,
  type StateDiff,
  type MergeConflictKind,
  type MergeConflict,
  type ConflictResolution,
  type MergeResult,
} from './time-travel'

// Performance Observatory
export {
  RingBuffer,
  DEFAULT_WINDOW_SIZE,
  FRAME_BUDGET_60FPS,
  DROPPED_FRAME_THRESHOLD,
  LEAK_THRESHOLD_BYTES,
  PerformanceCollector,
  ScopedTimer,
  FrameTimer,
  type FrameSample,
  type MemorySample,
  type NetworkSample,
  type SolverSample,
  type IncrementalSample,
  type FrameStats,
  type MemoryStats,
  type NetworkStats,
  type PerformanceSnapshot,
} from './perf-observatory'

// NVIDIA integrations (T9/T10/T11)
export {
  // Cosmos (T9)
  serializeScene,
  countByType as cosmosCountByType,
  estimateCapacity as cosmosEstimateCapacity,
  formatFurnitureSummary,
  detectLayoutStyle,
  MockCosmosClient,
  createCosmosClient,
  type CosmosClient,
  // Omniverse (T10)
  toOmniverseScene,
  toOmniverseItem,
  computeSceneDiff,
  MockOmniverseClient,
  createOmniverseClient,
  type SceneDiffOp,
  type OmniverseStreamingClient,
  // ACE (T11)
  buildConciergeContext,
  buildSystemPrompt,
  MockConciergeClient,
  createConciergeClient,
  type ConciergeChatClient,
  // Shared types
  type JobStatus,
  type JobResult,
  type SceneDescription,
  type CosmosRequest,
  type CosmosResult,
  type OmniverseSessionConfig,
  type OmniverseSession,
  type OmniverseSceneItem,
  type OmniverseSceneState,
  type ConciergeContext,
  type ConciergeMessage,
  type ConciergeSession,
  type ConciergeResponse,
} from './nvidia'
