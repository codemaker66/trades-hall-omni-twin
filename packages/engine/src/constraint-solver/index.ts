export {
  solve,
  validate,
  score,
} from './solver'

export {
  validateLayout,
  validateLayoutSpatial,
  checkNoOverlap,
  checkNoOverlapSpatial,
  checkBounds,
  checkObstacles,
  checkExitClearance,
  checkAisleWidth,
  checkAisleWidthSpatial,
  validateSinglePlacement,
  violationSeverity,
} from './constraints'

export {
  scoreLayout,
  scoreCapacity,
  scoreSpaceUtilization,
  scoreSightlines,
  scoreSymmetry,
  scoreExitAccess,
  DEFAULT_WEIGHTS,
} from './objectives'

export { LayoutGrid } from './grid'
export { SolverSpatialHash } from './spatial-hash'
export { IncrementalConstraintGraph } from './incremental'
export { generateChairPositions, placeChairGroups } from './chair-grouping'

export {
  markValidated,
  toCardinalRotation,
  cardinalToRadians,
  assertNeverViolation,
  CellState,
  VIOLATION_TYPES,
} from './types'

export type {
  RoomConfig,
  FurnitureSpec,
  Placement,
  LayoutRequest,
  LayoutResult,
  LayoutScores,
  ObjectiveWeights,
  SolverOptions,
  ValidationResult,
  Violation,
  ViolationType,
  Exit,
  Point2D,
  Rect,
  SolverFurnitureType,
  ValidatedLayout,
  CardinalRotation,
  CellStateValue,
  SolverPhase,
  TableGrouping,
} from './types'
