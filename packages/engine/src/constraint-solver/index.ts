export {
  solve,
  validate,
  score,
} from './solver'

export {
  validateLayout,
  checkNoOverlap,
  checkBounds,
  checkObstacles,
  checkExitClearance,
  checkAisleWidth,
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
  Exit,
  Point2D,
  Rect,
  SolverFurnitureType,
} from './types'
