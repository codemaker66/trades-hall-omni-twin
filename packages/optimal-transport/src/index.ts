/**
 * @omni-twin/optimal-transport
 *
 * Mathematically principled venue-event matching using Optimal Transport.
 * Wasserstein distances, Sinkhorn algorithm, barycenters, partial transport,
 * displacement interpolation, and GPU-accelerated browser computation.
 */

// ─── OT-1: Types & Configuration ──────────────────────────────────────────
export type {
  SinkhornConfig,
  TransportResult,
  VenueFeatures,
  EventRequirements,
  CostWeights,
  FurniturePosition,
  BarycenterConfig,
  SinkhornSolver,
  ObservedMatching,
  InverseOTConfig,
} from './types'

export {
  DEFAULT_SINKHORN_CONFIG,
  DEFAULT_COST_WEIGHTS,
  DEFAULT_BARYCENTER_CONFIG,
  DEFAULT_INVERSE_OT_CONFIG,
} from './types'

// ─── OT-1: Core Sinkhorn Solver ───────────────────────────────────────────
export { sinkhorn, sinkhornCost } from './sinkhorn'
export { sinkhornLog, sinkhornLogCost } from './sinkhorn-log'
export { SinkhornGPU, SinkhornCPU, createSolver } from './sinkhorn-gpu'

// ─── OT-1: Numerical Utilities ────────────────────────────────────────────
export {
  logSumExp,
  logSumExpSlice,
  normalize01,
  toRad,
  computeRowSums,
  computeColSums,
  matVecMul,
  matTransVecMul,
  normalizeDistribution,
  uniformDistribution,
  l1Distance,
  dot,
  clampFloor,
} from './utils'

// ─── OT-2: Cost Matrix & Sinkhorn Divergence ─────────────────────────────
export {
  capacityDistance,
  amenityDistance,
  locationDistance,
  priceDistance,
  buildCostMatrix,
  buildSelfCostMatrix,
  sinkhornDivergence,
  sinkhornDivergenceSymmetric,
} from './cost-matrix'

// ─── OT-3: Wasserstein Barycenters ───────────────────────────────────────
export {
  fixedSupportBarycenter,
  scoreAgainstBarycenter,
  featuresToDistribution,
} from './barycenter'

// ─── OT-4: Partial & Unbalanced Transport ─────────────────────────────────
export {
  partialSinkhorn,
  unbalancedSinkhorn,
} from './partial'

// ─── OT-5: Displacement Interpolation ─────────────────────────────────────
export {
  displacementInterpolation,
  generateTransitionKeyframes,
  extractAssignment,
  buildPositionCostMatrix,
} from './interpolation'

// ─── OT-7: Inverse Optimal Transport ──────────────────────────────────────
export {
  learnCostWeights,
  evaluateWeights,
  buildObservedPlan,
} from './inverse-ot'
