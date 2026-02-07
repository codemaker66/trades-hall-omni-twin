/**
 * Core types for the Optimal Transport package.
 */

// ─── Sinkhorn Configuration ────────────────────────────────────────────────

export interface SinkhornConfig {
  /** Regularization parameter (default: 0.01) */
  epsilon: number
  /** Max iterations (default: 100) */
  maxIterations: number
  /** Convergence tolerance on marginals (default: 1e-6) */
  tolerance: number
  /** Algorithm variant */
  method: 'standard' | 'log' | 'stabilized'
}

export const DEFAULT_SINKHORN_CONFIG: SinkhornConfig = {
  epsilon: 0.01,
  maxIterations: 100,
  tolerance: 1e-6,
  method: 'standard',
}

// ─── Transport Result ──────────────────────────────────────────────────────

export interface TransportResult {
  /** N×M transport plan (row-major) */
  plan: Float64Array
  /** Total transport cost <T, C> */
  cost: number
  /** Dual potential f (N) */
  dualF: Float64Array
  /** Dual potential g (M) */
  dualG: Float64Array
  /** Iterations to converge */
  iterations: number
  /** Whether the algorithm converged within tolerance */
  converged: boolean
  /** Source dimension */
  N: number
  /** Target dimension */
  M: number
}

// ─── Venue-Event Feature Types ─────────────────────────────────────────────

export interface VenueFeatures {
  capacity: number
  amenities: boolean[]
  location: { lat: number; lng: number }
  pricePerEvent: number
  sqFootage: number
  venueType: string
}

export interface EventRequirements {
  guestCount: number
  requiredAmenities: boolean[]
  preferredLocation: { lat: number; lng: number }
  budget: number
  minSqFootage: number
  eventType: string
}

// ─── Cost Matrix Types ─────────────────────────────────────────────────────

export interface CostWeights {
  capacity: number
  price: number
  amenity: number
  location: number
}

export const DEFAULT_COST_WEIGHTS: CostWeights = {
  capacity: 0.30,
  price: 0.30,
  amenity: 0.25,
  location: 0.15,
}

// ─── Furniture Positions (for displacement interpolation) ──────────────────

export interface FurniturePosition {
  id: string
  x: number
  z: number
  rotation: number
  type: string
  /** Opacity for fade in/out during partial transport transitions */
  opacity?: number
}

// ─── Barycenter Types ──────────────────────────────────────────────────────

export interface BarycenterConfig {
  epsilon: number
  maxIterations: number
  tolerance: number
}

export const DEFAULT_BARYCENTER_CONFIG: BarycenterConfig = {
  epsilon: 0.01,
  maxIterations: 100,
  tolerance: 1e-8,
}

// ─── Solver Interface ──────────────────────────────────────────────────────

export interface SinkhornSolver {
  solve(
    a: Float64Array | Float32Array,
    b: Float64Array | Float32Array,
    C: Float64Array | Float32Array,
    config: Partial<SinkhornConfig>,
  ): Promise<TransportResult> | TransportResult
}

// ─── Inverse OT Types ──────────────────────────────────────────────────────

export interface ObservedMatching {
  eventIndex: number
  venueIndex: number
  success: boolean
}

export interface InverseOTConfig {
  learningRate: number
  iterations: number
  epsilon: number
  finiteDiffStep: number
}

export const DEFAULT_INVERSE_OT_CONFIG: InverseOTConfig = {
  learningRate: 0.01,
  iterations: 100,
  epsilon: 0.05,
  finiteDiffStep: 1e-4,
}
