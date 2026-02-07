/**
 * Types for the TDA WASM package (TDA-7).
 */

/** Result from persistent homology computation. */
export interface PersistenceResult {
  /** Persistence diagrams per dimension: H0, H1, H2, etc. */
  diagrams: Record<string, [number, number][]>
  /** Computation time in milliseconds */
  computeTimeMs: number
  /** Number of input points */
  numPoints: number
  /** Maximum dimension computed */
  maxDim: number
}

/** Configuration for browser-side persistence computation. */
export interface BrowserPersistenceConfig {
  /** Maximum homology dimension (default 1, max 2 in browser) */
  maxDim?: number
  /** Filtration threshold (limits complex size) */
  threshold?: number
  /** Whether to use a Web Worker for non-blocking computation */
  useWorker?: boolean
  /** Callback for progress updates (0 to 1) */
  onProgress?: (progress: number) => void
}

/** Message sent to the Ripser Web Worker. */
export interface WorkerRequest {
  type: 'compute'
  distanceMatrix: Float64Array
  maxDim: number
  threshold?: number
}

/** Message received from the Ripser Web Worker. */
export interface WorkerResponse {
  type: 'result' | 'error' | 'progress'
  result?: PersistenceResult
  error?: string
  progress?: number
}

/** Persistence statistics for a single dimension. */
export interface DimensionStats {
  count: number
  meanLifespan: number
  stdLifespan: number
  medianLifespan: number
  maxLifespan: number
  iqrLifespan: number
  entropy: number
}

/** Layout point for dead-space analysis. */
export interface LayoutPoint {
  x: number
  y: number
  width?: number
  depth?: number
  type?: string
}

/** Dead space detection result. */
export interface DeadSpaceResult {
  deadSpaces: Array<{
    birthRadius: number
    deathRadius: number
    persistence: number
    approxDiameterFt: number
    severity: 'high' | 'medium'
  }>
  coverageScore: number
  connectivityScore: number
  numPoints: number
}
