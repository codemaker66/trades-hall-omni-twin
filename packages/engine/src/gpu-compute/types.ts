/**
 * Shared types for GPU compute spatial analysis.
 */

// ─── Geometry ───────────────────────────────────────────────────────────────

/** 2D axis-aligned bounding box. */
export interface AABB2D {
  minX: number
  minZ: number
  maxX: number
  maxZ: number
}

/** 2D point. */
export interface Point2D {
  x: number
  z: number
}

/** Room geometry for analysis. */
export interface RoomGeometry {
  width: number
  depth: number
  exits: Point2D[]
  walls: Array<{ a: Point2D; b: Point2D }>
}

/** Furniture item for analysis. */
export interface AnalysisItem {
  id: string
  x: number
  z: number
  halfWidth: number
  halfDepth: number
  isChair: boolean
}

// ─── Results ────────────────────────────────────────────────────────────────

/** Collision detection result. */
export interface CollisionResult {
  /** Pairs of colliding item indices. */
  pairs: Array<[number, number]>
  /** Total number of collisions. */
  count: number
  /** Per-item collision flags. */
  colliding: boolean[]
}

/** Sightline analysis result. */
export interface SightlineResult {
  /** Per-chair sightline score (0 = blocked, 1 = clear). */
  perChair: number[]
  /** Overall coverage: fraction of chairs with clear sightlines. */
  coverage: number
  /** Heatmap: per-grid-cell sightline quality (0-1). */
  heatmap: Float32Array
  heatmapWidth: number
  heatmapHeight: number
}

/** Crowd flow simulation result. */
export interface CrowdFlowResult {
  /** Per-agent evacuation time in seconds. */
  evacuationTimes: number[]
  /** Max evacuation time (bottleneck indicator). */
  maxTime: number
  /** Average evacuation time. */
  avgTime: number
  /** Density heatmap: per-grid-cell peak agent count. */
  densityHeatmap: Float32Array
  heatmapWidth: number
  heatmapHeight: number
  /** Bottleneck cells (density > threshold). */
  bottlenecks: Point2D[]
}

// ─── GPU Capabilities ───────────────────────────────────────────────────────

export interface GpuCapabilities {
  available: boolean
  adapterName: string
  maxBufferSize: number
  maxComputeWorkgroupSize: [number, number, number]
  maxStorageBuffersPerShaderStage: number
}
