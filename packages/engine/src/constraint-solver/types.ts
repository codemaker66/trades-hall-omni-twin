/**
 * Types for the constraint-based layout solver (T1).
 *
 * Follows Jane Street "make illegal states unrepresentable" principle:
 * branded types, discriminated unions, and exhaustive matching throughout.
 */

// ─── Branded Types ──────────────────────────────────────────────────────────

declare const _validated: unique symbol

/**
 * A layout that has passed full hard-constraint validation.
 * Can only be produced via `markValidated()` — prevents passing
 * unvalidated layouts where validated ones are required.
 */
export type ValidatedLayout = Placement[] & { readonly [_validated]: true }

/** Brand a layout as validated (only call after validation passes). */
export function markValidated(placements: Placement[]): ValidatedLayout {
  return placements as ValidatedLayout
}

// ─── Violation Types ────────────────────────────────────────────────────────

export const VIOLATION_TYPES = [
  'overlap',
  'out-of-bounds',
  'aisle-too-narrow',
  'exit-blocked',
  'obstacle-overlap',
] as const

export type ViolationType = (typeof VIOLATION_TYPES)[number]

/** Exhaustive check — compile error if a new ViolationType is added without handling. */
export function assertNeverViolation(type: never): never {
  throw new Error(`Unhandled violation type: ${String(type)}`)
}

// ─── Cell States ────────────────────────────────────────────────────────────

export const CellState = {
  EMPTY: 0,
  WALL: 1,
  OBSTACLE: 2,
  OCCUPIED: 3,
  EXIT_ZONE: 4,
} as const

export type CellStateValue = (typeof CellState)[keyof typeof CellState]

// ─── Cardinal Rotation ──────────────────────────────────────────────────────

/** 0 = 0°, 1 = 90°, 2 = 180°, 3 = 270°. */
export type CardinalRotation = 0 | 1 | 2 | 3

/** Convert arbitrary radians to nearest cardinal rotation. */
export function toCardinalRotation(radians: number): CardinalRotation {
  const TWO_PI = Math.PI * 2
  const normalized = ((radians % TWO_PI) + TWO_PI) % TWO_PI
  return (Math.round(normalized / (Math.PI / 2)) % 4) as CardinalRotation
}

/** Convert cardinal rotation back to radians. */
export function cardinalToRadians(r: CardinalRotation): number {
  return r * (Math.PI / 2)
}

// ─── Solver Phase (discriminated union) ─────────────────────────────────────

export type SolverPhase =
  | { phase: 'greedy'; placed: number; total: number }
  | { phase: 'annealing'; iteration: number; temperature: number; bestScore: number }
  | { phase: 'complete'; result: LayoutResult }

// ─── Room ───────────────────────────────────────────────────────────────────

/** A 2D point in room-local coordinates (meters). */
export interface Point2D {
  x: number
  z: number
}

/** An axis-aligned rectangle obstacle or zone. */
export interface Rect {
  x: number
  z: number
  width: number
  depth: number
}

/** Exit/door definition. */
export interface Exit {
  position: Point2D
  /** Width of the exit in meters. */
  width: number
  /** Direction the exit faces (in radians from +X axis). */
  facing: number
}

/** Room configuration — the space to fill. */
export interface RoomConfig {
  /** Room width in meters (X axis). */
  width: number
  /** Room depth in meters (Z axis). */
  depth: number
  /** Exit locations. */
  exits: Exit[]
  /** Obstacle zones (columns, fixtures, etc.) that cannot be used. */
  obstacles: Rect[]
  /** Optional focal point (stage, podium) for sightline scoring. */
  focalPoint?: Point2D
}

// ─── Furniture ──────────────────────────────────────────────────────────────

/** Furniture type identifier. */
export type SolverFurnitureType =
  | 'chair'
  | 'round-table'
  | 'rect-table'
  | 'trestle-table'
  | 'podium'
  | 'stage'
  | 'bar'

/** Specification for a type of furniture to place. */
export interface FurnitureSpec {
  type: SolverFurnitureType
  /** Width in meters (X axis extent). */
  width: number
  /** Depth in meters (Z axis extent). */
  depth: number
  /** How many to place. */
  count: number
  /** Number of chairs to associate around each table (0 for non-tables). */
  chairsPerUnit: number
  /** If true, must be placed against a wall. */
  wallAdjacent?: boolean
  /** Optional: fix to a specific zone (e.g., "stage at north wall"). */
  fixedZone?: 'north' | 'south' | 'east' | 'west' | 'center'
}

// ─── Placement ──────────────────────────────────────────────────────────────

/** A placed furniture item in the solution. */
export interface Placement {
  /** Reference back to the furniture spec index. */
  specIndex: number
  /** Instance index (0..count-1). */
  instanceIndex: number
  /** Center X position in meters. */
  x: number
  /** Center Z position in meters. */
  z: number
  /** Rotation in radians. */
  rotation: number
  /** Furniture type (for convenience). */
  type: SolverFurnitureType
  /** Width after rotation consideration (for AABB). */
  effectiveWidth: number
  /** Depth after rotation consideration (for AABB). */
  effectiveDepth: number
}

// ─── Constraints & Objectives ───────────────────────────────────────────────

/** Hard constraint violation. */
export interface Violation {
  type: ViolationType
  message: string
  /** Indices of placements involved. */
  placements: number[]
}

/** Table-to-chair grouping produced by the solver. */
export interface TableGrouping {
  /** Index of the table placement in the placements array. */
  tableIndex: number
  /** Indices of chair placements grouped around this table. */
  chairIndices: number[]
  /** How many chairs were requested per table. */
  chairsPerUnit: number
}

/** Soft objective weights (0-1, sum not required to be 1). */
export interface ObjectiveWeights {
  /** Minimize wasted space. */
  spaceUtilization: number
  /** Maximize sightline coverage to focal point. */
  sightlineCoverage: number
  /** Maximize symmetry along primary axis. */
  symmetry: number
  /** Minimize max distance to nearest exit. */
  exitAccess: number
}

/** Scored layout quality metrics. */
export interface LayoutScores {
  /** 0-1: fraction of placed items vs requested. */
  capacityUtilization: number
  /** 0-1: fraction of floor area used. */
  spaceUtilization: number
  /** 0-1: fraction of seats with unobstructed sightline. */
  sightlineCoverage: number
  /** 0-1: mirror symmetry score. */
  symmetry: number
  /** 0-1: exit accessibility (inverse of max distance). */
  exitAccess: number
  /** Weighted total score. */
  total: number
}

// ─── Solver I/O ─────────────────────────────────────────────────────────────

/** Full layout request. */
export interface LayoutRequest {
  room: RoomConfig
  furniture: FurnitureSpec[]
  objectives?: Partial<ObjectiveWeights>
  options?: SolverOptions
}

/** Solver tuning options. */
export interface SolverOptions {
  /** Grid cell size in meters (default 0.15 = ~6 inches). */
  gridCellSize?: number
  /** Minimum aisle width in meters (default 0.914 = 36 inches ADA). */
  minAisleWidth?: number
  /** Exit clearance zone in meters (default 1.12 = 44 inches fire code). */
  exitClearance?: number
  /** Simulated annealing iterations (default 2000). */
  annealingIterations?: number
  /** Annealing initial temperature (default 10). */
  annealingInitialTemp?: number
  /** Annealing cooling rate (default 0.995). */
  annealingCoolingRate?: number
  /** Maximum placement attempts per item in greedy phase (default 200). */
  maxPlacementAttempts?: number
  /** PRNG seed for deterministic results (default 42). */
  seed?: number
  /** Enable limited backtracking in greedy phase (default true). */
  enableBacktracking?: boolean
  /** Number of SA restarts from best known solution (default 3). */
  maxRestarts?: number
}

/** Solver result. */
export interface LayoutResult {
  /** Whether a feasible layout was found. */
  feasible: boolean
  /** Placed items. */
  placements: Placement[]
  /** Layout quality scores. */
  scores: LayoutScores
  /** Hard constraint violations (empty if feasible). */
  violations: Violation[]
  /** Table-to-chair groupings (empty if no chairsPerUnit). */
  groupings: TableGrouping[]
  /** Solver statistics. */
  stats: {
    /** Total solver time in ms. */
    solveTimeMs: number
    /** Number of items successfully placed. */
    placedCount: number
    /** Number of items requested. */
    requestedCount: number
    /** Simulated annealing iterations run. */
    annealingIterations: number
    /** Number of SA restarts performed. */
    restarts: number
    /** Number of backtracks in greedy phase. */
    backtracks: number
  }
}

/** Validation-only result. */
export interface ValidationResult {
  valid: boolean
  violations: Violation[]
}
