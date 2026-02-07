/**
 * Layout solver: greedy constraint-aware placement + simulated annealing optimization.
 *
 * Phase 1: Greedy placement with MRV heuristic (most constrained items first).
 * Phase 2: Simulated annealing to optimize soft objectives while maintaining feasibility.
 */

import type {
  LayoutRequest, LayoutResult, Placement, FurnitureSpec,
  RoomConfig, SolverOptions, ObjectiveWeights, Point2D,
} from './types'
import { LayoutGrid } from './grid'
import { validateLayout } from './constraints'
import { scoreLayout, DEFAULT_WEIGHTS } from './objectives'

// ─── Defaults ───────────────────────────────────────────────────────────────

const DEFAULT_OPTIONS: Required<SolverOptions> = {
  gridCellSize: 0.15,
  minAisleWidth: 0.914, // 36" ADA
  exitClearance: 1.12,  // 44" fire code
  annealingIterations: 2000,
  annealingInitialTemp: 10,
  annealingCoolingRate: 0.995,
  maxPlacementAttempts: 200,
}

// ─── Deterministic PRNG ─────────────────────────────────────────────────────

/** Simple Mulberry32 PRNG for reproducible results. */
function createRng(seed: number) {
  let state = seed | 0
  return () => {
    state = (state + 0x6D2B79F5) | 0
    let t = Math.imul(state ^ (state >>> 15), 1 | state)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// ─── Effective Dimensions ───────────────────────────────────────────────────

/** Compute AABB dimensions after rotation (0 or 90 degrees). */
function effectiveDims(spec: FurnitureSpec, rotation: number): [number, number] {
  // Snap to nearest 90 degrees
  const r = Math.round(rotation / (Math.PI / 2)) % 2
  if (r === 0) return [spec.width, spec.depth]
  return [spec.depth, spec.width]
}

// ─── Placement helpers ──────────────────────────────────────────────────────

function makePlacement(
  spec: FurnitureSpec,
  specIndex: number,
  instanceIndex: number,
  x: number,
  z: number,
  rotation: number,
): Placement {
  const [ew, ed] = effectiveDims(spec, rotation)
  return {
    specIndex,
    instanceIndex,
    x,
    z,
    rotation,
    type: spec.type,
    effectiveWidth: ew,
    effectiveDepth: ed,
  }
}

// ─── Fixed zone placement ───────────────────────────────────────────────────

function getFixedZonePosition(
  zone: 'north' | 'south' | 'east' | 'west' | 'center',
  room: RoomConfig,
  halfW: number,
  halfD: number,
): Point2D {
  const margin = 0.3
  switch (zone) {
    case 'north': return { x: room.width / 2, z: halfD + margin }
    case 'south': return { x: room.width / 2, z: room.depth - halfD - margin }
    case 'east':  return { x: room.width - halfW - margin, z: room.depth / 2 }
    case 'west':  return { x: halfW + margin, z: room.depth / 2 }
    case 'center': return { x: room.width / 2, z: room.depth / 2 }
  }
}

// ─── Phase 1: Greedy Placement ──────────────────────────────────────────────

interface PlacementTask {
  specIndex: number
  instanceIndex: number
  spec: FurnitureSpec
  priority: number // lower = place first (MRV)
}

function buildTaskList(specs: FurnitureSpec[]): PlacementTask[] {
  const tasks: PlacementTask[] = []
  for (let si = 0; si < specs.length; si++) {
    const spec = specs[si]!
    for (let ii = 0; ii < spec.count; ii++) {
      // MRV: fixed-zone items first, then wall-adjacent, then large items, then small
      let priority = 100
      if (spec.fixedZone) priority = 0
      else if (spec.wallAdjacent) priority = 10
      else priority = 100 - (spec.width * spec.depth * 10) // larger items first
      tasks.push({ specIndex: si, instanceIndex: ii, spec, priority })
    }
  }
  tasks.sort((a, b) => a.priority - b.priority)
  return tasks
}

function greedyPlace(
  room: RoomConfig,
  specs: FurnitureSpec[],
  grid: LayoutGrid,
  opts: Required<SolverOptions>,
  rng: () => number,
): Placement[] {
  const placements: Placement[] = []
  const tasks = buildTaskList(specs)

  for (const task of tasks) {
    const { spec, specIndex, instanceIndex } = task
    const placed = tryPlaceItem(room, spec, specIndex, instanceIndex, grid, opts, rng)
    if (placed) {
      placements.push(placed)
      const hw = placed.effectiveWidth / 2
      const hd = placed.effectiveDepth / 2
      grid.occupy(placed.x, placed.z, hw, hd)
    }
  }

  return placements
}

function tryPlaceItem(
  room: RoomConfig,
  spec: FurnitureSpec,
  specIndex: number,
  instanceIndex: number,
  grid: LayoutGrid,
  opts: Required<SolverOptions>,
  rng: () => number,
): Placement | null {
  const rotations = [0, Math.PI / 2]

  // Fixed zone: try the specific position
  if (spec.fixedZone) {
    for (const rot of rotations) {
      const [ew, ed] = effectiveDims(spec, rot)
      const pos = getFixedZonePosition(spec.fixedZone, room, ew / 2, ed / 2)
      if (grid.canPlace(pos.x, pos.z, ew / 2, ed / 2)) {
        return makePlacement(spec, specIndex, instanceIndex, pos.x, pos.z, rot)
      }
    }
  }

  // Wall-adjacent: try along walls first
  if (spec.wallAdjacent) {
    for (const rot of rotations) {
      const [ew, ed] = effectiveDims(spec, rot)
      const hw = ew / 2
      const hd = ed / 2
      const margin = 0.3

      // Try each wall
      const wallPositions: Point2D[] = []
      for (let i = 0; i < opts.maxPlacementAttempts / 4; i++) {
        const t = rng()
        wallPositions.push(
          { x: hw + margin, z: hd + t * (room.depth - ed) },             // west
          { x: room.width - hw - margin, z: hd + t * (room.depth - ed) }, // east
          { x: hw + t * (room.width - ew), z: hd + margin },             // north
          { x: hw + t * (room.width - ew), z: room.depth - hd - margin }, // south
        )
      }

      for (const pos of wallPositions) {
        if (grid.canPlace(pos.x, pos.z, hw, hd)) {
          return makePlacement(spec, specIndex, instanceIndex, pos.x, pos.z, rot)
        }
      }
    }
  }

  // General placement: random positions with grid snapping
  for (let attempt = 0; attempt < opts.maxPlacementAttempts; attempt++) {
    const rot = rotations[Math.floor(rng() * rotations.length)]!
    const [ew, ed] = effectiveDims(spec, rot)
    const hw = ew / 2
    const hd = ed / 2

    // Random position within room bounds
    const x = hw + rng() * (room.width - ew)
    const z = hd + rng() * (room.depth - ed)

    // Snap to grid
    const sx = Math.round(x / opts.gridCellSize) * opts.gridCellSize
    const sz = Math.round(z / opts.gridCellSize) * opts.gridCellSize

    if (grid.canPlace(sx, sz, hw, hd) && grid.hasAisleClearance(sx, sz, hw, hd, opts.minAisleWidth)) {
      return makePlacement(spec, specIndex, instanceIndex, sx, sz, rot)
    }
  }

  return null // Could not place this item
}

// ─── Phase 2: Simulated Annealing ──────────────────────────────────────────

function simulatedAnnealing(
  room: RoomConfig,
  specs: FurnitureSpec[],
  placements: Placement[],
  grid: LayoutGrid,
  opts: Required<SolverOptions>,
  weights: ObjectiveWeights,
  rng: () => number,
): { placements: Placement[]; iterations: number } {
  if (placements.length < 2) return { placements, iterations: 0 }

  let current = [...placements]
  let currentScore = scoreLayout(room, specs, current, weights).total
  let best = [...current]
  let bestScore = currentScore
  let temp = opts.annealingInitialTemp
  let iterations = 0

  for (let i = 0; i < opts.annealingIterations; i++) {
    iterations++
    temp *= opts.annealingCoolingRate

    // Pick a random item to perturb
    const idx = Math.floor(rng() * current.length)
    const old = current[idx]!

    // Generate a neighbor: small random displacement or rotation
    const moveType = rng()
    let candidate: Placement

    if (moveType < 0.7) {
      // Positional perturbation (scaled by temperature)
      const dx = (rng() - 0.5) * temp * 0.2
      const dz = (rng() - 0.5) * temp * 0.2
      const nx = Math.max(old.effectiveWidth / 2, Math.min(room.width - old.effectiveWidth / 2, old.x + dx))
      const nz = Math.max(old.effectiveDepth / 2, Math.min(room.depth - old.effectiveDepth / 2, old.z + dz))
      // Snap
      const sx = Math.round(nx / opts.gridCellSize) * opts.gridCellSize
      const sz = Math.round(nz / opts.gridCellSize) * opts.gridCellSize
      candidate = { ...old, x: sx, z: sz }
    } else {
      // Rotation (90 degree flip)
      const newRot = old.rotation === 0 ? Math.PI / 2 : 0
      const spec = specs[old.specIndex]!
      const [ew, ed] = effectiveDims(spec, newRot)
      candidate = { ...old, rotation: newRot, effectiveWidth: ew, effectiveDepth: ed }
    }

    // Temporarily vacate old, check if new position is valid
    const hw = old.effectiveWidth / 2
    const hd = old.effectiveDepth / 2
    grid.vacate(old.x, old.z, hw, hd)

    const nhw = candidate.effectiveWidth / 2
    const nhd = candidate.effectiveDepth / 2

    if (grid.canPlace(candidate.x, candidate.z, nhw, nhd)) {
      // Evaluate new layout
      const trial = [...current]
      trial[idx] = candidate
      const trialViolations = validateLayout(room, trial, opts.minAisleWidth, opts.exitClearance)

      if (trialViolations.length === 0) {
        const trialScore = scoreLayout(room, specs, trial, weights).total

        // Acceptance criterion
        const delta = trialScore - currentScore
        if (delta > 0 || rng() < Math.exp(delta / Math.max(temp, 0.001))) {
          current = trial
          currentScore = trialScore
          grid.occupy(candidate.x, candidate.z, nhw, nhd)

          if (trialScore > bestScore) {
            best = [...trial]
            bestScore = trialScore
          }
          continue
        }
      }
    }

    // Revert: re-occupy old position
    grid.occupy(old.x, old.z, hw, hd)
  }

  return { placements: best, iterations }
}

// ─── Public API ─────────────────────────────────────────────────────────────

/**
 * Solve a layout problem: place furniture in a room satisfying all hard constraints
 * and optimizing soft objectives.
 */
export function solve(request: LayoutRequest): LayoutResult {
  const start = performance.now()
  const opts: Required<SolverOptions> = { ...DEFAULT_OPTIONS, ...request.options }
  const weights: ObjectiveWeights = { ...DEFAULT_WEIGHTS, ...request.objectives }
  const rng = createRng(42) // deterministic seed for reproducibility

  // Phase 1: Greedy placement
  const grid = new LayoutGrid(request.room, opts.gridCellSize)
  let placements = greedyPlace(request.room, request.furniture, grid, opts, rng)

  // Phase 2: Simulated annealing (only if we have multiple items)
  const annealResult = simulatedAnnealing(
    request.room, request.furniture, placements, grid, opts, weights, rng,
  )
  placements = annealResult.placements

  // Validate final layout
  const violations = validateLayout(request.room, placements, opts.minAisleWidth, opts.exitClearance)
  const scores = scoreLayout(request.room, request.furniture, placements, weights)
  const requestedCount = request.furniture.reduce((sum, s) => sum + s.count, 0)

  return {
    feasible: violations.length === 0,
    placements,
    scores,
    violations,
    stats: {
      solveTimeMs: performance.now() - start,
      placedCount: placements.length,
      requestedCount,
      annealingIterations: annealResult.iterations,
    },
  }
}

/**
 * Validate an existing layout against hard constraints.
 */
export function validate(
  room: RoomConfig,
  placements: Placement[],
  options?: Partial<SolverOptions>,
) {
  const opts = { ...DEFAULT_OPTIONS, ...options }
  const violations = validateLayout(room, placements, opts.minAisleWidth, opts.exitClearance)
  return { valid: violations.length === 0, violations }
}

/**
 * Score an existing layout's soft objectives.
 */
export function score(
  room: RoomConfig,
  specs: FurnitureSpec[],
  placements: Placement[],
  weights?: Partial<ObjectiveWeights>,
) {
  return scoreLayout(room, specs, placements, { ...DEFAULT_WEIGHTS, ...weights })
}
