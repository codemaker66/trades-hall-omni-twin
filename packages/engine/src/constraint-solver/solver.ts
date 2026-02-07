/**
 * Layout solver: greedy constraint-aware placement + simulated annealing optimization.
 *
 * Phase 1: Greedy placement with MRV heuristic + limited backtracking.
 * Phase 2: Adaptive simulated annealing (room-scaled displacement, swap moves,
 *          early termination, random restarts) with spatial-hash acceleration.
 * Phase 3: Chair-table grouping (chairsPerUnit).
 *
 * Jane Street principles applied:
 * - Incremental validation via spatial hash (O(k) per iteration instead of O(n²))
 * - Room-adaptive displacement scaling (sqrt(w*d)/10)
 * - Three move types: translate (60%), rotate (20%), swap (20%)
 * - Convergence-based early termination (200-iteration window, 0.001 threshold)
 * - Random restarts from best known solution
 */

import type {
  LayoutRequest, LayoutResult, Placement, FurnitureSpec,
  RoomConfig, SolverOptions, ObjectiveWeights, Point2D,
} from './types'
import { LayoutGrid } from './grid'
import { validateLayout, validateSinglePlacement } from './constraints'
import { scoreLayout, DEFAULT_WEIGHTS } from './objectives'
import { SolverSpatialHash } from './spatial-hash'
import { placeChairGroups } from './chair-grouping'

// ─── Defaults ───────────────────────────────────────────────────────────────

const DEFAULT_OPTIONS: Required<SolverOptions> = {
  gridCellSize: 0.15,
  minAisleWidth: 0.914, // 36" ADA
  exitClearance: 1.12,  // 44" fire code
  annealingIterations: 2000,
  annealingInitialTemp: 10,
  annealingCoolingRate: 0.995,
  maxPlacementAttempts: 200,
  seed: 42,
  enableBacktracking: true,
  maxRestarts: 3,
}

/** Maximum backtrack attempts in greedy phase. */
const MAX_BACKTRACKS = 20

/** Convergence window for early termination (iterations). */
const CONVERGENCE_WINDOW = 200

/** Minimum score improvement to avoid convergence. */
const CONVERGENCE_THRESHOLD = 0.001

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

// ─── Phase 1: Greedy Placement with Backtracking ───────────────────────────

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
  spatialHash: SolverSpatialHash,
  opts: Required<SolverOptions>,
  rng: () => number,
): { placements: Placement[]; backtracks: number } {
  const placements: Placement[] = []
  const tasks = buildTaskList(specs)
  let backtracks = 0

  for (let ti = 0; ti < tasks.length; ti++) {
    const task = tasks[ti]!
    const { spec, specIndex, instanceIndex } = task
    const placed = tryPlaceItem(room, spec, specIndex, instanceIndex, grid, spatialHash, opts, rng)

    if (placed) {
      placements.push(placed)
      const hw = placed.effectiveWidth / 2
      const hd = placed.effectiveDepth / 2
      grid.occupy(placed.x, placed.z, hw, hd)
      spatialHash.insert(placements.length - 1, placed.x, placed.z, hw, hd)
    } else if (opts.enableBacktracking && backtracks < MAX_BACKTRACKS && placements.length > 0) {
      // Backtrack: remove last placed item and retry
      backtracks++
      const removed = placements.pop()!
      const rhw = removed.effectiveWidth / 2
      const rhd = removed.effectiveDepth / 2
      grid.vacate(removed.x, removed.z, rhw, rhd)
      spatialHash.remove(placements.length) // was at the end
      ti -= 2 // retry previous task (will be incremented by loop)
      if (ti < -1) ti = -1
    }
    // If no backtracking or max backtracks reached, skip this item
  }

  return { placements, backtracks }
}

function tryPlaceItem(
  room: RoomConfig,
  spec: FurnitureSpec,
  specIndex: number,
  instanceIndex: number,
  grid: LayoutGrid,
  spatialHash: SolverSpatialHash,
  opts: Required<SolverOptions>,
  rng: () => number,
): Placement | null {
  const rotations = [0, Math.PI / 2]

  // Fixed zone: try the specific position, then fall back to general placement
  if (spec.fixedZone) {
    for (const rot of rotations) {
      const [ew, ed] = effectiveDims(spec, rot)
      const pos = getFixedZonePosition(spec.fixedZone, room, ew / 2, ed / 2)
      if (grid.canPlace(pos.x, pos.z, ew / 2, ed / 2)) {
        return makePlacement(spec, specIndex, instanceIndex, pos.x, pos.z, rot)
      }
    }
    // Fall through to general placement if fixed zone is occupied
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

// ─── Phase 2: Adaptive Simulated Annealing ─────────────────────────────────

function simulatedAnnealing(
  room: RoomConfig,
  specs: FurnitureSpec[],
  placements: Placement[],
  grid: LayoutGrid,
  spatialHash: SolverSpatialHash,
  opts: Required<SolverOptions>,
  weights: ObjectiveWeights,
  rng: () => number,
): { placements: Placement[]; iterations: number; restarts: number } {
  if (placements.length < 2) return { placements, iterations: 0, restarts: 0 }

  // Room-adaptive displacement scaling
  const roomScale = Math.sqrt(room.width * room.depth) / 10
  const displacementFactor = roomScale * 0.05

  let current = [...placements]
  let currentScore = scoreLayout(room, specs, current, weights).total
  let best = [...current]
  let bestScore = currentScore
  let temp = opts.annealingInitialTemp
  let totalIterations = 0
  let restartCount = 0

  // Convergence tracking
  let windowBestScore = currentScore

  for (let restart = 0; restart <= opts.maxRestarts; restart++) {
    if (restart > 0) {
      // Restart from best known solution
      current = [...best]
      currentScore = bestScore
      temp = opts.annealingInitialTemp * (1 - restart / (opts.maxRestarts + 1))
      windowBestScore = currentScore
      restartCount++

      // Rebuild spatial hash for restarted layout
      spatialHash.buildFromPlacements(current)

      // Rebuild grid for restarted layout
      grid.rebuildOccupancy(room, current)
    }

    for (let i = 0; i < opts.annealingIterations; i++) {
      totalIterations++
      temp *= opts.annealingCoolingRate

      // Early termination: check convergence every CONVERGENCE_WINDOW iterations
      if (i > 0 && i % CONVERGENCE_WINDOW === 0) {
        const improvement = currentScore - windowBestScore
        if (Math.abs(improvement) < CONVERGENCE_THRESHOLD) break
        windowBestScore = currentScore
      }

      // Pick move type: translate (60%), rotate (20%), swap (20%)
      const moveRoll = rng()
      const idx = Math.floor(rng() * current.length)
      const old = current[idx]!

      let candidate: Placement

      if (moveRoll < 0.6) {
        // Positional perturbation (room-adaptive displacement)
        const dx = (rng() - 0.5) * temp * displacementFactor
        const dz = (rng() - 0.5) * temp * displacementFactor
        const nx = Math.max(old.effectiveWidth / 2, Math.min(room.width - old.effectiveWidth / 2, old.x + dx))
        const nz = Math.max(old.effectiveDepth / 2, Math.min(room.depth - old.effectiveDepth / 2, old.z + dz))
        // Snap to grid
        const sx = Math.round(nx / opts.gridCellSize) * opts.gridCellSize
        const sz = Math.round(nz / opts.gridCellSize) * opts.gridCellSize
        candidate = { ...old, x: sx, z: sz }
      } else if (moveRoll < 0.8) {
        // Rotation (90 degree flip)
        const newRot = old.rotation === 0 ? Math.PI / 2 : 0
        const spec = specs[old.specIndex]!
        const [ew, ed] = effectiveDims(spec, newRot)
        candidate = { ...old, rotation: newRot, effectiveWidth: ew, effectiveDepth: ed }
      } else {
        // Swap: exchange positions of two items
        const idx2 = Math.floor(rng() * current.length)
        if (idx2 === idx) continue // skip self-swap

        const other = current[idx2]!
        // Swap positions only (keep original rotations and specs)
        const cand1 = { ...old, x: other.x, z: other.z }
        const cand2 = { ...other, x: old.x, z: old.z }

        // Temporarily vacate both
        grid.vacate(old.x, old.z, old.effectiveWidth / 2, old.effectiveDepth / 2)
        grid.vacate(other.x, other.z, other.effectiveWidth / 2, other.effectiveDepth / 2)

        const canPlace1 = grid.canPlace(cand1.x, cand1.z, cand1.effectiveWidth / 2, cand1.effectiveDepth / 2)
        const canPlace2 = grid.canPlace(cand2.x, cand2.z, cand2.effectiveWidth / 2, cand2.effectiveDepth / 2)

        if (canPlace1 && canPlace2) {
          const trial = [...current]
          trial[idx] = cand1
          trial[idx2] = cand2

          // Update spatial hash
          spatialHash.update(idx, cand1.x, cand1.z, cand1.effectiveWidth / 2, cand1.effectiveDepth / 2)
          spatialHash.update(idx2, cand2.x, cand2.z, cand2.effectiveWidth / 2, cand2.effectiveDepth / 2)

          // Incremental validation: check both swapped items
          const v1 = validateSinglePlacement(room, trial, idx, spatialHash, opts.minAisleWidth, opts.exitClearance)
          const v2 = validateSinglePlacement(room, trial, idx2, spatialHash, opts.minAisleWidth, opts.exitClearance)

          if (v1.length === 0 && v2.length === 0) {
            const trialScore = scoreLayout(room, specs, trial, weights).total
            const delta = trialScore - currentScore
            if (delta > 0 || rng() < Math.exp(delta / Math.max(temp, 0.001))) {
              current = trial
              currentScore = trialScore
              grid.occupy(cand1.x, cand1.z, cand1.effectiveWidth / 2, cand1.effectiveDepth / 2)
              grid.occupy(cand2.x, cand2.z, cand2.effectiveWidth / 2, cand2.effectiveDepth / 2)
              if (trialScore > bestScore) {
                best = [...trial]
                bestScore = trialScore
              }
              continue
            }
          }

          // Revert spatial hash
          spatialHash.update(idx, old.x, old.z, old.effectiveWidth / 2, old.effectiveDepth / 2)
          spatialHash.update(idx2, other.x, other.z, other.effectiveWidth / 2, other.effectiveDepth / 2)
        }

        // Revert grid
        grid.occupy(old.x, old.z, old.effectiveWidth / 2, old.effectiveDepth / 2)
        grid.occupy(other.x, other.z, other.effectiveWidth / 2, other.effectiveDepth / 2)
        continue
      }

      // For translate/rotate moves: vacate old, try candidate
      const hw = old.effectiveWidth / 2
      const hd = old.effectiveDepth / 2
      grid.vacate(old.x, old.z, hw, hd)

      const nhw = candidate.effectiveWidth / 2
      const nhd = candidate.effectiveDepth / 2

      if (grid.canPlace(candidate.x, candidate.z, nhw, nhd)) {
        // Incremental validation: only check the moved item and its neighbors
        const trial = [...current]
        trial[idx] = candidate

        // Update spatial hash for the candidate position
        spatialHash.update(idx, candidate.x, candidate.z, nhw, nhd)
        const localViolations = validateSinglePlacement(
          room, trial, idx, spatialHash, opts.minAisleWidth, opts.exitClearance,
        )

        if (localViolations.length === 0) {
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

        // Revert spatial hash
        spatialHash.update(idx, old.x, old.z, hw, hd)
      }

      // Revert: re-occupy old position
      grid.occupy(old.x, old.z, hw, hd)
    }
  }

  return { placements: best, iterations: totalIterations, restarts: restartCount }
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
  const rng = createRng(opts.seed)

  // Initialize data structures
  const grid = new LayoutGrid(request.room, opts.gridCellSize)
  const spatialHash = new SolverSpatialHash(Math.max(2, opts.gridCellSize * 10))

  // Phase 1: Greedy placement with backtracking
  const greedyResult = greedyPlace(request.room, request.furniture, grid, spatialHash, opts, rng)
  let placements = greedyResult.placements

  // Phase 2: Simulated annealing
  const annealResult = simulatedAnnealing(
    request.room, request.furniture, placements, grid, spatialHash, opts, weights, rng,
  )
  placements = annealResult.placements

  // Phase 3: Chair-table grouping
  // Rebuild grid and spatial hash for final layout
  const finalGrid = new LayoutGrid(request.room, opts.gridCellSize)
  const finalSpatialHash = new SolverSpatialHash(Math.max(2, opts.gridCellSize * 10))
  for (let i = 0; i < placements.length; i++) {
    const p = placements[i]!
    finalGrid.occupy(p.x, p.z, p.effectiveWidth / 2, p.effectiveDepth / 2)
    finalSpatialHash.insertPlacement(i, p)
  }

  const { chairs, groupings } = placeChairGroups(
    request.room,
    request.furniture,
    placements,
    finalGrid,
    finalSpatialHash,
    { minAisle: opts.minAisleWidth, gridCellSize: opts.gridCellSize },
  )

  // Merge chair placements
  if (chairs.length > 0) {
    placements = [...placements, ...chairs]
  }

  // Validate final layout
  const violations = validateLayout(request.room, placements, opts.minAisleWidth, opts.exitClearance)
  const scores = scoreLayout(request.room, request.furniture, placements, weights)
  const requestedCount = request.furniture.reduce((sum, s) => sum + s.count, 0)

  return {
    feasible: violations.length === 0,
    placements,
    scores,
    violations,
    groupings,
    stats: {
      solveTimeMs: performance.now() - start,
      placedCount: placements.length,
      requestedCount,
      annealingIterations: annealResult.iterations,
      restarts: annealResult.restarts,
      backtracks: greedyResult.backtracks,
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
