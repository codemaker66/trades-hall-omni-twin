/**
 * Incremental Constraint Graph — the Jane Street killer feature.
 *
 * Wraps IncrementalGraph to model per-item constraint satisfaction.
 * When one item moves, only affected constraints re-evaluate via
 * cutoff propagation. If a constraint's status doesn't change,
 * the global violation node skips recomputation entirely.
 *
 * Architecture:
 * - One InputNode<Placement> per placement (position/rotation)
 * - One DerivedNode<Violation[]> per placement (local violations)
 * - One DerivedNode<Violation[]> for aggregated global violations
 * - One DerivedNode<number> for weighted total score
 */

import { IncrementalGraph, type InputNode, type DerivedNode } from '../incremental'
import type { Placement, RoomConfig, Violation, ObjectiveWeights, FurnitureSpec } from './types'
import type { SolverSpatialHash } from './spatial-hash'
import { validateSinglePlacement } from './constraints'
import { scoreLayout } from './objectives'

// ─── Equality Functions ────────────────────────────────────────────────────

/** Structural equality for Violation arrays — enables cutoff propagation. */
function violationsEqual(a: Violation[], b: Violation[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) {
    const va = a[i]!
    const vb = b[i]!
    if (va.type !== vb.type || va.message !== vb.message) return false
    if (va.placements.length !== vb.placements.length) return false
    for (let j = 0; j < va.placements.length; j++) {
      if (va.placements[j] !== vb.placements[j]) return false
    }
  }
  return true
}

/** Numeric equality with tolerance for score cutoff. */
const SCORE_EPSILON = 1e-10
function scoreEqual(a: number, b: number): boolean {
  return Math.abs(a - b) < SCORE_EPSILON
}

// ─── IncrementalConstraintGraph ────────────────────────────────────────────

export class IncrementalConstraintGraph {
  private graph: IncrementalGraph
  private placementInputs: InputNode<Placement>[]
  private localViolationNodes: DerivedNode<Violation[]>[]
  private globalViolationsNode: DerivedNode<Violation[]>
  private scoreNode: DerivedNode<number>

  private room: RoomConfig
  private specs: FurnitureSpec[]
  private spatialHash: SolverSpatialHash
  private weights: ObjectiveWeights
  private minAisle: number
  private exitClearance: number
  private currentPlacements: Placement[]

  constructor(
    room: RoomConfig,
    specs: FurnitureSpec[],
    placements: Placement[],
    spatialHash: SolverSpatialHash,
    weights: ObjectiveWeights,
    minAisle: number,
    exitClearance: number,
  ) {
    this.room = room
    this.specs = specs
    this.spatialHash = spatialHash
    this.weights = weights
    this.minAisle = minAisle
    this.exitClearance = exitClearance
    this.currentPlacements = [...placements]
    this.graph = new IncrementalGraph()

    // Create input nodes for each placement
    this.placementInputs = placements.map((p) => this.graph.input(p))

    // Create local violation nodes for each placement
    this.localViolationNodes = placements.map((_p, i) =>
      this.graph.derive(
        [this.placementInputs[i]!],
        () => validateSinglePlacement(
          this.room,
          this.currentPlacements,
          i,
          this.spatialHash,
          this.minAisle,
          this.exitClearance,
        ),
        violationsEqual,
      ),
    )

    // Global violations: aggregate all local violations
    this.globalViolationsNode = this.graph.derive(
      this.localViolationNodes,
      (...localArrays: Violation[][]) => {
        const all: Violation[] = []
        // Deduplicate: for pair violations (overlap, aisle), only include once
        const seen = new Set<string>()
        for (const locals of localArrays) {
          for (const v of locals) {
            const key = `${v.type}:${v.placements.sort().join(',')}`
            if (!seen.has(key)) {
              seen.add(key)
              all.push(v)
            }
          }
        }
        return all
      },
      violationsEqual,
    )

    // Score node: weighted total
    this.scoreNode = this.graph.derive(
      [this.globalViolationsNode],
      (violations: Violation[]) => {
        if (violations.length > 0) return -violations.length
        return scoreLayout(this.room, this.specs, this.currentPlacements, this.weights).total
      },
      scoreEqual,
    )
  }

  /** Update a single placement and rebuild affected constraints. */
  updatePlacement(index: number, newPlacement: Placement): void {
    this.currentPlacements[index] = newPlacement
    this.spatialHash.update(
      index, newPlacement.x, newPlacement.z,
      newPlacement.effectiveWidth / 2, newPlacement.effectiveDepth / 2,
    )
    this.graph.set(this.placementInputs[index]!, newPlacement)

    // Also mark neighbors dirty (their local violations may change)
    const hw = newPlacement.effectiveWidth / 2
    const hd = newPlacement.effectiveDepth / 2
    const expand = this.minAisle + 1
    const neighbors = this.spatialHash.queryAABB(
      newPlacement.x - hw - expand,
      newPlacement.z - hd - expand,
      newPlacement.x + hw + expand,
      newPlacement.z + hd + expand,
    )
    for (const ni of neighbors) {
      if (ni !== index && ni < this.placementInputs.length) {
        // Force re-evaluation by re-setting to same value (will mark dependents dirty)
        const input = this.placementInputs[ni]!
        this.graph.set(input, { ...this.currentPlacements[ni]! })
      }
    }
  }

  /** Stabilize the graph (propagate all pending changes). */
  stabilize(): void {
    this.graph.stabilize()
  }

  /** Read current violations. */
  get violations(): Violation[] {
    return this.globalViolationsNode.value
  }

  /** Read current total score. */
  get totalScore(): number {
    return this.scoreNode.value
  }

  /** Get recomputation count since last reset. */
  get recomputations(): number {
    return this.graph.totalRecomputations()
  }

  /** Reset counters for benchmarking. */
  resetCounters(): void {
    this.graph.resetCounters()
  }

  /** Current placements snapshot. */
  get placements(): Placement[] {
    return this.currentPlacements
  }
}
