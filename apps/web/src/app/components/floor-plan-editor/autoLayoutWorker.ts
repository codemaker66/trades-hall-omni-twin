/**
 * Web Worker for physics-based auto-layout optimization.
 *
 * Runs simulated annealing off the main thread to avoid UI jank.
 * Receives solver-format items + room boundary, posts back optimized positions.
 */
import {
  simulatedAnnealing,
  computeLayoutEnergy,
  generateLayoutNeighbor,
  itemsToState,
  stateToItems,
  createPRNG,
  CoolingSchedule,
  DEFAULT_WEIGHTS,
  type FurnitureItem,
  type RoomBoundary,
  type LayoutWeights,
  type SAConfig,
} from '@omni-twin/physics-solvers'

// ── Message types ────────────────────────────────────────────────────────────

export interface WorkerInput {
  items: FurnitureItem[]
  room: RoomBoundary
  weights: LayoutWeights
  targetCapacity: number
  seed: number
  maxIterations: number
}

export interface WorkerOutput {
  type: 'result'
  items: FurnitureItem[]
  bestEnergy: number
  iterations: number
}

export interface WorkerError {
  type: 'error'
  message: string
}

// ── Worker entry point ───────────────────────────────────────────────────────

self.onmessage = (e: MessageEvent<WorkerInput>) => {
  try {
    const { items, room, weights, targetCapacity, seed, maxIterations } = e.data

    // Reconstruct Float64Arrays from transferred data
    // (structured clone may convert them to regular arrays)
    const roomBoundary: RoomBoundary = {
      vertices: new Float64Array(room.vertices),
      exits: new Float64Array(room.exits),
      stageArea: room.stageArea ? new Float64Array(room.stageArea) : undefined,
      width: room.width,
      height: room.height,
    }

    const initialState = itemsToState(items)

    // Energy function: evaluates a state vector against the layout energy model
    const energyFn = (state: Float64Array): number => {
      const tempItems = stateToItems(state, items)
      return computeLayoutEnergy(tempItems, roomBoundary, weights, targetCapacity)
    }

    // Neighbor function: generates a perturbation
    const neighborFn = (state: Float64Array, rng: ReturnType<typeof createPRNG>): Float64Array => {
      const tempItems = stateToItems(state, items)
      const newItems = generateLayoutNeighbor(tempItems, rng)
      return itemsToState(newItems)
    }

    const config: SAConfig = {
      initialTemp: 100,
      finalTemp: 0.01,
      cooling: CoolingSchedule.Geometric,
      alpha: 0.995,
      maxIterations,
      reheatInterval: Math.floor(maxIterations / 5),
      reheatTempFraction: 0.4,
      seed,
    }

    const result = simulatedAnnealing(initialState, config, energyFn, neighborFn)

    const optimizedItems = stateToItems(result.bestState, items)

    const output: WorkerOutput = {
      type: 'result',
      items: optimizedItems,
      bestEnergy: result.bestEnergy,
      iterations: result.iterations,
    }

    self.postMessage(output)
  } catch (err) {
    const output: WorkerError = {
      type: 'error',
      message: err instanceof Error ? err.message : 'Unknown worker error',
    }
    self.postMessage(output)
  }
}
