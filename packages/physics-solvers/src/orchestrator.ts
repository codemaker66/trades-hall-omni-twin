/**
 * PS-12: The Layered Solver Pipeline (Orchestrator)
 *
 * Five-layer optimization pipeline:
 *
 * Layer 1 — Initial generation:
 *   LayoutGPT for warm start from natural language description
 *   OR template selection based on event type + capacity
 *
 * Layer 2 — Hard constraint enforcement:
 *   HiGHS-js (WASM MIP) for scheduling (event-room-timeslot)
 *
 * Layer 3 — Soft optimization:
 *   Parallel tempering SA (multiple replicas, geometric temperature spacing)
 *   Energy function: PS-5 complete venue layout energy
 *
 * Layer 4 — Diversity sampling:
 *   MCMC at fixed temperature for 20-50 diverse layouts
 *
 * Layer 5 — Multi-objective trade-offs:
 *   NSGA-II for Pareto fronts (cost vs flow vs compliance)
 */

import type {
  SolverPipeline, PlanningRequest, PlanningResult,
  Layout, LayoutWeights, RoomBoundary, ScheduleResult,
  EventSpec, RoomSpec, TimeslotSpec, ParetoSolution,
  FurnitureItem,
} from './types.js'
import { DEFAULT_WEIGHTS, CoolingSchedule, TempSpacing, CrossoverType, createPRNG } from './types.js'
import { simulatedAnnealing } from './sa.js'
import { parallelTempering } from './parallel-tempering.js'
import { computeLayoutEnergy, generateLayoutNeighbor } from './energy/layout-energy.js'
import { solveScheduleMIP } from './mip-scheduler.js'
import { sampleLayoutsMH } from './mcmc.js'
import { nsga2 } from './nsga2.js'
import { cmaes } from './cmaes.js'
import { generateTemplateLayout, generateLayoutLLM } from './layout-generation.js'

// ---------------------------------------------------------------------------
// State <-> FurnitureItem[] conversion
// ---------------------------------------------------------------------------

/** Pack FurnitureItem[] into flat Float64Array: [x1,y1,rot1, x2,y2,rot2, ...] */
export function itemsToState(items: FurnitureItem[]): Float64Array {
  const state = new Float64Array(items.length * 3)
  for (let i = 0; i < items.length; i++) {
    const item = items[i]!
    state[i * 3] = item.x
    state[i * 3 + 1] = item.y
    state[i * 3 + 2] = item.rotation
  }
  return state
}

/** Unpack Float64Array back to FurnitureItem[], using template for width/depth/type/seats */
export function stateToItems(state: Float64Array, template: FurnitureItem[]): FurnitureItem[] {
  return template.map((item, i) => ({
    ...item,
    x: state[i * 3]!,
    y: state[i * 3 + 1]!,
    rotation: state[i * 3 + 2]!,
  }))
}

// ---------------------------------------------------------------------------
// Pipeline Implementation
// ---------------------------------------------------------------------------

export class VenueSolverPipeline implements SolverPipeline {
  /**
   * Layer 1: Generate initial layout from description.
   * Tries LLM generation, falls back to template-based.
   */
  async generateInitial(description: string, room: RoomBoundary): Promise<Layout> {
    const items = await generateLayoutLLM({ description, room })
    const energy = computeLayoutEnergy(items, room, DEFAULT_WEIGHTS, 0)
    return { items, room, energy }
  }

  /**
   * Layer 2: Schedule events to rooms/timeslots via MIP.
   */
  async scheduleEvents(
    events: EventSpec[],
    rooms: RoomSpec[],
    timeslots: TimeslotSpec[],
  ): Promise<ScheduleResult> {
    return solveScheduleMIP(events, rooms, timeslots)
  }

  /**
   * Layer 3: Optimize furniture layout via parallel tempering SA.
   * 8 replicas, geometric temperature spacing.
   */
  async optimizeLayout(
    layout: Layout,
    weights: LayoutWeights,
    targetCapacity: number,
  ): Promise<Layout> {
    const { items, room } = layout
    const initialState = itemsToState(items)

    // Energy function wrapping layout energy
    const energyFn = (state: Float64Array) => {
      const tempItems = stateToItems(state, items)
      return computeLayoutEnergy(tempItems, room, weights, targetCapacity)
    }

    // Neighbor function wrapping layout perturbation
    const neighborFn = (state: Float64Array, rng: ReturnType<typeof createPRNG>) => {
      const tempItems = stateToItems(state, items)
      const newItems = generateLayoutNeighbor(tempItems, rng)
      return itemsToState(newItems)
    }

    // Run parallel tempering
    const ptResult = parallelTempering(initialState, energyFn, neighborFn, {
      nReplicas: 8,
      tMin: 0.1,
      tMax: 100.0,
      spacing: TempSpacing.Geometric,
      sweepsPerSwap: 50,
      totalSwaps: 200,
      seed: 42,
    })

    // Refine best with CMA-ES for fine-tuning
    const refined = cmaes(ptResult.bestState, {
      initialSigma: 0.5,
      maxEvaluations: 1000,
      seed: 123,
    }, energyFn)

    const finalItems = stateToItems(refined.bestState, items)
    return {
      items: finalItems,
      room,
      energy: refined.bestEnergy,
    }
  }

  /**
   * Layer 4: Sample N diverse alternatives via MCMC at fixed temperature.
   */
  async sampleAlternatives(
    layout: Layout,
    weights: LayoutWeights,
    n: number,
  ): Promise<Layout[]> {
    const { items, room } = layout
    const state = itemsToState(items)

    const energyFn = (s: Float64Array) => {
      const tempItems = stateToItems(s, items)
      return computeLayoutEnergy(tempItems, room, weights, 0)
    }
    const neighborFn = (s: Float64Array, rng: ReturnType<typeof createPRNG>) => {
      const tempItems = stateToItems(s, items)
      return itemsToState(generateLayoutNeighbor(tempItems, rng))
    }

    const mcmcResult = sampleLayoutsMH(state, {
      temperature: 10.0,
      nSamples: n,
      thin: 20,
      burnIn: 500,
      seed: 42,
    }, energyFn, neighborFn)

    return mcmcResult.samples.map((sample, i) => ({
      items: stateToItems(sample, items),
      room,
      energy: mcmcResult.energies[i]!,
    }))
  }

  /**
   * Layer 5: Compute Pareto front for multi-objective trade-offs.
   * Objectives: (1) cost, (2) attendee flow, (3) compliance.
   */
  async computeParetoFront(layout: Layout, room: RoomBoundary): Promise<ParetoSolution[]> {
    const { items } = layout
    const state = itemsToState(items)

    // Create initial population (50 perturbations of current layout)
    const rng = createPRNG(42)
    const population: Float64Array[] = [state]
    for (let i = 1; i < 50; i++) {
      const perturbed = new Float64Array(state)
      for (let j = 0; j < perturbed.length; j++) {
        perturbed[j] = perturbed[j]! + (rng.random() - 0.5) * 2.0
      }
      population.push(perturbed)
    }

    // 3 objectives
    const objectiveFn = (s: Float64Array) => {
      const tempItems = stateToItems(s, items)
      const objectives = new Float64Array(3)

      // Objective 1: Cost (proportional to item count * distance from origin)
      let cost = 0
      for (const item of tempItems) {
        cost += Math.sqrt(item.x * item.x + item.y * item.y) * 0.1
      }
      objectives[0] = cost

      // Objective 2: Negative flow (higher avg distance between items = worse flow)
      let totalDist = 0
      let pairs = 0
      for (let i = 0; i < tempItems.length; i++) {
        for (let j = i + 1; j < tempItems.length; j++) {
          const dx = tempItems[i]!.x - tempItems[j]!.x
          const dy = tempItems[i]!.y - tempItems[j]!.y
          totalDist += Math.sqrt(dx * dx + dy * dy)
          pairs++
        }
      }
      objectives[1] = pairs > 0 ? totalDist / pairs : 0

      // Objective 3: Non-compliance (energy from safety terms)
      const safetyWeights = { ...DEFAULT_WEIGHTS, aesthetic: 0, service: 0, capacity: 0, sightline: 0 }
      objectives[2] = computeLayoutEnergy(tempItems, room, safetyWeights, 0)

      return objectives
    }

    return nsga2(population, objectiveFn, {
      populationSize: 50,
      generations: 100,
      crossoverRate: 0.9,
      mutationRate: 0.1,
      crossoverType: CrossoverType.SBX,
      seed: 42,
    })
  }

  /**
   * Run the full 5-layer pipeline.
   */
  async runFullPipeline(request: PlanningRequest): Promise<PlanningResult> {
    // Layer 1: Generate initial layout
    const initial = await this.generateInitial(request.description, request.room)

    // Layer 2: Schedule events
    const schedule = await this.scheduleEvents(request.events, request.rooms, request.timeslots)

    // Layer 3: Optimize layout
    const optimized = await this.optimizeLayout(initial, request.weights, request.targetCapacity)

    // Layer 4: Sample diverse alternatives
    const alternatives = await this.sampleAlternatives(optimized, request.weights, 30)

    // Layer 5: Compute Pareto front
    const pareto = await this.computeParetoFront(optimized, request.room)

    return { schedule, optimized, alternatives, pareto }
  }
}
