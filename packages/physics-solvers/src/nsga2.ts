/**
 * PS-8: NSGA-II Multi-Objective Optimization
 *
 * Implements the Non-dominated Sorting Genetic Algorithm II (Deb et al. 2002)
 * with SBX crossover, uniform crossover, polynomial mutation, crowding
 * distance assignment, and deterministic tournament selection.
 *
 * Reference: Deb, Pratap, Agarwal & Meyarivan,
 *   "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II",
 *   IEEE Transactions on Evolutionary Computation, 6(2), 2002.
 */

import type { NSGA2Config, ParetoSolution, ObjectiveFunction, PRNG } from './types.js'
import { CrossoverType, createPRNG } from './types.js'

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

interface Individual {
  state: Float64Array
  objectives: Float64Array
  frontRank: number
  crowdingDistance: number
}

// ---------------------------------------------------------------------------
// Non-dominated sorting (fast approach from Deb et al. 2002)
// ---------------------------------------------------------------------------

/**
 * Returns true if solution `a` dominates solution `b`, i.e.
 * a is no worse on every objective AND strictly better on at least one.
 * (All objectives are minimized.)
 */
function dominates(a: Float64Array, b: Float64Array): boolean {
  let strictlyBetter = false
  for (let m = 0; m < a.length; m++) {
    const va = a[m]!
    const vb = b[m]!
    if (va > vb) return false
    if (va < vb) strictlyBetter = true
  }
  return strictlyBetter
}

/**
 * Performs non-dominated sorting on the population.
 * Returns an array of fronts, where fronts[0] is the Pareto-optimal front.
 * Each individual's `frontRank` is updated in place.
 */
function nonDominatedSort(population: Individual[]): Individual[][] {
  const n = population.length
  const dominated: number[][] = new Array<number[]>(n)
  const dominationCount = new Int32Array(n)

  for (let i = 0; i < n; i++) {
    dominated[i] = []
  }

  const fronts: Individual[][] = []
  const firstFront: Individual[] = []

  for (let p = 0; p < n; p++) {
    const pObj = population[p]!.objectives
    for (let q = p + 1; q < n; q++) {
      const qObj = population[q]!.objectives
      if (dominates(pObj, qObj)) {
        dominated[p]!.push(q)
        dominationCount[q] = (dominationCount[q] ?? 0) + 1
      } else if (dominates(qObj, pObj)) {
        dominated[q]!.push(p)
        dominationCount[p] = (dominationCount[p] ?? 0) + 1
      }
    }
  }

  for (let i = 0; i < n; i++) {
    if (dominationCount[i] === 0) {
      population[i]!.frontRank = 0
      firstFront.push(population[i]!)
    }
  }

  fronts.push(firstFront)

  let currentFrontIdx = 0
  while (fronts[currentFrontIdx]!.length > 0) {
    const nextFront: Individual[] = []
    for (const p of fronts[currentFrontIdx]!) {
      const pIdx = population.indexOf(p)
      for (const qIdx of dominated[pIdx]!) {
        dominationCount[qIdx] = (dominationCount[qIdx] ?? 0) - 1
        if (dominationCount[qIdx]! === 0) {
          population[qIdx]!.frontRank = currentFrontIdx + 1
          nextFront.push(population[qIdx]!)
        }
      }
    }
    if (nextFront.length === 0) break
    fronts.push(nextFront)
    currentFrontIdx++
  }

  return fronts
}

// ---------------------------------------------------------------------------
// Crowding distance assignment (Deb et al. 2002, Section III-B)
// ---------------------------------------------------------------------------

/**
 * Assigns crowding distances to individuals within a single front.
 * Individuals at the boundary of objective space get Infinity.
 */
function assignCrowdingDistance(front: Individual[]): void {
  const n = front.length
  if (n <= 2) {
    for (const ind of front) ind.crowdingDistance = Infinity
    return
  }

  for (const ind of front) ind.crowdingDistance = 0

  const nObj = front[0]!.objectives.length
  for (let m = 0; m < nObj; m++) {
    // Sort by objective m
    front.sort((a, b) => a.objectives[m]! - b.objectives[m]!)

    // Boundary solutions get infinite distance
    front[0]!.crowdingDistance = Infinity
    front[n - 1]!.crowdingDistance = Infinity

    const fMin = front[0]!.objectives[m]!
    const fMax = front[n - 1]!.objectives[m]!
    const range = fMax - fMin
    if (range === 0) continue

    for (let i = 1; i < n - 1; i++) {
      front[i]!.crowdingDistance += (front[i + 1]!.objectives[m]! - front[i - 1]!.objectives[m]!) / range
    }
  }
}

// ---------------------------------------------------------------------------
// Crossover operators
// ---------------------------------------------------------------------------

/**
 * Simulated Binary Crossover (SBX) — Deb & Agrawal 1995.
 * Distribution index eta_c = 20 (commonly used).
 */
function sbxCrossover(
  p1: Float64Array,
  p2: Float64Array,
  rng: PRNG,
): [Float64Array, Float64Array] {
  const n = p1.length
  const c1 = new Float64Array(n)
  const c2 = new Float64Array(n)
  const etaC = 20

  for (let i = 0; i < n; i++) {
    if (rng.random() < 0.5) {
      // SBX for this variable
      const u = rng.random()
      let beta: number
      if (u <= 0.5) {
        beta = Math.pow(2 * u, 1 / (etaC + 1))
      } else {
        beta = Math.pow(1 / (2 * (1 - u)), 1 / (etaC + 1))
      }
      c1[i] = 0.5 * ((1 + beta) * p1[i]! + (1 - beta) * p2[i]!)
      c2[i] = 0.5 * ((1 - beta) * p1[i]! + (1 + beta) * p2[i]!)
    } else {
      c1[i] = p1[i]!
      c2[i] = p2[i]!
    }
  }
  return [c1, c2]
}

/**
 * Uniform crossover — each gene independently from either parent.
 */
function uniformCrossover(
  p1: Float64Array,
  p2: Float64Array,
  rng: PRNG,
): [Float64Array, Float64Array] {
  const n = p1.length
  const c1 = new Float64Array(n)
  const c2 = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    if (rng.random() < 0.5) {
      c1[i] = p1[i]!
      c2[i] = p2[i]!
    } else {
      c1[i] = p2[i]!
      c2[i] = p1[i]!
    }
  }
  return [c1, c2]
}

// ---------------------------------------------------------------------------
// Polynomial mutation (Deb & Goyal 1996)
// ---------------------------------------------------------------------------

/**
 * Polynomial mutation with distribution index eta_m = 20.
 * Each gene has independent probability `rate` of being mutated.
 */
function polynomialMutation(
  state: Float64Array,
  rate: number,
  rng: PRNG,
): Float64Array {
  const result = new Float64Array(state.length)
  const etaM = 20

  for (let i = 0; i < state.length; i++) {
    if (rng.random() < rate) {
      const u = rng.random()
      let delta: number
      if (u < 0.5) {
        delta = Math.pow(2 * u, 1 / (etaM + 1)) - 1
      } else {
        delta = 1 - Math.pow(2 * (1 - u), 1 / (etaM + 1))
      }
      result[i] = state[i]! + delta
    } else {
      result[i] = state[i]!
    }
  }
  return result
}

// ---------------------------------------------------------------------------
// Tournament selection
// ---------------------------------------------------------------------------

/**
 * Binary tournament: pick two random individuals, return the one with
 * lower front rank (or higher crowding distance if same rank).
 */
function binaryTournament(
  population: Individual[],
  rng: PRNG,
): Individual {
  const n = population.length
  const i = Math.floor(rng.random() * n)
  const j = Math.floor(rng.random() * n)
  const a = population[i]!
  const b = population[j]!

  if (a.frontRank < b.frontRank) return a
  if (b.frontRank < a.frontRank) return b
  // Same front — prefer higher crowding distance
  return a.crowdingDistance >= b.crowdingDistance ? a : b
}

// ---------------------------------------------------------------------------
// Main NSGA-II
// ---------------------------------------------------------------------------

/**
 * NSGA-II multi-objective optimizer.
 *
 * @param initialPopulation Array of starting state vectors
 * @param objectiveFn       Maps state → Float64Array of objectives (all minimized)
 * @param config            NSGA-II configuration
 * @returns                 Pareto-optimal solutions from the final population
 */
export function nsga2(
  initialPopulation: Float64Array[],
  objectiveFn: ObjectiveFunction,
  config: NSGA2Config,
): ParetoSolution[] {
  const {
    populationSize,
    generations,
    crossoverRate,
    mutationRate,
    crossoverType,
    seed = 42,
  } = config

  const rng = createPRNG(seed)

  // Crossover function selection
  const crossover = crossoverType === CrossoverType.SBX ? sbxCrossover : uniformCrossover

  // Initialize population
  let population: Individual[] = []
  for (let i = 0; i < populationSize; i++) {
    const state = i < initialPopulation.length
      ? new Float64Array(initialPopulation[i]!)
      : (() => {
          // Random initialization using first state's dimensions
          const dim = initialPopulation[0]!.length
          const s = new Float64Array(dim)
          for (let d = 0; d < dim; d++) {
            s[d] = rng.random() * 2 - 1
          }
          return s
        })()
    const objectives = objectiveFn(state)
    population.push({
      state,
      objectives,
      frontRank: 0,
      crowdingDistance: 0,
    })
  }

  // Initial non-dominated sorting + crowding
  const initFronts = nonDominatedSort(population)
  for (const front of initFronts) {
    assignCrowdingDistance(front)
  }

  // Main generational loop
  for (let gen = 0; gen < generations; gen++) {
    // Generate offspring population
    const offspring: Individual[] = []
    while (offspring.length < populationSize) {
      const p1 = binaryTournament(population, rng)
      const p2 = binaryTournament(population, rng)

      let c1State: Float64Array
      let c2State: Float64Array

      if (rng.random() < crossoverRate) {
        [c1State, c2State] = crossover(p1.state, p2.state, rng)
      } else {
        c1State = new Float64Array(p1.state)
        c2State = new Float64Array(p2.state)
      }

      // Mutation
      c1State = polynomialMutation(c1State, mutationRate, rng)
      c2State = polynomialMutation(c2State, mutationRate, rng)

      offspring.push({
        state: c1State,
        objectives: objectiveFn(c1State),
        frontRank: 0,
        crowdingDistance: 0,
      })
      if (offspring.length < populationSize) {
        offspring.push({
          state: c2State,
          objectives: objectiveFn(c2State),
          frontRank: 0,
          crowdingDistance: 0,
        })
      }
    }

    // Combine parent + offspring (2N)
    const combined = [...population, ...offspring]

    // Non-dominated sorting on combined
    const fronts = nonDominatedSort(combined)

    // Fill next population from fronts until full
    const nextPop: Individual[] = []
    for (const front of fronts) {
      assignCrowdingDistance(front)
      if (nextPop.length + front.length <= populationSize) {
        nextPop.push(...front)
      } else {
        // Partial fill: sort by crowding distance (descending), take what fits
        front.sort((a, b) => b.crowdingDistance - a.crowdingDistance)
        const remaining = populationSize - nextPop.length
        for (let i = 0; i < remaining; i++) {
          nextPop.push(front[i]!)
        }
        break
      }
    }

    population = nextPop
  }

  // Final sort to identify Pareto front
  const finalFronts = nonDominatedSort(population)
  for (const front of finalFronts) {
    assignCrowdingDistance(front)
  }

  // Return all solutions, sorted by front rank then crowding distance
  return population
    .sort((a, b) => {
      if (a.frontRank !== b.frontRank) return a.frontRank - b.frontRank
      return b.crowdingDistance - a.crowdingDistance
    })
    .map(ind => ({
      state: new Float64Array(ind.state),
      objectives: new Float64Array(ind.objectives),
      frontRank: ind.frontRank,
      crowdingDistance: ind.crowdingDistance,
    }))
}
