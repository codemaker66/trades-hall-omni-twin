/**
 * Core types for all physics-inspired solvers.
 *
 * Covers: SA, PT, Ising/Potts/QUBO, Simulated Bifurcation,
 * layout energy, RBM, MCMC, NSGA-II, CMA-ES, MIP scheduling,
 * and the orchestrator pipeline.
 */

// ---------------------------------------------------------------------------
// Random number generation
// ---------------------------------------------------------------------------

/** Seedable PRNG interface used throughout all solvers */
export interface PRNG {
  /** Returns a uniform random number in [0, 1) */
  random(): number
}

/** Simple mulberry32 PRNG for reproducible runs */
export function createPRNG(seed: number): PRNG {
  let s = seed | 0
  return {
    random() {
      s = (s + 0x6d2b79f5) | 0
      let t = Math.imul(s ^ (s >>> 15), 1 | s)
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296
    },
  }
}

// ---------------------------------------------------------------------------
// PS-1: Simulated Annealing
// ---------------------------------------------------------------------------

export enum CoolingSchedule {
  /** T(k+1) = alpha * T(k), alpha typically 0.95-0.999 */
  Geometric = 'geometric',
  /**
   * Targets 44% acceptance ratio, adjusts T using variance/mean of energy.
   * (Lam & Delosme, DAC 1988) — 2-24x speedup over fixed schedules.
   */
  LamDelosme = 'lam-delosme',
  /**
   * T_{k+1} = T_k * exp(-T_k * lambda / sigma_k)
   * Decreases faster when landscape is flat, slower near phase transitions.
   * (Huang, Romeo & Sangiovanni-Vincentelli, ICCAD 1986)
   */
  Huang = 'huang',
}

export interface SAConfig {
  initialTemp: number
  finalTemp: number
  cooling: CoolingSchedule
  /** Geometric cooling rate (0.95-0.999) */
  alpha: number
  maxIterations: number
  /** 0 = no reheating, else reheat every N iterations */
  reheatInterval: number
  /** Fraction of initial temp to reheat to (0.3-0.7) */
  reheatTempFraction: number
  seed?: number
}

export interface SAResult {
  bestEnergy: number
  bestState: Float64Array
  iterations: number
  accepts: number
  rejects: number
  reheats: number
  energyHistory: Float64Array
  tempHistory: Float64Array
}

export type EnergyFunction = (state: Float64Array) => number
export type NeighborFunction = (state: Float64Array, rng: PRNG) => Float64Array

// ---------------------------------------------------------------------------
// PS-2: Parallel Tempering
// ---------------------------------------------------------------------------

export enum TempSpacing {
  /**
   * T_i = T_min * (T_max/T_min)^((i-1)/(N-1))
   * Standard, works well when heat capacity is roughly constant.
   */
  Geometric = 'geometric',
  /**
   * Vousden, Farr & Mandel (arXiv:1501.05823)
   * Dynamically adjusts temperatures to equalize swap acceptance rates.
   * 1.2-5x efficiency improvement over geometric.
   */
  Adaptive = 'adaptive',
}

export interface PTConfig {
  nReplicas: number
  tMin: number
  tMax: number
  spacing: TempSpacing
  sweepsPerSwap: number
  totalSwaps: number
  seed?: number
}

export interface PTResult {
  bestEnergy: number
  bestState: Float64Array
  replicaEnergies: Float64Array
  swapAcceptanceRates: Float64Array
  energyTraces: Float64Array[]
}

// ---------------------------------------------------------------------------
// PS-3: QUBO / Ising / Potts
// ---------------------------------------------------------------------------

export interface EventSpec {
  id: string
  guests: number
  duration: number
  preferences: Record<string, number>
}

export interface RoomSpec {
  id: string
  capacity: number
  amenities: string[]
}

export interface TimeslotSpec {
  id: string
  start: number
  end: number
  day: number
}

/** Upper-triangular QUBO matrix Q: H = x^T Q x */
export interface QUBOMatrix {
  n: number
  /** Flattened upper-triangular entries [n*(n+1)/2] */
  data: Float64Array
}

/** Ising model: H = -1/2 sum J_ij s_i s_j - sum h_i s_i */
export interface IsingModel {
  n: number
  /** Coupling matrix [n*n] */
  couplings: Float64Array
  /** External field [n] */
  field: Float64Array
}

/** Potts model: s_i in {0,...,K-1} */
export interface PottsModel {
  n: number
  k: number
  /** Coupling [n*n] — energy bonus when s_i == s_j */
  couplings: Float64Array
  /** Local field [n*k] — bias for spin i to take value q */
  field: Float64Array
}

// ---------------------------------------------------------------------------
// PS-4: Simulated Bifurcation
// ---------------------------------------------------------------------------

export enum SBVariant {
  /** f(x) = x — smooth, good for structured problems */
  Ballistic = 'ballistic',
  /** f(x) = sign(x) — quasi-quantum tunneling, best overall */
  Discrete = 'discrete',
}

export interface SBConfig {
  variant: SBVariant
  nSteps: number
  dt: number
  pumpRate: number
  kerr: number
  seed?: number
}

// ---------------------------------------------------------------------------
// PS-5: Layout Energy Function
// ---------------------------------------------------------------------------

export enum ItemType {
  Chair = 'chair',
  RoundTable = 'round-table',
  RectTable = 'rect-table',
  Stage = 'stage',
  Bar = 'bar',
  Podium = 'podium',
  DanceFloor = 'dance-floor',
  AVBooth = 'av-booth',
  ServiceStation = 'service-station',
}

export interface FurnitureItem {
  x: number
  y: number
  width: number
  depth: number
  rotation: number
  itemType: ItemType
  seats: number
}

export interface RoomBoundary {
  /** Flattened polygon vertices [x1,y1, x2,y2, ...] */
  vertices: Float64Array
  /** Exit locations [x1,y1,w1, x2,y2,w2, ...] */
  exits: Float64Array
  /** Optional stage area polygon */
  stageArea?: Float64Array
  width: number
  height: number
}

export interface LayoutWeights {
  overlap: number
  aisle: number
  egress: number
  sightline: number
  capacity: number
  ada: number
  aesthetic: number
  service: number
}

export const DEFAULT_WEIGHTS: LayoutWeights = {
  overlap: 1e6,
  aisle: 1e4,
  egress: 1e6,
  sightline: 100,
  capacity: 1e4,
  ada: 1e6,
  aesthetic: 10,
  service: 50,
}

// ---------------------------------------------------------------------------
// PS-6: Restricted Boltzmann Machine
// ---------------------------------------------------------------------------

export interface RBMConfig {
  nVisible: number
  nHidden: number
  cdK: number
  learningRate: number
  epochs: number
  momentum: number
  weightDecay: number
  seed?: number
}

// ---------------------------------------------------------------------------
// PS-7: MCMC Layout Sampling
// ---------------------------------------------------------------------------

export interface MCMCConfig {
  temperature: number
  nSamples: number
  /** Keep every thin-th sample */
  thin: number
  /** Burn-in iterations to discard */
  burnIn: number
  seed?: number
}

export interface MCMCResult {
  samples: Float64Array[]
  energies: Float64Array
  acceptanceRate: number
}

// ---------------------------------------------------------------------------
// PS-8: NSGA-II
// ---------------------------------------------------------------------------

export enum CrossoverType {
  /** Partially Mapped Crossover */
  PMX = 'pmx',
  /** Simulated Binary Crossover */
  SBX = 'sbx',
  /** Uniform crossover */
  Uniform = 'uniform',
}

export interface NSGA2Config {
  populationSize: number
  generations: number
  crossoverRate: number
  mutationRate: number
  crossoverType: CrossoverType
  seed?: number
}

export interface ParetoSolution {
  state: Float64Array
  objectives: Float64Array
  frontRank: number
  crowdingDistance: number
}

export type ObjectiveFunction = (state: Float64Array) => Float64Array

// ---------------------------------------------------------------------------
// PS-9: CMA-ES
// ---------------------------------------------------------------------------

export interface CMAESConfig {
  initialSigma: number
  maxEvaluations: number
  seed?: number
}

// ---------------------------------------------------------------------------
// PS-10: MIP Scheduling
// ---------------------------------------------------------------------------

export interface ScheduleAssignment {
  eventId: string
  roomId: string
  timeslotId: string
}

export interface ScheduleResult {
  assignments: ScheduleAssignment[]
  objectiveValue: number
  feasible: boolean
  solveDurationMs: number
}

// ---------------------------------------------------------------------------
// PS-12: Orchestrator Pipeline
// ---------------------------------------------------------------------------

export interface Layout {
  items: FurnitureItem[]
  room: RoomBoundary
  energy: number
}

export interface PlanningRequest {
  description: string
  room: RoomBoundary
  events: EventSpec[]
  rooms: RoomSpec[]
  timeslots: TimeslotSpec[]
  weights: LayoutWeights
  targetCapacity: number
}

export interface PlanningResult {
  schedule: ScheduleResult
  optimized: Layout
  alternatives: Layout[]
  pareto: ParetoSolution[]
}

export interface SolverPipeline {
  generateInitial(description: string, room: RoomBoundary): Promise<Layout>
  scheduleEvents(events: EventSpec[], rooms: RoomSpec[], timeslots: TimeslotSpec[]): Promise<ScheduleResult>
  optimizeLayout(layout: Layout, weights: LayoutWeights, targetCapacity: number): Promise<Layout>
  sampleAlternatives(layout: Layout, weights: LayoutWeights, n: number): Promise<Layout[]>
  computeParetoFront(layout: Layout, room: RoomBoundary): Promise<ParetoSolution[]>
  runFullPipeline(request: PlanningRequest): Promise<PlanningResult>
}
