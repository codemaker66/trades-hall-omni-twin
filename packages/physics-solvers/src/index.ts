/**
 * @omni-twin/physics-solvers
 *
 * Physics-inspired optimization suite for venue layout and scheduling.
 *
 * PS-1:  Simulated Annealing (3 cooling schedules + reheating)
 * PS-2:  Parallel Tempering / Replica Exchange Monte Carlo
 * PS-3:  QUBO + Ising + Potts spin glass models
 * PS-4:  Simulated Bifurcation (Toshiba SB, ballistic + discrete)
 * PS-5:  Complete venue layout energy function (8 terms)
 * PS-6:  Restricted Boltzmann Machine
 * PS-7:  MCMC Layout Sampling (MH + HMC)
 * PS-8:  NSGA-II Multi-Objective Optimization
 * PS-9:  CMA-ES Continuous Refinement
 * PS-10: MIP Scheduling (HiGHS + greedy fallback)
 * PS-11: Layout Generation (templates + LLM + diffusion stubs)
 * PS-12: Layered Solver Pipeline (orchestrator)
 */

// Types
export * from './types.js'

// PS-1: Simulated Annealing
export { simulatedAnnealing } from './sa.js'

// PS-2: Parallel Tempering
export { parallelTempering } from './parallel-tempering.js'

// PS-3: QUBO / Ising / Potts
export {
  buildSchedulingQUBO,
  quboToIsing,
  buildPottsScheduling,
  evaluateQUBO,
  evaluateIsing,
  solveQUBOSA,
  solveQUBOPT,
} from './qubo.js'

// PS-4: Simulated Bifurcation
export { simulatedBifurcation, simulatedBifurcationQUBO } from './simulated-bifurcation.js'

// PS-5: Layout Energy
export {
  computeLayoutEnergy,
  eOverlap, eAisle, eEgress, eSightline, eAda, eAesthetic, eService,
  generateLayoutNeighbor,
} from './energy/layout-energy.js'

// PS-6: RBM
export { RBM } from './rbm.js'

// PS-7: MCMC
export {
  sampleLayoutsMH,
  sampleLayoutsHMC,
  layoutDiversity,
  effectiveSampleSize,
} from './mcmc.js'

// PS-8: NSGA-II
export { nsga2 } from './nsga2.js'

// PS-9: CMA-ES
export { cmaes } from './cmaes.js'

// PS-10: MIP Scheduling
export {
  buildScheduleLP,
  solveScheduleGreedy,
  solveScheduleMIP,
  validateSchedule,
  computeScheduleEnergy,
} from './mip-scheduler.js'

// PS-11: Layout Generation
export {
  generateTemplateLayout,
  generateLayoutLLM,
  generateLayoutDiffusion,
  perturbLayout,
} from './layout-generation.js'
export type { LayoutStyle, LLMLayoutOptions, DiffusionLayoutOptions } from './layout-generation.js'

// PS-12: Orchestrator
export { VenueSolverPipeline, itemsToState, stateToItems } from './orchestrator.js'

// Schedule energy
export { computeScheduleEnergy as computeScheduleEnergyFn, scheduleNeighbor } from './energy/schedule-energy.js'
