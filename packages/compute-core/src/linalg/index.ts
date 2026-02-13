// ---------------------------------------------------------------------------
// @omni-twin/compute-core — HPC-12: Numerical Linear Algebra (barrel)
// ---------------------------------------------------------------------------

export {
  f64ToF32,
  f32ToF64,
  precisionLoss,
  isSafeForF32,
  estimateConditionNumber,
  needsDoublePrecision,
  mixedPrecisionSolve,
} from './mixed-precision.js';

export {
  analyzeMatrix,
  recommendSolver,
  estimateSolveTimeMs,
  selectPrecision,
} from './solver-select.js';

export {
  luFactorize,
  luSolve,
  iterativeRefinement,
  computeResidual,
  residualNorm,
  conjugateGradient,
} from './iterative-refine.js';
