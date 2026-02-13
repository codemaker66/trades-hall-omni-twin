// ---------------------------------------------------------------------------
// Gaussian Processes — Barrel Export
// ---------------------------------------------------------------------------

export {
  rbfKernel,
  matern32Kernel,
  matern52Kernel,
  periodicKernel,
  linearKernel,
  computeKernelMatrix,
  evaluateKernel,
  compositeKernel,
} from './kernel.js';

export { GPRegressor } from './gp-regression.js';

export {
  normalPDF,
  normalCDF,
  expectedImprovement,
  upperConfidenceBound,
  probabilityOfImprovement,
  bayesianOptimize,
} from './bayesian-opt.js';
