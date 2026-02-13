// ---------------------------------------------------------------------------
// Bayesian Optimization using GP Surrogate
// ---------------------------------------------------------------------------

import type { PRNG, BayesOptConfig, BayesOptResult, GPConfig } from '../types.js';
import { GPRegressor } from './gp-regression.js';

// ---------------------------------------------------------------------------
// Helper: Standard Normal PDF and CDF
// ---------------------------------------------------------------------------

/**
 * Standard normal probability density function.
 * phi(x) = (1/sqrt(2*pi)) * exp(-0.5 * x^2)
 */
export function normalPDF(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

/**
 * Standard normal cumulative distribution function.
 * Uses the Abramowitz & Stegun approximation (formula 7.1.26) via the
 * error function. Maximum error ~1.5e-7.
 *
 * Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
 */
export function normalCDF(x: number): number {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

/**
 * Error function approximation (Abramowitz & Stegun 7.1.26).
 * |epsilon| <= 1.5e-7
 */
function erf(x: number): number {
  const sign = x >= 0 ? 1 : -1;
  const a = Math.abs(x);

  // Constants
  const p = 0.3275911;
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;

  const t = 1 / (1 + p * a);
  const t2 = t * t;
  const t3 = t2 * t;
  const t4 = t3 * t;
  const t5 = t4 * t;

  const result = 1 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * Math.exp(-a * a);
  return sign * result;
}

// ---------------------------------------------------------------------------
// Acquisition Functions
// ---------------------------------------------------------------------------

/**
 * Expected Improvement (EI) acquisition function.
 *
 * EI(x) = (mu - f_best - xi) * Phi(Z) + sigma * phi(Z)
 * where Z = (mu - f_best - xi) / sigma
 *
 * When sigma == 0, returns max(mu - f_best - xi, 0).
 *
 * @param mean Predicted mean at x
 * @param std Predicted standard deviation at x
 * @param bestY Best observed value so far
 * @param xi Exploration-exploitation tradeoff parameter (>= 0)
 * @returns Expected improvement value
 */
export function expectedImprovement(
  mean: number,
  std: number,
  bestY: number,
  xi: number,
): number {
  if (std < 1e-12) {
    return Math.max(mean - bestY - xi, 0);
  }
  const z = (mean - bestY - xi) / std;
  return (mean - bestY - xi) * normalCDF(z) + std * normalPDF(z);
}

/**
 * Upper Confidence Bound (UCB) acquisition function.
 *
 * UCB(x) = mu + kappa * sigma
 *
 * @param mean Predicted mean at x
 * @param std Predicted standard deviation at x
 * @param kappa Exploration parameter (>= 0)
 * @returns UCB value
 */
export function upperConfidenceBound(
  mean: number,
  std: number,
  kappa: number,
): number {
  return mean + kappa * std;
}

/**
 * Probability of Improvement (PI) acquisition function.
 *
 * PI(x) = Phi((mu - f_best - xi) / sigma)
 *
 * When sigma == 0, returns 1 if mu > bestY + xi, else 0.
 *
 * @param mean Predicted mean at x
 * @param std Predicted standard deviation at x
 * @param bestY Best observed value so far
 * @param xi Exploration-exploitation tradeoff parameter (>= 0)
 * @returns Probability of improvement
 */
export function probabilityOfImprovement(
  mean: number,
  std: number,
  bestY: number,
  xi: number,
): number {
  if (std < 1e-12) {
    return mean > bestY + xi ? 1 : 0;
  }
  const z = (mean - bestY - xi) / std;
  return normalCDF(z);
}

// ---------------------------------------------------------------------------
// Bayesian Optimization Loop
// ---------------------------------------------------------------------------

/**
 * Full Bayesian Optimization loop.
 *
 * Algorithm:
 *   1. Evaluate objective at nInitial random points
 *   2. For each iteration:
 *      a. Fit a GP to observed (x, y) pairs
 *      b. Maximize the acquisition function over a grid of candidates
 *      c. Evaluate the objective at the best candidate
 *      d. Update observations
 *   3. Return the best observed point
 *
 * This is a 1D optimizer (operates on the first dimension of bounds).
 * For multi-dimensional problems, bounds[0] is used.
 *
 * @param objectiveFn The black-box function to optimize
 * @param config Bayesian optimization configuration
 * @param rng Seedable PRNG
 * @returns Best point found and optimization history
 */
export function bayesianOptimize(
  objectiveFn: (x: number[]) => number,
  config: BayesOptConfig,
  rng: PRNG,
): BayesOptResult {
  const nDims = config.bounds.length;
  const history: Array<{ x: number[]; y: number; acquisition: number }> = [];

  const xObs: number[][] = [];
  const yObs: number[] = [];

  // --- Phase 1: Initial random evaluations ---
  for (let i = 0; i < config.nInitial; i++) {
    const x: number[] = new Array<number>(nDims);
    for (let d = 0; d < nDims; d++) {
      const bound = config.bounds[d]!;
      const lo = bound[0];
      const hi = bound[1];
      x[d] = lo + rng() * (hi - lo);
    }
    const y = objectiveFn(x);
    xObs.push(x);
    yObs.push(y);
    history.push({ x: [...x], y, acquisition: 0 });
  }

  // --- Phase 2: GP-based optimization iterations ---
  // Default GP config for the surrogate
  const gpConfig: GPConfig = {
    kernel: { type: 'matern52', lengthscale: 1, variance: 1 },
    noiseVariance: 1e-6,
    meanFunction: 'zero',
  };

  // Number of candidate points for acquisition maximization
  const nCandidates = 200;

  for (let iter = 0; iter < config.maxIterations; iter++) {
    // For 1D GP surrogate, project all data to first dimension
    const xTrain1D: number[] = new Array<number>(xObs.length);
    for (let j = 0; j < xObs.length; j++) {
      xTrain1D[j] = xObs[j]![0]!;
    }

    // Estimate lengthscale from data range
    let xMin = Infinity;
    let xMax = -Infinity;
    for (let j = 0; j < xTrain1D.length; j++) {
      const v = xTrain1D[j] ?? 0;
      if (v < xMin) xMin = v;
      if (v > xMax) xMax = v;
    }
    const dataRange = Math.max(xMax - xMin, 1e-6);
    gpConfig.kernel = {
      type: 'matern52',
      lengthscale: dataRange / 3,
      variance: 1,
    };

    // Fit GP
    const gp = new GPRegressor(gpConfig);
    gp.fit(xTrain1D, yObs);

    // Find best observed y
    let bestY = -Infinity;
    for (let j = 0; j < yObs.length; j++) {
      const yj = yObs[j] ?? -Infinity;
      if (yj > bestY) bestY = yj;
    }

    // Generate candidate points (random + grid)
    const candidates: number[][] = [];
    for (let c = 0; c < nCandidates; c++) {
      const x: number[] = new Array<number>(nDims);
      for (let d = 0; d < nDims; d++) {
        const bound = config.bounds[d]!;
        const lo = bound[0];
        const hi = bound[1];
        x[d] = lo + rng() * (hi - lo);
      }
      candidates.push(x);
    }

    // Predict at all candidates (1D projection for GP)
    const candidateX1D: number[] = new Array<number>(nCandidates);
    for (let c = 0; c < nCandidates; c++) {
      candidateX1D[c] = candidates[c]![0]!;
    }
    const pred = gp.predict(candidateX1D, true);

    // Evaluate acquisition function at each candidate
    let bestAcq = -Infinity;
    let bestCandidateIdx = 0;

    for (let c = 0; c < nCandidates; c++) {
      const mu = pred.mean[c] ?? 0;
      const v = pred.variance[c] ?? 0;
      const sigma = Math.sqrt(Math.max(v, 0));

      let acqVal: number;
      switch (config.acquisitionFn) {
        case 'ei':
          acqVal = expectedImprovement(mu, sigma, bestY, config.xi);
          break;
        case 'ucb':
          acqVal = upperConfidenceBound(mu, sigma, config.kappa);
          break;
        case 'pi':
          acqVal = probabilityOfImprovement(mu, sigma, bestY, config.xi);
          break;
        default:
          acqVal = expectedImprovement(mu, sigma, bestY, config.xi);
      }

      if (acqVal > bestAcq) {
        bestAcq = acqVal;
        bestCandidateIdx = c;
      }
    }

    // Evaluate objective at best candidate
    const nextX = candidates[bestCandidateIdx]!;
    const nextY = objectiveFn(nextX);

    xObs.push(nextX);
    yObs.push(nextY);
    history.push({ x: [...nextX], y: nextY, acquisition: bestAcq });
  }

  // Find the overall best
  let bestIdx = 0;
  let bestYFinal = -Infinity;
  for (let i = 0; i < yObs.length; i++) {
    const yi = yObs[i] ?? -Infinity;
    if (yi > bestYFinal) {
      bestYFinal = yi;
      bestIdx = i;
    }
  }

  return {
    bestX: [...xObs[bestIdx]!],
    bestY: bestYFinal,
    history,
  };
}
