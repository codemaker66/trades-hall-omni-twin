// ---------------------------------------------------------------------------
// Gaussian Process Regression with Full Posterior Inference
// ---------------------------------------------------------------------------

import type {
  Matrix,
  GPConfig,
  GPPrediction,
  KernelConfig,
  CompositeKernelConfig,
} from '../types.js';
import { cholesky, solveLower, solveUpper, matGet, matSet } from '../types.js';
import { computeKernelMatrix, evaluateKernel, compositeKernel } from './kernel.js';

/**
 * Helper: check if config is a CompositeKernelConfig.
 */
function isComposite(config: KernelConfig | CompositeKernelConfig): config is CompositeKernelConfig {
  return 'op' in config;
}

/**
 * Evaluate any kernel config (simple or composite).
 */
function evalKernel(x1: number, x2: number, config: KernelConfig | CompositeKernelConfig): number {
  return isComposite(config)
    ? compositeKernel(x1, x2, config)
    : evaluateKernel(x1, x2, config);
}

/**
 * GP Regressor — full Gaussian Process regression with Cholesky-based inference.
 *
 * Follows the standard GP textbook (Rasmussen & Williams 2006, Algorithm 2.1):
 *   1. Compute K + sigma^2 I
 *   2. L = cholesky(K + sigma^2 I)
 *   3. alpha = L^T \ (L \ y)
 *   4. Predictive mean: k_* ^T alpha
 *   5. Predictive variance: k(x_*, x_*) - v^T v  where v = L \ k_*
 *   6. Log marginal likelihood: -0.5 * (y^T alpha + sum(log(diag(L))) + n*log(2*pi))
 */
export class GPRegressor {
  private readonly config: GPConfig;

  // Stored after fit()
  private xTrain: number[] = [];
  private yTrain: number[] = [];
  private L: Matrix | null = null;
  private alpha: Float64Array | null = null;
  private n: number = 0;
  private meanVal: number = 0;

  constructor(config: GPConfig) {
    this.config = config;
    this.meanVal = config.meanFunction === 'constant' ? (config.meanValue ?? 0) : 0;
  }

  /**
   * Fit the GP to training data.
   *
   * Computes the kernel matrix K, adds noise variance to the diagonal,
   * performs Cholesky decomposition, and solves for the alpha vector.
   *
   * @param xTrain Training inputs (1D array of scalars)
   * @param yTrain Training targets
   */
  fit(xTrain: number[], yTrain: number[]): void {
    this.xTrain = xTrain;
    this.yTrain = yTrain;
    this.n = xTrain.length;

    if (this.n === 0) {
      this.L = null;
      this.alpha = null;
      return;
    }

    // Compute kernel matrix K
    const K = computeKernelMatrix(xTrain, this.config.kernel);

    // Add noise variance to diagonal: K + sigma_n^2 I
    for (let i = 0; i < this.n; i++) {
      const current = matGet(K, i, i);
      matSet(K, i, i, current + this.config.noiseVariance);
    }

    // Cholesky decomposition: K + sigma_n^2 I = L L^T
    this.L = cholesky(K);

    // Compute y - mean
    const yMean = new Float64Array(this.n);
    for (let i = 0; i < this.n; i++) {
      yMean[i] = (yTrain[i] ?? 0) - this.meanVal;
    }

    // alpha = L^T \ (L \ y)
    const z = solveLower(this.L, yMean);
    this.alpha = solveUpper(this.L, z);
  }

  /**
   * Make predictions at new test points.
   *
   * @param xNew Test inputs
   * @param returnVar Whether to compute predictive variance (default true)
   * @returns GPPrediction with mean, variance, and confidence bounds
   */
  predict(xNew: number[], returnVar: boolean = true): GPPrediction {
    const nNew = xNew.length;
    const mean = new Array<number>(nNew);
    const variance = new Array<number>(nNew);
    const lower = new Array<number>(nNew);
    const upper = new Array<number>(nNew);

    // If not fitted, return prior
    if (this.L === null || this.alpha === null || this.n === 0) {
      for (let i = 0; i < nNew; i++) {
        const xi = xNew[i] ?? 0;
        mean[i] = this.meanVal;
        const priorVar = evalKernel(xi, xi, this.config.kernel);
        variance[i] = priorVar;
        const std = Math.sqrt(Math.max(priorVar, 0));
        lower[i] = this.meanVal - 1.96 * std;
        upper[i] = this.meanVal + 1.96 * std;
      }
      return {
        mean,
        variance,
        lower,
        upper,
        logMarginalLikelihood: 0,
      };
    }

    for (let i = 0; i < nNew; i++) {
      const xStar = xNew[i] ?? 0;

      // k_* = kernel vector between x_* and all training points
      const kStar = new Float64Array(this.n);
      for (let j = 0; j < this.n; j++) {
        kStar[j] = evalKernel(xStar, this.xTrain[j] ?? 0, this.config.kernel);
      }

      // Predictive mean: mu_* = k_*^T alpha + mean
      let mu = this.meanVal;
      for (let j = 0; j < this.n; j++) {
        mu += (kStar[j] ?? 0) * (this.alpha[j] ?? 0);
      }
      mean[i] = mu;

      if (returnVar) {
        // v = L \ k_*
        const v = solveLower(this.L, kStar);

        // Predictive variance: k(x_*, x_*) - v^T v
        const kss = evalKernel(xStar, xStar, this.config.kernel);
        let vTv = 0;
        for (let j = 0; j < this.n; j++) {
          const vj = v[j] ?? 0;
          vTv += vj * vj;
        }
        const var_i = Math.max(kss - vTv, 1e-12);
        variance[i] = var_i;

        const std = Math.sqrt(var_i);
        lower[i] = mu - 1.96 * std;
        upper[i] = mu + 1.96 * std;
      } else {
        variance[i] = 0;
        lower[i] = mu;
        upper[i] = mu;
      }
    }

    return {
      mean,
      variance,
      lower,
      upper,
      logMarginalLikelihood: this.logMarginalLikelihood(),
    };
  }

  /**
   * Compute the log marginal likelihood of the fitted GP.
   *
   * log p(y|X) = -0.5 * (y^T alpha + sum(log(diag(L))) + n * log(2*pi))
   *
   * Note: The standard formula uses 2*sum(log(diag(L))) = log(det(K+sigma^2 I)),
   * but since L L^T = K + sigma^2 I, log(det(K+sigma^2 I)) = 2*sum(log(L_ii)).
   *
   * @returns Log marginal likelihood
   */
  logMarginalLikelihood(): number {
    if (this.L === null || this.alpha === null || this.n === 0) {
      return 0;
    }

    // Data fit term: y^T alpha
    let dataFit = 0;
    for (let i = 0; i < this.n; i++) {
      const yi = (this.yTrain[i] ?? 0) - this.meanVal;
      dataFit += yi * (this.alpha[i] ?? 0);
    }

    // Complexity penalty: sum of log diagonal of L
    // (This is 0.5 * log|K + sigma^2 I| since det(L) = prod(L_ii))
    let logDetTerm = 0;
    for (let i = 0; i < this.n; i++) {
      const Lii = matGet(this.L, i, i);
      logDetTerm += Math.log(Math.max(Lii, 1e-15));
    }

    // Constant term
    const constTerm = this.n * Math.log(2 * Math.PI);

    return -0.5 * (dataFit + 2 * logDetTerm + constTerm);
  }
}
