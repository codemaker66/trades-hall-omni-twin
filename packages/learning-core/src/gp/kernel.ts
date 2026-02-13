// ---------------------------------------------------------------------------
// Kernel Functions for Gaussian Process Regression
// ---------------------------------------------------------------------------

import type { Matrix, KernelConfig, CompositeKernelConfig } from '../types.js';
import { matSet } from '../types.js';

/**
 * Radial Basis Function (Squared Exponential) kernel.
 * k(x1, x2) = variance * exp(-0.5 * ((x1 - x2) / lengthscale)^2)
 */
export function rbfKernel(
  x1: number,
  x2: number,
  lengthscale: number,
  variance: number,
): number {
  const diff = x1 - x2;
  const scaled = diff / lengthscale;
  return variance * Math.exp(-0.5 * scaled * scaled);
}

/**
 * Matern kernel with nu = 3/2.
 * k(x1, x2) = variance * (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)
 * where r = |x1 - x2|
 */
export function matern32Kernel(
  x1: number,
  x2: number,
  lengthscale: number,
  variance: number,
): number {
  const r = Math.abs(x1 - x2);
  const sqrt3r_l = Math.sqrt(3) * r / lengthscale;
  return variance * (1 + sqrt3r_l) * Math.exp(-sqrt3r_l);
}

/**
 * Matern kernel with nu = 5/2.
 * k(x1, x2) = variance * (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)
 * where r = |x1 - x2|
 */
export function matern52Kernel(
  x1: number,
  x2: number,
  lengthscale: number,
  variance: number,
): number {
  const r = Math.abs(x1 - x2);
  const sqrt5r_l = Math.sqrt(5) * r / lengthscale;
  const r2_l2 = (r * r) / (lengthscale * lengthscale);
  return variance * (1 + sqrt5r_l + (5 / 3) * r2_l2) * Math.exp(-sqrt5r_l);
}

/**
 * Periodic kernel.
 * k(x1, x2) = variance * exp(-2 * sin^2(pi * |x1 - x2| / period) / lengthscale^2)
 */
export function periodicKernel(
  x1: number,
  x2: number,
  lengthscale: number,
  variance: number,
  period: number,
): number {
  const r = Math.abs(x1 - x2);
  const sinArg = Math.sin(Math.PI * r / period);
  return variance * Math.exp(-2 * sinArg * sinArg / (lengthscale * lengthscale));
}

/**
 * Linear kernel.
 * k(x1, x2) = variance * x1 * x2
 */
export function linearKernel(
  x1: number,
  x2: number,
  variance: number,
): number {
  return variance * x1 * x2;
}

/**
 * Evaluate a single kernel given a KernelConfig.
 * Dispatches to the appropriate kernel function based on config.type.
 */
export function evaluateKernel(
  x1: number,
  x2: number,
  config: KernelConfig,
): number {
  switch (config.type) {
    case 'rbf':
      return rbfKernel(x1, x2, config.lengthscale, config.variance);
    case 'matern32':
      return matern32Kernel(x1, x2, config.lengthscale, config.variance);
    case 'matern52':
      return matern52Kernel(x1, x2, config.lengthscale, config.variance);
    case 'periodic':
      return periodicKernel(x1, x2, config.lengthscale, config.variance, config.period ?? 1);
    case 'linear':
      return linearKernel(x1, x2, config.variance);
    case 'spectral_mixture':
      return spectralMixtureKernel(x1, x2, config);
    default:
      return rbfKernel(x1, x2, config.lengthscale, config.variance);
  }
}

/**
 * Spectral Mixture kernel (Wilson & Adams 2013).
 * k(x1, x2) = sum_q weight_q * exp(-2 pi^2 tau^2 v_q) * cos(2 pi tau mu_q)
 * where tau = x1 - x2
 */
function spectralMixtureKernel(x1: number, x2: number, config: KernelConfig): number {
  const weights = config.weights ?? [1];
  const means = config.means ?? [0];
  const variances = config.variances ?? [1];
  const tau = x1 - x2;
  const twoPi = 2 * Math.PI;
  const twoPiSq = twoPi * Math.PI; // 2 * pi^2

  let result = 0;
  for (let q = 0; q < weights.length; q++) {
    const w = weights[q] ?? 1;
    const mu = means[q] ?? 0;
    const v = variances[q] ?? 1;
    result += w * Math.exp(-twoPiSq * tau * tau * v) * Math.cos(twoPi * tau * mu);
  }
  return config.variance * result;
}

/**
 * Helper type guard: check if config is a CompositeKernelConfig.
 */
function isComposite(config: KernelConfig | CompositeKernelConfig): config is CompositeKernelConfig {
  return 'op' in config;
}

/**
 * Evaluate a composite kernel recursively.
 * Supports 'add' and 'mul' operations over sub-kernels.
 *
 * @param x1 First input
 * @param x2 Second input
 * @param config Composite kernel configuration
 * @returns Kernel value
 */
export function compositeKernel(
  x1: number,
  x2: number,
  config: CompositeKernelConfig,
): number {
  const leftVal = isComposite(config.left)
    ? compositeKernel(x1, x2, config.left)
    : evaluateKernel(x1, x2, config.left);

  const rightVal = isComposite(config.right)
    ? compositeKernel(x1, x2, config.right)
    : evaluateKernel(x1, x2, config.right);

  if (config.op === 'add') {
    return leftVal + rightVal;
  }
  // config.op === 'mul'
  return leftVal * rightVal;
}

/**
 * Evaluate any kernel config (either simple or composite).
 */
function evaluateAnyKernel(
  x1: number,
  x2: number,
  config: KernelConfig | CompositeKernelConfig,
): number {
  return isComposite(config)
    ? compositeKernel(x1, x2, config)
    : evaluateKernel(x1, x2, config);
}

/**
 * Compute the full N x N kernel matrix for a set of input points.
 *
 * @param xs Input points
 * @param config Kernel configuration (simple or composite)
 * @returns N x N kernel matrix
 */
export function computeKernelMatrix(
  xs: number[],
  config: KernelConfig | CompositeKernelConfig,
): Matrix {
  const n = xs.length;
  const data = new Float64Array(n * n);
  const K: Matrix = { data, rows: n, cols: n };

  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const xi = xs[i] ?? 0;
      const xj = xs[j] ?? 0;
      const val = evaluateAnyKernel(xi, xj, config);
      matSet(K, i, j, val);
      if (i !== j) {
        matSet(K, j, i, val); // Symmetric
      }
    }
  }

  return K;
}
