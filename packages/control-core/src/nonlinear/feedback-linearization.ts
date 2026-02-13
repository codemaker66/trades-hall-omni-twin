// ---------------------------------------------------------------------------
// OC-6: Feedback Linearization
// ---------------------------------------------------------------------------

import type { FeedbackLinConfig } from '../types.js';
import { vecClone } from '../types.js';

// ---------------------------------------------------------------------------
// Lie Derivative
// ---------------------------------------------------------------------------

/**
 * Compute the Lie derivative L_f h(x) = (dh/dx) * f(x).
 *
 * The gradient dh/dx is approximated via central finite differences:
 *   dh/dx_i ~ (h(x + eps*e_i) - h(x - eps*e_i)) / (2 * eps)
 *
 * @param f   Drift vector field f: R^n -> R^n
 * @param h   Scalar output function h: R^n -> R
 * @param x   State vector
 * @param eps Finite difference step size (default 1e-7)
 * @returns   Scalar value of L_f h(x)
 */
export function lieDerivative(
  f: (x: Float64Array) => Float64Array,
  h: (x: Float64Array) => number,
  x: Float64Array,
  eps: number = 1e-7,
): number {
  const nx = x.length;
  const fVal = f(x);

  // Compute gradient dh/dx via central differences
  const grad = new Float64Array(nx);
  const xp = vecClone(x);
  const xm = vecClone(x);

  for (let i = 0; i < nx; i++) {
    const xi = x[i]!;
    xp[i] = xi + eps;
    xm[i] = xi - eps;
    grad[i] = (h(xp) - h(xm)) / (2 * eps);
    // Restore
    xp[i] = xi;
    xm[i] = xi;
  }

  // Dot product: dh/dx . f(x)
  let result = 0;
  for (let i = 0; i < nx; i++) {
    result += grad[i]! * fVal[i]!;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Higher-order Lie derivative (iterated)
// ---------------------------------------------------------------------------

/**
 * Compute L_f^k h(x) — the k-th iterated Lie derivative of h along f.
 *
 * L_f^0 h = h
 * L_f^k h = L_f (L_f^{k-1} h)
 *
 * @param f   Drift vector field
 * @param h   Output function
 * @param x   State vector
 * @param k   Order of iterated Lie derivative
 * @param eps Finite difference step
 * @returns   Scalar L_f^k h(x)
 */
function iteratedLieDerivative(
  f: (x: Float64Array) => Float64Array,
  h: (x: Float64Array) => number,
  x: Float64Array,
  k: number,
  eps: number = 1e-7,
): number {
  if (k === 0) return h(x);

  // Build the (k-1)-th Lie derivative as a function of state
  const hPrev = (xv: Float64Array): number =>
    iteratedLieDerivative(f, h, xv, k - 1, eps);

  return lieDerivative(f, hPrev, x, eps);
}

// ---------------------------------------------------------------------------
// Relative Degree Computation
// ---------------------------------------------------------------------------

/**
 * Compute the relative degree of the system at a given operating point x0.
 *
 * The relative degree r is the smallest integer such that
 *   L_g L_f^{r-1} h(x0) != 0
 *
 * @param config  Feedback linearization configuration
 * @param x0      Operating point to check
 * @param maxDeg  Maximum degree to search (default 10)
 * @returns       Relative degree, or maxDeg+1 if not found
 */
export function computeRelativeDegree(
  config: FeedbackLinConfig,
  x0: Float64Array,
  maxDeg: number = 10,
): number {
  const { f, g, h } = config;
  const eps = 1e-7;
  const threshold = 1e-10;

  for (let r = 1; r <= maxDeg; r++) {
    // Compute L_g L_f^{r-1} h(x0)
    // This is the Lie derivative of L_f^{r-1} h along g
    const hPrev = (xv: Float64Array): number =>
      iteratedLieDerivative(f, h, xv, r - 1, eps);

    const lgVal = lieDerivative(g, hPrev, x0, eps);

    if (Math.abs(lgVal) > threshold) {
      return r;
    }
  }

  return maxDeg + 1;
}

// ---------------------------------------------------------------------------
// Feedback Linearizing Control Law
// ---------------------------------------------------------------------------

/**
 * Compute the feedback linearizing control input.
 *
 * For a system with relative degree r:
 *   y^(r) = L_f^r h(x) + L_g L_f^{r-1} h(x) * u
 *
 * Setting y^(r) = v (new synthetic input):
 *   u = alpha(x) + beta(x) * v
 *
 * where:
 *   alpha(x) = -L_f^r h(x) / (L_g L_f^{r-1} h(x))
 *   beta(x)  = 1 / (L_g L_f^{r-1} h(x))
 *
 * @param config  Feedback linearization configuration
 * @param x       Current state vector
 * @param v       New (linear) input
 * @returns       Actual control input u
 */
export function feedbackLinearize(
  config: FeedbackLinConfig,
  x: Float64Array,
  v: number,
): number {
  const { f, g, h, relativeDegree: r } = config;
  const eps = 1e-7;

  // Build L_f^{r-1} h as a function of state
  const hPrev = (xv: Float64Array): number =>
    iteratedLieDerivative(f, h, xv, r - 1, eps);

  // L_f^r h(x) = L_f (L_f^{r-1} h)(x)
  const LfRh = lieDerivative(f, hPrev, x, eps);

  // L_g L_f^{r-1} h(x)
  const LgLfR1h = lieDerivative(g, hPrev, x, eps);

  // Guard against singular decoupling matrix
  if (Math.abs(LgLfR1h) < 1e-12) {
    throw new Error(
      'Feedback linearization: L_g L_f^{r-1} h(x) is zero — system loses relative degree at this state.',
    );
  }

  const alpha = -LfRh / LgLfR1h;
  const beta = 1 / LgLfR1h;

  return alpha + beta * v;
}
