// ---------------------------------------------------------------------------
// SP-4: Savitzky-Golay Filter
// ---------------------------------------------------------------------------
// Least-squares polynomial fit within a moving window.
// Preserves peaks better than moving average.
// With deriv > 0: computes smoothed derivatives (velocity, acceleration).

import type { SavitzkyGolayConfig } from '../types.js';

/**
 * Compute Savitzky-Golay convolution coefficients via polynomial regression.
 *
 * For window of size 2m+1 and polynomial order p:
 * Fit y = a₀ + a₁x + ... + aₚxᵖ at points -m,...,0,...,m
 * The smoothed value at center is the fitted value at x=0.
 *
 * For derivative d: the d-th coefficient × d! gives the derivative.
 */
export function sgCoefficients(windowLength: number, polyOrder: number, deriv: number = 0): Float64Array {
  if (windowLength % 2 === 0) throw new Error('windowLength must be odd');
  if (polyOrder >= windowLength) throw new Error('polyOrder must be less than windowLength');
  if (deriv > polyOrder) throw new Error('deriv must be <= polyOrder');

  const m = Math.floor(windowLength / 2);
  const nPoly = polyOrder + 1;

  // Build Vandermonde matrix J: J[i][j] = (i-m)^j
  const J = new Float64Array(windowLength * nPoly);
  for (let i = 0; i < windowLength; i++) {
    const x = i - m;
    let xPow = 1;
    for (let j = 0; j < nPoly; j++) {
      J[i * nPoly + j] = xPow;
      xPow *= x;
    }
  }

  // Compute (JᵀJ)⁻¹Jᵀ via normal equations
  // JᵀJ is (nPoly × nPoly)
  const JtJ = new Float64Array(nPoly * nPoly);
  for (let i = 0; i < nPoly; i++) {
    for (let j = 0; j < nPoly; j++) {
      let sum = 0;
      for (let k = 0; k < windowLength; k++) {
        sum += J[k * nPoly + i]! * J[k * nPoly + j]!;
      }
      JtJ[i * nPoly + j] = sum;
    }
  }

  // Invert JᵀJ (small matrix, Gauss-Jordan)
  const inv = invertSmall(JtJ, nPoly);

  // Compute row `deriv` of (JᵀJ)⁻¹Jᵀ → coefficients
  // c = factorial(deriv) × row[deriv] of (JᵀJ)⁻¹Jᵀ
  const coeffs = new Float64Array(windowLength);
  let factorial = 1;
  for (let i = 2; i <= deriv; i++) factorial *= i;

  for (let k = 0; k < windowLength; k++) {
    let sum = 0;
    for (let j = 0; j < nPoly; j++) {
      sum += inv[deriv * nPoly + j]! * J[k * nPoly + j]!;
    }
    coeffs[k] = sum * factorial;
  }

  return coeffs;
}

/** Gauss-Jordan inversion for small matrices. */
function invertSmall(A: Float64Array, n: number): Float64Array {
  const aug = new Float64Array(n * 2 * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      aug[i * 2 * n + j] = A[i * n + j]!;
    }
    aug[i * 2 * n + n + i] = 1;
  }

  for (let col = 0; col < n; col++) {
    // Partial pivot
    let maxRow = col;
    let maxVal = Math.abs(aug[col * 2 * n + col]!);
    for (let row = col + 1; row < n; row++) {
      const val = Math.abs(aug[row * 2 * n + col]!);
      if (val > maxVal) { maxVal = val; maxRow = row; }
    }
    if (maxRow !== col) {
      for (let j = 0; j < 2 * n; j++) {
        const tmp = aug[col * 2 * n + j]!;
        aug[col * 2 * n + j] = aug[maxRow * 2 * n + j]!;
        aug[maxRow * 2 * n + j] = tmp;
      }
    }

    const pivot = aug[col * 2 * n + col]!;
    for (let j = 0; j < 2 * n; j++) aug[col * 2 * n + j] = aug[col * 2 * n + j]! / pivot;

    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row * 2 * n + col]!;
      for (let j = 0; j < 2 * n; j++) {
        aug[row * 2 * n + j] = aug[row * 2 * n + j]! - factor * aug[col * 2 * n + j]!;
      }
    }
  }

  const result = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      result[i * n + j] = aug[i * 2 * n + n + j]!;
    }
  }
  return result;
}

/**
 * Apply Savitzky-Golay filter to a signal.
 */
export function savitzkyGolayFilter(
  signal: Float64Array,
  config: SavitzkyGolayConfig,
): Float64Array {
  const { windowLength, polyOrder, deriv = 0 } = config;
  const coeffs = sgCoefficients(windowLength, polyOrder, deriv);
  const m = Math.floor(windowLength / 2);
  const N = signal.length;
  const result = new Float64Array(N);

  for (let i = 0; i < N; i++) {
    let sum = 0;
    for (let j = 0; j < windowLength; j++) {
      const idx = i + j - m;
      // Mirror boundary conditions
      const mirroredIdx = idx < 0
        ? -idx
        : idx >= N
          ? 2 * (N - 1) - idx
          : idx;
      const clampedIdx = Math.max(0, Math.min(N - 1, mirroredIdx));
      sum += coeffs[j]! * signal[clampedIdx]!;
    }
    result[i] = sum;
  }

  return result;
}
