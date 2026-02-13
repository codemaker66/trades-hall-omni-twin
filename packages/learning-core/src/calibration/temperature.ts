// ---------------------------------------------------------------------------
// Temperature Scaling (Guo et al. 2017)
// A single-parameter calibration method for neural network logits.
// ---------------------------------------------------------------------------

import type { TemperatureParams } from '../types.js';

/**
 * Sigmoid function: σ(x) = 1 / (1 + exp(-x))
 */
function sigmoid(x: number): number {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }
  const ex = Math.exp(x);
  return ex / (1 + ex);
}

/**
 * Fit temperature scaling parameter T by minimizing negative log-likelihood.
 *
 * Given logits z and binary labels y, we want T such that σ(z/T) is well-calibrated.
 * The NLL is:
 *   L(T) = -Σ [ y_i * log(σ(z_i/T)) + (1 - y_i) * log(1 - σ(z_i/T)) ]
 *
 * We optimize T via gradient descent on log(T) to ensure T > 0:
 *   Let u = log(T), then T = exp(u).
 *   dL/du = dL/dT * dT/du = dL/dT * T
 *
 * @param logits - Raw model logits (before sigmoid)
 * @param labels - Binary labels (0 or 1)
 * @param maxIter - Maximum gradient descent iterations
 * @param lr - Learning rate
 * @returns TemperatureParams with fitted temperature
 */
export function temperatureFit(
  logits: number[],
  labels: number[],
  maxIter: number = 100,
  lr: number = 0.01,
): TemperatureParams {
  const n = Math.min(logits.length, labels.length);
  if (n === 0) return { temperature: 1 };

  // Optimize in log-space: u = log(T)
  let u = 0; // T = exp(0) = 1 initially

  for (let iter = 0; iter < maxIter; iter++) {
    const T = Math.exp(u);
    let gradU = 0;

    for (let i = 0; i < n; i++) {
      const z = logits[i] ?? 0;
      const y = labels[i] ?? 0;
      const scaled = z / T;
      const p = sigmoid(scaled);
      const pClamped = Math.max(1e-15, Math.min(1 - 1e-15, p));

      // dNLL/dT = Σ (p_i - y_i) * (-z_i / T²)
      // dNLL/du = dNLL/dT * T = Σ (p_i - y_i) * (-z_i / T)
      gradU += (pClamped - y) * (-z / T);
    }

    gradU /= n;
    u -= lr * gradU;

    // Clamp u to prevent extreme temperatures
    u = Math.max(-5, Math.min(5, u)); // T in [exp(-5), exp(5)] ≈ [0.007, 148.4]
  }

  return { temperature: Math.exp(u) };
}

/**
 * Apply temperature scaling to logits.
 *
 * P(y=1|z) = σ(z/T)
 *
 * @param logits - Raw model logits
 * @param params - Fitted temperature parameter
 * @returns Calibrated probabilities
 */
export function temperatureTransform(
  logits: number[],
  params: TemperatureParams,
): number[] {
  const T = Math.max(params.temperature, 1e-10); // Prevent division by zero
  const result: number[] = new Array<number>(logits.length);
  for (let i = 0; i < logits.length; i++) {
    const z = logits[i] ?? 0;
    result[i] = sigmoid(z / T);
  }
  return result;
}
