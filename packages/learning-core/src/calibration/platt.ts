// ---------------------------------------------------------------------------
// Platt Scaling (Platt 1999)
// Logistic calibration: P(y=1|f) = 1 / (1 + exp(a*f + b))
// ---------------------------------------------------------------------------

import type { PlattParams } from '../types.js';

/**
 * Sigmoid function: σ(x) = 1 / (1 + exp(-x))
 */
function sigmoid(x: number): number {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }
  // Numerically stable for negative x
  const ex = Math.exp(x);
  return ex / (1 + ex);
}

/**
 * Fit Platt scaling parameters via gradient descent on the negative
 * log-likelihood of the logistic model.
 *
 * Model: P(y=1|f) = σ(-(a*f + b)) = 1 / (1 + exp(a*f + b))
 *
 * We optimize a and b to minimize:
 *   NLL = -Σ [ t_i * log(p_i) + (1 - t_i) * log(1 - p_i) ]
 *
 * where t_i are target probabilities (smoothed labels per Platt's recipe):
 *   t_i = (y_i * N+ + 1) / (N+ + 2)  for positive examples
 *   t_i = 1 / (N- + 2)                for negative examples
 *
 * @param predictions - Uncalibrated model scores/predictions
 * @param labels - Binary labels (0 or 1)
 * @param maxIter - Maximum gradient descent iterations
 * @param lr - Learning rate
 * @returns Fitted PlattParams { a, b }
 */
export function plattFit(
  predictions: number[],
  labels: number[],
  maxIter: number = 100,
  lr: number = 0.01,
): PlattParams {
  const n = Math.min(predictions.length, labels.length);
  if (n === 0) return { a: 0, b: 0 };

  // Count positives and negatives for target probability smoothing
  let nPos = 0;
  let nNeg = 0;
  for (let i = 0; i < n; i++) {
    if ((labels[i] ?? 0) > 0.5) {
      nPos++;
    } else {
      nNeg++;
    }
  }

  // Compute smoothed targets (Platt's recipe to avoid overfitting)
  const targets: number[] = new Array<number>(n);
  const hiTarget = (nPos + 1) / (nPos + 2);
  const loTarget = 1 / (nNeg + 2);
  for (let i = 0; i < n; i++) {
    targets[i] = (labels[i] ?? 0) > 0.5 ? hiTarget : loTarget;
  }

  // Initialize parameters
  let a = 0;
  let b = 0;

  // Gradient descent
  for (let iter = 0; iter < maxIter; iter++) {
    let gradA = 0;
    let gradB = 0;

    for (let i = 0; i < n; i++) {
      const f = predictions[i] ?? 0;
      const t = targets[i] ?? 0;

      // p_i = σ(-(a*f + b)) = 1/(1+exp(a*f+b))
      const p = sigmoid(-(a * f + b));

      // Clamp to avoid log(0)
      const pClamped = Math.max(1e-15, Math.min(1 - 1e-15, p));

      // Gradient of NLL w.r.t. a and b
      // dNLL/da = Σ (p_i - t_i) * f_i
      // dNLL/db = Σ (p_i - t_i)
      const diff = pClamped - t;
      gradA += diff * f;
      gradB += diff;
    }

    // Normalize gradients by n
    gradA /= n;
    gradB /= n;

    // Update parameters
    a -= lr * gradA;
    b -= lr * gradB;
  }

  return { a, b };
}

/**
 * Apply Platt scaling to transform scores into calibrated probabilities.
 *
 * P(y=1|f) = 1 / (1 + exp(a*f + b))
 *
 * @param scores - Uncalibrated model scores
 * @param params - Fitted Platt parameters { a, b }
 * @returns Calibrated probabilities
 */
export function plattTransform(scores: number[], params: PlattParams): number[] {
  const result: number[] = new Array<number>(scores.length);
  for (let i = 0; i < scores.length; i++) {
    const f = scores[i] ?? 0;
    result[i] = sigmoid(-(params.a * f + params.b));
  }
  return result;
}
