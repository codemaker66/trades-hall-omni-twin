// ---------------------------------------------------------------------------
// MC Dropout BNN Approximation (Gal & Ghahramani 2016)
// ---------------------------------------------------------------------------

import type { PRNG, MCDropoutPrediction } from '../types.js';

/**
 * Perform MC Dropout inference through a simple 2-layer fully-connected
 * network with ReLU activations.
 *
 * Architecture:
 *   Layer 1: z1 = ReLU(W1 * x + b1)    with dropout
 *   Layer 2: out = W2 * z1 + b2         (no dropout on output)
 *
 * MC Dropout keeps dropout active at inference time and performs
 * multiple stochastic forward passes. The variance of outputs across
 * passes captures epistemic uncertainty (model uncertainty), while the
 * mean of per-pass variances (if the model has multiple outputs)
 * captures aleatoric uncertainty (data/noise uncertainty).
 *
 * For a single-output regression model, we estimate aleatoric uncertainty
 * by treating the network as predicting both mean and log-variance
 * (heteroscedastic regression). If the output dimension is 1, we
 * use a fixed aleatoric estimate based on residual spread.
 *
 * @param weights     weights[0] = W1 (hidden x input), weights[1] = W2 (output x hidden)
 * @param biases      biases[0] = b1 (hidden), biases[1] = b2 (output)
 * @param x           Input feature vector
 * @param dropoutRate Probability of zeroing a hidden unit
 * @param nPasses     Number of stochastic forward passes
 * @param rng         Seedable PRNG
 * @returns MCDropoutPrediction with mean, epistemic, aleatoric, total uncertainty
 */
export function mcDropoutPredict(
  weights: number[][],
  biases: number[][],
  x: number[],
  dropoutRate: number,
  nPasses: number,
  rng: PRNG,
): MCDropoutPrediction {
  if (!Number.isInteger(nPasses) || nPasses <= 0) {
    throw new Error('nPasses must be a positive integer');
  }

  const W1 = weights[0]!; // flattened hidden x input
  const W2 = weights[1]!; // flattened output x hidden
  const b1 = biases[0]!;
  const b2 = biases[1]!;

  const inputDim = x.length;
  const hiddenDim = b1.length;
  const outputDim = b2.length;

  // Collect outputs from multiple stochastic forward passes
  const allOutputs: number[][] = [];

  for (let pass = 0; pass < nPasses; pass++) {
    // Create dropout mask for hidden layer
    const mask = createDropoutMask(hiddenDim, dropoutRate, rng);

    // Layer 1: z1 = ReLU(W1 * x + b1) with dropout
    const z1 = new Array<number>(hiddenDim);
    for (let h = 0; h < hiddenDim; h++) {
      let sum = b1[h] ?? 0;
      for (let j = 0; j < inputDim; j++) {
        // W1 is stored row-major: W1[h * inputDim + j]
        sum += (W1[h * inputDim + j] ?? 0) * (x[j] ?? 0);
      }
      // ReLU + dropout (scale by 1/(1-p) to maintain expected value)
      const activated = Math.max(0, sum);
      const scale = dropoutRate < 1 ? 1 / (1 - dropoutRate) : 0;
      z1[h] = activated * (mask[h] ?? 0) * scale;
    }

    // Layer 2: out = W2 * z1 + b2
    const out = new Array<number>(outputDim);
    for (let o = 0; o < outputDim; o++) {
      let sum = b2[o] ?? 0;
      for (let h = 0; h < hiddenDim; h++) {
        // W2 is stored row-major: W2[o * hiddenDim + h]
        sum += (W2[o * hiddenDim + h] ?? 0) * (z1[h] ?? 0);
      }
      out[o] = sum;
    }

    allOutputs.push(out);
  }

  // -----------------------------------------------------------------------
  // Compute uncertainty decomposition
  // -----------------------------------------------------------------------

  // Mean prediction across passes (per output dimension)
  const meanPred = new Array<number>(outputDim).fill(0);
  for (let pass = 0; pass < nPasses; pass++) {
    for (let o = 0; o < outputDim; o++) {
      meanPred[o] = (meanPred[o] ?? 0) + (allOutputs[pass]![o] ?? 0);
    }
  }
  for (let o = 0; o < outputDim; o++) {
    meanPred[o] = (meanPred[o] ?? 0) / nPasses;
  }

  // Epistemic uncertainty: variance of means across passes
  // (since each pass gives a single output, variance of outputs = epistemic)
  const epistemicVar = new Array<number>(outputDim).fill(0);
  for (let pass = 0; pass < nPasses; pass++) {
    for (let o = 0; o < outputDim; o++) {
      const diff = (allOutputs[pass]![o] ?? 0) - (meanPred[o] ?? 0);
      epistemicVar[o] = (epistemicVar[o] ?? 0) + diff * diff;
    }
  }
  for (let o = 0; o < outputDim; o++) {
    epistemicVar[o] = (epistemicVar[o] ?? 0) / nPasses;
  }

  const epistemicStd = epistemicVar.map((v) => Math.sqrt(Math.max(v, 0)));

  // Aleatoric uncertainty: estimated from the spread within passes.
  // For a deterministic output per pass, we approximate aleatoric noise
  // as the mean absolute deviation of individual passes from the
  // overall mean, scaled by sqrt(pi/2) to convert MAD to std.
  // This is a heuristic since we don't have a separate noise head.
  const aleatoricVar = new Array<number>(outputDim).fill(0);
  for (let pass = 0; pass < nPasses; pass++) {
    for (let o = 0; o < outputDim; o++) {
      const absDev = Math.abs((allOutputs[pass]![o] ?? 0) - (meanPred[o] ?? 0));
      aleatoricVar[o] = (aleatoricVar[o] ?? 0) + absDev;
    }
  }
  for (let o = 0; o < outputDim; o++) {
    // MAD -> std approximation: std ~ MAD * sqrt(pi/2) for normal
    const mad = (aleatoricVar[o] ?? 0) / nPasses;
    const stdApprox = mad * Math.sqrt(Math.PI / 2);
    // Aleatoric = min of estimated noise, capped to not exceed total variance
    aleatoricVar[o] = Math.min(stdApprox * stdApprox, epistemicVar[o] ?? 0);
  }
  const aleatoricStd = aleatoricVar.map((v) => Math.sqrt(Math.max(v, 0)));

  // Total uncertainty: sqrt(epistemic^2 + aleatoric^2)
  const totalStd = new Array<number>(outputDim);
  for (let o = 0; o < outputDim; o++) {
    totalStd[o] = Math.sqrt(
      (epistemicVar[o] ?? 0) + (aleatoricVar[o] ?? 0),
    );
  }

  return {
    mean: meanPred,
    epistemicStd,
    aleatoricStd,
    totalStd,
    samples: allOutputs,
  };
}

/**
 * Create a binary dropout mask.
 *
 * Each element is 1 with probability (1 - rate) and 0 with probability rate.
 *
 * @param size  Number of units
 * @param rate  Dropout probability (0 to 1)
 * @param rng   Seedable PRNG
 * @returns Binary mask array of length `size`
 */
export function createDropoutMask(
  size: number,
  rate: number,
  rng: PRNG,
): number[] {
  const mask = new Array<number>(size);
  for (let i = 0; i < size; i++) {
    mask[i] = rng() >= rate ? 1 : 0;
  }
  return mask;
}
