// ---------------------------------------------------------------------------
// OC-12  Edge/Browser Deployment -- Neural Network Policy Inference
// ---------------------------------------------------------------------------

import type { MLPWeights, NeuralPolicyConfig } from '../types.js';

// ---------------------------------------------------------------------------
// mlpLayer
// ---------------------------------------------------------------------------

/**
 * Forward pass through a single MLP layer:
 *
 *   y = activation( W * x + b )
 *
 * @param weight     Weight matrix (outDim x inDim, row-major flat Float64Array).
 * @param bias       Bias vector (outDim).
 * @param input      Input vector (inDim).
 * @param inDim      Input dimension.
 * @param outDim     Output dimension.
 * @param activation Activation function ('relu' | 'tanh' | 'linear').
 * @returns          Output vector (outDim).
 */
export function mlpLayer(
  weight: Float64Array,
  bias: Float64Array,
  input: Float64Array,
  inDim: number,
  outDim: number,
  activation: 'relu' | 'tanh' | 'linear',
) {
  const y = new Float64Array(outDim);

  for (let i = 0; i < outDim; i++) {
    let sum = bias[i]!;
    for (let j = 0; j < inDim; j++) {
      sum += weight[i * inDim + j]! * input[j]!;
    }

    if (activation === 'relu') {
      y[i] = sum > 0 ? sum : 0;
    } else if (activation === 'tanh') {
      y[i] = Math.tanh(sum);
    } else {
      // linear
      y[i] = sum;
    }
  }

  return y;
}

// ---------------------------------------------------------------------------
// neuralPolicyForward
// ---------------------------------------------------------------------------

/**
 * Evaluate a neural network policy on a given state.
 *
 * Pipeline:
 *   1. Normalize input:  z = (x - mean) / std
 *   2. Forward through MLP (hidden layers with activation, final layer linear)
 *   3. Denormalize output:  u = actionScale * y + actionBias
 *
 * @param config  Neural policy configuration with weights and normalization.
 * @param state   Current state vector.
 * @returns       Control action vector.
 */
export function neuralPolicyForward(
  config: NeuralPolicyConfig,
  state: Float64Array,
): Float64Array {
  const { weights, stateNormMean, stateNormStd, actionScale, actionBias } =
    config;

  // 1. Normalize input
  const z = new Float64Array(state.length);
  for (let i = 0; i < state.length; i++) {
    const std = stateNormStd[i]!;
    z[i] = std > 1e-15 ? (state[i]! - stateNormMean[i]!) / std : 0;
  }

  // 2. Forward through MLP
  let current = z;
  const nLayers = weights.layers.length;

  for (let l = 0; l < nLayers; l++) {
    const layer = weights.layers[l]!;
    // Hidden layers use the configured activation; final layer is linear
    const act = l < nLayers - 1 ? weights.activation : 'linear';
    current = mlpLayer(layer.weight, layer.bias, current, layer.inDim, layer.outDim, act);
  }

  // 3. Denormalize output
  const u = new Float64Array(current.length);
  for (let i = 0; i < current.length; i++) {
    u[i] = actionScale[i]! * current[i]! + actionBias[i]!;
  }

  return u;
}

// ---------------------------------------------------------------------------
// loadNeuralPolicy
// ---------------------------------------------------------------------------

/**
 * Parse a JSON object into a NeuralPolicyConfig.
 *
 * Expected shape:
 * ```json
 * {
 *   "weights": {
 *     "layers": [
 *       { "weight": number[], "bias": number[], "inDim": number, "outDim": number }
 *     ],
 *     "activation": "relu" | "tanh"
 *   },
 *   "stateNormMean": number[],
 *   "stateNormStd": number[],
 *   "actionScale": number[],
 *   "actionBias": number[]
 * }
 * ```
 *
 * @param json  Parsed JSON object encoding the neural policy configuration.
 * @returns     NeuralPolicyConfig ready for inference.
 */
export function loadNeuralPolicy(json: {
  weights: {
    layers: Array<{
      weight: number[];
      bias: number[];
      inDim: number;
      outDim: number;
    }>;
    activation: 'relu' | 'tanh';
  };
  stateNormMean: number[];
  stateNormStd: number[];
  actionScale: number[];
  actionBias: number[];
}): NeuralPolicyConfig {
  const mlpWeights: MLPWeights = {
    layers: json.weights.layers.map((l) => ({
      weight: new Float64Array(l.weight),
      bias: new Float64Array(l.bias),
      inDim: l.inDim,
      outDim: l.outDim,
    })),
    activation: json.weights.activation,
  };

  return {
    weights: mlpWeights,
    stateNormMean: new Float64Array(json.stateNormMean),
    stateNormStd: new Float64Array(json.stateNormStd),
    actionScale: new Float64Array(json.actionScale),
    actionBias: new Float64Array(json.actionBias),
  };
}
