// ---------------------------------------------------------------------------
// OC-7: Multi-Agent PPO (MAPPO) — Centralized Training, Decentralized Execution
// ---------------------------------------------------------------------------

import type { MAPPOConfig, MLPWeights, PRNG } from '../types.js';
import { createPRNG } from '../types.js';

// ---------------------------------------------------------------------------
// Xavier-like Weight Initialization (internal)
// ---------------------------------------------------------------------------

/**
 * Initialize a weight matrix with Xavier-like uniform random values.
 * Samples from U(-scale, +scale) where scale = sqrt(6 / (fanIn + fanOut)).
 */
function initXavier(fanIn: number, fanOut: number, rng: PRNG): Float64Array {
  const scale = Math.sqrt(6.0 / (fanIn + fanOut));
  const w = new Float64Array(fanOut * fanIn);
  for (let i = 0; i < w.length; i++) {
    w[i] = (rng() * 2 - 1) * scale;
  }
  return w;
}

// ---------------------------------------------------------------------------
// MLP Forward Pass (inline)
// ---------------------------------------------------------------------------

/**
 * Forward pass through a multi-layer perceptron.
 *
 * For each hidden layer: y = activation(W * x + b)
 * For the last layer: y = W * x + b (linear output)
 */
function mlpForward(weights: MLPWeights, input: Float64Array): Float64Array {
  let current = input;
  const nLayers = weights.layers.length;

  for (let l = 0; l < nLayers; l++) {
    const layer = weights.layers[l]!;
    const { weight, bias, inDim, outDim } = layer;

    const y = new Float64Array(outDim);
    for (let i = 0; i < outDim; i++) {
      let sum = bias[i]!;
      for (let j = 0; j < inDim; j++) {
        sum += weight[i * inDim + j]! * current[j]!;
      }
      y[i] = sum;
    }

    // Apply activation for all layers except the last
    if (l < nLayers - 1) {
      if (weights.activation === 'relu') {
        for (let i = 0; i < outDim; i++) {
          if (y[i]! < 0) {
            y[i] = 0;
          }
        }
      } else {
        // tanh
        for (let i = 0; i < outDim; i++) {
          y[i] = Math.tanh(y[i]!);
        }
      }
    }

    current = y;
  }

  return current;
}

// ---------------------------------------------------------------------------
// createMAPPOAgents
// ---------------------------------------------------------------------------

/**
 * Create MAPPO agent networks: one actor MLP per agent and one centralized
 * critic MLP.
 *
 * Actor architecture (per agent):
 *   input(obsPerAgent) -> hidden(actorHiddenDim) -> hidden(actorHiddenDim)
 *   -> output(actionPerAgent)
 *
 * Critic architecture:
 *   input(centralStateDim) -> hidden(criticHiddenDim) -> hidden(criticHiddenDim)
 *   -> output(1)
 *
 * All networks use ReLU activation between hidden layers and a linear
 * output layer. Weights are initialized with Xavier-like uniform sampling.
 *
 * @param config  MAPPO configuration.
 * @returns       Object with `actors` (MLPWeights[]) and `critic` (MLPWeights).
 */
export function createMAPPOAgents(
  config: MAPPOConfig,
): { actors: MLPWeights[]; critic: MLPWeights } {
  const {
    nAgents,
    obsPerAgent,
    actionPerAgent,
    centralStateDim,
    criticHiddenDim,
    actorHiddenDim,
  } = config;

  const rng = createPRNG(42);

  // Create one actor per agent
  const actors: MLPWeights[] = [];
  for (let a = 0; a < nAgents; a++) {
    const actorLayers: MLPWeights['layers'] = [
      {
        weight: initXavier(obsPerAgent, actorHiddenDim, rng),
        bias: new Float64Array(actorHiddenDim),
        inDim: obsPerAgent,
        outDim: actorHiddenDim,
      },
      {
        weight: initXavier(actorHiddenDim, actorHiddenDim, rng),
        bias: new Float64Array(actorHiddenDim),
        inDim: actorHiddenDim,
        outDim: actorHiddenDim,
      },
      {
        weight: initXavier(actorHiddenDim, actionPerAgent, rng),
        bias: new Float64Array(actionPerAgent),
        inDim: actorHiddenDim,
        outDim: actionPerAgent,
      },
    ];
    actors.push({ layers: actorLayers, activation: 'relu' });
  }

  // Create centralized critic
  const criticLayers: MLPWeights['layers'] = [
    {
      weight: initXavier(centralStateDim, criticHiddenDim, rng),
      bias: new Float64Array(criticHiddenDim),
      inDim: centralStateDim,
      outDim: criticHiddenDim,
    },
    {
      weight: initXavier(criticHiddenDim, criticHiddenDim, rng),
      bias: new Float64Array(criticHiddenDim),
      inDim: criticHiddenDim,
      outDim: criticHiddenDim,
    },
    {
      weight: initXavier(criticHiddenDim, 1, rng),
      bias: new Float64Array(1),
      inDim: criticHiddenDim,
      outDim: 1,
    },
  ];
  const critic: MLPWeights = { layers: criticLayers, activation: 'relu' };

  return { actors, critic };
}

// ---------------------------------------------------------------------------
// mappoSelectActions
// ---------------------------------------------------------------------------

/**
 * Select actions for all agents using their respective actor networks.
 *
 * Each agent performs a forward pass through its own actor MLP on its
 * local observation. Small Gaussian exploration noise is added using the
 * provided PRNG.
 *
 * @param actors        Array of actor MLP weights (one per agent).
 * @param observations  Local observation vector per agent.
 * @param rng           PRNG for exploration noise.
 * @returns             Action vector per agent.
 */
export function mappoSelectActions(
  actors: MLPWeights[],
  observations: Float64Array[],
  rng: PRNG,
): Float64Array[] {
  const actions: Float64Array[] = [];

  for (let a = 0; a < actors.length; a++) {
    const actor = actors[a]!;
    const obs = observations[a]!;

    // Forward pass through actor network
    const mean = mlpForward(actor, obs);

    // Add Gaussian exploration noise (Box-Muller)
    const action = new Float64Array(mean.length);
    for (let i = 0; i < mean.length; i++) {
      const u1 = Math.max(rng(), 1e-10);
      const u2 = rng();
      const noise = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      action[i] = mean[i]! + 0.1 * noise;
    }

    actions.push(action);
  }

  return actions;
}

// ---------------------------------------------------------------------------
// mappoCriticValue
// ---------------------------------------------------------------------------

/**
 * Evaluate the centralized critic on a global state vector.
 *
 * The critic takes a concatenation of all agents' observations (or any
 * global state representation) and outputs a single scalar value estimate.
 *
 * @param critic       Critic MLP weights.
 * @param globalState  Global state vector (centralStateDim).
 * @returns            Scalar value estimate.
 */
export function mappoCriticValue(
  critic: MLPWeights,
  globalState: Float64Array,
): number {
  const output = mlpForward(critic, globalState);
  return output[0]!;
}
