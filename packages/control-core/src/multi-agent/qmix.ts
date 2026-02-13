// ---------------------------------------------------------------------------
// OC-7: QMIX — Monotonic Value Function Factorization
// ---------------------------------------------------------------------------

import type { MLPWeights, QMIXConfig } from '../types.js';
import { createPRNG } from '../types.js';
import type { PRNG } from '../types.js';

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
// Mixer Weights Type
// ---------------------------------------------------------------------------

/** Mixing network weights for QMIX. */
export interface QMIXMixerWeights {
  /** First layer weights (mixingHiddenDim x nAgents, row-major). */
  w1: Float64Array;
  /** First layer bias (mixingHiddenDim). */
  b1: Float64Array;
  /** Second layer weights (1 x mixingHiddenDim, row-major). */
  w2: Float64Array;
  /** Second layer bias (1). */
  b2: Float64Array;
}

// ---------------------------------------------------------------------------
// createQMIX
// ---------------------------------------------------------------------------

/**
 * Create QMIX agent networks and mixing network weights.
 *
 * Agent Q-networks:
 *   input(obsPerAgent) -> hidden(agentHiddenDim) -> hidden(agentHiddenDim)
 *   -> output(nActions)
 *
 * Mixing network:
 *   Input: per-agent Q-values (nAgents)
 *   Layer 1: hidden = |w1| * Q_agents + b1  (abs-constrained for monotonicity)
 *   Activation: ReLU
 *   Layer 2: output = |w2| * hidden + b2
 *
 * The absolute value constraint on w1 and w2 ensures that the joint Q-value
 * is monotonically increasing in each agent's Q-value, which is the key QMIX
 * structural constraint enabling decentralized execution.
 *
 * @param config  QMIX configuration.
 * @returns       Object with `agentNets` (MLPWeights[]) and `mixerWeights`.
 */
export function createQMIX(
  config: QMIXConfig,
): { agentNets: MLPWeights[]; mixerWeights: QMIXMixerWeights } {
  const { nAgents, obsPerAgent, nActions, mixingHiddenDim, agentHiddenDim } = config;

  const rng = createPRNG(42);

  // Create one Q-network per agent
  const agentNets: MLPWeights[] = [];
  for (let a = 0; a < nAgents; a++) {
    const layers: MLPWeights['layers'] = [
      {
        weight: initXavier(obsPerAgent, agentHiddenDim, rng),
        bias: new Float64Array(agentHiddenDim),
        inDim: obsPerAgent,
        outDim: agentHiddenDim,
      },
      {
        weight: initXavier(agentHiddenDim, agentHiddenDim, rng),
        bias: new Float64Array(agentHiddenDim),
        inDim: agentHiddenDim,
        outDim: agentHiddenDim,
      },
      {
        weight: initXavier(agentHiddenDim, nActions, rng),
        bias: new Float64Array(nActions),
        inDim: agentHiddenDim,
        outDim: nActions,
      },
    ];
    agentNets.push({ layers, activation: 'relu' });
  }

  // Create mixing network weights
  // w1: (mixingHiddenDim x nAgents) — will be abs-constrained during mixing
  // w2: (1 x mixingHiddenDim) — will be abs-constrained during mixing
  const w1 = initXavier(nAgents, mixingHiddenDim, rng);
  const b1 = new Float64Array(mixingHiddenDim);
  const w2 = initXavier(mixingHiddenDim, 1, rng);
  const b2 = new Float64Array(1);

  const mixerWeights: QMIXMixerWeights = { w1, b1, w2, b2 };

  return { agentNets, mixerWeights };
}

// ---------------------------------------------------------------------------
// qmixAgentQValues
// ---------------------------------------------------------------------------

/**
 * Compute Q-values for all actions given a single agent's observation.
 *
 * Runs a forward pass through the agent's Q-network. The output is a vector
 * of Q-values, one per discrete action.
 *
 * @param agentNet  Agent Q-network MLP weights.
 * @param obs       Agent's observation vector (obsPerAgent).
 * @returns         Q-value per action (nActions).
 */
export function qmixAgentQValues(
  agentNet: MLPWeights,
  obs: Float64Array,
): Float64Array {
  // Inline MLP forward pass
  let current = obs;
  const nLayers = agentNet.layers.length;

  for (let l = 0; l < nLayers; l++) {
    const layer = agentNet.layers[l]!;
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
      if (agentNet.activation === 'relu') {
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
// qmixMix
// ---------------------------------------------------------------------------

/**
 * Mix individual agent Q-values into a joint Q-value using the QMIX
 * hypernetwork-style mixer.
 *
 * The mixing is computed as:
 *   hidden = abs(w1) * Q_agents + b1   (element-wise abs on weights)
 *   hidden = relu(hidden)
 *   Q_tot  = abs(w2) * hidden + b2
 *
 * The absolute value on weights ensures monotonicity: dQ_tot/dQ_i >= 0,
 * enabling decentralized greedy action selection.
 *
 * Note: In the full QMIX algorithm, w1, b1, w2, b2 are produced by
 * hypernetworks conditioned on global state. Here we use them directly.
 *
 * @param agentQValues   Per-agent Q-values (one Float64Array per agent,
 *                       each containing a single chosen Q-value or the
 *                       selected Q for that agent).
 * @param mixerWeights   Mixing network weights.
 * @param state          Global state vector (used conceptually; in the full
 *                       QMIX this conditions the hypernetwork).
 * @returns              Scalar joint Q-value (Q_tot).
 */
export function qmixMix(
  agentQValues: Float64Array[],
  mixerWeights: QMIXMixerWeights,
  state: Float64Array,
): number {
  const { w1, b1, w2, b2 } = mixerWeights;
  const nAgents = agentQValues.length;
  const mixingHiddenDim = b1.length;

  // Flatten agent Q-values into a single vector
  // Each entry is the selected Q-value for that agent (scalar)
  const qFlat = new Float64Array(nAgents);
  for (let a = 0; a < nAgents; a++) {
    // If the agent provides a single Q-value, use it; if multiple, take max
    const qv = agentQValues[a]!;
    if (qv.length === 1) {
      qFlat[a] = qv[0]!;
    } else {
      // Take argmax Q-value (greedy)
      let maxQ = qv[0]!;
      for (let i = 1; i < qv.length; i++) {
        if (qv[i]! > maxQ) {
          maxQ = qv[i]!;
        }
      }
      qFlat[a] = maxQ;
    }
  }

  // Layer 1: hidden = |w1| * qFlat + b1
  // w1 is (mixingHiddenDim x nAgents), row-major
  const hidden = new Float64Array(mixingHiddenDim);
  for (let i = 0; i < mixingHiddenDim; i++) {
    let sum = b1[i]!;
    for (let j = 0; j < nAgents; j++) {
      sum += Math.abs(w1[i * nAgents + j]!) * qFlat[j]!;
    }
    hidden[i] = sum;
  }

  // ReLU activation
  for (let i = 0; i < mixingHiddenDim; i++) {
    if (hidden[i]! < 0) {
      hidden[i] = 0;
    }
  }

  // Layer 2: output = |w2| * hidden + b2
  // w2 is (1 x mixingHiddenDim), row-major
  let qTot = b2[0]!;
  for (let i = 0; i < mixingHiddenDim; i++) {
    qTot += Math.abs(w2[i]!) * hidden[i]!;
  }

  return qTot;
}
