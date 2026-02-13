// ---------------------------------------------------------------------------
// OC-5: Soft Actor-Critic (SAC)
// ---------------------------------------------------------------------------

import type { MLPWeights, PRNG, ReplayBuffer, SACConfig } from '../types.js';
import { arrayToMatrix, matVecMul } from '../types.js';

// ---------------------------------------------------------------------------
// MLP Forward Pass (shared helper)
// ---------------------------------------------------------------------------

/**
 * Forward pass through a multi-layer perceptron.
 *
 * For each hidden layer: y = activation(W * x + b)
 * For the last layer: y = W * x + b (linear, no activation)
 *
 * @param weights  MLP layer weights and activation type.
 * @param input    Input vector.
 * @returns        Network output vector.
 */
export function mlpForward(
  weights: MLPWeights,
  input: Float64Array,
): Float64Array {
  let current = input;
  const nLayers = weights.layers.length;

  for (let l = 0; l < nLayers; l++) {
    const layer = weights.layers[l]!;
    const { weight, bias, inDim, outDim } = layer;

    // y = W * x + b
    const Wmat = arrayToMatrix(weight, outDim, inDim);
    const Wx = matVecMul(Wmat, current);

    const y = new Float64Array(outDim);
    for (let i = 0; i < outDim; i++) {
      y[i] = Wx[i]! + bias[i]!;
    }

    // Apply activation for all layers except the last (linear output)
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
// Xavier-like Weight Initialization
// ---------------------------------------------------------------------------

/**
 * Initialize a weight matrix with Xavier-like uniform random values.
 *
 * Samples from U(-scale, +scale) where scale = sqrt(6 / (fanIn + fanOut)).
 */
function initXavier(
  fanIn: number,
  fanOut: number,
  rng: PRNG,
): Float64Array {
  const scale = Math.sqrt(6.0 / (fanIn + fanOut));
  const w = new Float64Array(fanOut * fanIn);
  for (let i = 0; i < w.length; i++) {
    w[i] = (rng() * 2 - 1) * scale;
  }
  return w;
}

// ---------------------------------------------------------------------------
// SAC Agent Creation
// ---------------------------------------------------------------------------

/**
 * Create a Soft Actor-Critic agent as an MLP with Xavier-like initialization.
 *
 * Architecture: input(stateDim) -> hidden(hiddenDim) -> hidden(hiddenDim) -> output(actionDim)
 * Uses ReLU activation between hidden layers; last layer is linear.
 *
 * @param config  SAC configuration (extends RLConfig with tau, alphaEntropy, replayCapacity).
 * @returns       Initialized MLP weights for the actor network.
 */
export function createSACAgent(config: SACConfig): MLPWeights {
  const { stateDim, actionDim, hiddenDim } = config;
  const rng = seedRNG(42);

  const layers: MLPWeights['layers'] = [
    {
      weight: initXavier(stateDim, hiddenDim, rng),
      bias: new Float64Array(hiddenDim),
      inDim: stateDim,
      outDim: hiddenDim,
    },
    {
      weight: initXavier(hiddenDim, hiddenDim, rng),
      bias: new Float64Array(hiddenDim),
      inDim: hiddenDim,
      outDim: hiddenDim,
    },
    {
      weight: initXavier(hiddenDim, actionDim, rng),
      bias: new Float64Array(actionDim),
      inDim: hiddenDim,
      outDim: actionDim,
    },
  ];

  return { layers, activation: 'relu' };
}

/**
 * Simple internal deterministic PRNG for weight initialization.
 * Uses mulberry32 algorithm.
 */
function seedRNG(seed: number): PRNG {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// SAC Action Selection
// ---------------------------------------------------------------------------

/**
 * Select an action using the SAC actor network with Gaussian exploration noise.
 *
 * Runs a forward pass through the actor MLP to compute the mean action,
 * then adds Gaussian noise sampled via the Box-Muller transform for exploration.
 *
 * @param actor  Actor MLP weights.
 * @param state  Current state vector.
 * @param rng    PRNG for sampling exploration noise.
 * @returns      Action vector with exploration noise.
 */
export function sacSelectAction(
  actor: MLPWeights,
  state: Float64Array,
  rng: PRNG,
): Float64Array {
  const mean = mlpForward(actor, state);
  const action = new Float64Array(mean.length);

  for (let i = 0; i < mean.length; i++) {
    // Box-Muller transform for Gaussian noise
    const u1 = Math.max(rng(), 1e-10);
    const u2 = rng();
    const noise = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    action[i] = mean[i]! + 0.1 * noise; // small exploration std = 0.1
  }

  return action;
}

// ---------------------------------------------------------------------------
// Replay Buffer
// ---------------------------------------------------------------------------

/**
 * Create an empty circular experience replay buffer.
 *
 * The buffer stores (s, a, r, s', done) tuples and overwrites the oldest
 * entries once capacity is reached.
 *
 * @param capacity  Maximum number of transitions to store.
 * @returns         An empty ReplayBuffer.
 */
export function createReplayBuffer(capacity: number): ReplayBuffer {
  return {
    states: [],
    actions: [],
    nextStates: [],
    rewards: new Float64Array(capacity),
    dones: new Uint8Array(capacity),
    size: 0,
    capacity,
    ptr: 0,
  };
}

/**
 * Push a single transition into the replay buffer using circular insertion.
 *
 * Overwrites the oldest entry when the buffer is full.
 *
 * @param buf    Replay buffer.
 * @param s      State vector.
 * @param a      Action vector.
 * @param r      Scalar reward.
 * @param sNext  Next-state vector.
 * @param done   Whether the episode ended.
 */
export function pushReplayBuffer(
  buf: ReplayBuffer,
  s: Float64Array,
  a: Float64Array,
  r: number,
  sNext: Float64Array,
  done: boolean,
): void {
  const idx = buf.ptr;

  // Overwrite or append at the current pointer position
  buf.states[idx] = new Float64Array(s);
  buf.actions[idx] = new Float64Array(a);
  buf.nextStates[idx] = new Float64Array(sNext);
  buf.rewards[idx] = r;
  buf.dones[idx] = done ? 1 : 0;

  buf.ptr = (idx + 1) % buf.capacity;
  if (buf.size < buf.capacity) {
    buf.size++;
  }
}

/**
 * Sample a random mini-batch from the replay buffer.
 *
 * Draws `batchSize` random indices from the stored transitions and returns
 * them as parallel arrays.
 *
 * @param buf        Replay buffer.
 * @param batchSize  Number of transitions to sample.
 * @param rng        PRNG for random index generation.
 * @returns          Batch of (states, actions, rewards, nextStates, dones).
 */
export function sampleReplayBuffer(
  buf: ReplayBuffer,
  batchSize: number,
  rng: PRNG,
): {
  states: Float64Array[];
  actions: Float64Array[];
  rewards: Float64Array;
  nextStates: Float64Array[];
  dones: Uint8Array;
} {
  const n = Math.min(batchSize, buf.size);
  const states: Float64Array[] = [];
  const actions: Float64Array[] = [];
  const nextStates: Float64Array[] = [];
  const rewards = new Float64Array(n);
  const dones = new Uint8Array(n);

  for (let i = 0; i < n; i++) {
    const idx = Math.floor(rng() * buf.size);
    states.push(new Float64Array(buf.states[idx]!));
    actions.push(new Float64Array(buf.actions[idx]!));
    nextStates.push(new Float64Array(buf.nextStates[idx]!));
    rewards[i] = buf.rewards[idx]!;
    dones[i] = buf.dones[idx]!;
  }

  return { states, actions, rewards, nextStates, dones };
}
