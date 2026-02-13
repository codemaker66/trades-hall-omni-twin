// ---------------------------------------------------------------------------
// OC-5: Proximal Policy Optimization (PPO)
// ---------------------------------------------------------------------------

import type { MLPWeights, PRNG, PPOConfig } from '../types.js';
import { mlpForward } from './sac.js';

// ---------------------------------------------------------------------------
// Xavier-like Weight Initialization (local copy)
// ---------------------------------------------------------------------------

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
// PPO Agent Creation
// ---------------------------------------------------------------------------

/**
 * Create a PPO agent with separate actor and critic MLPs.
 *
 * Both networks share the same architecture:
 *   input -> hidden(hiddenDim) -> hidden(hiddenDim) -> output
 *
 * Actor output dimension = actionDim (mean of Gaussian policy).
 * Critic output dimension = 1 (state value estimate).
 *
 * @param config  PPO configuration.
 * @returns       Actor and critic MLP weights.
 */
export function createPPOAgent(
  config: PPOConfig,
): { actor: MLPWeights; critic: MLPWeights } {
  const { stateDim, actionDim, hiddenDim } = config;

  const actorRNG = seedRNG(123);
  const criticRNG = seedRNG(456);

  function buildMLP(
    inputDim: number,
    outputDim: number,
    hDim: number,
    rng: PRNG,
  ): MLPWeights {
    return {
      layers: [
        {
          weight: initXavier(inputDim, hDim, rng),
          bias: new Float64Array(hDim),
          inDim: inputDim,
          outDim: hDim,
        },
        {
          weight: initXavier(hDim, hDim, rng),
          bias: new Float64Array(hDim),
          inDim: hDim,
          outDim: hDim,
        },
        {
          weight: initXavier(hDim, outputDim, rng),
          bias: new Float64Array(outputDim),
          inDim: hDim,
          outDim: outputDim,
        },
      ],
      activation: 'tanh',
    };
  }

  const actor = buildMLP(stateDim, actionDim, hiddenDim, actorRNG);
  const critic = buildMLP(stateDim, 1, hiddenDim, criticRNG);

  return { actor, critic };
}

// ---------------------------------------------------------------------------
// PPO Action Selection
// ---------------------------------------------------------------------------

/**
 * Select an action from the PPO actor with a Gaussian policy.
 *
 * The actor network outputs the mean of a Gaussian distribution with
 * fixed standard deviation (0.5). An action is sampled and its log
 * probability is computed.
 *
 * log_prob = -0.5 * sum_i [ ((a_i - mu_i) / sigma)^2 + log(2*pi*sigma^2) ]
 *
 * @param actor  Actor MLP weights.
 * @param state  Current state vector.
 * @param rng    PRNG for sampling.
 * @returns      Sampled action and its log probability under the policy.
 */
export function ppoSelectAction(
  actor: MLPWeights,
  state: Float64Array,
  rng: PRNG,
): { action: Float64Array; logProb: number } {
  const mean = mlpForward(actor, state);
  const sigma = 0.5; // fixed standard deviation
  const actionDim = mean.length;
  const action = new Float64Array(actionDim);

  let logProb = 0;
  const logConst = Math.log(2 * Math.PI * sigma * sigma);

  for (let i = 0; i < actionDim; i++) {
    // Box-Muller transform
    const u1 = Math.max(rng(), 1e-10);
    const u2 = rng();
    const noise = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);

    action[i] = mean[i]! + sigma * noise;

    // log probability of this sample
    const diff = (action[i]! - mean[i]!) / sigma;
    logProb += -0.5 * (diff * diff + logConst);
  }

  return { action, logProb };
}

// ---------------------------------------------------------------------------
// Generalized Advantage Estimation (GAE)
// ---------------------------------------------------------------------------

/**
 * Compute Generalized Advantage Estimation (GAE) for a trajectory.
 *
 * Uses the backward recurrence:
 *   delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t)
 *   A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
 *
 * @param rewards  Reward at each timestep (T).
 * @param values   Value estimates V(s_t) at each timestep (T+1, including terminal V(s_T)).
 * @param dones    Done flags at each timestep (T), 1.0 = terminal, 0.0 = non-terminal.
 * @param gamma    Discount factor.
 * @param lambda   GAE lambda for bias-variance tradeoff.
 * @returns        Advantage estimates (T).
 */
export function computeGAE(
  rewards: Float64Array,
  values: Float64Array,
  dones: Float64Array,
  gamma: number,
  lambda: number,
): Float64Array {
  const T = rewards.length;
  const advantages = new Float64Array(T);

  let lastAdv = 0;

  for (let t = T - 1; t >= 0; t--) {
    const notDone = 1.0 - dones[t]!;
    const delta = rewards[t]! + gamma * notDone * values[t + 1]! - values[t]!;
    lastAdv = delta + gamma * lambda * notDone * lastAdv;
    advantages[t] = lastAdv;
  }

  return advantages;
}
