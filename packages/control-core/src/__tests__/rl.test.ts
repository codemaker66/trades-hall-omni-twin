// ---------------------------------------------------------------------------
// Tests for OC-5: Reinforcement Learning
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';
import {
  createSACAgent,
  createReplayBuffer,
  pushReplayBuffer,
  sampleReplayBuffer,
  createPPOAgent,
  computeGAE,
  cqlLoss,
  safeProject,
  computeVenueReward,
  potentialShaping,
} from '../rl/index.js';
import { createPRNG } from '../types.js';
import type { SACConfig, PPOConfig, RewardConfig } from '../types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const baseSACConfig: SACConfig = {
  stateDim: 4,
  actionDim: 2,
  hiddenDim: 16,
  lr: 3e-4,
  gamma: 0.99,
  batchSize: 32,
  tau: 0.005,
  alphaEntropy: 0.2,
  replayCapacity: 1000,
};

const basePPOConfig: PPOConfig = {
  stateDim: 4,
  actionDim: 2,
  hiddenDim: 16,
  lr: 3e-4,
  gamma: 0.99,
  batchSize: 64,
  clipEpsilon: 0.2,
  epochsPerBatch: 4,
  gaeLambda: 0.95,
  entropyCoeff: 0.01,
};

const baseRewardConfig: RewardConfig = {
  revenueWeight: 1.0,
  overcrowdingPenalty: 10.0,
  overcrowdingSafe: 2.0,
  priceStabilityPenalty: 0.5,
  queuePenalty: 1.0,
  queueMax: 5,
  churnPenalty: 2.0,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('SAC Agent', () => {
  it('createSACAgent returns MLPWeights with correct layer dimensions', () => {
    const agent = createSACAgent(baseSACConfig);

    // 3 layers: input->hidden, hidden->hidden, hidden->output
    expect(agent.layers).toHaveLength(3);
    expect(agent.activation).toBe('relu');

    // Layer 0: stateDim -> hiddenDim
    const l0 = agent.layers[0]!;
    expect(l0.inDim).toBe(4);
    expect(l0.outDim).toBe(16);
    expect(l0.weight).toHaveLength(16 * 4);
    expect(l0.bias).toHaveLength(16);

    // Layer 1: hiddenDim -> hiddenDim
    const l1 = agent.layers[1]!;
    expect(l1.inDim).toBe(16);
    expect(l1.outDim).toBe(16);
    expect(l1.weight).toHaveLength(16 * 16);

    // Layer 2: hiddenDim -> actionDim
    const l2 = agent.layers[2]!;
    expect(l2.inDim).toBe(16);
    expect(l2.outDim).toBe(2);
    expect(l2.weight).toHaveLength(16 * 2);
    expect(l2.bias).toHaveLength(2);
  });
});

describe('Replay Buffer', () => {
  it('circular buffer works and size tracks correctly', () => {
    const buf = createReplayBuffer(3);
    expect(buf.size).toBe(0);
    expect(buf.capacity).toBe(3);

    const s1 = new Float64Array([1, 0, 0, 0]);
    const a1 = new Float64Array([0.5, -0.5]);
    const s2 = new Float64Array([0, 1, 0, 0]);

    // Push 3 transitions -- fills buffer
    pushReplayBuffer(buf, s1, a1, 1.0, s2, false);
    expect(buf.size).toBe(1);

    pushReplayBuffer(buf, s2, a1, 2.0, s1, false);
    expect(buf.size).toBe(2);

    pushReplayBuffer(buf, s1, a1, 3.0, s2, true);
    expect(buf.size).toBe(3);

    // Push one more -- overwrites oldest (index 0)
    const s3 = new Float64Array([0, 0, 1, 0]);
    pushReplayBuffer(buf, s3, a1, 4.0, s1, false);
    expect(buf.size).toBe(3); // capped at capacity

    // The overwritten slot (index 0) should now contain s3
    expect(buf.states[0]![0]).toBe(0);
    expect(buf.states[0]![2]).toBe(1);
    expect(buf.rewards[0]).toBe(4.0);
  });

  it('sampleReplayBuffer returns correct batch size', () => {
    const buf = createReplayBuffer(100);
    const rng = createPRNG(42);

    const s = new Float64Array([1, 2, 3, 4]);
    const a = new Float64Array([0.1, 0.2]);
    const sn = new Float64Array([5, 6, 7, 8]);

    // Fill buffer with 20 transitions
    for (let i = 0; i < 20; i++) {
      pushReplayBuffer(buf, s, a, i * 0.1, sn, i === 19);
    }

    const batch = sampleReplayBuffer(buf, 8, rng);
    expect(batch.states).toHaveLength(8);
    expect(batch.actions).toHaveLength(8);
    expect(batch.nextStates).toHaveLength(8);
    expect(batch.rewards).toHaveLength(8);
    expect(batch.dones).toHaveLength(8);

    // Each sampled state should match the original shape
    expect(batch.states[0]!).toHaveLength(4);
    expect(batch.actions[0]!).toHaveLength(2);

    // Requesting more than buffer size returns buf.size samples
    const largeBatch = sampleReplayBuffer(buf, 100, rng);
    expect(largeBatch.states).toHaveLength(20);
  });
});

describe('PPO Agent', () => {
  it('createPPOAgent returns actor and critic with correct architecture', () => {
    const { actor, critic } = createPPOAgent(basePPOConfig);

    // Both should have 3 layers
    expect(actor.layers).toHaveLength(3);
    expect(critic.layers).toHaveLength(3);

    // Actor uses tanh
    expect(actor.activation).toBe('tanh');
    expect(critic.activation).toBe('tanh');

    // Actor output = actionDim
    expect(actor.layers[2]!.outDim).toBe(2);
    // Critic output = 1 (state value)
    expect(critic.layers[2]!.outDim).toBe(1);

    // Input dim = stateDim for both
    expect(actor.layers[0]!.inDim).toBe(4);
    expect(critic.layers[0]!.inDim).toBe(4);
  });
});

describe('Generalized Advantage Estimation', () => {
  it('computeGAE matches manual calculation for a simple 3-step trajectory', () => {
    // 3-step trajectory, no terminal
    const rewards = new Float64Array([1, 2, 3]);
    const values = new Float64Array([10, 20, 30, 40]); // T+1 values
    const dones = new Float64Array([0, 0, 0]);
    const gamma = 0.5;
    const lambda = 0.5;

    const advantages = computeGAE(rewards, values, dones, gamma, lambda);
    expect(advantages).toHaveLength(3);

    // Manual backward calculation:
    // t=2: delta_2 = r_2 + gamma * V(s_3) - V(s_2) = 3 + 0.5*40 - 30 = -7
    //       A_2 = delta_2 = -7
    const delta2 = 3 + 0.5 * 40 - 30;
    const A2 = delta2;

    // t=1: delta_1 = r_1 + gamma * V(s_2) - V(s_1) = 2 + 0.5*30 - 20 = -3
    //       A_1 = delta_1 + gamma * lambda * A_2 = -3 + 0.5*0.5*(-7) = -4.75
    const delta1 = 2 + 0.5 * 30 - 20;
    const A1 = delta1 + 0.5 * 0.5 * A2;

    // t=0: delta_0 = r_0 + gamma * V(s_1) - V(s_0) = 1 + 0.5*20 - 10 = 1
    //       A_0 = delta_0 + gamma * lambda * A_1 = 1 + 0.5*0.5*(-4.75) = -0.1875
    const delta0 = 1 + 0.5 * 20 - 10;
    const A0 = delta0 + 0.5 * 0.5 * A1;

    expect(advantages[0]).toBeCloseTo(A0, 10);
    expect(advantages[1]).toBeCloseTo(A1, 10);
    expect(advantages[2]).toBeCloseTo(A2, 10);
  });
});

describe('CQL Loss', () => {
  it('penalizes OOD actions (cqlLoss > 0 when Q-values are uniformly high)', () => {
    // High Q-values for all actions
    const qValues = new Float64Array([10, 10, 10, 10, 10]);
    // Only action 0 is in the dataset
    const dataActions = new Int32Array([0]);
    const alpha = 1.0;

    const loss = cqlLoss(qValues, dataActions, alpha);

    // logsumexp of five 10s = 10 + log(5)
    // data mean = 10
    // loss = alpha * (10 + log(5) - 10) = log(5) ~ 1.609
    expect(loss).toBeGreaterThan(0);
    expect(loss).toBeCloseTo(Math.log(5), 5);
  });
});

describe('Safe RL', () => {
  it('safeProject projects violated constraint back to feasible', () => {
    // Action in 2D
    const action = new Float64Array([3, 4]);
    // Constraint gradient pointing in the direction [1, 0]
    const constraintGrad = new Float64Array([1, 0]);
    // Constraint is violated (positive value)
    const constraintVal = 1.0;

    const safe = safeProject(action, constraintGrad, constraintVal);

    // Projection: a_safe = a - (g / ||grad||^2) * grad
    // ||grad||^2 = 1, so a_safe = [3 - 1*1, 4 - 1*0] = [2, 4]
    expect(safe[0]).toBeCloseTo(2.0, 10);
    expect(safe[1]).toBeCloseTo(4.0, 10);

    // When constraint is satisfied, original action is returned
    const feasible = safeProject(action, constraintGrad, -1.0);
    expect(feasible[0]).toBeCloseTo(3.0, 10);
    expect(feasible[1]).toBeCloseTo(4.0, 10);
  });
});

describe('Reward Shaping', () => {
  it('computeVenueReward: high revenue yields higher reward than low revenue', () => {
    const highRevReward = computeVenueReward(
      baseRewardConfig,
      /* revenue */ 1000,
      /* occupancy */ 1.0, // below safe threshold
      /* priceChange */ 0,
      /* queueLength */ 0,
      /* churnRate */ 0,
    );

    const lowRevReward = computeVenueReward(
      baseRewardConfig,
      /* revenue */ 100,
      /* occupancy */ 1.0,
      /* priceChange */ 0,
      /* queueLength */ 0,
      /* churnRate */ 0,
    );

    expect(highRevReward).toBeGreaterThan(lowRevReward);

    // Overcrowding penalty should reduce reward
    const crowdedReward = computeVenueReward(
      baseRewardConfig,
      /* revenue */ 1000,
      /* occupancy */ 5.0, // well above safe threshold of 2.0
      /* priceChange */ 0,
      /* queueLength */ 0,
      /* churnRate */ 0,
    );

    expect(crowdedReward).toBeLessThan(highRevReward);
  });

  it('potentialShaping: gamma * curr - prev formula', () => {
    const prevPotential = 2.0;
    const currPotential = 5.0;
    const gamma = 0.99;

    const shaping = potentialShaping(prevPotential, currPotential, gamma);

    // F = gamma * Phi(s') - Phi(s) = 0.99 * 5 - 2 = 2.95
    expect(shaping).toBeCloseTo(0.99 * 5.0 - 2.0, 10);

    // When both potentials are the same and gamma=1, shaping is 0
    const zeroShaping = potentialShaping(3.0, 3.0, 1.0);
    expect(zeroShaping).toBeCloseTo(0.0, 10);
  });
});
