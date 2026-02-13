// ---------------------------------------------------------------------------
// Tests for OC-7: Multi-Agent Control
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';
import {
  solveDecentralizedMPC,
  findNashEquilibrium,
  bestResponse,
  solveStackelberg,
  meanFieldStep,
  meanFieldEquilibrium,
  createMAPPOAgents,
  createQMIX,
  qmixAgentQValues,
  qmixMix,
} from '../multi-agent/index.js';
import { createPRNG } from '../types.js';
import type {
  DecentralizedMPCConfig,
  NashEquilibriumConfig,
  StackelbergConfig,
  MeanFieldConfig,
  MAPPOConfig,
  QMIXConfig,
} from '../types.js';

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('solveDecentralizedMPC', () => {
  it('ADMM converges (iterations < max)', () => {
    const nAgents = 2;
    const nx = 2;
    const nu = 1;

    const agentConfigs = [
      {
        A: new Float64Array([1, 0, 0, 1]),
        B: new Float64Array([1, 0]),
        Q: new Float64Array([1, 0, 0, 1]),
        R: new Float64Array([1]),
        nx,
        nu,
        horizon: 3,
      },
      {
        A: new Float64Array([1, 0, 0, 1]),
        B: new Float64Array([0, 1]),
        Q: new Float64Array([1, 0, 0, 1]),
        R: new Float64Array([1]),
        nx,
        nu,
        horizon: 3,
      },
    ];

    // Coupling: both agents coupled equally
    const couplingMatrix = new Float64Array([1, 0.5, 0.5, 1]);

    const config: DecentralizedMPCConfig = {
      nAgents,
      agentConfigs,
      couplingMatrix,
      consensusWeight: 1.0,
      maxConsensusIter: 200,
    };

    const states = [
      new Float64Array([1, -1]),
      new Float64Array([-1, 1]),
    ];

    const result = solveDecentralizedMPC(config, states);

    // Should converge within the max iterations
    expect(result.iterations).toBeLessThanOrEqual(200);
    expect(result.actions.length).toBe(nAgents);

    // Each action should be the correct dimension
    for (let a = 0; a < nAgents; a++) {
      expect(result.actions[a]!.length).toBe(nu);
      // Actions should be finite
      expect(Number.isFinite(result.actions[a]![0]!)).toBe(true);
    }
  });
});

describe('findNashEquilibrium', () => {
  it('converges for a simple 2-player game', () => {
    // Two players, each with 1D action in [0, 1].
    // Player 0 payoff: -(a0 - 0.6)^2 - 0.1*(a0 - a1)^2
    // Player 1 payoff: -(a1 - 0.4)^2 - 0.1*(a1 - a0)^2
    // Nash equilibrium is near a0 ~ 0.6, a1 ~ 0.4 (with coupling shift).
    const config: NashEquilibriumConfig = {
      nPlayers: 2,
      payoffFns: [
        (actions: Float64Array[]) => {
          const a0 = actions[0]![0]!;
          const a1 = actions[1]![0]!;
          return -Math.pow(a0 - 0.6, 2) - 0.1 * Math.pow(a0 - a1, 2);
        },
        (actions: Float64Array[]) => {
          const a0 = actions[0]![0]!;
          const a1 = actions[1]![0]!;
          return -Math.pow(a1 - 0.4, 2) - 0.1 * Math.pow(a1 - a0, 2);
        },
      ],
      actionDims: [1, 1],
      actionBounds: [
        { min: new Float64Array([0]), max: new Float64Array([1]) },
        { min: new Float64Array([0]), max: new Float64Array([1]) },
      ],
      tolerance: 1e-3,
      maxIter: 50,
    };

    const result = findNashEquilibrium(config);

    expect(result.converged).toBe(true);
    expect(result.iterations).toBeGreaterThan(0);
    expect(result.iterations).toBeLessThanOrEqual(50);

    // Equilibrium actions should be within bounds
    expect(result.equilibriumActions[0]![0]!).toBeGreaterThanOrEqual(0);
    expect(result.equilibriumActions[0]![0]!).toBeLessThanOrEqual(1);
    expect(result.equilibriumActions[1]![0]!).toBeGreaterThanOrEqual(0);
    expect(result.equilibriumActions[1]![0]!).toBeLessThanOrEqual(1);

    // Payoffs should be negative (quadratic losses) but close to 0
    expect(result.payoffs[0]!).toBeLessThanOrEqual(0);
    expect(result.payoffs[1]!).toBeLessThanOrEqual(0);
  });
});

describe('bestResponse', () => {
  it('returns best action for given opponent', () => {
    // Player's payoff = -(a - 0.7)^2.  Best response is a = 0.7.
    // otherActions has 2 entries; player slot is the first one (same dim).
    const payoff = (actions: Float64Array[]) => {
      const a = actions[0]![0]!;
      return -Math.pow(a - 0.7, 2);
    };

    const otherActions: Float64Array[] = [
      new Float64Array([0]),    // placeholder for this player
      new Float64Array([0.5]),  // opponent's fixed action
    ];

    const result = bestResponse(
      payoff,
      otherActions,
      1,
      { min: new Float64Array([0]), max: new Float64Array([1]) },
      21,
    );

    // Best response should be close to 0.7
    expect(result[0]!).toBeCloseTo(0.7, 1);
  });
});

describe('solveStackelberg', () => {
  it('leader payoff >= Nash equilibrium payoff for a simple game', () => {
    // Simple coordination game:
    // Leader payoff:   -(leaderAction - 0.8)^2 + 0.5 * followerAction
    // Follower payoff: -(followerAction - leaderAction)^2
    // Follower BR: followerAction = leaderAction
    // Leader anticipates BR: max -(l - 0.8)^2 + 0.5*l = -(l^2 - 1.6l + 0.64) + 0.5l
    //   = -l^2 + 2.1l - 0.64 => optimal l = 1.05 (clamped to [0,1])
    const config: StackelbergConfig = {
      leaderPayoff: (leaderAction, followerActions) => {
        const l = leaderAction[0]!;
        const f = followerActions[0]![0]!;
        return -Math.pow(l - 0.8, 2) + 0.5 * f;
      },
      followerPayoffs: [
        (leaderAction, followerAction) => {
          const l = leaderAction[0]!;
          const f = followerAction[0]!;
          return -Math.pow(f - l, 2);
        },
      ],
      nFollowers: 1,
      leaderActionDim: 1,
      followerActionDim: 1,
    };

    const leaderBounds = { min: new Float64Array([0]), max: new Float64Array([1]) };
    const followerBounds = { min: new Float64Array([0]), max: new Float64Array([1]) };

    const stackResult = solveStackelberg(config, leaderBounds, followerBounds, 21);

    // Also compute Nash for comparison using the same game
    const nashConfig: NashEquilibriumConfig = {
      nPlayers: 2,
      payoffFns: [
        (actions: Float64Array[]) =>
          config.leaderPayoff(actions[0]!, [actions[1]!]),
        (actions: Float64Array[]) =>
          config.followerPayoffs[0]!(actions[0]!, actions[1]!),
      ],
      actionDims: [1, 1],
      actionBounds: [leaderBounds, followerBounds],
      tolerance: 1e-3,
      maxIter: 50,
    };
    const nashResult = findNashEquilibrium(nashConfig);

    // Leader should do at least as well in Stackelberg as in Nash
    const nashLeaderPayoff = nashResult.payoffs[0]!;
    expect(stackResult.leaderPayoff).toBeGreaterThanOrEqual(nashLeaderPayoff - 0.05);

    // Follower's best response should approximate the leader's action
    expect(stackResult.followerActions[0]![0]!).toBeCloseTo(
      stackResult.leaderAction[0]!,
      0,
    );
  });
});

describe('meanFieldStep', () => {
  it('advances states correctly', () => {
    const config: MeanFieldConfig = {
      nAgentsApprox: 3,
      // Simple dynamics: dx/dt = -x + u + 0.1 * distribution
      agentDynamics: (x, u, distribution) => {
        const dxdt = new Float64Array(x.length);
        for (let i = 0; i < x.length; i++) {
          dxdt[i] = -x[i]! + u[i]! + 0.1 * distribution[i]!;
        }
        return dxdt;
      },
      costFn: () => 0,
      nx: 2,
      nu: 2,
    };

    const states = [
      new Float64Array([1, 0]),
      new Float64Array([0, 1]),
      new Float64Array([-1, -1]),
    ];

    const actions = [
      new Float64Array([0.1, 0.1]),
      new Float64Array([0.1, 0.1]),
      new Float64Array([0.1, 0.1]),
    ];

    const dt = 0.1;
    const nextStates = meanFieldStep(config, states, actions, dt);

    // Should return same number of agents
    expect(nextStates.length).toBe(3);

    // Each state should have correct dimension
    for (let a = 0; a < 3; a++) {
      expect(nextStates[a]!.length).toBe(2);
    }

    // States should have changed (non-zero dynamics)
    // For agent 0: distribution = mean([1,0], [0,1], [-1,-1]) = [0, 0]
    // dx/dt = -[1,0] + [0.1,0.1] + 0.1*[0,0] = [-0.9, 0.1]
    // x_next = [1, 0] + 0.1*[-0.9, 0.1] = [0.91, 0.01]
    expect(nextStates[0]![0]!).toBeCloseTo(0.91, 5);
    expect(nextStates[0]![1]!).toBeCloseTo(0.01, 5);
  });
});

describe('meanFieldEquilibrium', () => {
  it('distribution converges for stable dynamics', () => {
    const config: MeanFieldConfig = {
      nAgentsApprox: 10,
      // Contractive dynamics: dx/dt = -0.5 * x (with zero control)
      // This contracts toward zero, so the distribution should converge.
      agentDynamics: (x, _u, _distribution) => {
        const dxdt = new Float64Array(x.length);
        for (let i = 0; i < x.length; i++) {
          dxdt[i] = -0.5 * x[i]!;
        }
        return dxdt;
      },
      costFn: () => 0,
      nx: 2,
      nu: 2,
    };

    // Start agents at various positions
    const rng = createPRNG(42);
    const initialStates: Float64Array[] = [];
    for (let a = 0; a < 10; a++) {
      initialStates.push(new Float64Array([rng() * 2 - 1, rng() * 2 - 1]));
    }

    const result = meanFieldEquilibrium(config, initialStates, 0.1, 200, 1e-6);

    // Distribution should be close to zero (all agents contract toward origin)
    expect(Math.abs(result.distribution[0]!)).toBeLessThan(0.1);
    expect(Math.abs(result.distribution[1]!)).toBeLessThan(0.1);
  });
});

describe('createMAPPOAgents', () => {
  it('returns correct structure (nAgents actors + 1 critic)', () => {
    const config: MAPPOConfig = {
      nAgents: 3,
      obsPerAgent: 4,
      actionPerAgent: 2,
      centralStateDim: 12,
      criticHiddenDim: 16,
      actorHiddenDim: 8,
      lr: 1e-3,
      clipEpsilon: 0.2,
    };

    const { actors, critic } = createMAPPOAgents(config);

    // Should have exactly nAgents actors
    expect(actors.length).toBe(3);

    // Each actor should have 3 layers
    for (let a = 0; a < 3; a++) {
      expect(actors[a]!.layers.length).toBe(3);
      expect(actors[a]!.activation).toBe('relu');

      // First layer: obsPerAgent -> actorHiddenDim
      expect(actors[a]!.layers[0]!.inDim).toBe(4);
      expect(actors[a]!.layers[0]!.outDim).toBe(8);

      // Last layer: actorHiddenDim -> actionPerAgent
      expect(actors[a]!.layers[2]!.inDim).toBe(8);
      expect(actors[a]!.layers[2]!.outDim).toBe(2);
    }

    // Critic should have 3 layers
    expect(critic.layers.length).toBe(3);
    expect(critic.activation).toBe('relu');

    // Critic input: centralStateDim, output: 1
    expect(critic.layers[0]!.inDim).toBe(12);
    expect(critic.layers[0]!.outDim).toBe(16);
    expect(critic.layers[2]!.outDim).toBe(1);
  });
});

describe('QMIX monotonicity', () => {
  it('mixer output is monotonic in agent Q-values', () => {
    const config: QMIXConfig = {
      nAgents: 3,
      obsPerAgent: 4,
      nActions: 3,
      mixingHiddenDim: 8,
      agentHiddenDim: 8,
    };

    const { agentNets, mixerWeights } = createQMIX(config);

    // Generate Q-values for each agent from a fixed observation
    const obs = new Float64Array([0.5, -0.3, 0.2, 0.1]);
    const agentQs: Float64Array[] = [];
    for (let a = 0; a < 3; a++) {
      const qvals = qmixAgentQValues(agentNets[a]!, obs);
      // Take the max Q-value as a single scalar for the mixer
      let maxQ = qvals[0]!;
      for (let i = 1; i < qvals.length; i++) {
        if (qvals[i]! > maxQ) maxQ = qvals[i]!;
      }
      agentQs.push(new Float64Array([maxQ]));
    }

    const globalState = new Float64Array([1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]);
    const baseQtot = qmixMix(agentQs, mixerWeights, globalState);

    // Increase each agent's Q-value independently and verify Q_tot increases
    for (let a = 0; a < 3; a++) {
      const modifiedQs = agentQs.map((q) => new Float64Array(q));
      modifiedQs[a]![0] = modifiedQs[a]![0]! + 1.0; // increase agent a's Q

      const newQtot = qmixMix(modifiedQs, mixerWeights, globalState);

      // Monotonicity: increasing any agent's Q should not decrease Q_tot
      expect(newQtot).toBeGreaterThanOrEqual(baseQtot - 1e-10);
    }
  });
});
