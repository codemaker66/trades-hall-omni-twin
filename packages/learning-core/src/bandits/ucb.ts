// ---------------------------------------------------------------------------
// UCB1 (Auer et al. 2002) — Stochastic Multi-Armed Bandits
// ---------------------------------------------------------------------------

import type { BanditState, BanditArm } from '../types.js';

/**
 * Create initial UCB state with the given number of arms.
 * All arms start with 0 pulls and 0 reward.
 * @param nArms Number of arms
 * @returns Initial bandit state
 */
export function createUCBState(nArms: number): BanditState {
  const arms: BanditArm[] = new Array<BanditArm>(nArms);
  for (let i = 0; i < nArms; i++) {
    arms[i] = {
      index: i,
      pulls: 0,
      totalReward: 0,
      meanReward: 0,
    };
  }
  return {
    arms,
    totalPulls: 0,
    cumulativeRegret: 0,
  };
}

/**
 * Select an arm using UCB1 policy.
 * UCB1 index: mean_i + sqrt(2 * ln(t) / n_i)
 *
 * Arms that have never been pulled are selected first (round-robin).
 * @param state Current bandit state
 * @returns Index of the arm to pull
 */
export function ucbSelect(state: BanditState): number {
  const { arms, totalPulls } = state;

  // First, pull any arm that has never been tried
  for (let i = 0; i < arms.length; i++) {
    const arm = arms[i]!;
    if (arm.pulls === 0) {
      return i;
    }
  }

  // All arms have been tried at least once; use UCB1 formula
  let bestIndex = 0;
  let bestValue = -Infinity;

  const logT = Math.log(totalPulls);

  for (let i = 0; i < arms.length; i++) {
    const arm = arms[i]!;
    const explorationBonus = Math.sqrt((2 * logT) / arm.pulls);
    const ucbValue = arm.meanReward + explorationBonus;

    if (ucbValue > bestValue) {
      bestValue = ucbValue;
      bestIndex = i;
    }
  }

  return bestIndex;
}

/**
 * Update bandit state after pulling an arm and observing a reward.
 * Returns a new state (immutable update).
 * @param state Current bandit state
 * @param arm Index of the arm that was pulled
 * @param reward Observed reward
 * @returns Updated bandit state
 */
export function ucbUpdate(state: BanditState, arm: number, reward: number): BanditState {
  const newArms: BanditArm[] = new Array<BanditArm>(state.arms.length);

  for (let i = 0; i < state.arms.length; i++) {
    const existing = state.arms[i]!;
    if (i === arm) {
      const newPulls = existing.pulls + 1;
      const newTotalReward = existing.totalReward + reward;
      newArms[i] = {
        index: i,
        pulls: newPulls,
        totalReward: newTotalReward,
        meanReward: newTotalReward / newPulls,
      };
    } else {
      newArms[i] = { ...existing };
    }
  }

  return {
    arms: newArms,
    totalPulls: state.totalPulls + 1,
    cumulativeRegret: state.cumulativeRegret, // Caller can compute regret externally
  };
}
