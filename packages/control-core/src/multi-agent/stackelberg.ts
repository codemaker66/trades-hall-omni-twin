// ---------------------------------------------------------------------------
// OC-7: Stackelberg Game Solver
// ---------------------------------------------------------------------------

import type { StackelbergConfig } from '../types.js';

// ---------------------------------------------------------------------------
// solveStackelberg
// ---------------------------------------------------------------------------

/**
 * Solve a Stackelberg game with one leader and multiple followers.
 *
 * The leader commits to an action first. Each follower then independently
 * chooses a best response to the leader's action. The leader selects the
 * action that maximizes its own payoff anticipating the followers'
 * best responses.
 *
 * Both the leader and followers are optimized via grid search over their
 * respective action spaces.
 *
 * Algorithm:
 *   1. Enumerate candidate leader actions (grid over leaderBounds).
 *   2. For each leader candidate, compute each follower's best response
 *      via grid search over followerBounds.
 *   3. Evaluate the leader's payoff given the candidate and followers'
 *      best responses.
 *   4. Return the leader action with the highest payoff.
 *
 * @param config          Stackelberg game configuration.
 * @param leaderBounds    Per-dimension min/max bounds for the leader's actions.
 * @param followerBounds  Per-dimension min/max bounds for each follower's actions.
 * @param nSamples        Number of samples per dimension for grid search (default 11).
 * @returns               Optimal leader action, follower best responses, and leader payoff.
 */
export function solveStackelberg(
  config: StackelbergConfig,
  leaderBounds: { min: Float64Array; max: Float64Array },
  followerBounds: { min: Float64Array; max: Float64Array },
  nSamples: number = 11,
): {
  leaderAction: Float64Array;
  followerActions: Float64Array[];
  leaderPayoff: number;
} {
  const { leaderPayoff, followerPayoffs, nFollowers, leaderActionDim, followerActionDim } = config;

  const totalLeaderCandidates = Math.pow(nSamples, leaderActionDim);

  let bestLeaderAction = new Float64Array(leaderActionDim);
  let bestFollowerActions: Float64Array[] = [];
  let bestLeaderPayoff = -Infinity;

  for (let li = 0; li < totalLeaderCandidates; li++) {
    // Decode leader candidate from flat index
    const leaderCandidate = new Float64Array(leaderActionDim);
    let rem = li;
    for (let d = leaderActionDim - 1; d >= 0; d--) {
      const dimIdx = rem % nSamples;
      rem = Math.floor(rem / nSamples);
      const lo = leaderBounds.min[d]!;
      const hi = leaderBounds.max[d]!;
      leaderCandidate[d] = nSamples > 1
        ? lo + (hi - lo) * dimIdx / (nSamples - 1)
        : (lo + hi) / 2;
    }

    // Compute each follower's best response to this leader candidate
    const followerResponses: Float64Array[] = [];
    for (let f = 0; f < nFollowers; f++) {
      const followerBR = followerBestResponse(
        followerPayoffs[f]!,
        leaderCandidate,
        followerActionDim,
        followerBounds,
        nSamples,
      );
      followerResponses.push(followerBR);
    }

    // Evaluate leader payoff
    const lp = leaderPayoff(leaderCandidate, followerResponses);

    if (lp > bestLeaderPayoff) {
      bestLeaderPayoff = lp;
      bestLeaderAction = new Float64Array(leaderCandidate);
      bestFollowerActions = followerResponses.map((a) => new Float64Array(a));
    }
  }

  return {
    leaderAction: bestLeaderAction,
    followerActions: bestFollowerActions,
    leaderPayoff: bestLeaderPayoff,
  };
}

// ---------------------------------------------------------------------------
// followerBestResponse (internal helper)
// ---------------------------------------------------------------------------

/**
 * Internal: grid-search best response for a single follower given a fixed
 * leader action.
 */
function followerBestResponse(
  payoff: (leaderAction: Float64Array, followerAction: Float64Array) => number,
  leaderAction: Float64Array,
  followerActionDim: number,
  followerBounds: { min: Float64Array; max: Float64Array },
  nSamples: number,
): Float64Array {
  const totalCandidates = Math.pow(nSamples, followerActionDim);
  let bestAction = new Float64Array(followerActionDim);
  let bestValue = -Infinity;

  for (let idx = 0; idx < totalCandidates; idx++) {
    const candidate = new Float64Array(followerActionDim);
    let rem = idx;
    for (let d = followerActionDim - 1; d >= 0; d--) {
      const dimIdx = rem % nSamples;
      rem = Math.floor(rem / nSamples);
      const lo = followerBounds.min[d]!;
      const hi = followerBounds.max[d]!;
      candidate[d] = nSamples > 1
        ? lo + (hi - lo) * dimIdx / (nSamples - 1)
        : (lo + hi) / 2;
    }

    const value = payoff(leaderAction, candidate);
    if (value > bestValue) {
      bestValue = value;
      bestAction = new Float64Array(candidate);
    }
  }

  return bestAction;
}
