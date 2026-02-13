// ---------------------------------------------------------------------------
// OC-7: Nash Equilibrium via Iterated Best Response
// ---------------------------------------------------------------------------

import type { NashEquilibriumConfig, NashResult } from '../types.js';
import { vecNorm, vecSub } from '../types.js';

// ---------------------------------------------------------------------------
// bestResponse
// ---------------------------------------------------------------------------

/**
 * Compute a player's best response given the other players' fixed actions.
 *
 * Performs a grid search over the player's action space (discretized into
 * `nSamples` points per dimension) and returns the action that maximizes the
 * player's payoff.
 *
 * @param payoff        Payoff function: (allActions) => scalar. The player's
 *                      actions are placed at the position determined by the caller.
 * @param otherActions  Actions of all other players (the player's own slot will
 *                      be filled during search).
 * @param actionDim     Dimension of the player's action space.
 * @param actionBounds  Per-dimension min/max bounds for the player's actions.
 * @param nSamples      Number of samples per dimension (default 11).
 * @returns             Action vector that maximizes payoff.
 */
export function bestResponse(
  payoff: (actions: Float64Array[]) => number,
  otherActions: Float64Array[],
  actionDim: number,
  actionBounds: { min: Float64Array; max: Float64Array },
  nSamples: number = 11,
): Float64Array {
  // For multi-dimensional action spaces, enumerate a grid over each dimension.
  // Total candidates = nSamples^actionDim. To keep computation bounded, if
  // actionDim > 1 we still iterate over a full grid (practical for small dims).

  const totalCandidates = Math.pow(nSamples, actionDim);
  let bestAction = new Float64Array(actionDim);
  let bestValue = -Infinity;

  for (let idx = 0; idx < totalCandidates; idx++) {
    // Decode multi-index from flat index
    const candidate = new Float64Array(actionDim);
    let rem = idx;
    for (let d = actionDim - 1; d >= 0; d--) {
      const dimIdx = rem % nSamples;
      rem = Math.floor(rem / nSamples);
      const lo = actionBounds.min[d]!;
      const hi = actionBounds.max[d]!;
      candidate[d] = nSamples > 1 ? lo + (hi - lo) * dimIdx / (nSamples - 1) : (lo + hi) / 2;
    }

    // Build the full action profile with this candidate inserted
    // otherActions already has all players' actions; we build a copy
    // placing the candidate at each position in the call context.
    // The caller is responsible for arranging otherActions such that
    // the candidate is inserted at the correct player index.
    // We build allActions = [...otherActions] but splice candidate in.
    // Convention: otherActions has length = nPlayers, with a placeholder
    // at the current player's index. We replace it.
    const allActions: Float64Array[] = [];
    let inserted = false;
    for (let p = 0; p < otherActions.length; p++) {
      // Detect the placeholder: the slot with the same length as actionDim
      // is the player's own slot
      if (!inserted && otherActions[p]!.length === actionDim) {
        // Check if this is the player slot by comparing dimensions
        // Actually, multiple players can have the same dimension.
        // We use a simpler convention: the caller sets the player's
        // entry to a zero-filled array of the right dimension, and
        // we just replace every entry that matches actionDim.
        // Better approach: we accept the full array and a playerIndex.
        // But to match the specified API, we replace the first match.
        allActions.push(candidate);
        inserted = true;
      } else {
        allActions.push(otherActions[p]!);
      }
    }

    // If we never inserted (e.g., all dims match), just replace index 0
    if (!inserted) {
      allActions[0] = candidate;
    }

    const value = payoff(allActions);
    if (value > bestValue) {
      bestValue = value;
      bestAction = new Float64Array(candidate);
    }
  }

  return bestAction;
}

// ---------------------------------------------------------------------------
// findNashEquilibrium
// ---------------------------------------------------------------------------

/**
 * Find a Nash equilibrium via iterated best response.
 *
 * Starting from an initial action profile (midpoint of each player's bounds),
 * each player in turn updates their action to the best response given all
 * other players' current actions. The process repeats until the maximum change
 * in any player's action falls below `tolerance` or `maxIter` is reached.
 *
 * @param config  Nash equilibrium configuration with payoff functions, action
 *                dimensions, bounds, tolerance, and iteration limit.
 * @returns       NashResult with equilibrium actions, payoffs, convergence, and
 *                iteration count.
 */
export function findNashEquilibrium(config: NashEquilibriumConfig): NashResult {
  const { nPlayers, payoffFns, actionDims, actionBounds, tolerance, maxIter } = config;

  // Initialize each player's action to the midpoint of their bounds
  const actions: Float64Array[] = [];
  for (let p = 0; p < nPlayers; p++) {
    const dim = actionDims[p]!;
    const bounds = actionBounds[p]!;
    const a = new Float64Array(dim);
    for (let d = 0; d < dim; d++) {
      a[d] = (bounds.min[d]! + bounds.max[d]!) / 2;
    }
    actions.push(a);
  }

  let converged = false;
  let iterations = 0;

  for (let iter = 0; iter < maxIter; iter++) {
    iterations = iter + 1;
    let maxChange = 0;

    for (let p = 0; p < nPlayers; p++) {
      const dim = actionDims[p]!;
      const bounds = actionBounds[p]!;
      const payoffFn = payoffFns[p]!;

      // Build the "otherActions" array with a placeholder for player p
      const allForBR: Float64Array[] = [];
      for (let q = 0; q < nPlayers; q++) {
        if (q === p) {
          // Placeholder that will be replaced inside bestResponse
          allForBR.push(new Float64Array(dim));
        } else {
          allForBR.push(actions[q]!);
        }
      }

      // Wrap payoff so it receives the full action profile
      const wrappedPayoff = (acts: Float64Array[]): number => {
        return payoffFn(acts);
      };

      const oldAction = actions[p]!;

      // We need a custom best response here that knows the player index
      // to correctly place the candidate action.
      const newAction = bestResponseForPlayer(
        wrappedPayoff,
        actions,
        p,
        dim,
        bounds,
      );

      // Measure change
      const diff = vecSub(newAction, oldAction);
      const change = vecNorm(diff);
      if (change > maxChange) {
        maxChange = change;
      }

      actions[p] = newAction;
    }

    if (maxChange < tolerance) {
      converged = true;
      break;
    }
  }

  // Compute final payoffs
  const payoffs = new Float64Array(nPlayers);
  for (let p = 0; p < nPlayers; p++) {
    payoffs[p] = payoffFns[p]!(actions);
  }

  return {
    equilibriumActions: actions,
    payoffs,
    converged,
    iterations,
  };
}

// ---------------------------------------------------------------------------
// bestResponseForPlayer (internal helper)
// ---------------------------------------------------------------------------

/**
 * Internal: grid-search best response that correctly splices the candidate
 * action at the specified player index.
 */
function bestResponseForPlayer(
  payoff: (actions: Float64Array[]) => number,
  currentActions: Float64Array[],
  playerIndex: number,
  actionDim: number,
  actionBounds: { min: Float64Array; max: Float64Array },
  nSamples: number = 11,
): Float64Array {
  const totalCandidates = Math.pow(nSamples, actionDim);
  let bestAction = new Float64Array(actionDim);
  let bestValue = -Infinity;

  for (let idx = 0; idx < totalCandidates; idx++) {
    // Decode multi-index
    const candidate = new Float64Array(actionDim);
    let rem = idx;
    for (let d = actionDim - 1; d >= 0; d--) {
      const dimIdx = rem % nSamples;
      rem = Math.floor(rem / nSamples);
      const lo = actionBounds.min[d]!;
      const hi = actionBounds.max[d]!;
      candidate[d] = nSamples > 1
        ? lo + (hi - lo) * dimIdx / (nSamples - 1)
        : (lo + hi) / 2;
    }

    // Build action profile with candidate at playerIndex
    const allActions: Float64Array[] = [];
    for (let p = 0; p < currentActions.length; p++) {
      if (p === playerIndex) {
        allActions.push(candidate);
      } else {
        allActions.push(currentActions[p]!);
      }
    }

    const value = payoff(allActions);
    if (value > bestValue) {
      bestValue = value;
      bestAction = new Float64Array(candidate);
    }
  }

  return bestAction;
}
