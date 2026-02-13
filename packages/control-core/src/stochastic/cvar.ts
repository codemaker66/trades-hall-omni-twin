// ---------------------------------------------------------------------------
// OC-9  Stochastic Optimal Control -- Conditional Value-at-Risk (CVaR)
// ---------------------------------------------------------------------------

import type { CVaRResult } from '../types.js';

// ---------------------------------------------------------------------------
// CVaR Computation
// ---------------------------------------------------------------------------

/**
 * Compute the Conditional Value-at-Risk (CVaR) of a loss distribution.
 *
 * CVaR_alpha = E[ X | X >= VaR_alpha ]
 *
 * This is the expected loss in the worst (1-alpha) fraction of outcomes.
 * Also called Expected Shortfall (ES) or Tail Value-at-Risk (TVaR).
 *
 * Implementation sorts losses, finds VaR as the alpha-quantile, then
 * computes CVaR as the probability-weighted conditional mean above VaR.
 *
 * @param losses - Loss values for each scenario
 * @param alpha  - Confidence level in (0, 1), e.g. 0.95 for 95% CVaR
 * @param probs  - Scenario probabilities (optional; uniform if omitted)
 * @returns Object with cvar and var_ (VaR)
 */
export function computeCVaR(
  losses: Float64Array,
  alpha: number,
  probs?: Float64Array,
): { cvar: number; var_: number } {
  const n = losses.length;

  if (n === 0) {
    return { cvar: 0, var_: 0 };
  }

  // Build index-probability pairs and sort by loss value
  const indices = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    indices[i] = i;
  }

  // Sort indices by loss value ascending
  const sortedIndices = Array.from(indices).sort(
    (a, b) => losses[a]! - losses[b]!,
  );

  // Compute uniform probabilities if not provided
  const uniform = 1 / n;

  // Find VaR: smallest loss such that cumulative probability >= alpha
  let cumProb = 0;
  let varValue = losses[sortedIndices[0]!]!;

  for (let i = 0; i < n; i++) {
    const idx = sortedIndices[i]!;
    const p = probs ? probs[idx]! : uniform;
    cumProb += p;

    if (cumProb >= alpha) {
      varValue = losses[idx]!;
      break;
    }
  }

  // Compute CVaR using the Rockafellar-Uryasev formulation:
  //   CVaR_alpha = min_t { t + 1/(1-alpha) * E[ max(loss - t, 0) ] }
  //
  // At the optimum, t* = VaR, so:
  //   CVaR_alpha = VaR + 1/(1-alpha) * E[ max(loss - VaR, 0) ]
  const tailWeight = 1 / (1 - alpha);
  let expectedExcess = 0;

  for (let i = 0; i < n; i++) {
    const p = probs ? probs[i]! : uniform;
    const excess = losses[i]! - varValue;
    if (excess > 0) {
      expectedExcess += p * excess;
    }
  }

  const cvar = varValue + tailWeight * expectedExcess;

  return { cvar, var_: varValue };
}

// ---------------------------------------------------------------------------
// CVaR Optimization over Candidate Actions
// ---------------------------------------------------------------------------

/**
 * Find the action that minimises CVaR across a finite set of scenarios.
 *
 * For each candidate action, evaluates the cost under every scenario,
 * computes CVaR of the resulting loss distribution, and selects the
 * action with the smallest CVaR.
 *
 * Uses the Rockafellar-Uryasev reformulation internally:
 *   CVaR_alpha(action) = min_t { t + 1/(1-alpha) * E[ max(cost(action,w) - t, 0) ] }
 *
 * @param costFn    - Cost function mapping (action, scenario) -> scalar loss
 * @param actions   - Array of candidate action vectors
 * @param scenarios - Array of scenario vectors
 * @param alpha     - Confidence level in (0, 1)
 * @param probs     - Scenario probabilities (optional; uniform if omitted)
 * @returns CVaRResult with optimal action, cvar, var, and per-scenario costs
 */
export function cvarOptimize(
  costFn: (action: Float64Array, scenario: Float64Array) => number,
  actions: Float64Array[],
  scenarios: Float64Array[],
  alpha: number,
  probs?: Float64Array,
): CVaRResult {
  const nScenarios = scenarios.length;
  const nActions = actions.length;

  if (nActions === 0 || nScenarios === 0) {
    return {
      cvar: 0,
      var: 0,
      optimalAction: new Float64Array(0),
      scenarioCosts: new Float64Array(0),
    };
  }

  let bestCVaR = Infinity;
  let bestVaR = 0;
  let bestActionIdx = 0;
  let bestCosts = new Float64Array(nScenarios);

  for (let ai = 0; ai < nActions; ai++) {
    const action = actions[ai]!;

    // Evaluate cost under each scenario
    const costs = new Float64Array(nScenarios);
    for (let wi = 0; wi < nScenarios; wi++) {
      costs[wi] = costFn(action, scenarios[wi]!);
    }

    // Compute CVaR for this action
    const { cvar, var_ } = computeCVaR(costs, alpha, probs);

    if (cvar < bestCVaR) {
      bestCVaR = cvar;
      bestVaR = var_;
      bestActionIdx = ai;
      bestCosts = costs;
    }
  }

  return {
    cvar: bestCVaR,
    var: bestVaR,
    optimalAction: new Float64Array(actions[bestActionIdx]!),
    scenarioCosts: bestCosts,
  };
}
