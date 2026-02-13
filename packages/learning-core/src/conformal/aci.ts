// ---------------------------------------------------------------------------
// Adaptive Conformal Inference (Gibbs & Candes 2021, arXiv:2106.00170)
// ---------------------------------------------------------------------------

import type { ACIState } from '../types.js';

/**
 * Create initial ACI state.
 *
 * ACI adapts the significance level alpha_t online to maintain coverage
 * for non-exchangeable (e.g. time series) data.
 *
 * Update rule: alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)
 * where err_t = 1{Y_t not in C_t(X_t)}
 *
 * Guarantee (Gibbs & Candes 2021):
 *   |sum(err_t)/T - alpha| <= (max(alpha_1, 1-alpha_1) + 1) / (gamma * T)
 *
 * @param alpha - Target miscoverage rate (e.g. 0.1 for 90% nominal coverage)
 * @param gamma - Step size for alpha updates. Smaller = more stable, larger = faster adaptation.
 *                Typical values: 0.001 to 0.01.
 * @returns Initial ACI state
 */
export function createACIState(alpha: number, gamma: number): ACIState {
  return {
    alphaTarget: alpha,
    alphaT: alpha,
    gamma,
    coverageHistory: [],
  };
}

/**
 * Update ACI state after observing a new (y_true, interval) pair.
 *
 * If the true value y_true falls outside [lower, upper]:
 *   - err_t = 1 (miscoverage)
 *   - alpha_t increases -> wider future intervals
 *
 * If y_true falls inside:
 *   - err_t = 0 (coverage)
 *   - alpha_t decreases -> tighter future intervals
 *
 * alpha_t is clamped to [0.001, 0.999] for numerical stability.
 *
 * @param state - Current ACI state
 * @param yTrue - Observed true value
 * @param lower - Lower bound of prediction interval
 * @param upper - Upper bound of prediction interval
 * @returns Updated ACI state (immutable — returns new object)
 */
export function aciUpdate(
  state: ACIState,
  yTrue: number,
  lower: number,
  upper: number,
): ACIState {
  // Compute coverage indicator: 1 if covered, 0 if not
  const covered = yTrue >= lower && yTrue <= upper ? 1 : 0;
  const errT = 1 - covered;

  // Update alpha_t:
  // alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)
  //
  // When errT = 1 (miscoverage): alpha_t decreases by gamma*(alpha_target - 1)
  //   Since alpha_target < 1, this is negative, so alpha_t decreases.
  //   Wait — lower alpha_t means the conformal quantile uses (1 - alpha_t) which
  //   increases, giving wider intervals. So:
  //   Miscoverage -> alpha_t decreases -> wider intervals. Correct.
  //
  // When errT = 0 (coverage): alpha_t increases by gamma*alpha_target
  //   alpha_t increases -> narrower intervals. Correct.
  const newAlphaT = state.alphaT + state.gamma * (state.alphaTarget - errT);

  // Clamp to valid range
  const clampedAlpha = Math.max(0.001, Math.min(0.999, newAlphaT));

  return {
    alphaTarget: state.alphaTarget,
    alphaT: clampedAlpha,
    gamma: state.gamma,
    coverageHistory: [...state.coverageHistory, covered],
  };
}

/**
 * Get the current adaptive alpha value from ACI state.
 *
 * Use this alpha when computing the next conformal quantile:
 *   quantile = conformalQuantile(residuals, aciGetAlpha(state))
 *
 * @param state - Current ACI state
 * @returns The current adaptive miscoverage rate alpha_t
 */
export function aciGetAlpha(state: ACIState): number {
  return state.alphaT;
}
