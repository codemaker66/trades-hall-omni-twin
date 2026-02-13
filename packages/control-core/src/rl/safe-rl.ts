// ---------------------------------------------------------------------------
// OC-5: Safe / Constrained Reinforcement Learning
// ---------------------------------------------------------------------------

import { vecDot } from '../types.js';

// ---------------------------------------------------------------------------
// Safe Action Projection
// ---------------------------------------------------------------------------

/**
 * Project an action onto the feasible set using gradient projection.
 *
 * If the constraint is violated (constraintVal > 0), the action is projected
 * along the negative constraint gradient direction to bring the constraint
 * back to zero:
 *
 *   a_safe = a - (g(a) / ||grad||^2) * grad
 *
 * If the constraint is satisfied (constraintVal <= 0), the original action
 * is returned unchanged.
 *
 * @param action          Proposed action vector (actionDim).
 * @param constraintGrad  Gradient of the constraint function w.r.t. action (actionDim).
 * @param constraintVal   Scalar constraint value g(a). Violated when > 0.
 * @returns               Safe (projected) action vector.
 */
export function safeProject(
  action: Float64Array,
  constraintGrad: Float64Array,
  constraintVal: number,
): Float64Array {
  // Constraint is satisfied — no projection needed
  if (constraintVal <= 0) {
    return new Float64Array(action);
  }

  // Compute ||grad||^2
  const gradNormSq = vecDot(constraintGrad, constraintGrad);

  // Guard against zero gradient (degenerate case)
  if (gradNormSq < 1e-15) {
    return new Float64Array(action);
  }

  // Project: a_safe = a - (g(a) / ||grad||^2) * grad
  const scale = constraintVal / gradNormSq;
  const aSafe = new Float64Array(action.length);

  for (let i = 0; i < action.length; i++) {
    aSafe[i] = action[i]! - scale * constraintGrad[i]!;
  }

  return aSafe;
}

// ---------------------------------------------------------------------------
// Lagrange Multiplier Update (Dual Ascent)
// ---------------------------------------------------------------------------

/**
 * Update Lagrange multipliers via dual ascent for constrained RL.
 *
 * Each multiplier is updated according to:
 *
 *   lambda_{k+1} = max(0, lambda_k + lr * (cost_k - threshold_k))
 *
 * When cost exceeds the threshold, the multiplier increases, penalising
 * constraint violations more heavily. When cost is below threshold, the
 * multiplier decreases (but never below zero).
 *
 * @param lambdas     Current Lagrange multipliers (constraintDim).
 * @param costs       Observed constraint costs (constraintDim).
 * @param thresholds  Per-constraint violation thresholds (constraintDim).
 * @param lr          Lagrangian learning rate.
 * @returns           Updated Lagrange multipliers (constraintDim).
 */
export function updateLagrangeMultipliers(
  lambdas: Float64Array,
  costs: Float64Array,
  thresholds: Float64Array,
  lr: number,
): Float64Array {
  const n = lambdas.length;
  const updated = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const newLambda = lambdas[i]! + lr * (costs[i]! - thresholds[i]!);
    updated[i] = Math.max(0, newLambda);
  }

  return updated;
}
