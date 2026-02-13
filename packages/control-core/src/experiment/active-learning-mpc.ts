// ---------------------------------------------------------------------------
// OC-10: Optimal Experiment Design -- Active Learning MPC
// ---------------------------------------------------------------------------
//
// Extends the standard linear MPC objective with an information-value term
// that encourages control actions which reduce parameter uncertainty.
//
// Modified objective:
//   min sum( x^T Q x + u^T R u ) - informationValueWeight * uncertainty_reduction
//
// Implementation: solve the nominal MPC, then add a perturbation in the
// direction of highest parameter uncertainty.  The perturbation magnitude
// scales with informationValueWeight * norm(parameterUncertainty).
// ---------------------------------------------------------------------------

import type { ActiveLearningMPCConfig } from '../types.js';
import { vecAdd, vecScale, vecNorm, vecClone } from '../types.js';
import { solveLinearMPC } from '../mpc/linear-mpc.js';

// ---------------------------------------------------------------------------
// Active Learning MPC solver
// ---------------------------------------------------------------------------

/**
 * Solve an MPC problem augmented with an active-learning exploration bonus.
 *
 * Steps:
 * 1. Solve the standard linear MPC (exploitation).
 * 2. Compute the exploration direction: the unit vector in the direction of
 *    the parameter uncertainty, projected onto the control space.
 * 3. Add a perturbation to the nominal control proportional to
 *    `informationValueWeight * ||parameterUncertainty||`.
 *
 * The exploration bonus quantifies how much additional uncertainty reduction
 * the perturbation provides.
 *
 * @param config  Active-learning MPC configuration (extends MPCConfig)
 * @param x0      Current state vector (nx)
 * @returns The modified control action and exploration bonus magnitude
 */
export function solveActiveLearningMPC(
  config: ActiveLearningMPCConfig,
  x0: Float64Array,
): { action: Float64Array; explorationBonus: number } {
  const { nu, informationValueWeight, parameterUncertainty } = config;

  // -------------------------------------------------------------------------
  // Step 1: Solve nominal MPC (exploitation)
  // -------------------------------------------------------------------------
  const mpcResult = solveLinearMPC(config, x0);
  const nominalAction = vecClone(mpcResult.uOptimal);

  // -------------------------------------------------------------------------
  // Step 2: Compute exploration perturbation direction
  // -------------------------------------------------------------------------
  // The parameter uncertainty vector has length nParams (may differ from nu).
  // We project it onto the control space by taking the first nu components
  // or, if nParams < nu, zero-padding.
  const uncertNorm = vecNorm(parameterUncertainty);

  if (uncertNorm < 1e-15 || Math.abs(informationValueWeight) < 1e-15) {
    // No uncertainty or no exploration weight — return nominal action
    return { action: nominalAction, explorationBonus: 0 };
  }

  // Build the exploration direction in control space
  const perturbation = new Float64Array(nu);
  const projLen = Math.min(parameterUncertainty.length, nu);
  for (let i = 0; i < projLen; i++) {
    perturbation[i] = parameterUncertainty[i]! / uncertNorm;
  }

  // -------------------------------------------------------------------------
  // Step 3: Scale perturbation and add to nominal action
  // -------------------------------------------------------------------------
  const perturbMagnitude = informationValueWeight * uncertNorm;
  const scaledPerturbation = vecScale(perturbation, perturbMagnitude);
  const action = vecAdd(nominalAction, scaledPerturbation);

  // -------------------------------------------------------------------------
  // Clamp action to control bounds if present
  // -------------------------------------------------------------------------
  if (config.uMin || config.uMax) {
    for (let i = 0; i < nu; i++) {
      if (config.uMin) {
        const lo = config.uMin[i]!;
        if (action[i]! < lo) {
          action[i] = lo;
        }
      }
      if (config.uMax) {
        const hi = config.uMax[i]!;
        if (action[i]! > hi) {
          action[i] = hi;
        }
      }
    }
  }

  return {
    action,
    explorationBonus: perturbMagnitude,
  };
}
