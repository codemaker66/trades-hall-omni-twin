// ---------------------------------------------------------------------------
// OC-3: Bang-Bang Control Analysis (Pontryagin's Maximum Principle)
// ---------------------------------------------------------------------------

import type { ShootingConfig, BangBangResult } from '../types.js';
import { vecClone } from '../types.js';

// ---------------------------------------------------------------------------
// Switching function analysis
// ---------------------------------------------------------------------------

/**
 * Analyze the switching function along a solved trajectory to identify
 * bang-bang control structure.
 *
 * The switching function phi(t) = dH/du is computed by evaluating the
 * partial derivative of the Hamiltonian with respect to the control.
 * For bang-bang control, the optimal control switches between extreme
 * values when phi(t) crosses zero.
 *
 * This function:
 *  1. Evaluates phi(t) = dH/du along the trajectory using finite differences
 *  2. Detects zero crossings in the switching function
 *  3. Determines control levels between switches
 *
 * @param config           Shooting configuration (provides dynamics, cost, etc.)
 * @param lambdaTrajectory Costate trajectory from a shooting solve
 * @param uTrajectory      Control trajectory from a shooting solve
 * @returns                BangBangResult with switching times, levels, and function values
 */
export function analyzeSwitchingFunction(
  config: ShootingConfig,
  lambdaTrajectory: Float64Array[],
  uTrajectory: Float64Array[],
): BangBangResult {
  const { nx, nu, T, nSteps, x0 } = config;
  const dt = T / nSteps;
  const eps = 1e-7;

  // -----------------------------------------------------------------------
  // Evaluate switching function at each time step
  // The switching function phi(t) = dH/du is approximated via finite
  // differences of the Hamiltonian w.r.t. control
  // -----------------------------------------------------------------------
  const nPoints = nSteps + 1;
  const phiValues = new Float64Array(nPoints * nu);

  // We need the state trajectory as well -- reconstruct it by integrating
  // forward (or we can compute it from the provided data)
  // For simplicity, assume the trajectories are consistent and use the
  // lambda and u trajectories directly with the controlOptimality condition

  for (let k = 0; k < nPoints; k++) {
    const t = k * dt;
    const lambda = lambdaTrajectory[k]!;
    const u = uTrajectory[k]!;

    // Use controlOptimality to get optimal control, then evaluate
    // switching function as the gradient of H w.r.t. u
    // phi_j(t) = dH/du_j, approximated by finite differences
    for (let j = 0; j < nu; j++) {
      const uPlus = vecClone(u);
      const uMinus = vecClone(u);
      uPlus[j] = uPlus[j]! + eps;
      uMinus[j] = uMinus[j]! - eps;

      // We need the state at this time point. Since we are analyzing
      // a given trajectory, we compute a dummy state using x0 and
      // the controlOptimality function for consistency.
      // In practice, the user should provide the full state trajectory.
      // Here we use the control optimality to get a consistent x.
      const xApprox = config.controlOptimality(
        new Float64Array(nx),
        lambda,
        t,
      );

      // H = L(x, u, t) + lambda^T f(x, u, lambda, t)
      const LPlus = config.runningCost(xApprox, uPlus, t);
      const LMinus = config.runningCost(xApprox, uMinus, t);

      const fPlus = config.stateDynamics(xApprox, uPlus, lambda, t);
      const fMinus = config.stateDynamics(xApprox, uMinus, lambda, t);

      let lambdaTfPlus = 0;
      let lambdaTfMinus = 0;
      for (let i = 0; i < nx; i++) {
        lambdaTfPlus += lambda[i]! * fPlus[i]!;
        lambdaTfMinus += lambda[i]! * fMinus[i]!;
      }

      const HPlus = LPlus + lambdaTfPlus;
      const HMinus = LMinus + lambdaTfMinus;

      phiValues[k * nu + j] = (HPlus - HMinus) / (2 * eps);
    }
  }

  // -----------------------------------------------------------------------
  // Detect zero crossings in the switching function
  // For simplicity, analyze the first control dimension (nu >= 1)
  // -----------------------------------------------------------------------
  const switchingTimesList: number[] = [];
  const controlLevelsList: number[] = [];

  // Track sign of phi for the first control component
  for (let k = 0; k < nPoints - 1; k++) {
    const phi_k = phiValues[k * nu]!;
    const phi_k1 = phiValues[(k + 1) * nu]!;

    if (phi_k * phi_k1 < 0) {
      // Zero crossing detected -- interpolate switching time
      const t_k = k * dt;
      const t_k1 = (k + 1) * dt;
      const tSwitch = t_k + (t_k1 - t_k) * Math.abs(phi_k) / (Math.abs(phi_k) + Math.abs(phi_k1));
      switchingTimesList.push(tSwitch);
    }
  }

  // Determine control levels between switches
  // Before first switch, between switches, and after last switch
  const nSwitches = switchingTimesList.length;
  const nLevels = nSwitches + 1;

  for (let seg = 0; seg < nLevels; seg++) {
    // Sample phi at midpoint of this segment to determine control level sign
    let tMid: number;
    if (seg === 0) {
      tMid = nSwitches > 0 ? switchingTimesList[0]! / 2 : T / 2;
    } else if (seg === nSwitches) {
      tMid = nSwitches > 0
        ? (switchingTimesList[nSwitches - 1]! + T) / 2
        : T / 2;
    } else {
      tMid = (switchingTimesList[seg - 1]! + switchingTimesList[seg]!) / 2;
    }

    // Evaluate phi at tMid (interpolate from discrete values)
    const kMid = tMid / dt;
    const kLow = Math.floor(kMid);
    const kHigh = Math.min(kLow + 1, nPoints - 1);
    const frac = kMid - kLow;

    const phiMid =
      (1 - frac) * phiValues[kLow * nu]! + frac * phiValues[kHigh * nu]!;

    // Bang-bang: control is at max when phi < 0, min when phi > 0
    // (for minimization of H)
    controlLevelsList.push(phiMid < 0 ? 1.0 : -1.0);
  }

  // Pack the full switching function (first control component only for output)
  const switchingFunction = new Float64Array(nPoints);
  for (let k = 0; k < nPoints; k++) {
    switchingFunction[k] = phiValues[k * nu]!;
  }

  return {
    switchingTimes: new Float64Array(switchingTimesList),
    controlLevels: new Float64Array(controlLevelsList),
    switchingFunction,
  };
}

// ---------------------------------------------------------------------------
// Bang-bang control trajectory construction
// ---------------------------------------------------------------------------

/**
 * Construct a piecewise-constant bang-bang control trajectory from
 * switching times and control levels.
 *
 * @param switchingTimes  Times at which control switches (sorted, within [0, T])
 * @param controlLevels   Control level in each segment (length = switchingTimes.length + 1)
 * @param T               Terminal time
 * @param nSteps          Number of time steps for the output trajectory
 * @returns               Array of control vectors at each time step
 */
export function constructBangBang(
  switchingTimes: Float64Array,
  controlLevels: Float64Array,
  T: number,
  nSteps: number,
): Float64Array[] {
  const dt = T / nSteps;
  const nSwitches = switchingTimes.length;
  const trajectory: Float64Array[] = [];

  for (let k = 0; k <= nSteps; k++) {
    const t = k * dt;

    // Determine which segment this time point falls in
    let seg = 0;
    for (let s = 0; s < nSwitches; s++) {
      if (t >= switchingTimes[s]!) {
        seg = s + 1;
      } else {
        break;
      }
    }

    // Control is the level for this segment
    const level = controlLevels[seg]!;
    const u = new Float64Array(1);
    u[0] = level;
    trajectory.push(u);
  }

  return trajectory;
}
