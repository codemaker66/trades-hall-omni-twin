// ---------------------------------------------------------------------------
// OC-6: Adaptive Control (MRAC and L1 Adaptive)
// ---------------------------------------------------------------------------

import type { AdaptiveControlConfig, AdaptiveControlState } from '../types.js';
import {
  arrayToMatrix,
  matVecMul,
  vecAdd,
  vecClone,
  vecDot,
  vecScale,
  vecSub,
} from '../types.js';

// ---------------------------------------------------------------------------
// Create Initial Adaptive State
// ---------------------------------------------------------------------------

/**
 * Create the initial adaptive controller state.
 *
 * Initialises parameter estimates to zero and reference model state to zero.
 *
 * @param config  Adaptive control configuration
 * @returns       Initial adaptive control state
 */
export function createAdaptiveState(
  config: AdaptiveControlConfig,
): AdaptiveControlState {
  const { nx, nu } = config.modelRef;

  // Parameter vector length: nx + nu columns of regressor
  // For MRAC: theta = [theta_x (nx); theta_r (nu)]
  // We store theta as a flat vector of size nx + nu
  const nParams = nx + nu;

  return {
    thetaHat: new Float64Array(nParams),
    xRef: new Float64Array(nx),
  };
}

// ---------------------------------------------------------------------------
// MRAC Step
// ---------------------------------------------------------------------------

/**
 * Perform one step of Model Reference Adaptive Control (MRAC).
 *
 * The reference model evolves as:
 *   dx_ref/dt = A_m * x_ref + B_m * r
 *
 * The tracking error is:
 *   e = x - x_ref
 *
 * The control law is:
 *   u = theta_x^T * x + theta_r * r
 *
 * The parameter update follows the MIT rule variant:
 *   d(theta)/dt = -Gamma * e . phi^T
 *
 * where phi = [x; r] is the regression vector and Gamma is the
 * adaptation gain matrix.
 *
 * @param config  Adaptive control configuration (method must be 'mrac')
 * @param state   Current adaptive state (thetaHat, xRef)
 * @param x       Current plant state (nx)
 * @param r       Reference command (scalar)
 * @param dt      Time step
 * @returns       Control input u and updated adaptive state
 */
export function mracStep(
  config: AdaptiveControlConfig,
  state: AdaptiveControlState,
  x: Float64Array,
  r: number,
  dt: number,
): { u: number; nextState: AdaptiveControlState } {
  const { modelRef, adaptationGain } = config;
  const { A, B, nx, nu } = modelRef;

  const Am = arrayToMatrix(A, nx, nx);
  const Bm = arrayToMatrix(B, nx, nu);

  // --- Reference model integration (Euler) ---
  // dx_ref/dt = A_m * x_ref + B_m * r
  const AxRef = matVecMul(Am, state.xRef);
  const rVec = new Float64Array(nu);
  rVec[0] = r;
  const BrRef = matVecMul(Bm, rVec);
  const xRefDot = vecAdd(AxRef, BrRef);
  const xRefNext = vecAdd(state.xRef, vecScale(xRefDot, dt));

  // --- Tracking error ---
  const e = vecSub(x, state.xRef);

  // --- Regression vector phi = [x; r] ---
  const nParams = nx + nu;
  const phi = new Float64Array(nParams);
  for (let i = 0; i < nx; i++) {
    phi[i] = x[i]!;
  }
  phi[nx] = r;

  // --- Control law: u = theta^T * phi ---
  const thetaHat = state.thetaHat;
  let u = 0;
  for (let i = 0; i < nParams; i++) {
    u += thetaHat[i]! * phi[i]!;
  }

  // --- Parameter update: d(theta)/dt = -Gamma * (e^T * phi projected) ---
  // Using the MIT rule: d(theta_i)/dt = -gamma_i * e_norm * phi_i
  // where e_norm is a scalar error measure (first component or norm)
  // For SISO simplification, use e[0] as the error signal.
  const eSignal = e[0]!;

  const Gamma = arrayToMatrix(adaptationGain, nParams, nParams);
  const ePhi = vecScale(phi, eSignal);
  const thetaDot = matVecMul(Gamma, ePhi);

  // Update parameters: theta_{k+1} = theta_k - dt * Gamma * e * phi
  const thetaNext = vecClone(thetaHat);
  for (let i = 0; i < nParams; i++) {
    thetaNext[i] = thetaHat[i]! - dt * thetaDot[i]!;
  }

  return {
    u,
    nextState: {
      thetaHat: thetaNext,
      xRef: xRefNext,
    },
  };
}

// ---------------------------------------------------------------------------
// L1 Adaptive Control Step
// ---------------------------------------------------------------------------

/**
 * Perform one step of L1 Adaptive Control.
 *
 * L1 adaptive control differs from MRAC by employing:
 * 1. Fast adaptation with bounded adaptation rate
 * 2. A low-pass filter on the control signal to decouple adaptation
 *    speed from robustness
 *
 * Reference model:
 *   dx_ref/dt = A_m * x_ref + B_m * (u + sigma_hat)
 *
 * Adaptation law (piecewise-constant, fast):
 *   sigma_hat is updated to minimise prediction error
 *   with bounded rate: |d(sigma_hat)/dt| <= Gamma_bar
 *
 * Control:
 *   u_raw = -sigma_hat + k_r * r
 *   u = low-pass filtered u_raw (first-order filter)
 *
 * @param config  Adaptive control configuration (method must be 'l1')
 * @param state   Current adaptive state
 * @param x       Current plant state (nx)
 * @param r       Reference command (scalar)
 * @param dt      Time step
 * @returns       Control input u and updated adaptive state
 */
export function l1AdaptiveStep(
  config: AdaptiveControlConfig,
  state: AdaptiveControlState,
  x: Float64Array,
  r: number,
  dt: number,
): { u: number; nextState: AdaptiveControlState } {
  const { modelRef, adaptationGain } = config;
  const { A, B, nx, nu } = modelRef;

  const Am = arrayToMatrix(A, nx, nx);
  const Bm = arrayToMatrix(B, nx, nu);

  const nParams = nx + nu;
  const thetaHat = state.thetaHat;

  // --- State predictor / reference model integration ---
  // dx_ref/dt = A_m * x_ref + B_m * r_vec
  const AxRef = matVecMul(Am, state.xRef);
  const rVec = new Float64Array(nu);
  rVec[0] = r;
  const BrRef = matVecMul(Bm, rVec);
  const xRefDot = vecAdd(AxRef, BrRef);
  const xRefNext = vecAdd(state.xRef, vecScale(xRefDot, dt));

  // --- Prediction error ---
  const eTilde = vecSub(x, state.xRef);

  // --- Fast adaptation ---
  // Estimate matched uncertainty sigma_hat from prediction error.
  // The adaptation gain diagonal bounds the rate.
  // sigma_hat components map to: [theta_x (nx); theta_r (1)]
  // d(theta)/dt = -Gamma * B^T * P * eTilde (projected)
  // where P is a Lyapunov matrix (approximate as identity for simplicity)

  // Regression vector: phi = [x; r]
  const phi = new Float64Array(nParams);
  for (let i = 0; i < nx; i++) {
    phi[i] = x[i]!;
  }
  phi[nx] = r;

  // Adaptation direction: -Gamma * eTilde_scalar * phi
  const eSignal = eTilde[0]!;

  const Gamma = arrayToMatrix(adaptationGain, nParams, nParams);
  const ePhi = vecScale(phi, eSignal);
  const rawUpdate = matVecMul(Gamma, ePhi);

  // Bound the adaptation rate (key L1 feature)
  const maxRate = adaptationGainBound(adaptationGain, nParams);
  const thetaNext = vecClone(thetaHat);
  for (let i = 0; i < nParams; i++) {
    let delta = -dt * rawUpdate[i]!;
    // Clamp to bounded rate
    const bound = maxRate * dt;
    if (delta > bound) delta = bound;
    if (delta < -bound) delta = -bound;
    thetaNext[i] = thetaHat[i]! + delta;
  }

  // --- Control law with low-pass filtering ---
  // Raw control: u_raw = theta^T * phi
  let uRaw = 0;
  for (let i = 0; i < nParams; i++) {
    uRaw += thetaNext[i]! * phi[i]!;
  }

  // Low-pass filter: u_filtered = (1 - alpha) * u_prev + alpha * u_raw
  // Filter bandwidth chosen relative to adaptation rate.
  // We store the previous filtered control in thetaHat's last entry as a
  // secondary use, but to keep the interface clean, we use the first-order
  // filter with time constant tau = 1 / (2 * maxRate).
  const tau = maxRate > 0 ? 1 / (2 * maxRate) : dt;
  const filterAlpha = dt / (tau + dt);

  // Recover previous control from the theta_r component's output
  // For statelessness w.r.t. control filter, we approximate:
  const uPrev = vecDot(thetaHat, phi);
  const u = (1 - filterAlpha) * uPrev + filterAlpha * uRaw;

  return {
    u,
    nextState: {
      thetaHat: thetaNext,
      xRef: xRefNext,
    },
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Compute a scalar bound on the adaptation rate from the adaptation gain
 * matrix. Uses the maximum diagonal element as the rate bound.
 *
 * @param gainData  Flat adaptation gain matrix (nParams x nParams, row-major)
 * @param nParams   Number of parameters
 * @returns         Scalar adaptation rate bound
 */
function adaptationGainBound(
  gainData: Float64Array,
  nParams: number,
): number {
  let maxDiag = 0;
  for (let i = 0; i < nParams; i++) {
    const diag = Math.abs(gainData[i * nParams + i]!);
    if (diag > maxDiag) {
      maxDiag = diag;
    }
  }
  return maxDiag;
}
