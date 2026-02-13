// ---------------------------------------------------------------------------
// OC-1: Linear-Quadratic-Gaussian (LQG) Design
// ---------------------------------------------------------------------------

import type { LQGConfig, LQGResult, LQGState } from '../types.js';
import {
  arrayToMatrix,
  matAdd,
  matIdentity,
  matInvert,
  matMul,
  matSub,
  matTranspose,
  matVecMul,
  vecAdd,
  vecSub,
} from '../types.js';

import { discreteLQR } from './dare.js';

// ---------------------------------------------------------------------------
// LQG design via separation principle
// ---------------------------------------------------------------------------

/**
 * Design a Linear-Quadratic-Gaussian controller by solving the LQR and
 * Kalman filter problems independently (separation principle).
 *
 * LQR: solves the regulator DARE for gain K using (A, B, Q, R).
 * Kalman: solves the filter DARE for the Kalman gain L using (A, C, Qn, Rn).
 *
 * Filter DARE (direct iteration):
 *   P_f_{k+1} = A P_f_k A^T + Qn - A P_f_k C^T (C P_f_k C^T + Rn)^{-1} C P_f_k A^T
 *
 * Kalman gain:
 *   L = P_f C^T (C P_f C^T + Rn)^{-1}
 *
 * @param config LQG configuration containing LQR params and noise covariances
 * @returns LQGResult with regulator gain K (in .lqr), estimator gain L, and
 *          filter error covariance Pf
 */
export function designLQG(config: LQGConfig): LQGResult {
  const { A, B, Q, R, N, C, Qn, Rn, nx, nu, ny } = config;

  // --- Regulator: solve LQR ---
  const lqr = discreteLQR({ A, B, Q, R, N, nx, nu });

  // --- Estimator: solve filter DARE directly ---
  const mA = arrayToMatrix(A, nx, nx);
  const mC = arrayToMatrix(C, ny, nx);
  const mQn = arrayToMatrix(Qn, nx, nx);
  const mRn = arrayToMatrix(Rn, ny, ny);

  const At = matTranspose(mA);
  const Ct = matTranspose(mC);

  // Initialise P_f = Qn
  let Pf = arrayToMatrix(Qn, nx, nx);

  const maxIter = 1000;
  const tolerance = 1e-10;

  for (let iter = 0; iter < maxIter; iter++) {
    // A P_f
    const APf = matMul(mA, Pf);
    // A P_f A^T
    const APfAt = matMul(APf, At);
    // C P_f
    const CPf = matMul(mC, Pf);
    // C P_f C^T
    const CPfCt = matMul(CPf, Ct);
    // A P_f C^T
    const APfCt = matMul(APf, Ct);
    // C P_f A^T
    const CPfAt = matMul(CPf, At);

    // Innovation covariance: S = C P_f C^T + Rn
    const S = matAdd(CPfCt, mRn);
    const Sinv = matInvert(S);

    // Correction: A P_f C^T S^{-1} C P_f A^T
    const correction = matMul(matMul(APfCt, Sinv), CPfAt);

    // P_f_new = A P_f A^T + Qn - correction
    const PfNew = matSub(matAdd(APfAt, mQn), correction);

    // Check convergence
    const diff = matSub(PfNew, Pf);
    let norm = 0;
    for (let i = 0; i < diff.data.length; i++) {
      norm += diff.data[i]! * diff.data[i]!;
    }
    norm = Math.sqrt(norm);

    Pf = PfNew;

    if (norm < tolerance) {
      break;
    }
  }

  // Kalman gain: L = P_f C^T (C P_f C^T + Rn)^{-1}
  const CPf = matMul(mC, Pf);
  const CPfCt = matMul(CPf, Ct);
  const Sinv = matInvert(matAdd(CPfCt, mRn));
  const PfCt = matMul(Pf, Ct);
  const L = matMul(PfCt, Sinv);

  return {
    lqr,
    L: new Float64Array(L.data),
    Pf: new Float64Array(Pf.data),
  };
}

// ---------------------------------------------------------------------------
// LQG state creation
// ---------------------------------------------------------------------------

/**
 * Create an initial LQG estimator state with zero state estimate and
 * identity error covariance.
 *
 * @param config LQG configuration (uses nx)
 * @returns Initial LQG state
 */
export function createLQGState(config: LQGConfig): LQGState {
  const { nx } = config;
  const I = matIdentity(nx);
  return {
    xHat: new Float64Array(nx),
    P: new Float64Array(I.data),
  };
}

// ---------------------------------------------------------------------------
// Single LQG step: predict + correct + control
// ---------------------------------------------------------------------------

/**
 * Execute one step of the LQG controller:
 *   1. Predict: xHat_pred = A xHat
 *   2. Correct: xHat_new = xHat_pred + L (y - C xHat_pred)
 *   3. Control: u = -K xHat_new
 *
 * Returns the control action and the next estimator state (immutable --
 * does not modify the input state).
 *
 * @param state      Current LQG estimator state
 * @param y          Current measurement vector (ny)
 * @param config     LQG configuration
 * @param lqgResult  LQG design result (contains K and L)
 * @returns Object with control action u (nu) and nextState (updated estimator)
 */
export function lqgStep(
  state: LQGState,
  y: Float64Array,
  config: LQGConfig,
  lqgResult: LQGResult,
): { u: Float64Array; nextState: LQGState } {
  const { A, B, C, nx, nu, ny } = config;

  const mA = arrayToMatrix(A, nx, nx);
  const mC = arrayToMatrix(C, ny, nx);
  const mK = arrayToMatrix(lqgResult.lqr.K, nu, nx);
  const mL = arrayToMatrix(lqgResult.L, nx, ny);

  // --- Predict ---
  // xHat_pred = A xHat
  const xPred = matVecMul(mA, state.xHat);

  // --- Correct ---
  // Innovation: y - C xHat_pred
  const yPred = matVecMul(mC, xPred);
  const innovation = vecSub(y, yPred);

  // xHat_new = xHat_pred + L * innovation
  const correction = matVecMul(mL, innovation);
  const xHatNew = vecAdd(xPred, correction);

  // --- Control ---
  // u = -K xHat_new
  const u = matVecMul(mK, xHatNew);
  for (let j = 0; j < nu; j++) {
    u[j] = -u[j]!;
  }

  const nextState: LQGState = {
    xHat: xHatNew,
    P: new Float64Array(state.P), // covariance unchanged in steady-state LQG
  };

  return { u, nextState };
}

// ---------------------------------------------------------------------------
// Full LQG simulation from measurements
// ---------------------------------------------------------------------------

/**
 * Simulate the LQG controller over a sequence of measurements.
 *
 * At each step:
 *   1. xHat_pred = A xHat + B u_prev
 *   2. xHat = xHat_pred + L (y_t - C xHat_pred)
 *   3. u_t = -K xHat
 *
 * @param config       LQG configuration
 * @param result       LQG design result (contains K and L)
 * @param measurements Array of measurement vectors y_t (ny each), one per step
 * @returns Array of state estimate vectors [xHat_0, xHat_1, ..., xHat_T]
 */
export function simulateLQG(
  config: LQGConfig,
  result: LQGResult,
  measurements: Float64Array[],
): Float64Array[] {
  const { A, B, C, nx, nu, ny } = config;

  const mA = arrayToMatrix(A, nx, nx);
  const mB = arrayToMatrix(B, nx, nu);
  const mC = arrayToMatrix(C, ny, nx);
  const mK = arrayToMatrix(result.lqr.K, nu, nx);
  const mL = arrayToMatrix(result.L, nx, ny);

  const T = measurements.length;

  let xHat = new Float64Array(nx); // initial estimate = zero
  const estimates: Float64Array[] = [new Float64Array(xHat)];
  let uPrev = new Float64Array(nu); // initial control = zero

  for (let t = 0; t < T; t++) {
    const y = measurements[t]!;

    // --- Predict ---
    // xHat_pred = A xHat + B u_prev
    const AxHat = matVecMul(mA, xHat);
    const BuPrev = matVecMul(mB, uPrev);
    const xPred = new Float64Array(nx);
    for (let i = 0; i < nx; i++) {
      xPred[i] = AxHat[i]! + BuPrev[i]!;
    }

    // --- Correct ---
    // Innovation: y - C xHat_pred
    const yPred = matVecMul(mC, xPred);
    const innovation = vecSub(y, yPred);
    const correction = matVecMul(mL, innovation);
    xHat = vecAdd(xPred, correction) as Float64Array<ArrayBuffer>;
    estimates.push(new Float64Array(xHat));

    // --- Control ---
    // u = -K xHat
    const u = matVecMul(mK, xHat);
    for (let j = 0; j < nu; j++) {
      u[j] = -u[j]!;
    }
    uPrev = u as Float64Array<ArrayBuffer>;
  }

  return estimates;
}
