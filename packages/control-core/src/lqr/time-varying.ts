// ---------------------------------------------------------------------------
// OC-1: Time-Varying (Finite-Horizon) LQR via Backward Riccati Recursion
// ---------------------------------------------------------------------------

import type {
  TimeVaryingLQRConfig,
  TimeVaryingLQRResult,
} from '../types.js';
import {
  arrayToMatrix,
  matAdd,
  matInvert,
  matMul,
  matSub,
  matTranspose,
  matVecMul,
} from '../types.js';

// ---------------------------------------------------------------------------
// Finite-horizon time-varying LQR
// ---------------------------------------------------------------------------

/**
 * Solve the finite-horizon time-varying LQR problem via backward Riccati
 * recursion.
 *
 * The cost-to-go matrices are computed backward from the terminal cost:
 *
 *   P_N = Qf
 *   P_t = Q_t + A_t^T P_{t+1} A_t
 *         - A_t^T P_{t+1} B_t (R_t + B_t^T P_{t+1} B_t)^{-1} B_t^T P_{t+1} A_t
 *
 * The gain matrices are:
 *
 *   K_t = (R_t + B_t^T P_{t+1} B_t)^{-1} B_t^T P_{t+1} A_t
 *
 * @param config Time-varying LQR configuration
 * @returns TimeVaryingLQRResult with gain and cost-to-go matrices at each step
 */
export function timeVaryingLQR(
  config: TimeVaryingLQRConfig,
): TimeVaryingLQRResult {
  const { As, Bs, Qs, Rs, Qf, nx, nu, horizon } = config;

  // Allocate arrays for results (one per time step)
  const Ks: Float64Array[] = new Array(horizon);
  const Ps: Float64Array[] = new Array(horizon + 1);

  // Terminal condition: P_N = Qf
  Ps[horizon] = new Float64Array(Qf);

  // Backward sweep: t = horizon-1, horizon-2, ..., 0
  for (let t = horizon - 1; t >= 0; t--) {
    const mA = arrayToMatrix(As[t]!, nx, nx);
    const mB = arrayToMatrix(Bs[t]!, nx, nu);
    const mQ = arrayToMatrix(Qs[t]!, nx, nx);
    const mR = arrayToMatrix(Rs[t]!, nu, nu);
    const Pnext = arrayToMatrix(Ps[t + 1]!, nx, nx);

    const At = matTranspose(mA);
    const Bt = matTranspose(mB);

    // A_t^T P_{t+1}
    const AtP = matMul(At, Pnext);
    // A_t^T P_{t+1} A_t
    const AtPA = matMul(AtP, mA);
    // A_t^T P_{t+1} B_t
    const AtPB = matMul(AtP, mB);
    // B_t^T P_{t+1}
    const BtP = matMul(Bt, Pnext);
    // B_t^T P_{t+1} B_t
    const BtPB = matMul(BtP, mB);
    // B_t^T P_{t+1} A_t
    const BtPA = matMul(BtP, mA);

    // (R_t + B_t^T P_{t+1} B_t)^{-1}
    const inv = matInvert(matAdd(mR, BtPB));

    // K_t = (R_t + B_t^T P_{t+1} B_t)^{-1} B_t^T P_{t+1} A_t
    const K = matMul(inv, BtPA);
    Ks[t] = new Float64Array(K.data);

    // P_t = Q_t + A_t^T P_{t+1} A_t
    //       - A_t^T P_{t+1} B_t (R_t + B_t^T P_{t+1} B_t)^{-1} B_t^T P_{t+1} A_t
    const correction = matMul(AtPB, matMul(inv, BtPA));
    const Pt = matSub(matAdd(mQ, AtPA), correction);
    Ps[t] = new Float64Array(Pt.data);
  }

  return { Ks, Ps };
}

// ---------------------------------------------------------------------------
// Time-varying LQR simulation
// ---------------------------------------------------------------------------

/**
 * Simulate the closed-loop time-varying LQR system.
 *
 *   u_t = -K_t x_t
 *   x_{t+1} = A_t x_t + B_t u_t
 *
 * @param config Time-varying LQR configuration (contains As, Bs, horizon)
 * @param Ks     Gain matrices from timeVaryingLQR result (one per time step)
 * @param x0     Initial state (nx)
 * @returns Array of state vectors [x_0, x_1, ..., x_horizon]
 */
export function simulateTimeVaryingLQR(
  config: TimeVaryingLQRConfig,
  Ks: Float64Array[],
  x0: Float64Array,
): Float64Array[] {
  const { As, Bs, nx, nu, horizon } = config;

  const trajectory: Float64Array[] = [new Float64Array(x0)];
  let x = new Float64Array(x0);

  for (let t = 0; t < horizon; t++) {
    const mA = arrayToMatrix(As[t]!, nx, nx);
    const mB = arrayToMatrix(Bs[t]!, nx, nu);
    const mK = arrayToMatrix(Ks[t]!, nu, nx);

    // u_t = -K_t x_t
    const u = matVecMul(mK, x);
    for (let j = 0; j < nu; j++) {
      u[j] = -u[j]!;
    }

    // x_{t+1} = A_t x_t + B_t u_t
    const Ax = matVecMul(mA, x);
    const Bu = matVecMul(mB, u);
    const xNext = new Float64Array(nx);
    for (let i = 0; i < nx; i++) {
      xNext[i] = Ax[i]! + Bu[i]!;
    }

    trajectory.push(new Float64Array(xNext));
    x = xNext;
  }

  return trajectory;
}
