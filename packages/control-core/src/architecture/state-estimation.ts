// ---------------------------------------------------------------------------
// OC-11  Real-Time Control Architecture -- State Estimation
// ---------------------------------------------------------------------------

import type { MultiSensorEstimateConfig } from '../types.js';
import {
  arrayToMatrix,
  matMul,
  matVecMul,
  matAdd,
  matSub,
  matTranspose,
  matIdentity,
  matInvert,
  createMatrix,
  matGet,
  matSet,
  vecAdd,
  vecSub,
} from '../types.js';

// ---------------------------------------------------------------------------
// multiSensorEstimate
// ---------------------------------------------------------------------------

/**
 * Sequential multi-sensor Kalman filter update.
 *
 * For each sensor that has a current measurement available in the
 * `measurements` map, the function performs a standard predict-then-update
 * cycle:
 *
 *  **Predict:**
 *    xHat_pred = F * xHat
 *    P_pred    = F * P * F^T + Q
 *
 *  **Update (per sensor i with measurement z_i):**
 *    S_i  = H_i * P_pred * H_i^T + R_i
 *    K_i  = P_pred * H_i^T * S_i^{-1}
 *    xHat = xHat_pred + K_i * (z_i - H_i * xHat_pred)
 *    P    = (I - K_i * H_i) * P_pred
 *
 * The prediction step is executed once; subsequent sensor updates are
 * applied sequentially on the predicted state.
 *
 * @param config        Multi-sensor configuration (F, Q, sensor list).
 * @param xHat          Prior state estimate (nx).
 * @param P             Prior error covariance (nx x nx, row-major flat).
 * @param measurements  Current sensor readings keyed by sensor name.
 * @returns             Updated `{ xHat, P }`.
 */
export function multiSensorEstimate(
  config: MultiSensorEstimateConfig,
  xHat: Float64Array,
  P: Float64Array,
  measurements: Map<string, Float64Array>,
): { xHat: Float64Array; P: Float64Array } {
  const { sensors, F: Fraw, Q: Qraw, nx } = config;

  const F = arrayToMatrix(Fraw, nx, nx);
  const Q = arrayToMatrix(Qraw, nx, nx);

  // ----- Predict -----
  let xPred = matVecMul(F, xHat);

  const Pmat = arrayToMatrix(P, nx, nx);
  const Ft = matTranspose(F);
  let Ppred = matAdd(matMul(matMul(F, Pmat), Ft), Q);

  // ----- Sequential sensor updates -----
  for (let s = 0; s < sensors.length; s++) {
    const sensor = sensors[s]!;
    const z = measurements.get(sensor.name);
    if (z === undefined) continue;

    const Hi = arrayToMatrix(sensor.H, sensor.dimZ, nx);
    const Ri = arrayToMatrix(sensor.R, sensor.dimZ, sensor.dimZ);

    // Innovation covariance: S = H * P_pred * H^T + R
    const Ht = matTranspose(Hi);
    const S = matAdd(matMul(matMul(Hi, Ppred), Ht), Ri);

    // Kalman gain: K = P_pred * H^T * S^{-1}
    const Sinv = matInvert(S);
    const K = matMul(matMul(Ppred, Ht), Sinv);

    // Innovation: y = z - H * xPred
    const zPred = matVecMul(Hi, xPred);
    const innovation = vecSub(z, zPred);

    // State update: xPred = xPred + K * innovation
    const Kinn = matVecMul(K, innovation);
    xPred = vecAdd(xPred, Kinn);

    // Covariance update: P = (I - K * H) * P_pred
    const I = matIdentity(nx);
    const KH = matMul(K, Hi);
    Ppred = matMul(matSub(I, KH), Ppred);
  }

  return {
    xHat: xPred,
    P: new Float64Array(Ppred.data),
  };
}

// ---------------------------------------------------------------------------
// movingHorizonEstimate
// ---------------------------------------------------------------------------

/**
 * Moving-horizon estimation (MHE) via least-squares over a sliding window.
 *
 * Solves the normal equations for:
 *
 *   min  sum_{k}  || z_k - H * x_k ||^2  +  || x_{k+1} - F * x_k ||^2
 *
 * over the most recent `windowSize` measurements.  This simplified
 * implementation treats only the latest time step's state as the decision
 * variable and folds all prior measurements into a single batch
 * least-squares problem:
 *
 *   x* = (A^T A)^{-1} A^T b
 *
 * where the stacked system incorporates observation and dynamics residuals.
 *
 * @param F             State transition matrix (nx x nx, row-major flat).
 * @param H             Observation matrix (nz x nx, row-major flat).
 * @param nx            State dimension.
 * @param nz            Observation dimension.
 * @param windowSize    Number of recent measurements to include.
 * @param measurements  Array of measurement vectors (most recent last).
 * @returns             State estimate (nx).
 */
export function movingHorizonEstimate(
  F: Float64Array,
  H: Float64Array,
  nx: number,
  nz: number,
  windowSize: number,
  measurements: Float64Array[],
): Float64Array {
  // Use at most the last `windowSize` measurements.
  const W = Math.min(windowSize, measurements.length);
  if (W === 0) {
    return new Float64Array(nx);
  }

  const startIdx = measurements.length - W;

  const Hmat = arrayToMatrix(H, nz, nx);
  const Fmat = arrayToMatrix(F, nx, nx);

  // Number of rows in the stacked system:
  //   W observation blocks (nz each) + (W-1) dynamics blocks (nx each)
  const nRows = W * nz + (W - 1) * nx;
  // Decision variable: [x_0, x_1, ..., x_{W-1}]  (W * nx unknowns)
  const nCols = W * nx;

  const A = createMatrix(nRows, nCols);
  const b = new Float64Array(nRows);

  // Fill observation blocks: z_k = H * x_k  =>  row block k: H * x_k = z_k
  for (let k = 0; k < W; k++) {
    const rowOff = k * nz;
    const colOff = k * nx;
    const zk = measurements[startIdx + k]!;

    for (let i = 0; i < nz; i++) {
      for (let j = 0; j < nx; j++) {
        matSet(A, rowOff + i, colOff + j, matGet(Hmat, i, j));
      }
      b[rowOff + i] = zk[i]!;
    }
  }

  // Fill dynamics blocks: x_{k+1} = F * x_k
  //   => -F * x_k + I * x_{k+1} = 0
  for (let k = 0; k < W - 1; k++) {
    const rowOff = W * nz + k * nx;
    const colOffCur = k * nx;
    const colOffNext = (k + 1) * nx;

    for (let i = 0; i < nx; i++) {
      // -F in columns for x_k
      for (let j = 0; j < nx; j++) {
        matSet(A, rowOff + i, colOffCur + j, -matGet(Fmat, i, j));
      }
      // +I in columns for x_{k+1}
      matSet(A, rowOff + i, colOffNext + i, 1);
      // RHS = 0 (already zero-initialised)
    }
  }

  // Normal equations: (A^T A) xVec = A^T b
  const At = matTranspose(A);
  const AtA = matMul(At, A);
  const Atb = matVecMul(At, b);

  const AtAinv = matInvert(AtA);
  const xVec = matVecMul(AtAinv, Atb);

  // Extract the last state block (most recent estimate).
  const result = new Float64Array(nx);
  const lastOff = (W - 1) * nx;
  for (let i = 0; i < nx; i++) {
    result[i] = xVec[lastOff + i]!;
  }

  return result;
}
