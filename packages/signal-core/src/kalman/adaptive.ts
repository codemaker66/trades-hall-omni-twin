// ---------------------------------------------------------------------------
// SP-3: Adaptive Kalman Filter
// ---------------------------------------------------------------------------
// Innovation-based Q/R estimation:
//   Ĉ(k) = α·Ĉ(k-1) + (1-α)·ỹ(k)·ỹ(k)ᵀ
//   R̂ = Ĉ - H·P·Hᵀ
// Exponential forgetting factor α ∈ [0.95, 0.99]

import type { AdaptiveKalmanConfig, KalmanState, KalmanResult } from '../types.js';
import {
  arrayToMatrix, matMul, matTranspose, matAdd, matSub,
  matInvert, matVecMul, matIdentity, matScale,
  createMatrix, matSet, matGet,
  type Matrix,
} from '../types.js';

export interface AdaptiveKalmanState extends KalmanState {
  /** Innovation covariance estimator Ĉ (dimZ × dimZ) */
  innovationCov: Float64Array;
  /** Estimated measurement noise R̂ (dimZ × dimZ) */
  estimatedR: Float64Array;
}

/**
 * Create adaptive Kalman filter state.
 */
export function createAdaptiveState(
  config: AdaptiveKalmanConfig,
  initialState?: Float64Array,
): AdaptiveKalmanState {
  const { dimX, dimZ, R } = config;
  const x = initialState ?? new Float64Array(dimX);
  const P = new Float64Array(dimX * dimX);
  for (let i = 0; i < dimX; i++) P[i * dimX + i] = 100;

  return {
    x: new Float64Array(x),
    P,
    dim: dimX,
    innovationCov: new Float64Array(R), // Initialize Ĉ = R
    estimatedR: new Float64Array(R),
  };
}

/**
 * Adaptive Kalman filter step with innovation-based noise estimation.
 */
export function adaptiveKalmanStep(
  state: AdaptiveKalmanState,
  measurement: Float64Array,
  config: AdaptiveKalmanConfig,
): { state: AdaptiveKalmanState; innovation: Float64Array } {
  const { dimX, dimZ, forgettingFactor: alpha } = config;
  const F = arrayToMatrix(config.F, dimX, dimX);
  const H = arrayToMatrix(config.H, dimZ, dimX);
  const Q = arrayToMatrix(config.Q, dimX, dimX);

  // --- Predict ---
  const xPred = matVecMul(F, state.x);
  const P = arrayToMatrix(state.P, dimX, dimX);
  const FT = matTranspose(F);
  const PPred = matAdd(matMul(matMul(F, P), FT), Q);

  // --- Innovation ---
  const Hx = matVecMul(H, xPred);
  const innovation = new Float64Array(dimZ);
  for (let i = 0; i < dimZ; i++) {
    innovation[i] = measurement[i]! - Hx[i]!;
  }

  // --- Update innovation covariance estimator ---
  // Ĉ(k) = α·Ĉ(k-1) + (1-α)·ỹ(k)·ỹ(k)ᵀ
  const newInnovCov = new Float64Array(dimZ * dimZ);
  for (let r = 0; r < dimZ; r++) {
    for (let c = 0; c < dimZ; c++) {
      newInnovCov[r * dimZ + c] =
        alpha * state.innovationCov[r * dimZ + c]! +
        (1 - alpha) * innovation[r]! * innovation[c]!;
    }
  }

  // --- Estimate R̂ = Ĉ - H·P·Hᵀ ---
  const HT = matTranspose(H);
  const HPH = matMul(matMul(H, PPred), HT);
  const estimatedR = new Float64Array(dimZ * dimZ);
  for (let i = 0; i < dimZ * dimZ; i++) {
    estimatedR[i] = Math.max(newInnovCov[i]! - HPH.data[i]!, 0.01);
  }

  // Ensure diagonal dominance (R should be positive definite)
  for (let i = 0; i < dimZ; i++) {
    const idx = i * dimZ + i;
    estimatedR[idx] = Math.max(estimatedR[idx]!, 0.1);
  }

  // --- Update with estimated R ---
  const RMat = arrayToMatrix(estimatedR, dimZ, dimZ);
  const S = matAdd(matMul(matMul(H, PPred), HT), RMat);
  const SInv = matInvert(S);
  const K = matMul(matMul(PPred, HT), SInv);

  const Ky = matVecMul(K, innovation);
  const xUpd = new Float64Array(dimX);
  for (let i = 0; i < dimX; i++) {
    xUpd[i] = xPred[i]! + Ky[i]!;
  }

  const I = matIdentity(dimX);
  const KH = matMul(K, H);
  const IminusKH = matSub(I, KH);
  const PUpd = matMul(IminusKH, PPred);

  return {
    state: {
      x: xUpd,
      P: PUpd.data,
      dim: dimX,
      innovationCov: newInnovCov,
      estimatedR,
    },
    innovation,
  };
}
