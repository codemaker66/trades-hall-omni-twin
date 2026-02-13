// ---------------------------------------------------------------------------
// SP-3: Rauch-Tung-Striebel (RTS) Smoother
// ---------------------------------------------------------------------------
// Backward pass for retrospective analysis:
//   C(k) = P(k|k)·Fᵀ·P(k+1|k)⁻¹
//   x̂ˢ(k) = x̂(k|k) + C(k)·(x̂ˢ(k+1) - F·x̂(k|k))
//   Pˢ(k) = P(k|k) + C(k)·(Pˢ(k+1) - P(k+1|k))·C(k)ᵀ

import type { KalmanConfig, KalmanState, RTSSmootherResult } from '../types.js';
import {
  arrayToMatrix, matMul, matTranspose, matAdd, matSub,
  matInvert, matVecMul,
  type Matrix,
} from '../types.js';

/**
 * RTS smoother: backward pass on Kalman filter results.
 *
 * @param filteredStates States from forward Kalman filter pass (x̂(k|k), P(k|k))
 * @param predictedStates Predicted states from forward pass (x̂(k|k-1), P(k|k-1))
 * @param config Kalman filter configuration
 */
export function rtsSmooth(
  filteredStates: KalmanState[],
  predictedStates: KalmanState[],
  config: KalmanConfig,
): RTSSmootherResult {
  const T = filteredStates.length;
  const { dimX } = config;
  const F = arrayToMatrix(config.F, dimX, dimX);
  const FT = matTranspose(F);

  const smoothedStates: Float64Array[] = new Array(T);
  const smoothedCovariances: Float64Array[] = new Array(T);

  // Initialize with last filtered state
  smoothedStates[T - 1] = new Float64Array(filteredStates[T - 1]!.x);
  smoothedCovariances[T - 1] = new Float64Array(filteredStates[T - 1]!.P);

  // Backward pass
  for (let k = T - 2; k >= 0; k--) {
    const Pk = arrayToMatrix(filteredStates[k]!.P, dimX, dimX);
    const PkPlus1Pred = arrayToMatrix(predictedStates[k + 1]!.P, dimX, dimX);

    // Smoother gain: C(k) = P(k|k)·Fᵀ·P(k+1|k)⁻¹
    const PkFT = matMul(Pk, FT);
    const PkPredInv = matInvert(PkPlus1Pred);
    const C = matMul(PkFT, PkPredInv);

    // Smoothed state: x̂ˢ(k) = x̂(k|k) + C·(x̂ˢ(k+1) - F·x̂(k|k))
    const FxK = matVecMul(F, filteredStates[k]!.x);
    const diff = new Float64Array(dimX);
    for (let i = 0; i < dimX; i++) {
      diff[i] = smoothedStates[k + 1]![i]! - FxK[i]!;
    }
    const Cdiff = matVecMul(C, diff);
    const xSmooth = new Float64Array(dimX);
    for (let i = 0; i < dimX; i++) {
      xSmooth[i] = filteredStates[k]!.x[i]! + Cdiff[i]!;
    }
    smoothedStates[k] = xSmooth;

    // Smoothed covariance: Pˢ(k) = P(k|k) + C·(Pˢ(k+1) - P(k+1|k))·Cᵀ
    const PkPlus1Smooth = arrayToMatrix(smoothedCovariances[k + 1]!, dimX, dimX);
    const Pdiff = matSub(PkPlus1Smooth, PkPlus1Pred);
    const CT = matTranspose(C);
    const CPdiff = matMul(C, Pdiff);
    const CPdiffCT = matMul(CPdiff, CT);
    const PSmooth = matAdd(Pk, CPdiffCT);
    smoothedCovariances[k] = PSmooth.data;
  }

  return {
    smoothedStates: smoothedStates as Float64Array[],
    smoothedCovariances: smoothedCovariances as Float64Array[],
  };
}
