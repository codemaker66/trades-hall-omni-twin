// ---------------------------------------------------------------------------
// SP-3: Unscented Kalman Filter (UKF)
// ---------------------------------------------------------------------------
// Avoids Jacobians via 2n+1 sigma points:
//   χ₀ = x̂
//   χᵢ = x̂ ± √((n+λ)P)ᵢ
// Merwe-Wan scaling: α ~1e-3 to 1, β=2 (Gaussian optimal), κ=0.
// Use when demand model is nonlinear.

import type { UKFConfig, KalmanState } from '../types.js';
import {
  arrayToMatrix, matMul, matTranspose, matAdd, matSub,
  matInvert, matVecMul, matIdentity, matScale,
  createMatrix, matSet, matGet,
  type Matrix,
} from '../types.js';

/**
 * Compute Cholesky decomposition: A = L·Lᵀ
 * Returns lower triangular L.
 */
function cholesky(A: Matrix): Matrix {
  const n = A.rows;
  const L = createMatrix(n, n);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) {
        sum += matGet(L, i, k) * matGet(L, j, k);
      }
      if (i === j) {
        const diag = matGet(A, i, i) - sum;
        if (diag < 0) {
          // Matrix not positive definite — regularize
          matSet(L, i, i, Math.sqrt(Math.max(diag, 1e-10)));
        } else {
          matSet(L, i, i, Math.sqrt(diag));
        }
      } else {
        const Ljj = matGet(L, j, j);
        matSet(L, i, j, Ljj !== 0 ? (matGet(A, i, j) - sum) / Ljj : 0);
      }
    }
  }
  return L;
}

/**
 * Generate sigma points using Van der Merwe's method.
 */
export function generateSigmaPoints(
  x: Float64Array,
  P: Float64Array,
  alpha: number,
  beta: number,
  kappa: number,
): { sigmaPoints: Float64Array[]; weightsMean: Float64Array; weightsCov: Float64Array } {
  const n = x.length;
  const lambda = alpha * alpha * (n + kappa) - n;
  const sqrtScale = Math.sqrt(n + lambda);

  const PMat = arrayToMatrix(P, n, n);
  const scaledP = matScale(PMat, n + lambda);
  const L = cholesky(scaledP);

  const nSigma = 2 * n + 1;
  const sigmaPoints: Float64Array[] = [];

  // χ₀ = x̂
  sigmaPoints.push(new Float64Array(x));

  // χᵢ = x̂ + √((n+λ)P)ᵢ for i = 1..n
  for (let i = 0; i < n; i++) {
    const pt = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      pt[j] = x[j]! + matGet(L, j, i);
    }
    sigmaPoints.push(pt);
  }

  // χᵢ = x̂ - √((n+λ)P)ᵢ for i = n+1..2n
  for (let i = 0; i < n; i++) {
    const pt = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      pt[j] = x[j]! - matGet(L, j, i);
    }
    sigmaPoints.push(pt);
  }

  // Weights
  const weightsMean = new Float64Array(nSigma);
  const weightsCov = new Float64Array(nSigma);

  weightsMean[0] = lambda / (n + lambda);
  weightsCov[0] = lambda / (n + lambda) + (1 - alpha * alpha + beta);

  const w = 1 / (2 * (n + lambda));
  for (let i = 1; i < nSigma; i++) {
    weightsMean[i] = w;
    weightsCov[i] = w;
  }

  return { sigmaPoints, weightsMean, weightsCov };
}

/**
 * UKF predict step.
 */
export function ukfPredict(state: KalmanState, config: UKFConfig): KalmanState {
  const { dimX, alpha, beta, kappa, stateTransitionFn, Q } = config;

  const { sigmaPoints, weightsMean, weightsCov } = generateSigmaPoints(
    state.x, state.P, alpha, beta, kappa,
  );

  // Transform sigma points through state transition
  const transformedPoints = sigmaPoints.map(sp => stateTransitionFn(sp));

  // Compute predicted mean
  const xPred = new Float64Array(dimX);
  for (let i = 0; i < transformedPoints.length; i++) {
    for (let j = 0; j < dimX; j++) {
      xPred[j] = xPred[j]! + weightsMean[i]! * transformedPoints[i]![j]!;
    }
  }

  // Compute predicted covariance
  const PPred = new Float64Array(dimX * dimX);
  for (let i = 0; i < transformedPoints.length; i++) {
    const diff = new Float64Array(dimX);
    for (let j = 0; j < dimX; j++) {
      diff[j] = transformedPoints[i]![j]! - xPred[j]!;
    }
    for (let r = 0; r < dimX; r++) {
      for (let c = 0; c < dimX; c++) {
        PPred[r * dimX + c] = PPred[r * dimX + c]! + weightsCov[i]! * diff[r]! * diff[c]!;
      }
    }
  }

  // Add process noise
  for (let i = 0; i < dimX * dimX; i++) {
    PPred[i] = PPred[i]! + Q[i]!;
  }

  return { x: xPred, P: PPred, dim: dimX };
}

/**
 * UKF update step.
 */
export function ukfUpdate(
  state: KalmanState,
  measurement: Float64Array,
  config: UKFConfig,
): { state: KalmanState; innovation: Float64Array } {
  const { dimX, dimZ, alpha, beta, kappa, observationFn, R } = config;

  const { sigmaPoints, weightsMean, weightsCov } = generateSigmaPoints(
    state.x, state.P, alpha, beta, kappa,
  );

  // Transform sigma points through observation model
  const observedPoints = sigmaPoints.map(sp => observationFn(sp));

  // Predicted measurement mean
  const zPred = new Float64Array(dimZ);
  for (let i = 0; i < observedPoints.length; i++) {
    for (let j = 0; j < dimZ; j++) {
      zPred[j] = zPred[j]! + weightsMean[i]! * observedPoints[i]![j]!;
    }
  }

  // Innovation
  const innovation = new Float64Array(dimZ);
  for (let i = 0; i < dimZ; i++) {
    innovation[i] = measurement[i]! - zPred[i]!;
  }

  // Innovation covariance S = Σ w_i (z_i - z̄)(z_i - z̄)ᵀ + R
  const S = new Float64Array(dimZ * dimZ);
  for (let i = 0; i < observedPoints.length; i++) {
    const dz = new Float64Array(dimZ);
    for (let j = 0; j < dimZ; j++) {
      dz[j] = observedPoints[i]![j]! - zPred[j]!;
    }
    for (let r = 0; r < dimZ; r++) {
      for (let c = 0; c < dimZ; c++) {
        S[r * dimZ + c] = S[r * dimZ + c]! + weightsCov[i]! * dz[r]! * dz[c]!;
      }
    }
  }
  for (let i = 0; i < dimZ * dimZ; i++) {
    S[i] = S[i]! + R[i]!;
  }

  // Cross-covariance Pxz = Σ w_i (x_i - x̄)(z_i - z̄)ᵀ
  const Pxz = new Float64Array(dimX * dimZ);
  for (let i = 0; i < sigmaPoints.length; i++) {
    const dx = new Float64Array(dimX);
    const dz = new Float64Array(dimZ);
    for (let j = 0; j < dimX; j++) dx[j] = sigmaPoints[i]![j]! - state.x[j]!;
    for (let j = 0; j < dimZ; j++) dz[j] = observedPoints[i]![j]! - zPred[j]!;
    for (let r = 0; r < dimX; r++) {
      for (let c = 0; c < dimZ; c++) {
        Pxz[r * dimZ + c] = Pxz[r * dimZ + c]! + weightsCov[i]! * dx[r]! * dz[c]!;
      }
    }
  }

  // Kalman gain K = Pxz · S⁻¹
  const SMat = arrayToMatrix(S, dimZ, dimZ);
  const SInv = matInvert(SMat);
  const PxzMat = arrayToMatrix(Pxz, dimX, dimZ);
  const K = matMul(PxzMat, SInv);

  // Updated state
  const Ky = matVecMul(K, innovation);
  const xUpd = new Float64Array(dimX);
  for (let i = 0; i < dimX; i++) {
    xUpd[i] = state.x[i]! + Ky[i]!;
  }

  // Updated covariance P = P - K·S·Kᵀ
  const KS = matMul(K, SMat);
  const KT = matTranspose(K);
  const KSKT = matMul(KS, KT);
  const PMat = arrayToMatrix(state.P, dimX, dimX);
  const PUpd = matSub(PMat, KSKT);

  return {
    state: { x: xUpd, P: PUpd.data, dim: dimX },
    innovation,
  };
}

/**
 * Full UKF step (predict + update).
 */
export function ukfStep(
  state: KalmanState,
  measurement: Float64Array,
  config: UKFConfig,
): { state: KalmanState; innovation: Float64Array } {
  const predicted = ukfPredict(state, config);
  return ukfUpdate(predicted, measurement, config);
}
