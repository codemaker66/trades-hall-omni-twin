// ---------------------------------------------------------------------------
// OC-1: Tracking LQR with Integral Action
// ---------------------------------------------------------------------------

import type {
  LQRConfig,
  TrackingLQRConfig,
  TrackingLQRResult,
} from '../types.js';
import {
  arrayToMatrix,
  createMatrix,
  matGet,
  matInvert,
  matMul,
  matSet,
  matTranspose,
  matVecMul,
  vecAdd,
  vecSub,
} from '../types.js';

import { solveDARE } from './dare.js';

// ---------------------------------------------------------------------------
// Compute steady-state input for a reference output
// ---------------------------------------------------------------------------

/**
 * Compute a steady-state input u_ss for the system x+ = Ax + Bu, y = Cx.
 *
 * At steady state: (I - A) x_ss = B u_ss, y_ss = C x_ss.
 * We build the system:
 *   [I - A, -B] [x_ss]   [0]
 *   [  C,    0] [u_ss] = [I]  (per output)
 *
 * and solve via least-squares (normal equations) to find the u_ss that
 * produces a unit output on average. The returned vector has length nu.
 *
 * @param A  State matrix (nx x nx, row-major)
 * @param B  Input matrix (nx x nu, row-major)
 * @param C  Output matrix (ny x nx, row-major)
 * @param nx State dimension
 * @param nu Control dimension
 * @param ny Output dimension
 * @returns Steady-state input u_ss (nu)
 */
export function computeSteadyState(
  A: Float64Array,
  B: Float64Array,
  C: Float64Array,
  nx: number,
  nu: number,
  ny: number,
): Float64Array {
  const dim = nx + nu;
  const mA = arrayToMatrix(A, nx, nx);
  const mB = arrayToMatrix(B, nx, nu);
  const mC = arrayToMatrix(C, ny, nx);

  // Build the combined matrix M of dimension (nx + ny) x (nx + nu)
  // Top: [I - A, -B]   (nx rows)
  // Bot: [C,      0]   (ny rows)
  const nRows = nx + ny;
  const M = createMatrix(nRows, dim);

  // Top-left: I - A
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < nx; j++) {
      const val = (i === j ? 1 : 0) - matGet(mA, i, j);
      matSet(M, i, j, val);
    }
  }
  // Top-right: -B
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < nu; j++) {
      matSet(M, i, nx + j, -matGet(mB, i, j));
    }
  }
  // Bottom-left: C
  for (let i = 0; i < ny; i++) {
    for (let j = 0; j < nx; j++) {
      matSet(M, nx + i, j, matGet(mC, i, j));
    }
  }
  // Bottom-right: 0 (already zero from createMatrix)

  // RHS: [0; 1, 1, ..., 1] -- unit reference for each output
  const rhs = new Float64Array(nRows);
  for (let i = 0; i < ny; i++) {
    rhs[nx + i] = 1.0;
  }

  // Solve via least-squares: M^T M z = M^T rhs (normal equations)
  const Mt = matTranspose(M);
  const MtM = matMul(Mt, M);
  const Mtrhs = matVecMul(Mt, rhs);
  const MtMInv = matInvert(MtM);
  const z = matVecMul(MtMInv, Mtrhs);

  // Extract u_ss from z = [x_ss; u_ss]
  const uSS = new Float64Array(nu);
  for (let i = 0; i < nu; i++) {
    uSS[i] = z[nx + i]!;
  }

  return uSS;
}

// ---------------------------------------------------------------------------
// Tracking LQR with integral action
// ---------------------------------------------------------------------------

/**
 * Design a tracking LQR controller with integral action.
 *
 * The system is augmented with ny integrator states to track C*x - r:
 *
 *   xi_{t+1} = xi_t + (C x_t - r_t)
 *
 * Augmented state z = [x; xi], augmented system:
 *
 *   z+ = [A, 0; C, I] z + [B; 0] u + [0; -r]
 *
 * The LQR is solved on the augmented system with cost:
 *
 *   Q_aug = blkdiag(Q, Q_i),  R_aug = R
 *
 * where Q_i adds quadratic cost on the integral states (scaled from the
 * mean diagonal of Q).
 *
 * @param config Tracking LQR config (extends LQRConfig with C and ny)
 * @returns TrackingLQRResult with gains K, Kaug (augmented), Kff (integral),
 *          DARE solution P, and eigenvalues
 */
export function trackingLQR(config: TrackingLQRConfig): TrackingLQRResult {
  const { A, B, Q, R, N, C, nx, nu, ny } = config;

  const nAug = nx + ny; // augmented state dimension

  const mA = arrayToMatrix(A, nx, nx);
  const mB = arrayToMatrix(B, nx, nu);
  const mQ = arrayToMatrix(Q, nx, nx);
  const mC = arrayToMatrix(C, ny, nx);

  // Build augmented state matrix A_aug = [A, 0; C, I]
  const Aaug = createMatrix(nAug, nAug);
  // Top-left: A
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < nx; j++) {
      matSet(Aaug, i, j, matGet(mA, i, j));
    }
  }
  // Bottom-left: C
  for (let i = 0; i < ny; i++) {
    for (let j = 0; j < nx; j++) {
      matSet(Aaug, nx + i, j, matGet(mC, i, j));
    }
  }
  // Bottom-right: I_{ny}
  for (let i = 0; i < ny; i++) {
    matSet(Aaug, nx + i, nx + i, 1);
  }

  // Build augmented input matrix B_aug = [B; 0]
  const Baug = createMatrix(nAug, nu);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < nu; j++) {
      matSet(Baug, i, j, matGet(mB, i, j));
    }
  }

  // Build augmented cost Q_aug = blkdiag(Q, Q_i)
  // Q_i weight: mean diagonal of Q, clamped to at least 1
  let qMean = 0;
  for (let i = 0; i < nx; i++) {
    qMean += matGet(mQ, i, i);
  }
  qMean /= nx;
  const qiWeight = Math.max(qMean, 1);

  const Qaug = createMatrix(nAug, nAug);
  // Top-left: Q
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < nx; j++) {
      matSet(Qaug, i, j, matGet(mQ, i, j));
    }
  }
  // Bottom-right: Q_i = qiWeight * I_{ny}
  for (let i = 0; i < ny; i++) {
    matSet(Qaug, nx + i, nx + i, qiWeight);
  }

  // Build augmented cross-coupling N_aug = [N; 0] or undefined
  let Naug: Float64Array | undefined;
  if (N) {
    const nAugData = new Float64Array(nAug * nu);
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < nu; j++) {
        nAugData[i * nu + j] = N[i * nu + j]!;
      }
    }
    Naug = nAugData;
  }

  // Build augmented LQRConfig and solve DARE
  const augConfig: LQRConfig = {
    A: new Float64Array(Aaug.data),
    B: new Float64Array(Baug.data),
    Q: new Float64Array(Qaug.data),
    R,
    N: Naug,
    nx: nAug,
    nu,
  };

  const augResult = solveDARE(augConfig);

  // augResult.K is (nu x nAug) -- extract sub-gains
  const mKfull = arrayToMatrix(augResult.K, nu, nAug);

  // Kx is nu x nx, Ki is nu x ny
  const Kx = new Float64Array(nu * nx);
  const Ki = new Float64Array(nu * ny);
  for (let i = 0; i < nu; i++) {
    for (let j = 0; j < nx; j++) {
      Kx[i * nx + j] = matGet(mKfull, i, j);
    }
    for (let j = 0; j < ny; j++) {
      Ki[i * ny + j] = matGet(mKfull, i, nx + j);
    }
  }

  return {
    K: Kx,
    P: augResult.P,
    eigenvalues: augResult.eigenvalues,
    Kaug: augResult.K,
    Kff: Ki,
  };
}

// ---------------------------------------------------------------------------
// Tracking simulation
// ---------------------------------------------------------------------------

/**
 * Simulate the tracking LQR controller with integral action.
 *
 * At each step:
 *   y_t = C x_t
 *   e_t = y_t - r_t                (tracking error)
 *   xi_{t+1} = xi_t + e_t          (integral state update)
 *   u_t = -K x_t - Ki xi_t         (control with integral action)
 *   x_{t+1} = A x_t + B u_t        (state update)
 *
 * @param A         State matrix (nx x nx, row-major)
 * @param B         Input matrix (nx x nu, row-major)
 * @param K         State feedback gain (nu x nx, row-major)
 * @param Ki        Integral feedback gain (nu x ny, row-major)
 * @param C         Output matrix (ny x nx, row-major)
 * @param x0        Initial state (nx)
 * @param reference Array of output references, one per step (ny each)
 * @param nx        State dimension
 * @param nu        Control dimension
 * @param ny        Output dimension
 * @returns Array of state vectors [x_0, x_1, ..., x_T]
 */
export function simulateTracking(
  A: Float64Array,
  B: Float64Array,
  K: Float64Array,
  Ki: Float64Array,
  C: Float64Array,
  x0: Float64Array,
  reference: Float64Array[],
  nx: number,
  nu: number,
  ny: number,
): Float64Array[] {
  const mA = arrayToMatrix(A, nx, nx);
  const mB = arrayToMatrix(B, nx, nu);
  const mK = arrayToMatrix(K, nu, nx);
  const mKi = arrayToMatrix(Ki, nu, ny);
  const mC = arrayToMatrix(C, ny, nx);

  const T = reference.length;

  const trajectory: Float64Array[] = [new Float64Array(x0)];
  let x = new Float64Array(x0);
  let xi = new Float64Array(ny); // integral error state

  for (let t = 0; t < T; t++) {
    const ref = reference[t]!;

    // Output: y_t = C x_t
    const y = matVecMul(mC, x);

    // Tracking error: e_t = y_t - r_t
    const e = vecSub(y, ref);

    // Update integral state: xi_{t+1} = xi_t + e_t
    xi = vecAdd(xi, e) as Float64Array<ArrayBuffer>;

    // Control: u_t = -K x_t - Ki xi_t
    const Kx_val = matVecMul(mK, x);
    const Ki_val = matVecMul(mKi, xi);
    const u = new Float64Array(nu);
    for (let j = 0; j < nu; j++) {
      u[j] = -Kx_val[j]! - Ki_val[j]!;
    }

    // State update: x_{t+1} = A x_t + B u_t
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
