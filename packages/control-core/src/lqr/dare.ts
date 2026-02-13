// ---------------------------------------------------------------------------
// OC-1: Discrete Algebraic Riccati Equation (DARE) Solver + LQR
// ---------------------------------------------------------------------------

import type { LQRConfig, LQRResult, Matrix } from '../types.js';
import {
  arrayToMatrix,
  createMatrix,
  matAdd,
  matGet,
  matInvert,
  matMul,
  matNorm,
  matSet,
  matSub,
  matTranspose,
  matVecMul,
} from '../types.js';

// ---------------------------------------------------------------------------
// Helper: wrap LQRConfig Float64Arrays into Matrix objects
// ---------------------------------------------------------------------------

/** Convert LQRConfig flat arrays into Matrix objects for internal computation. */
export function configToMatrices(config: LQRConfig): {
  A: Matrix;
  B: Matrix;
  Q: Matrix;
  R: Matrix;
  N: Matrix;
} {
  const { A, B, Q, R, N, nx, nu } = config;
  return {
    A: arrayToMatrix(A, nx, nx),
    B: arrayToMatrix(B, nx, nu),
    Q: arrayToMatrix(Q, nx, nx),
    R: arrayToMatrix(R, nu, nu),
    N: N ? arrayToMatrix(N, nx, nu) : createMatrix(nx, nu),
  };
}

// ---------------------------------------------------------------------------
// Approximate closed-loop eigenvalues via QR iteration
// ---------------------------------------------------------------------------

/**
 * Estimate the eigenvalues of a matrix using QR iteration (simplified).
 * Returns pairs [re, im] packed into a Float64Array of length 2 * n.
 *
 * This is a lightweight QR iteration suitable for small matrices arising
 * from LQR problems (typically nx <= 10). It does not implement implicit
 * shifts or deflation.
 */
function approximateEigenvalues(M: Matrix): Float64Array {
  const n = M.rows;
  const result = new Float64Array(2 * n);

  // Work on a copy -- we will overwrite during QR iterations
  let Ak = arrayToMatrix(M.data, n, n);
  const maxQRIter = 200;

  for (let iter = 0; iter < maxQRIter; iter++) {
    // Gram-Schmidt QR decomposition: A_k = Q R
    const Q = createMatrix(n, n);
    const Rk = createMatrix(n, n);

    for (let j = 0; j < n; j++) {
      // Copy column j of Ak into v
      const v = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        v[i] = matGet(Ak, i, j);
      }

      // Orthogonalise against previous columns
      for (let k = 0; k < j; k++) {
        let dot = 0;
        for (let i = 0; i < n; i++) {
          dot += matGet(Q, i, k) * v[i]!;
        }
        matSet(Rk, k, j, dot);
        for (let i = 0; i < n; i++) {
          v[i] = v[i]! - dot * matGet(Q, i, k);
        }
      }

      // Normalise
      let norm = 0;
      for (let i = 0; i < n; i++) {
        norm += v[i]! * v[i]!;
      }
      norm = Math.sqrt(norm);
      matSet(Rk, j, j, norm);
      if (norm > 1e-15) {
        for (let i = 0; i < n; i++) {
          matSet(Q, i, j, v[i]! / norm);
        }
      }
    }

    // A_{k+1} = R Q
    Ak = matMul(Rk, Q);
  }

  // Extract eigenvalues from the (quasi-upper-triangular) result
  let i = 0;
  while (i < n) {
    if (i + 1 < n && Math.abs(matGet(Ak, i + 1, i)) > 1e-10) {
      // 2x2 block -- complex conjugate pair
      const a = matGet(Ak, i, i);
      const b = matGet(Ak, i, i + 1);
      const c = matGet(Ak, i + 1, i);
      const d = matGet(Ak, i + 1, i + 1);
      const tr = a + d;
      const det = a * d - b * c;
      const disc = tr * tr - 4 * det;
      if (disc < 0) {
        const re = tr / 2;
        const im = Math.sqrt(-disc) / 2;
        result[2 * i] = re;
        result[2 * i + 1] = im;
        result[2 * (i + 1)] = re;
        result[2 * (i + 1) + 1] = -im;
      } else {
        const sq = Math.sqrt(disc);
        result[2 * i] = (tr + sq) / 2;
        result[2 * i + 1] = 0;
        result[2 * (i + 1)] = (tr - sq) / 2;
        result[2 * (i + 1) + 1] = 0;
      }
      i += 2;
    } else {
      // Real eigenvalue on the diagonal
      result[2 * i] = matGet(Ak, i, i);
      result[2 * i + 1] = 0;
      i += 1;
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// DARE solver
// ---------------------------------------------------------------------------

/**
 * Solve the Discrete Algebraic Riccati Equation via iterative fixed-point
 * (successive substitution).
 *
 *   P = Q + A^T P A - (A^T P B + N)(R + B^T P B)^{-1}(B^T P A + N^T)
 *
 * @param config  LQR configuration containing A, B, Q, R, optional N, nx, nu
 * @param tolerance Convergence tolerance (default 1e-10)
 * @param maxIter  Maximum iterations (default 1000)
 * @returns LQRResult with gain K, DARE solution P, and closed-loop eigenvalues
 */
export function solveDARE(
  config: LQRConfig,
  tolerance: number = 1e-10,
  maxIter: number = 1000,
): LQRResult {
  const { nx, nu } = config;
  const m = configToMatrices(config);

  const At = matTranspose(m.A);
  const Bt = matTranspose(m.B);
  const Nt = matTranspose(m.N);

  // Initialise P = Q
  let P = arrayToMatrix(config.Q, nx, nx);

  for (let iter = 0; iter < maxIter; iter++) {
    // AtP = A^T P
    const AtP = matMul(At, P);
    // AtPA = A^T P A
    const AtPA = matMul(AtP, m.A);
    // AtPB = A^T P B
    const AtPB = matMul(AtP, m.B);
    // BtP = B^T P
    const BtP = matMul(Bt, P);
    // BtPB = B^T P B
    const BtPB = matMul(BtP, m.B);
    // BtPA = B^T P A
    const BtPA = matMul(BtP, m.A);

    // left  = A^T P B + N
    const left = matAdd(AtPB, m.N);
    // inv   = (R + B^T P B)^{-1}
    const inv = matInvert(matAdd(m.R, BtPB));
    // right = B^T P A + N^T
    const right = matAdd(BtPA, Nt);

    // Pnew = Q + A^T P A - left * inv * right
    const correction = matMul(matMul(left, inv), right);
    const Pnew = matSub(matAdd(m.Q, AtPA), correction);

    // Check convergence: ||Pnew - P||_F < tolerance
    const diff = matSub(Pnew, P);
    const norm = matNorm(diff, 'frobenius');

    P = Pnew;

    if (norm < tolerance) {
      break;
    }
  }

  return computeLQRGain(P, config);
}

// ---------------------------------------------------------------------------
// LQR gain computation from DARE solution
// ---------------------------------------------------------------------------

/**
 * Compute the optimal LQR feedback gain from the DARE solution P.
 *
 *   K = (R + B^T P B)^{-1} (B^T P A + N^T)
 *
 * Also computes the closed-loop eigenvalues of (A - BK).
 *
 * @param P   DARE solution matrix (nx x nx, row-major Float64Array or Matrix)
 * @param config LQR configuration
 * @returns LQRResult with gain K, DARE solution P, and eigenvalues
 */
export function computeLQRGain(
  P: Matrix | Float64Array,
  config: LQRConfig,
): LQRResult {
  const { nx, nu } = config;
  const m = configToMatrices(config);

  const mP = P instanceof Float64Array ? arrayToMatrix(P, nx, nx) : P;

  const Nt = matTranspose(m.N);
  const Bt = matTranspose(m.B);
  const BtP = matMul(Bt, mP);
  const BtPB = matMul(BtP, m.B);
  const BtPA = matMul(BtP, m.A);

  const inv = matInvert(matAdd(m.R, BtPB));
  const right = matAdd(BtPA, Nt);

  const K = matMul(inv, right);

  // Closed-loop matrix: A - B K
  const Acl = matSub(m.A, matMul(m.B, K));
  const eigenvalues = approximateEigenvalues(Acl);

  return {
    K: new Float64Array(K.data),
    P: mP instanceof Float64Array ? mP : new Float64Array(mP.data),
    eigenvalues,
  };
}

// ---------------------------------------------------------------------------
// Full discrete LQR solver
// ---------------------------------------------------------------------------

/**
 * Solve the full infinite-horizon discrete LQR problem.
 *
 * Given: x_{t+1} = A x_t + B u_t
 * Minimise: sum_{t=0}^{inf} x^T Q x + u^T R u + 2 x^T N u
 *
 * Returns the optimal gain K, DARE solution P, and closed-loop eigenvalues.
 */
export function discreteLQR(config: LQRConfig): LQRResult {
  return solveDARE(config);
}

// ---------------------------------------------------------------------------
// LQR closed-loop simulation
// ---------------------------------------------------------------------------

/**
 * Simulate the closed-loop LQR system for a given number of steps.
 *
 *   u_t = -K x_t
 *   x_{t+1} = (A - B K) x_t
 *
 * @param A     State transition matrix (nx x nx, row-major)
 * @param B     Input matrix (nx x nu, row-major)
 * @param K     Feedback gain matrix (nu x nx, row-major)
 * @param x0    Initial state (nx)
 * @param steps Number of simulation steps
 * @param nx    State dimension
 * @param nu    Control dimension
 * @returns Array of state vectors [x_0, x_1, ..., x_steps]
 */
export function simulateLQR(
  A: Float64Array,
  B: Float64Array,
  K: Float64Array,
  x0: Float64Array,
  steps: number,
  nx: number,
  nu: number,
): Float64Array[] {
  const mA = arrayToMatrix(A, nx, nx);
  const mB = arrayToMatrix(B, nx, nu);
  const mK = arrayToMatrix(K, nu, nx);

  // Closed-loop matrix: Acl = A - B*K
  const Acl = matSub(mA, matMul(mB, mK));

  const trajectory: Float64Array[] = [new Float64Array(x0)];
  let x = new Float64Array(x0);

  for (let t = 0; t < steps; t++) {
    const xNext = matVecMul(Acl, x);
    trajectory.push(new Float64Array(xNext));
    x = xNext as Float64Array<ArrayBuffer>;
  }

  return trajectory;
}
