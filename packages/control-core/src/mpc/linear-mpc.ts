// ---------------------------------------------------------------------------
// OC-2: Linear Model Predictive Control
// ---------------------------------------------------------------------------
//
// Condensed QP formulation: eliminate predicted states via x_k = A^k x0 + ...,
// optimise over the stacked control vector U = [u_0, ..., u_{N-1}].
//
// QP solver: ADMM (Alternating Direction Method of Multipliers) for
// constrained problems, direct Cholesky solve for unconstrained.
// ---------------------------------------------------------------------------

import type { Matrix, MPCConfig, MPCResult } from '../types.js';
import {
  arrayToMatrix,
  createMatrix,
  matAdd,
  matGet,
  matMul,
  matScale,
  matSet,
  matTranspose,
  matVecMul,
  vecAdd,
  vecClone,
  vecNorm,
  vecScale,
  vecSub,
  matCholesky,
  matSolveLower,
  matSolveUpper,
  matIdentity,
} from '../types.js';

// ---------------------------------------------------------------------------
// Condensed QP construction
// ---------------------------------------------------------------------------

/**
 * Build the condensed QP for a linear MPC problem.
 *
 * The QP has the form:
 *   min  0.5 U^T H U + f^T U
 *   s.t. A_ineq U <= b_ineq
 *
 * where U = [u_0; u_1; ...; u_{N-1}] is the stacked control vector.
 *
 * State predictions are eliminated:
 *   x_k = A^k x0 + sum_{j=0}^{k-1} A^{k-1-j} B u_j
 *
 * @param config  MPC configuration
 * @param x0      Current state (nx)
 * @returns Condensed QP data: { H, f, A_ineq, b_ineq }
 */
export function buildCondensedQP(
  config: MPCConfig,
  x0: Float64Array,
): { H: Matrix; f: Float64Array; A_ineq: Matrix; b_ineq: Float64Array } {
  const { nx, nu, horizon: N } = config;
  const nU = nu * N; // total decision variables

  const A = arrayToMatrix(config.A, nx, nx);
  const B = arrayToMatrix(config.B, nx, nu);
  const Q = arrayToMatrix(config.Q, nx, nx);
  const R = arrayToMatrix(config.R, nu, nu);
  const Qf = config.Qf ? arrayToMatrix(config.Qf, nx, nx) : Q;

  // -----------------------------------------------------------------------
  // Build power matrices A^k  (k = 0 .. N)
  // -----------------------------------------------------------------------
  const Apow: Matrix[] = new Array<Matrix>(N + 1);
  Apow[0] = matIdentity(nx);
  for (let k = 1; k <= N; k++) {
    Apow[k] = matMul(Apow[k - 1]!, A);
  }

  // -----------------------------------------------------------------------
  // Build S matrix (state prediction from controls):
  //   x_k = Apow[k] x0 + S_k U     where S is (N*nx) x (N*nu)
  // S_{k,j} = A^{k-1-j} B   for j < k,  0 otherwise
  // -----------------------------------------------------------------------
  const S = createMatrix(N * nx, nU);
  for (let k = 1; k <= N; k++) {
    for (let j = 0; j < k; j++) {
      const Ab = matMul(Apow[k - 1 - j]!, B);
      for (let r = 0; r < nx; r++) {
        for (let c = 0; c < nu; c++) {
          matSet(S, (k - 1) * nx + r, j * nu + c, matGet(Ab, r, c));
        }
      }
    }
  }

  // -----------------------------------------------------------------------
  // Free (uncontrolled) state predictions: xfree_k = A^k x0
  // Stacked as vector of length N*nx
  // -----------------------------------------------------------------------
  const xFree = new Float64Array(N * nx);
  for (let k = 1; k <= N; k++) {
    const xk = matVecMul(Apow[k]!, x0);
    for (let i = 0; i < nx; i++) {
      xFree[(k - 1) * nx + i] = xk[i]!;
    }
  }

  // -----------------------------------------------------------------------
  // Build block-diagonal cost: Qbar = blkdiag(Q, Q, ..., Q, Qf)
  //   and Rbar = blkdiag(R, R, ..., R)
  // -----------------------------------------------------------------------
  const Qbar = createMatrix(N * nx, N * nx);
  for (let k = 0; k < N; k++) {
    const Qk = k === N - 1 ? Qf : Q;
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < nx; j++) {
        matSet(Qbar, k * nx + i, k * nx + j, matGet(Qk, i, j));
      }
    }
  }

  const Rbar = createMatrix(nU, nU);
  for (let k = 0; k < N; k++) {
    for (let i = 0; i < nu; i++) {
      for (let j = 0; j < nu; j++) {
        matSet(Rbar, k * nu + i, k * nu + j, matGet(R, i, j));
      }
    }
  }

  // -----------------------------------------------------------------------
  // H = S^T Qbar S + Rbar
  // f = S^T Qbar xFree
  // -----------------------------------------------------------------------
  const St = matTranspose(S);
  const StQ = matMul(St, Qbar);
  const H = matAdd(matMul(StQ, S), Rbar);
  const f = matVecMul(StQ, xFree);

  // -----------------------------------------------------------------------
  // Build inequality constraints: A_ineq U <= b_ineq
  // We stack: uMin/uMax, xMin/xMax (via S), duMax constraints
  // -----------------------------------------------------------------------
  const constraintRows: { row: Float64Array; rhs: number }[] = [];

  // Control bounds: uMin <= u_k <= uMax
  if (config.uMin || config.uMax) {
    for (let k = 0; k < N; k++) {
      for (let i = 0; i < nu; i++) {
        if (config.uMax) {
          // u_{k,i} <= uMax_i  =>  e_{k*nu+i}^T U <= uMax_i
          const row = new Float64Array(nU);
          row[k * nu + i] = 1;
          constraintRows.push({ row, rhs: config.uMax[i]! });
        }
        if (config.uMin) {
          // u_{k,i} >= uMin_i  =>  -e_{k*nu+i}^T U <= -uMin_i
          const row = new Float64Array(nU);
          row[k * nu + i] = -1;
          constraintRows.push({ row, rhs: -config.uMin[i]! });
        }
      }
    }
  }

  // State bounds: xMin <= x_k <= xMax  =>  xMin <= A^k x0 + S_k U <= xMax
  if (config.xMin || config.xMax) {
    for (let k = 0; k < N; k++) {
      for (let i = 0; i < nx; i++) {
        // S row for state k, dimension i
        const sRow = new Float64Array(nU);
        for (let j = 0; j < nU; j++) {
          sRow[j] = matGet(S, k * nx + i, j);
        }
        const xFreeKi = xFree[k * nx + i]!;

        if (config.xMax) {
          // S_k U <= xMax_i - xFree_k_i
          constraintRows.push({
            row: new Float64Array(sRow),
            rhs: config.xMax[i]! - xFreeKi,
          });
        }
        if (config.xMin) {
          // -S_k U <= -(xMin_i - xFree_k_i) = xFree_k_i - xMin_i
          const negRow = new Float64Array(nU);
          for (let j = 0; j < nU; j++) {
            negRow[j] = -sRow[j]!;
          }
          constraintRows.push({
            row: negRow,
            rhs: xFreeKi - config.xMin[i]!,
          });
        }
      }
    }
  }

  // Rate constraints: |u_k - u_{k-1}| <= duMax
  if (config.duMax) {
    for (let k = 0; k < N; k++) {
      for (let i = 0; i < nu; i++) {
        // u_k - u_{k-1} <= duMax  and  u_{k-1} - u_k <= duMax
        const rowPos = new Float64Array(nU);
        const rowNeg = new Float64Array(nU);
        rowPos[k * nu + i] = 1;
        rowNeg[k * nu + i] = -1;
        if (k > 0) {
          rowPos[(k - 1) * nu + i] = -1;
          rowNeg[(k - 1) * nu + i] = 1;
        }
        constraintRows.push({ row: rowPos, rhs: config.duMax[i]! });
        constraintRows.push({ row: rowNeg, rhs: config.duMax[i]! });
      }
    }
  }

  // Pack constraints into matrix form
  const nC = constraintRows.length;
  const A_ineq = createMatrix(nC > 0 ? nC : 1, nU);
  const b_ineq = new Float64Array(nC > 0 ? nC : 1);

  for (let r = 0; r < nC; r++) {
    const cr = constraintRows[r]!;
    for (let c = 0; c < nU; c++) {
      matSet(A_ineq, r, c, cr.row[c]!);
    }
    b_ineq[r] = cr.rhs;
  }

  return { H, f, A_ineq: nC > 0 ? A_ineq : createMatrix(0, nU), b_ineq: nC > 0 ? b_ineq : new Float64Array(0) };
}

// ---------------------------------------------------------------------------
// QP solver via ADMM
// ---------------------------------------------------------------------------

/**
 * Solve a convex QP:
 *   min  0.5 x^T H x + f^T x
 *   s.t. A_ineq x <= b_ineq
 *
 * For unconstrained (A_ineq = null): solve H x = -f directly via Cholesky.
 * For constrained: ADMM splitting with projection.
 *
 * @param H        Hessian (positive definite)
 * @param f        Linear cost vector
 * @param A_ineq   Inequality constraint matrix (or null)
 * @param b_ineq   Inequality constraint RHS (or null)
 * @param maxIter  Maximum ADMM iterations (default 500)
 * @returns Solution vector, convergence flag, iteration count
 */
export function solveQP(
  H: Matrix,
  f: Float64Array,
  A_ineq: Matrix | null,
  b_ineq: Float64Array | null,
  maxIter: number = 500,
): { x: Float64Array; converged: boolean; iterations: number } {
  const n = H.rows;

  // Unconstrained case: H x = -f
  const hasConstraints =
    A_ineq !== null &&
    b_ineq !== null &&
    A_ineq.rows > 0;

  if (!hasConstraints) {
    const negF = vecScale(f, -1);
    const L = matCholesky(H);
    const Lt = matTranspose(L);
    const y = matSolveLower(L, negF);
    const x = matSolveUpper(Lt, y);
    return { x, converged: true, iterations: 1 };
  }

  // ADMM for constrained QP
  // Reformulate: min 0.5 x^T H x + f^T x  s.t.  Gx + s = h, s >= 0
  //   where G = A_ineq, h = b_ineq, s = slack

  const m = A_ineq!.rows;
  const G = A_ineq!;
  const h = b_ineq!;

  // ADMM parameters
  const rho = 1.0;
  const absTol = 1e-8;
  const relTol = 1e-6;

  // Pre-factor: (H + rho G^T G)
  const Gt = matTranspose(G);
  const GtG = matMul(Gt, G);
  const KKT = matAdd(H, matScale(GtG, rho));

  let L: Matrix;
  try {
    L = matCholesky(KKT);
  } catch {
    // If H + rho G^T G is not PD, add regularization
    const reg = matIdentity(n);
    const KKTreg = matAdd(KKT, matScale(reg, 1e-6));
    L = matCholesky(KKTreg);
  }
  const Lt = matTranspose(L);

  // ADMM variables
  let x = new Float64Array(n);
  let z = new Float64Array(m); // slack (Gx - h mapped)
  let u = new Float64Array(m); // scaled dual

  let converged = false;
  let iter = 0;

  for (iter = 0; iter < maxIter; iter++) {
    // x-update: x = (H + rho G^T G)^{-1} (-f + rho G^T (h - z + u))
    const tmp = vecAdd(vecSub(h, z), u);
    const rhs = vecAdd(vecScale(f, -1), vecScale(matVecMul(Gt, tmp), rho));
    const yy = matSolveLower(L, rhs);
    x = matSolveUpper(Lt, yy) as Float64Array<ArrayBuffer>;

    // z-update: z = proj_{>= 0}(Gx - h + u)
    const Gx = matVecMul(G, x);
    const zOld = vecClone(z);
    for (let i = 0; i < m; i++) {
      // We want Gx <= h  <=>  Gx - h <= 0  <=>  slack s = h - Gx >= 0
      // ADMM form: z = h - Gx projected to >= 0 ... but let's be precise.
      // With variable z representing slack s = h - Gx:
      //   z_new = max(0, h_i - Gx_i + u_i)
      z[i] = Math.max(0, h[i]! - Gx[i]! + u[i]!);
    }

    // u-update (dual)
    for (let i = 0; i < m; i++) {
      u[i] = u[i]! + (h[i]! - Gx[i]! - z[i]!);
    }

    // Convergence check
    // Primal residual: r = h - Gx - z
    const primalRes = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      primalRes[i] = h[i]! - Gx[i]! - z[i]!;
    }
    const priNorm = vecNorm(primalRes);

    // Dual residual: rho * G^T (z - zOld)
    const dz = vecSub(z, zOld);
    const dualRes = vecScale(matVecMul(Gt, dz), rho);
    const dualNorm = vecNorm(dualRes);

    const epsPri = absTol * Math.sqrt(m) + relTol * Math.max(vecNorm(Gx), vecNorm(z), vecNorm(h));
    const epsDual = absTol * Math.sqrt(n) + relTol * vecNorm(matVecMul(Gt, u)) * rho;

    if (priNorm < epsPri && dualNorm < epsDual) {
      converged = true;
      break;
    }
  }

  return { x, converged, iterations: iter + 1 };
}

// ---------------------------------------------------------------------------
// Linear MPC solver
// ---------------------------------------------------------------------------

/**
 * Solve a linear MPC problem.
 *
 * Given the discrete-time linear system  x_{k+1} = A x_k + B u_k  and
 * the quadratic cost  sum_{k=0}^{N-1}[ x_k^T Q x_k + u_k^T R u_k ] + x_N^T Qf x_N,
 * find the optimal control sequence U* = [u_0*, ..., u_{N-1}*].
 *
 * Optionally enforces:
 * - Control bounds: uMin <= u_k <= uMax
 * - State bounds:   xMin <= x_k <= xMax
 * - Rate bounds:    |u_k - u_{k-1}| <= duMax
 *
 * @param config  MPC configuration
 * @param x0      Current state (nx)
 * @param uPrev   Previous control input for rate constraints (nu, optional)
 * @returns MPCResult with optimal control, predictions, cost, etc.
 */
export function solveLinearMPC(
  config: MPCConfig,
  x0: Float64Array,
  uPrev?: Float64Array,
): MPCResult {
  const { nx, nu, horizon: N } = config;

  // Build condensed QP
  const { H, f, A_ineq, b_ineq } = buildCondensedQP(config, x0);

  // If uPrev is provided and duMax constraints exist, adjust the first
  // rate constraint RHS to account for the previous control input
  let bAdj = b_ineq;
  if (uPrev && config.duMax && A_ineq.rows > 0) {
    bAdj = new Float64Array(b_ineq);
    // The rate constraints for k=0 reference u_{-1} = uPrev.
    // In buildCondensedQP, for k=0 the row has only u_0 terms (no u_{-1} term).
    // We need to find those rows and adjust the RHS.
    // Rate constraints are appended after uMin/uMax and xMin/xMax.
    // Count preceding constraints to find rate constraint offset.
    let offset = 0;
    if (config.uMin || config.uMax) {
      offset += N * nu * ((config.uMin ? 1 : 0) + (config.uMax ? 1 : 0));
    }
    if (config.xMin || config.xMax) {
      offset += N * nx * ((config.xMin ? 1 : 0) + (config.xMax ? 1 : 0));
    }
    // For k=0, the rate constraints are at rows [offset, offset + 2*nu)
    // row[offset + 2*i]:     u_0_i <= duMax_i    =>  adjust: + uPrev_i
    // row[offset + 2*i + 1]: -u_0_i <= duMax_i   =>  adjust: - uPrev_i
    for (let i = 0; i < nu; i++) {
      const idx0 = offset + 2 * i;
      const idx1 = offset + 2 * i + 1;
      if (idx0 < bAdj.length) {
        bAdj[idx0] = bAdj[idx0]! + uPrev[i]!;
      }
      if (idx1 < bAdj.length) {
        bAdj[idx1] = bAdj[idx1]! - uPrev[i]!;
      }
    }
  }

  // Solve QP
  const hasConstraints = A_ineq.rows > 0;
  const qpResult = solveQP(
    H,
    f,
    hasConstraints ? A_ineq : null,
    hasConstraints ? bAdj : null,
  );

  const U = qpResult.x;

  // Extract control sequence
  const uSequence: Float64Array[] = [];
  for (let k = 0; k < N; k++) {
    const uk = new Float64Array(nu);
    for (let i = 0; i < nu; i++) {
      uk[i] = U[k * nu + i]!;
    }
    uSequence.push(uk);
  }

  // Predict state trajectory
  const Amat = arrayToMatrix(config.A, nx, nx);
  const Bmat = arrayToMatrix(config.B, nx, nu);
  const xPredicted: Float64Array[] = [vecClone(x0)];
  let xCur = vecClone(x0);
  for (let k = 0; k < N; k++) {
    const Ax = matVecMul(Amat, xCur);
    const Bu = matVecMul(Bmat, uSequence[k]!);
    xCur = vecAdd(Ax, Bu);
    xPredicted.push(vecClone(xCur));
  }

  // Compute cost
  const Qmat = arrayToMatrix(config.Q, nx, nx);
  const Rmat = arrayToMatrix(config.R, nu, nu);
  const Qfmat = config.Qf ? arrayToMatrix(config.Qf, nx, nx) : Qmat;
  let cost = 0;
  for (let k = 0; k < N; k++) {
    const xk = xPredicted[k]!;
    const uk = uSequence[k]!;
    // x^T Q x
    const Qx = matVecMul(Qmat, xk);
    for (let i = 0; i < nx; i++) {
      cost += xk[i]! * Qx[i]!;
    }
    // u^T R u
    const Ru = matVecMul(Rmat, uk);
    for (let i = 0; i < nu; i++) {
      cost += uk[i]! * Ru[i]!;
    }
  }
  // Terminal cost
  const xN = xPredicted[N]!;
  const QfxN = matVecMul(Qfmat, xN);
  for (let i = 0; i < nx; i++) {
    cost += xN[i]! * QfxN[i]!;
  }
  cost *= 0.5;

  // Determine status
  let status: MPCResult['status'] = 'optimal';
  if (!qpResult.converged) {
    status = 'max_iter';
  }

  return {
    uOptimal: vecClone(uSequence[0]!),
    uSequence,
    xPredicted,
    cost,
    iterations: qpResult.iterations,
    status,
  };
}
