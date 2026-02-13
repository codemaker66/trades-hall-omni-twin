// ---------------------------------------------------------------------------
// OC-2: Nonlinear Model Predictive Control (via SQP)
// ---------------------------------------------------------------------------
//
// Sequential Quadratic Programming (SQP) for nonlinear MPC:
//   At each SQP iteration, linearise dynamics and quadratise cost around
//   the current trajectory guess, then solve the resulting QP for a
//   correction step. Repeat until convergence.
// ---------------------------------------------------------------------------

import type { Matrix, NonlinearMPCConfig, MPCResult } from '../types.js';
import {
  arrayToMatrix,
  createMatrix,
  matGet,
  matSet,
  vecAdd,
  vecClone,
  vecNorm,
  vecScale,
  vecSub,
} from '../types.js';

import { solveQP } from './linear-mpc.js';

// ---------------------------------------------------------------------------
// Numerical linearisation helpers
// ---------------------------------------------------------------------------

/**
 * Numerically compute the Jacobian df/dx of the dynamics f(x, u)
 * at the point (x0, u0). Returns a flat Float64Array of shape (nx x nx).
 */
function dynamicsJacobianX(
  dynamicsFn: (x: Float64Array, u: Float64Array) => Float64Array,
  x0: Float64Array,
  u0: Float64Array,
  nx: number,
  eps: number = 1e-6,
): Float64Array {
  const J = new Float64Array(nx * nx);
  for (let j = 0; j < nx; j++) {
    const xPlus = vecClone(x0);
    const xMinus = vecClone(x0);
    xPlus[j] = xPlus[j]! + eps;
    xMinus[j] = xMinus[j]! - eps;

    const fPlus = dynamicsFn(xPlus, u0);
    const fMinus = dynamicsFn(xMinus, u0);

    for (let i = 0; i < nx; i++) {
      J[i * nx + j] = (fPlus[i]! - fMinus[i]!) / (2 * eps);
    }
  }
  return J;
}

/**
 * Numerically compute the Jacobian df/du of the dynamics f(x, u)
 * at the point (x0, u0). Returns a flat Float64Array of shape (nx x nu).
 */
function dynamicsJacobianU(
  dynamicsFn: (x: Float64Array, u: Float64Array) => Float64Array,
  x0: Float64Array,
  u0: Float64Array,
  nx: number,
  nu: number,
  eps: number = 1e-6,
): Float64Array {
  const J = new Float64Array(nx * nu);
  for (let j = 0; j < nu; j++) {
    const uPlus = vecClone(u0);
    const uMinus = vecClone(u0);
    uPlus[j] = uPlus[j]! + eps;
    uMinus[j] = uMinus[j]! - eps;

    const fPlus = dynamicsFn(x0, uPlus);
    const fMinus = dynamicsFn(x0, uMinus);

    for (let i = 0; i < nx; i++) {
      J[i * nu + j] = (fPlus[i]! - fMinus[i]!) / (2 * eps);
    }
  }
  return J;
}

/**
 * Numerically compute the gradient of a scalar function f(x) at x0.
 */
function numericalGradient(
  fn: (x: Float64Array) => number,
  x0: Float64Array,
  eps: number = 1e-6,
): Float64Array {
  const n = x0.length;
  const g = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    const xPlus = vecClone(x0);
    const xMinus = vecClone(x0);
    xPlus[j] = xPlus[j]! + eps;
    xMinus[j] = xMinus[j]! - eps;
    g[j] = (fn(xPlus) - fn(xMinus)) / (2 * eps);
  }
  return g;
}

/**
 * Numerically compute the Hessian of a scalar function f(x) at x0.
 * Returns flat Float64Array of shape (n x n).
 */
function numericalHessian(
  fn: (x: Float64Array) => number,
  x0: Float64Array,
  eps: number = 1e-5,
): Float64Array {
  const n = x0.length;
  const H = new Float64Array(n * n);
  const f0 = fn(x0);

  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const xpp = vecClone(x0);
      const xpm = vecClone(x0);
      const xmp = vecClone(x0);
      const xmm = vecClone(x0);

      xpp[i] = xpp[i]! + eps;
      xpp[j] = xpp[j]! + eps;
      xpm[i] = xpm[i]! + eps;
      xpm[j] = xpm[j]! - eps;
      xmp[i] = xmp[i]! - eps;
      xmp[j] = xmp[j]! + eps;
      xmm[i] = xmm[i]! - eps;
      xmm[j] = xmm[j]! - eps;

      const hij = (fn(xpp) - fn(xpm) - fn(xmp) + fn(xmm)) / (4 * eps * eps);
      H[i * n + j] = hij;
      H[j * n + i] = hij;
    }
  }
  return H;
}

// ---------------------------------------------------------------------------
// SQP step
// ---------------------------------------------------------------------------

/**
 * Perform one SQP iteration: given the current trajectory guess
 * (xTraj, uTraj), linearise the dynamics and quadratise the cost,
 * then solve the resulting QP for corrections (dx, du).
 *
 * @param config  Nonlinear MPC configuration
 * @param xTraj   Current state trajectory guess [x_0, ..., x_N] (N+1 entries)
 * @param uTraj   Current control trajectory guess [u_0, ..., u_{N-1}] (N entries)
 * @returns Corrections { dx, du } to apply to the trajectory
 */
export function sqpStep(
  config: NonlinearMPCConfig,
  xTraj: Float64Array[],
  uTraj: Float64Array[],
): { dx: Float64Array[]; du: Float64Array[] } {
  const { nx, nu, horizon: N, dynamicsFn, costStageFn, costTerminalFn } = config;

  const nU = nu * N;

  // Linearise dynamics at each stage: x_{k+1} ~ A_k dx_k + B_k du_k + c_k
  const Ak: Float64Array[] = [];
  const Bk: Float64Array[] = [];
  const ck: Float64Array[] = []; // defect: f(xk, uk) - x_{k+1}

  for (let k = 0; k < N; k++) {
    const xk = xTraj[k]!;
    const uk = uTraj[k]!;

    Ak.push(dynamicsJacobianX(dynamicsFn, xk, uk, nx));
    Bk.push(dynamicsJacobianU(dynamicsFn, xk, uk, nx, nu));

    // Defect
    const fk = dynamicsFn(xk, uk);
    const xkp1 = xTraj[k + 1]!;
    const defect = vecSub(fk, xkp1);
    ck.push(defect);
  }

  // Quadratise stage costs: l(x,u) ~ 0.5 [dx; du]^T H_k [dx; du] + g_k^T [dx; du]
  const Qk: Float64Array[] = []; // Hessian w.r.t. x (nx x nx)
  const Rk_arr: Float64Array[] = []; // Hessian w.r.t. u (nu x nu)
  const Sk: Float64Array[] = []; // Cross-Hessian (nx x nu)
  const qk: Float64Array[] = []; // gradient w.r.t. x
  const rk: Float64Array[] = []; // gradient w.r.t. u

  for (let k = 0; k < N; k++) {
    const xk = xTraj[k]!;
    const uk = uTraj[k]!;

    // Gradient w.r.t. x
    qk.push(
      numericalGradient((x: Float64Array) => costStageFn(x, uk), xk),
    );

    // Gradient w.r.t. u
    rk.push(
      numericalGradient((u: Float64Array) => costStageFn(xk, u), uk),
    );

    // Hessian w.r.t. x
    const Hxx = numericalHessian(
      (x: Float64Array) => costStageFn(x, uk),
      xk,
    );
    // Regularise to ensure positive definiteness
    for (let i = 0; i < nx; i++) {
      Hxx[i * nx + i] = Hxx[i * nx + i]! + 1e-6;
    }
    Qk.push(Hxx);

    // Hessian w.r.t. u
    const Huu = numericalHessian(
      (u: Float64Array) => costStageFn(xk, u),
      uk,
    );
    for (let i = 0; i < nu; i++) {
      Huu[i * nu + i] = Huu[i * nu + i]! + 1e-6;
    }
    Rk_arr.push(Huu);

    // Cross-Hessian (approximate as zero for simplicity)
    Sk.push(new Float64Array(nx * nu));
  }

  // Terminal cost Hessian and gradient
  const xN = xTraj[N]!;
  const qN = numericalGradient(costTerminalFn, xN);
  const QN = numericalHessian(costTerminalFn, xN);
  for (let i = 0; i < nx; i++) {
    QN[i * nx + i] = QN[i * nx + i]! + 1e-6;
  }

  // -----------------------------------------------------------------------
  // Build condensed QP over du = [du_0; ...; du_{N-1}]
  //
  // Using the sensitivity approach: propagate dx through linearised dynamics
  //   dx_{k+1} = A_k dx_k + B_k du_k + c_k
  // with dx_0 = 0 (current state is fixed).
  //
  // Substitute forward to express dx_k = M_k du + d_k (affine in du)
  // where M_k, d_k accumulate the forward propagation.
  // -----------------------------------------------------------------------

  // Propagation matrices: dx_k = M_k * dU + d_k
  // M_k is (nx x nU), d_k is (nx)
  const Mk: Matrix[] = new Array<Matrix>(N + 1);
  const dk: Float64Array[] = new Array<Float64Array>(N + 1);

  Mk[0] = createMatrix(nx, nU); // all zeros
  dk[0] = new Float64Array(nx); // all zeros

  for (let k = 0; k < N; k++) {
    const Amat = arrayToMatrix(Ak[k]!, nx, nx);
    const Bmat = arrayToMatrix(Bk[k]!, nx, nu);

    // M_{k+1} = A_k M_k + [0 ... B_k ... 0] (B_k at column block k)
    const AMk = createMatrix(nx, nU);
    // A_k * M_k
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < nU; j++) {
        let sum = 0;
        for (let p = 0; p < nx; p++) {
          sum += matGet(Amat, i, p) * matGet(Mk[k]!, p, j);
        }
        matSet(AMk, i, j, sum);
      }
    }
    // Add B_k at column block k
    const Mkp1 = createMatrix(nx, nU);
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < nU; j++) {
        matSet(Mkp1, i, j, matGet(AMk, i, j));
      }
      for (let j = 0; j < nu; j++) {
        const cur = matGet(Mkp1, i, k * nu + j);
        matSet(Mkp1, i, k * nu + j, cur + matGet(Bmat, i, j));
      }
    }
    Mk[k + 1] = Mkp1;

    // d_{k+1} = A_k d_k + c_k
    const dkCur = dk[k]!;
    const dkp1 = new Float64Array(nx);
    for (let i = 0; i < nx; i++) {
      let sum = 0;
      for (let p = 0; p < nx; p++) {
        sum += matGet(Amat, i, p) * dkCur[p]!;
      }
      dkp1[i] = sum + ck[k]![i]!;
    }
    dk[k + 1] = dkp1;
  }

  // Build QP: H_qp dU = -g_qp
  // Cost = sum_k 0.5 dx_k^T Qk dx_k + qk^T dx_k + 0.5 du_k^T Rk du_k + rk^T du_k
  //       + 0.5 dx_N^T QN dx_N + qN^T dx_N
  // Substitute dx_k = M_k dU + d_k

  const H_qp = createMatrix(nU, nU);
  const g_qp = new Float64Array(nU);

  for (let k = 0; k <= N; k++) {
    const Qmat = k < N
      ? arrayToMatrix(Qk[k]!, nx, nx)
      : arrayToMatrix(QN, nx, nx);
    const qVec = k < N ? qk[k]! : qN;
    const MkMat = Mk[k]!;
    const dkVec = dk[k]!;

    // H += M_k^T Q_k M_k
    // g += M_k^T (Q_k d_k + q_k)
    for (let i = 0; i < nU; i++) {
      for (let j = 0; j < nU; j++) {
        let val = 0;
        for (let p = 0; p < nx; p++) {
          for (let q = 0; q < nx; q++) {
            val +=
              matGet(MkMat, p, i) *
              matGet(Qmat, p, q) *
              matGet(MkMat, q, j);
          }
        }
        matSet(H_qp, i, j, matGet(H_qp, i, j) + val);
      }
    }

    // g += M_k^T (Q_k d_k + q_k)
    const Qd = new Float64Array(nx);
    for (let p = 0; p < nx; p++) {
      let sum = 0;
      for (let q = 0; q < nx; q++) {
        sum += matGet(Qmat, p, q) * dkVec[q]!;
      }
      Qd[p] = sum + qVec[p]!;
    }
    for (let i = 0; i < nU; i++) {
      let sum = 0;
      for (let p = 0; p < nx; p++) {
        sum += matGet(MkMat, p, i) * Qd[p]!;
      }
      g_qp[i] = g_qp[i]! + sum;
    }

    // Control cost (only for k < N)
    if (k < N) {
      const Rmat = arrayToMatrix(Rk_arr[k]!, nu, nu);
      const rkVec = rk[k]!;
      for (let i = 0; i < nu; i++) {
        for (let j = 0; j < nu; j++) {
          const idx_i = k * nu + i;
          const idx_j = k * nu + j;
          matSet(
            H_qp,
            idx_i,
            idx_j,
            matGet(H_qp, idx_i, idx_j) + matGet(Rmat, i, j),
          );
        }
        g_qp[k * nu + i] = g_qp[k * nu + i]! + rkVec[i]!;
      }
    }
  }

  // Build inequality constraints from uMin/uMax
  const constraintRows: { row: Float64Array; rhs: number }[] = [];

  if (config.uMin || config.uMax) {
    for (let k = 0; k < N; k++) {
      const uk = uTraj[k]!;
      for (let i = 0; i < nu; i++) {
        if (config.uMax) {
          const row = new Float64Array(nU);
          row[k * nu + i] = 1;
          constraintRows.push({ row, rhs: config.uMax[i]! - uk[i]! });
        }
        if (config.uMin) {
          const row = new Float64Array(nU);
          row[k * nu + i] = -1;
          constraintRows.push({ row, rhs: -(config.uMin[i]! - uk[i]!) });
        }
      }
    }
  }

  // State constraints via M_k
  if (config.xMin || config.xMax) {
    for (let k = 1; k <= N; k++) {
      const xk = xTraj[k]!;
      for (let i = 0; i < nx; i++) {
        const mRow = new Float64Array(nU);
        for (let j = 0; j < nU; j++) {
          mRow[j] = matGet(Mk[k]!, i, j);
        }
        if (config.xMax) {
          constraintRows.push({
            row: new Float64Array(mRow),
            rhs: config.xMax[i]! - xk[i]! - dk[k]![i]!,
          });
        }
        if (config.xMin) {
          const negRow = new Float64Array(nU);
          for (let j = 0; j < nU; j++) {
            negRow[j] = -mRow[j]!;
          }
          constraintRows.push({
            row: negRow,
            rhs: -(config.xMin[i]! - xk[i]! - dk[k]![i]!),
          });
        }
      }
    }
  }

  // Pack constraints
  const nC = constraintRows.length;
  let A_ineq: Matrix | null = null;
  let b_ineq: Float64Array | null = null;

  if (nC > 0) {
    A_ineq = createMatrix(nC, nU);
    b_ineq = new Float64Array(nC);
    for (let r = 0; r < nC; r++) {
      const cr = constraintRows[r]!;
      for (let c = 0; c < nU; c++) {
        matSet(A_ineq, r, c, cr.row[c]!);
      }
      b_ineq[r] = cr.rhs;
    }
  }

  // Solve QP
  const qpResult = solveQP(H_qp, g_qp, A_ineq, b_ineq);
  const dU = qpResult.x;

  // Extract du corrections
  const du: Float64Array[] = [];
  for (let k = 0; k < N; k++) {
    const duk = new Float64Array(nu);
    for (let i = 0; i < nu; i++) {
      duk[i] = dU[k * nu + i]!;
    }
    du.push(duk);
  }

  // Compute dx from M_k dU + d_k
  const dx: Float64Array[] = [];
  for (let k = 0; k <= N; k++) {
    const dxk = new Float64Array(nx);
    for (let i = 0; i < nx; i++) {
      let sum = dk[k]![i]!;
      for (let j = 0; j < nU; j++) {
        sum += matGet(Mk[k]!, i, j) * dU[j]!;
      }
      dxk[i] = sum;
    }
    dx.push(dxk);
  }

  return { dx, du };
}

// ---------------------------------------------------------------------------
// Nonlinear MPC solver
// ---------------------------------------------------------------------------

/**
 * Solve a nonlinear MPC problem using Sequential Quadratic Programming (SQP).
 *
 * Given nonlinear discrete-time dynamics x_{k+1} = f(x_k, u_k) and
 * general stage/terminal costs, find the optimal control sequence.
 *
 * @param config  Nonlinear MPC configuration
 * @param x0      Current state (nx)
 * @returns MPCResult with optimal control, predictions, cost, etc.
 */
export function solveNonlinearMPC(
  config: NonlinearMPCConfig,
  x0: Float64Array,
): MPCResult {
  const { nx, nu, horizon: N, dynamicsFn, costStageFn, costTerminalFn } = config;
  const { maxSQPIterations, tolerance } = config;

  // Initialise trajectory guess: propagate with zero control
  let xTraj: Float64Array[] = [vecClone(x0)];
  let uTraj: Float64Array[] = [];

  for (let k = 0; k < N; k++) {
    const uk = new Float64Array(nu); // zero initial guess
    uTraj.push(uk);
    const xNext = dynamicsFn(xTraj[k]!, uk);
    xTraj.push(vecClone(xNext));
  }

  let converged = false;
  let totalIter = 0;

  for (let sqpIter = 0; sqpIter < maxSQPIterations; sqpIter++) {
    totalIter = sqpIter + 1;

    const { dx, du } = sqpStep(config, xTraj, uTraj);

    // Line search: use a simple backtracking with step size alpha
    let alpha = 1.0;
    const currentCost = evaluateCost(
      costStageFn,
      costTerminalFn,
      xTraj,
      uTraj,
      N,
    );

    let bestCost = Infinity;
    let bestXTraj = xTraj;
    let bestUTraj = uTraj;

    for (let ls = 0; ls < 10; ls++) {
      // Apply correction with step size alpha
      const newUTraj: Float64Array[] = [];
      for (let k = 0; k < N; k++) {
        newUTraj.push(vecAdd(uTraj[k]!, vecScale(du[k]!, alpha)));
      }

      // Clamp controls to bounds
      for (let k = 0; k < N; k++) {
        for (let i = 0; i < nu; i++) {
          if (config.uMin) {
            newUTraj[k]![i] = Math.max(newUTraj[k]![i]!, config.uMin[i]!);
          }
          if (config.uMax) {
            newUTraj[k]![i] = Math.min(newUTraj[k]![i]!, config.uMax[i]!);
          }
        }
      }

      // Forward simulate with new controls
      const newXTraj: Float64Array[] = [vecClone(x0)];
      for (let k = 0; k < N; k++) {
        const xNext = dynamicsFn(newXTraj[k]!, newUTraj[k]!);
        newXTraj.push(vecClone(xNext));
      }

      const newCost = evaluateCost(
        costStageFn,
        costTerminalFn,
        newXTraj,
        newUTraj,
        N,
      );

      if (newCost < bestCost) {
        bestCost = newCost;
        bestXTraj = newXTraj;
        bestUTraj = newUTraj;
      }

      if (newCost < currentCost) {
        break;
      }

      alpha *= 0.5;
    }

    xTraj = bestXTraj;
    uTraj = bestUTraj;

    // Check convergence: norm of correction
    let corrNorm = 0;
    for (let k = 0; k < N; k++) {
      corrNorm += vecNorm(du[k]!);
    }
    corrNorm /= N;

    if (corrNorm < tolerance) {
      converged = true;
      break;
    }
  }

  // Compute final cost
  const cost = evaluateCost(costStageFn, costTerminalFn, xTraj, uTraj, N);

  return {
    uOptimal: vecClone(uTraj[0]!),
    uSequence: uTraj,
    xPredicted: xTraj,
    cost,
    iterations: totalIter,
    status: converged ? 'optimal' : 'max_iter',
  };
}

// ---------------------------------------------------------------------------
// Helper: evaluate total cost
// ---------------------------------------------------------------------------

function evaluateCost(
  costStageFn: (x: Float64Array, u: Float64Array) => number,
  costTerminalFn: (x: Float64Array) => number,
  xTraj: Float64Array[],
  uTraj: Float64Array[],
  N: number,
): number {
  let cost = 0;
  for (let k = 0; k < N; k++) {
    cost += costStageFn(xTraj[k]!, uTraj[k]!);
  }
  cost += costTerminalFn(xTraj[N]!);
  return cost;
}
