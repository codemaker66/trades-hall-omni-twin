// ---------------------------------------------------------------------------
// OC-5: Model-Based RL — Linear Dynamics Fitting & MPC Safety Shield
// ---------------------------------------------------------------------------

import {
  createMatrix,
  matGet,
  matInvert,
  matMul,
  matSet,
  matTranspose,
  matVecMul,
} from '../types.js';

// ---------------------------------------------------------------------------
// Fit Linear Dynamics from Data
// ---------------------------------------------------------------------------

/**
 * Fit a linear dynamics model x' = Ax + Bu from trajectory data.
 *
 * Given N transition samples (x_i, u_i, x'_i), this constructs the
 * least-squares problem:
 *
 *   X = [x_1 u_1; x_2 u_2; ...; x_N u_N]  (N x (nx + nu))
 *   Y = [x'_1; x'_2; ...; x'_N]            (N x nx)
 *
 * and solves Y = X * Theta^T where Theta = [A | B] via the normal equations:
 *
 *   Theta^T = (X^T X)^{-1} X^T Y
 *
 * Returns A (nx x nx, row-major) and B (nx x nu, row-major).
 *
 * @param states      State vectors x_i (N samples).
 * @param actions     Action vectors u_i (N samples).
 * @param nextStates  Next-state vectors x'_i (N samples).
 * @param nx          State dimension.
 * @param nu          Control/action dimension.
 * @returns           Fitted {A, B} matrices as flat Float64Arrays.
 */
export function fitLinearDynamics(
  states: Float64Array[],
  actions: Float64Array[],
  nextStates: Float64Array[],
  nx: number,
  nu: number,
): { A: Float64Array; B: Float64Array } {
  const N = states.length;
  const d = nx + nu;

  // Build X (N x d) — each row is [x_i | u_i]
  const X = createMatrix(N, d);
  for (let i = 0; i < N; i++) {
    const xi = states[i]!;
    const ui = actions[i]!;
    for (let j = 0; j < nx; j++) {
      matSet(X, i, j, xi[j]!);
    }
    for (let j = 0; j < nu; j++) {
      matSet(X, i, nx + j, ui[j]!);
    }
  }

  // Build Y (N x nx) — each row is x'_i
  const Y = createMatrix(N, nx);
  for (let i = 0; i < N; i++) {
    const yi = nextStates[i]!;
    for (let j = 0; j < nx; j++) {
      matSet(Y, i, j, yi[j]!);
    }
  }

  // Normal equations: Theta^T = (X^T X)^{-1} X^T Y
  const XT = matTranspose(X);
  const XTX = matMul(XT, X);

  // Add small regularization for numerical stability
  for (let i = 0; i < d; i++) {
    matSet(XTX, i, i, matGet(XTX, i, i) + 1e-8);
  }

  const XTXinv = matInvert(XTX);
  const XTY = matMul(XT, Y);
  const ThetaT = matMul(XTXinv, XTY); // (d x nx)

  // Extract A (nx x nx) and B (nx x nu) from Theta^T
  // Theta^T is (d x nx), so Theta is (nx x d) = [A | B]
  // Row i of Theta^T corresponds to column i of Theta
  // Theta^T[j][k] = Theta[k][j]
  // A[k][j] = ThetaT[j][k] for j < nx
  // B[k][j-nx] = ThetaT[j][k] for j >= nx

  const A = new Float64Array(nx * nx);
  const B = new Float64Array(nx * nu);

  for (let k = 0; k < nx; k++) {
    for (let j = 0; j < nx; j++) {
      A[k * nx + j] = matGet(ThetaT, j, k);
    }
    for (let j = 0; j < nu; j++) {
      B[k * nu + j] = matGet(ThetaT, nx + j, k);
    }
  }

  return { A, B };
}

// ---------------------------------------------------------------------------
// MPC Safety Shield (One-Step Look-Ahead)
// ---------------------------------------------------------------------------

/**
 * One-step MPC safety shield: if the nominal action leads to a state
 * that violates box constraints, project the action to keep the
 * predicted next state within bounds.
 *
 * Predicted next state: x_next = A * x + B * u
 *
 * For each state dimension i, if x_next[i] violates [xMin[i], xMax[i]],
 * the action is adjusted along the B column to bring x_next[i] back
 * to the nearest boundary.
 *
 * @param nominalAction  Proposed action from the RL policy (nu).
 * @param model          Linear dynamics model {A, B, nx, nu}.
 * @param x              Current state (nx).
 * @param xMin           Lower state bounds (nx).
 * @param xMax           Upper state bounds (nx).
 * @returns              Safe action that respects one-step state constraints.
 */
export function mpcSafetyShield(
  nominalAction: Float64Array,
  model: { A: Float64Array; B: Float64Array; nx: number; nu: number },
  x: Float64Array,
  xMin: Float64Array,
  xMax: Float64Array,
): Float64Array {
  const { A, B, nx, nu } = model;

  // Build matrices for matrix-vector operations
  const Amat = { data: new Float64Array(A), rows: nx, cols: nx };
  const Bmat = { data: new Float64Array(B), rows: nx, cols: nu };

  // Predict next state: x_next = A * x + B * u
  const Ax = matVecMul(Amat, x);
  const Bu = matVecMul(Bmat, nominalAction);

  const xNext = new Float64Array(nx);
  for (let i = 0; i < nx; i++) {
    xNext[i] = Ax[i]! + Bu[i]!;
  }

  // Check if the prediction is within bounds
  let feasible = true;
  for (let i = 0; i < nx; i++) {
    if (xNext[i]! < xMin[i]! || xNext[i]! > xMax[i]!) {
      feasible = false;
      break;
    }
  }

  if (feasible) {
    return new Float64Array(nominalAction);
  }

  // Project: adjust action to bring each violated dimension to its nearest bound.
  // For each violated state dimension i, compute the correction needed
  // and distribute it proportionally across action dimensions via B[i,:].
  const safeAction = new Float64Array(nominalAction);

  for (let i = 0; i < nx; i++) {
    let target: number | null = null;

    if (xNext[i]! < xMin[i]!) {
      target = xMin[i]!;
    } else if (xNext[i]! > xMax[i]!) {
      target = xMax[i]!;
    }

    if (target === null) continue;

    // We need: A_i * x + B_i * u_safe = target
    // Correction in predicted state: delta = target - xNext[i]
    const delta = target - xNext[i]!;

    // Find the B row for this state dimension and compute the projection
    // u_safe = u + delta * B_i^T / ||B_i||^2
    let bRowNormSq = 0;
    for (let j = 0; j < nu; j++) {
      const bVal = matGet(Bmat, i, j);
      bRowNormSq += bVal * bVal;
    }

    if (bRowNormSq < 1e-15) continue; // this state dim is not controllable

    const scale = delta / bRowNormSq;
    for (let j = 0; j < nu; j++) {
      safeAction[j] = safeAction[j]! + scale * matGet(Bmat, i, j);
    }
  }

  return safeAction;
}
