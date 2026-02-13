// ---------------------------------------------------------------------------
// OC-2: Economic Model Predictive Control
// ---------------------------------------------------------------------------
//
// Economic MPC maximises revenue (or minimises negative revenue) while
// satisfying operational constraints. The demand function is linearised
// around the current operating point, producing a modified quadratic cost
// that includes revenue terms. The resulting QP is solved via the standard
// linear MPC solver.
// ---------------------------------------------------------------------------

import type { EconomicMPCConfig, MPCConfig, MPCResult } from '../types.js';
import {
  arrayToMatrix,
  createMatrix,
  matGet,
  matSet,
  matVecMul,
  vecClone,
} from '../types.js';

import { solveLinearMPC } from './linear-mpc.js';

// ---------------------------------------------------------------------------
// Numerical Jacobian helper
// ---------------------------------------------------------------------------

/**
 * Approximate the Jacobian of f(x) at point x0 via central differences.
 * Returns a flat Float64Array of shape (m x n) in row-major order,
 * where m = dim(f) and n = dim(x).
 */
function numericalJacobian(
  fn: (x: Float64Array) => Float64Array,
  x0: Float64Array,
  eps: number = 1e-6,
): Float64Array {
  const n = x0.length;
  const f0 = fn(x0);
  const m = f0.length;
  const J = new Float64Array(m * n);

  for (let j = 0; j < n; j++) {
    const xPlus = vecClone(x0);
    const xMinus = vecClone(x0);
    xPlus[j] = xPlus[j]! + eps;
    xMinus[j] = xMinus[j]! - eps;

    const fPlus = fn(xPlus);
    const fMinus = fn(xMinus);

    for (let i = 0; i < m; i++) {
      J[i * n + j] = (fPlus[i]! - fMinus[i]!) / (2 * eps);
    }
  }

  return J;
}

// ---------------------------------------------------------------------------
// Economic MPC solver
// ---------------------------------------------------------------------------

/**
 * Solve an economic MPC problem.
 *
 * The objective combines a standard quadratic tracking cost with a revenue
 * term derived from the demand function:
 *
 *   revenue_k = price_k^T * demand(price_k, x_k)
 *
 * The demand function is linearised around the current state to produce a
 * quadratic approximation of the revenue, which is folded into the
 * Q and R matrices of a standard linear MPC formulation.
 *
 * @param config  Economic MPC configuration (extends MPCConfig with demandFn and costFn)
 * @param x0      Current state (nx)
 * @returns MPCResult with optimal control, predictions, cost, etc.
 */
export function solveEconomicMPC(
  config: EconomicMPCConfig,
  x0: Float64Array,
): MPCResult {
  const { nx, nu, horizon, demandFn, costFn } = config;

  // Current operating point: use zero control as nominal
  const u0 = new Float64Array(nu);
  const price0 = new Float64Array(nu);
  // Treat control inputs as price-like variables
  for (let i = 0; i < nu; i++) {
    price0[i] = u0[i]!;
  }

  // Evaluate demand and cost at current operating point
  const demand0 = demandFn(price0, x0);
  const cost0 = costFn(x0, u0);

  // Linearise demand w.r.t. price (control): dDemand/dPrice
  const demandJacPrice = numericalJacobian(
    (p: Float64Array) => demandFn(p, x0),
    price0,
  );
  const nDemand = demand0.length;

  // Linearise demand w.r.t. state: dDemand/dState
  const demandJacState = numericalJacobian(
    (s: Float64Array) => demandFn(price0, s),
    x0,
  );

  // Build modified cost matrices that incorporate the revenue objective.
  //
  // Revenue at stage k (linearised):
  //   rev_k ~ price^T demand0 + price^T dD/dp dp + demand0^T dp + price^T dD/dx dx
  //
  // We want to maximise revenue => minimise -revenue.
  //
  // Modify Q: add penalty on states that reduce demand (via dD/dx)
  // Modify R: add terms from -d(price^T demand)/d(price)
  //
  // Quadratic approximation of negative revenue in control:
  //   -rev ~ -u^T demand0 - u^T (dD/dp) u   (ignoring constants)
  //   => Hessian contribution to R: -(dD/dp + (dD/dp)^T) / 2  (if negative definite)
  //   => Linear contribution to cost: -demand0

  // Modified R: original R + revenue Hessian (we add a positive semi-definite approximation)
  const Rmod = new Float64Array(config.R);
  const dDdp = arrayToMatrix(demandJacPrice, nDemand, nu);

  // For the revenue Hessian, we use: -2 * (dD/dp) symmetrised
  // But we must ensure R stays positive definite, so we only add the
  // positive part (clamp eigenvalues). As a practical approximation,
  // we scale the demand jacobian contribution.
  for (let i = 0; i < nu; i++) {
    for (let j = 0; j < nu; j++) {
      let revHess = 0;
      // Approximate: d^2(rev)/d(u_i)d(u_j) ~ dD_i/dp_j + dD_j/dp_i
      for (let d = 0; d < nDemand; d++) {
        revHess += matGet(dDdp, d, i) * (i === j ? 1 : 0);
        revHess += matGet(dDdp, d, j) * (j === i ? 1 : 0);
      }
      // We subtract because we maximise revenue (minimise -revenue),
      // but keep R positive definite by using absolute value dampening.
      const contribution = -revHess * 0.5;
      if (contribution > 0) {
        Rmod[i * nu + j] = Rmod[i * nu + j]! + contribution;
      }
    }
  }

  // Ensure R diagonal is sufficiently positive
  for (let i = 0; i < nu; i++) {
    const diagVal = Rmod[i * nu + i]!;
    if (diagVal < 1e-4) {
      Rmod[i * nu + i] = 1e-4;
    }
  }

  // Modified Q: add small state-dependent revenue gradient contribution
  const Qmod = new Float64Array(config.Q);
  const dDdx = arrayToMatrix(demandJacState, nDemand, nx);

  // Revenue gradient w.r.t. state: d(rev)/dx ~ price^T dD/dx
  // This provides a linear cost term; to incorporate in quadratic form,
  // add a scaled identity-like contribution
  for (let i = 0; i < nx; i++) {
    let stateRevGrad = 0;
    for (let d = 0; d < nDemand; d++) {
      stateRevGrad += matGet(dDdx, d, i);
    }
    // Add small contribution to Q diagonal (positive to penalise state deviations
    // that reduce demand)
    const contribution = Math.abs(stateRevGrad) * 0.1;
    Qmod[i * nx + i] = Qmod[i * nx + i]! + contribution;
  }

  // Build modified MPC config
  const modConfig: MPCConfig = {
    A: config.A,
    B: config.B,
    Q: Qmod,
    R: Rmod,
    Qf: config.Qf,
    nx,
    nu,
    horizon,
    uMin: config.uMin,
    uMax: config.uMax,
    xMin: config.xMin,
    xMax: config.xMax,
    duMax: config.duMax,
  };

  // Solve modified linear MPC
  const result = solveLinearMPC(modConfig, x0);

  // Adjust cost to reflect economic objective (revenue minus operational cost)
  let economicCost = 0;
  for (let k = 0; k < horizon; k++) {
    const xk = result.xPredicted[k]!;
    const uk = result.uSequence[k]!;

    // Operational cost
    economicCost += costFn(xk, uk);

    // Subtract revenue (since we want to report net cost = opCost - revenue)
    const demand = demandFn(uk, xk);
    let revenue = 0;
    const nDem = Math.min(nu, demand.length);
    for (let i = 0; i < nDem; i++) {
      revenue += uk[i]! * demand[i]!;
    }
    economicCost -= revenue;
  }

  return {
    ...result,
    cost: economicCost,
  };
}
