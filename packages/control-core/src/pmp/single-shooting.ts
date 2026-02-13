// ---------------------------------------------------------------------------
// OC-3: Single Shooting Method (Pontryagin's Maximum Principle)
// ---------------------------------------------------------------------------

import type { ShootingConfig, ShootingResult } from '../types.js';
import { vecClone, vecSub, vecNorm, vecAdd, vecScale } from '../types.js';
import { rk4Step, integrateODE } from './hamiltonian.js';

// ---------------------------------------------------------------------------
// Single shooting solver
// ---------------------------------------------------------------------------

/**
 * Solve a two-point boundary value problem arising from PMP using single
 * shooting with Newton iteration on the initial costate.
 *
 * Algorithm:
 *  1. Guess initial costate lambda_0
 *  2. Forward integrate state + costate from t=0 to T
 *  3. Evaluate boundary condition residual at t=T (costate should equal zero
 *     for free terminal state, or match transversality conditions)
 *  4. Newton update on lambda_0 using finite-difference Jacobian
 *  5. Iterate until boundary conditions are met or maxIter reached
 *
 * @param config  Shooting configuration
 * @returns       ShootingResult with trajectories, cost, convergence info
 */
export function singleShooting(config: ShootingConfig): ShootingResult {
  const { nx, nu, x0, T, nSteps, tolerance, maxIter } = config;
  const dt = T / nSteps;

  // Guess initial costate as zero
  let lambda0 = new Float64Array(nx);

  // Perturbation size for finite-difference Jacobian
  const eps = 1e-7;

  let converged = false;
  let iterations = 0;

  // Storage for best trajectories
  let bestXTraj: Float64Array[] = [];
  let bestUTraj: Float64Array[] = [];
  let bestLambdaTraj: Float64Array[] = [];
  let bestCost = Infinity;

  for (let iter = 0; iter < maxIter; iter++) {
    iterations = iter + 1;

    // -----------------------------------------------------------------------
    // Forward integrate with current lambda_0
    // -----------------------------------------------------------------------
    const { xTraj, uTraj, lambdaTraj, totalCost } = forwardIntegrate(
      config, x0, lambda0, nSteps, dt,
    );

    bestXTraj = xTraj;
    bestUTraj = uTraj;
    bestLambdaTraj = lambdaTraj;
    bestCost = totalCost;

    // Boundary condition residual: lambda(T) should be zero (free final state)
    const lambdaT = lambdaTraj[nSteps]!;
    const residual = vecClone(lambdaT);
    const residualNorm = vecNorm(residual);

    if (residualNorm < tolerance) {
      converged = true;
      break;
    }

    // -----------------------------------------------------------------------
    // Build finite-difference Jacobian: d(residual) / d(lambda_0)
    // -----------------------------------------------------------------------
    const J = new Float64Array(nx * nx);

    for (let j = 0; j < nx; j++) {
      // Perturb lambda_0[j]
      const lambda0Pert = vecClone(lambda0);
      lambda0Pert[j] = lambda0Pert[j]! + eps;

      const pertResult = forwardIntegrate(config, x0, lambda0Pert, nSteps, dt);
      const lambdaTPert = pertResult.lambdaTraj[nSteps]!;

      // Column j of Jacobian: (residualPert - residual) / eps
      for (let i = 0; i < nx; i++) {
        J[i * nx + j] = (lambdaTPert[i]! - residual[i]!) / eps;
      }
    }

    // -----------------------------------------------------------------------
    // Newton update: delta = -J^{-1} * residual
    // Solve J * delta = -residual via Gauss elimination
    // -----------------------------------------------------------------------
    const negResidual = vecScale(residual, -1);
    const delta = solveLinearSystem(J, negResidual, nx);

    // Update lambda_0
    lambda0 = vecAdd(lambda0, delta) as Float64Array<ArrayBuffer>;
  }

  return {
    xTrajectory: bestXTraj,
    uTrajectory: bestUTraj,
    lambdaTrajectory: bestLambdaTraj,
    cost: bestCost,
    converged,
    iterations,
  };
}

// ---------------------------------------------------------------------------
// Forward integration of state + costate
// ---------------------------------------------------------------------------

/**
 * Forward integrate the augmented state-costate system from t=0 to T.
 */
function forwardIntegrate(
  config: ShootingConfig,
  x0: Float64Array,
  lambda0: Float64Array,
  nSteps: number,
  dt: number,
): {
  xTraj: Float64Array[];
  uTraj: Float64Array[];
  lambdaTraj: Float64Array[];
  totalCost: number;
} {
  const { nx, nu } = config;

  const xTraj: Float64Array[] = [vecClone(x0)];
  const lambdaTraj: Float64Array[] = [vecClone(lambda0)];
  const uTraj: Float64Array[] = [];

  let x = vecClone(x0);
  let lambda = vecClone(lambda0);
  let totalCost = 0;

  for (let step = 0; step < nSteps; step++) {
    const t = step * dt;

    // Compute optimal control from stationarity condition
    const u = config.controlOptimality(x, lambda, t);
    uTraj.push(vecClone(u));

    // Accumulate running cost
    totalCost += config.runningCost(x, u, t) * dt;

    // Build augmented system [x; lambda] and integrate
    const augmented = new Float64Array(2 * nx);
    augmented.set(x, 0);
    augmented.set(lambda, nx);

    const augDynamics = (y: Float64Array, time: number): Float64Array => {
      const xCur = y.subarray(0, nx);
      const lambdaCur = y.subarray(nx, 2 * nx);
      const uCur = config.controlOptimality(
        new Float64Array(xCur),
        new Float64Array(lambdaCur),
        time,
      );

      const dx = config.stateDynamics(
        new Float64Array(xCur),
        uCur,
        new Float64Array(lambdaCur),
        time,
      );
      const dLambda = config.costateDynamics(
        new Float64Array(xCur),
        uCur,
        new Float64Array(lambdaCur),
        time,
      );

      const result = new Float64Array(2 * nx);
      result.set(dx, 0);
      result.set(dLambda, nx);
      return result;
    };

    const augNext = rk4Step(augDynamics, augmented, t, dt);

    x = new Float64Array(augNext.subarray(0, nx));
    lambda = new Float64Array(augNext.subarray(nx, 2 * nx));

    xTraj.push(vecClone(x));
    lambdaTraj.push(vecClone(lambda));
  }

  // Final control at terminal time
  const tFinal = nSteps * dt;
  const uFinal = config.controlOptimality(x, lambda, tFinal);
  uTraj.push(vecClone(uFinal));

  return { xTraj, uTraj, lambdaTraj, totalCost };
}

// ---------------------------------------------------------------------------
// Small dense linear system solver (Gauss elimination with partial pivoting)
// ---------------------------------------------------------------------------

/**
 * Solve A * x = b for x, where A is stored as row-major Float64Array of
 * size n x n. Destructive to internal copies.
 */
function solveLinearSystem(
  A: Float64Array,
  b: Float64Array,
  n: number,
): Float64Array {
  // Build augmented matrix [A | b]
  const aug = new Float64Array(n * (n + 1));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      aug[i * (n + 1) + j] = A[i * n + j]!;
    }
    aug[i * (n + 1) + n] = b[i]!;
  }

  // Forward elimination with partial pivoting
  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxRow = col;
    let maxVal = Math.abs(aug[col * (n + 1) + col]!);
    for (let row = col + 1; row < n; row++) {
      const val = Math.abs(aug[row * (n + 1) + col]!);
      if (val > maxVal) {
        maxVal = val;
        maxRow = row;
      }
    }

    // Swap rows
    if (maxRow !== col) {
      for (let j = 0; j <= n; j++) {
        const tmp = aug[col * (n + 1) + j]!;
        aug[col * (n + 1) + j] = aug[maxRow * (n + 1) + j]!;
        aug[maxRow * (n + 1) + j] = tmp;
      }
    }

    const pivot = aug[col * (n + 1) + col]!;
    if (Math.abs(pivot) < 1e-15) {
      // Singular -- return zero delta
      return new Float64Array(n);
    }

    // Eliminate below
    for (let row = col + 1; row < n; row++) {
      const factor = aug[row * (n + 1) + col]! / pivot;
      for (let j = col; j <= n; j++) {
        aug[row * (n + 1) + j] = aug[row * (n + 1) + j]! - factor * aug[col * (n + 1) + j]!;
      }
    }
  }

  // Back substitution
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = aug[i * (n + 1) + n]!;
    for (let j = i + 1; j < n; j++) {
      sum -= aug[i * (n + 1) + j]! * x[j]!;
    }
    const diag = aug[i * (n + 1) + i]!;
    x[i] = Math.abs(diag) > 1e-15 ? sum / diag : 0;
  }

  return x;
}
