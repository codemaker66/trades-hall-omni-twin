// ---------------------------------------------------------------------------
// OC-3: Multiple Shooting Method (Pontryagin's Maximum Principle)
// ---------------------------------------------------------------------------

import type { ShootingConfig, ShootingResult } from '../types.js';
import { vecClone, vecSub, vecNorm, vecAdd, vecScale } from '../types.js';
import { rk4Step } from './hamiltonian.js';

// ---------------------------------------------------------------------------
// Multiple shooting solver
// ---------------------------------------------------------------------------

/**
 * Solve a two-point BVP arising from PMP using multiple shooting.
 *
 * Algorithm:
 *  1. Divide [0, T] into nSegments intervals
 *  2. Decision variables: initial costate lambda_0 plus intermediate state
 *     values at each segment boundary
 *  3. Forward integrate each segment independently
 *  4. Enforce continuity constraints at segment boundaries
 *  5. Newton method on the full defect vector (all boundary mismatches)
 *
 * Multiple shooting improves numerical stability over single shooting by
 * reducing the sensitivity of the residual to the initial guess.
 *
 * @param config     Shooting configuration
 * @param nSegments  Number of shooting segments (default 4)
 * @returns          ShootingResult with trajectories, cost, convergence info
 */
export function multipleShooting(
  config: ShootingConfig,
  nSegments: number = 4,
): ShootingResult {
  const { nx, nu, x0, T, nSteps, tolerance, maxIter } = config;

  // Steps per segment (distribute evenly, last segment takes remainder)
  const baseStepsPerSeg = Math.floor(nSteps / nSegments);
  const segSteps: number[] = [];
  for (let s = 0; s < nSegments; s++) {
    segSteps.push(
      s < nSegments - 1 ? baseStepsPerSeg : nSteps - baseStepsPerSeg * (nSegments - 1),
    );
  }

  // Segment time boundaries
  const segTimes: number[] = [0];
  for (let s = 0; s < nSegments; s++) {
    segTimes.push(segTimes[s]! + (segSteps[s]! * T) / nSteps);
  }

  // -----------------------------------------------------------------------
  // Decision variables layout:
  //   [ lambda_0 (nx) | x_1 (nx) | x_2 (nx) | ... | x_{nSegments-1} (nx) ]
  // Total decision variables: nx + (nSegments - 1) * nx = nSegments * nx
  // -----------------------------------------------------------------------
  const nVars = nSegments * nx;
  let vars = new Float64Array(nVars);

  // Initialize: lambda_0 = 0, intermediate x values via linear interpolation
  // from x0 (a simple initial guess)
  for (let s = 1; s < nSegments; s++) {
    const offset = s * nx;
    for (let i = 0; i < nx; i++) {
      vars[offset + i] = x0[i]!;
    }
  }

  const eps = 1e-7;
  let converged = false;
  let iterations = 0;

  for (let iter = 0; iter < maxIter; iter++) {
    iterations = iter + 1;

    // Compute residual (defect at each segment boundary + terminal costate)
    const residual = computeResidual(config, vars, x0, segSteps, segTimes, nSegments);
    const residualNorm = vecNorm(residual);

    if (residualNorm < tolerance) {
      converged = true;
      break;
    }

    // Build finite-difference Jacobian
    const nRes = residual.length;
    const J = new Float64Array(nRes * nVars);

    for (let j = 0; j < nVars; j++) {
      const varsPert = vecClone(vars);
      varsPert[j] = varsPert[j]! + eps;

      const residualPert = computeResidual(
        config, varsPert, x0, segSteps, segTimes, nSegments,
      );

      for (let i = 0; i < nRes; i++) {
        J[i * nVars + j] = (residualPert[i]! - residual[i]!) / eps;
      }
    }

    // Newton step: solve J * delta = -residual
    const negRes = vecScale(residual, -1);
    const delta = solveLinearLeastSquares(J, negRes, nRes, nVars);

    // Damped update to improve convergence
    const alpha = 1.0;
    vars = vecAdd(vars, vecScale(delta, alpha)) as Float64Array<ArrayBuffer>;
  }

  // -----------------------------------------------------------------------
  // Reconstruct full trajectories from converged solution
  // -----------------------------------------------------------------------
  return reconstructTrajectories(config, vars, x0, segSteps, segTimes, nSegments, converged, iterations);
}

// ---------------------------------------------------------------------------
// Residual computation
// ---------------------------------------------------------------------------

/**
 * Compute the defect residual vector for the multiple shooting formulation.
 *
 * Residual layout:
 *  - Segments 0..nSegments-2: continuity defect = x_end(seg) - x_start(seg+1)
 *  - Terminal costate: lambda(T)
 *
 * Total residual size: nSegments * nx
 */
function computeResidual(
  config: ShootingConfig,
  vars: Float64Array,
  x0: Float64Array,
  segSteps: number[],
  segTimes: number[],
  nSegments: number,
): Float64Array {
  const { nx } = config;
  const dt = config.T / config.nSteps;
  const nRes = nSegments * nx;
  const residual = new Float64Array(nRes);

  // Extract lambda_0 from vars
  const lambda0 = new Float64Array(vars.subarray(0, nx));

  // Integrate each segment
  let xStart = vecClone(x0);
  let lambdaStart = vecClone(lambda0);

  for (let s = 0; s < nSegments; s++) {
    const steps = segSteps[s]!;
    const t0 = segTimes[s]!;

    // Forward integrate this segment
    let x = vecClone(xStart);
    let lambda = vecClone(lambdaStart);

    for (let step = 0; step < steps; step++) {
      const t = t0 + step * dt;
      const u = config.controlOptimality(x, lambda, t);

      const augmented = new Float64Array(2 * nx);
      augmented.set(x, 0);
      augmented.set(lambda, nx);

      const augDyn = (y: Float64Array, time: number): Float64Array => {
        const xC = new Float64Array(y.subarray(0, nx));
        const lC = new Float64Array(y.subarray(nx, 2 * nx));
        const uC = config.controlOptimality(xC, lC, time);

        const dxdt = config.stateDynamics(xC, uC, lC, time);
        const dldt = config.costateDynamics(xC, uC, lC, time);

        const r = new Float64Array(2 * nx);
        r.set(dxdt, 0);
        r.set(dldt, nx);
        return r;
      };

      const augNext = rk4Step(augDyn, augmented, t, dt);
      x = new Float64Array(augNext.subarray(0, nx));
      lambda = new Float64Array(augNext.subarray(nx, 2 * nx));
    }

    if (s < nSegments - 1) {
      // Continuity defect: x_end(seg_s) - x_start(seg_{s+1})
      const nextXStart = new Float64Array(
        vars.subarray((s + 1) * nx, (s + 2) * nx),
      );
      const defect = vecSub(x, nextXStart);
      residual.set(defect, s * nx);

      // Next segment starts at the decision variable value
      xStart = vecClone(nextXStart);
      lambdaStart = vecClone(lambda);
    } else {
      // Terminal segment: costate boundary condition lambda(T) = 0
      residual.set(lambda, s * nx);
    }
  }

  return residual;
}

// ---------------------------------------------------------------------------
// Trajectory reconstruction
// ---------------------------------------------------------------------------

/**
 * Reconstruct the full state, control, and costate trajectories from
 * the converged decision variables.
 */
function reconstructTrajectories(
  config: ShootingConfig,
  vars: Float64Array,
  x0: Float64Array,
  segSteps: number[],
  segTimes: number[],
  nSegments: number,
  converged: boolean,
  iterations: number,
): ShootingResult {
  const { nx } = config;
  const dt = config.T / config.nSteps;

  const xTrajectory: Float64Array[] = [];
  const uTrajectory: Float64Array[] = [];
  const lambdaTrajectory: Float64Array[] = [];
  let totalCost = 0;

  const lambda0 = new Float64Array(vars.subarray(0, nx));

  let xStart = vecClone(x0);
  let lambdaStart = vecClone(lambda0);

  for (let s = 0; s < nSegments; s++) {
    const steps = segSteps[s]!;
    const t0 = segTimes[s]!;

    let x = vecClone(xStart);
    let lambda = vecClone(lambdaStart);

    // Record start of segment (avoid duplicates except first segment)
    if (s === 0) {
      xTrajectory.push(vecClone(x));
      lambdaTrajectory.push(vecClone(lambda));
    }

    for (let step = 0; step < steps; step++) {
      const t = t0 + step * dt;
      const u = config.controlOptimality(x, lambda, t);
      uTrajectory.push(vecClone(u));

      totalCost += config.runningCost(x, u, t) * dt;

      const augmented = new Float64Array(2 * nx);
      augmented.set(x, 0);
      augmented.set(lambda, nx);

      const augDyn = (y: Float64Array, time: number): Float64Array => {
        const xC = new Float64Array(y.subarray(0, nx));
        const lC = new Float64Array(y.subarray(nx, 2 * nx));
        const uC = config.controlOptimality(xC, lC, time);

        const dxdt = config.stateDynamics(xC, uC, lC, time);
        const dldt = config.costateDynamics(xC, uC, lC, time);

        const r = new Float64Array(2 * nx);
        r.set(dxdt, 0);
        r.set(dldt, nx);
        return r;
      };

      const augNext = rk4Step(augDyn, augmented, t, dt);
      x = new Float64Array(augNext.subarray(0, nx));
      lambda = new Float64Array(augNext.subarray(nx, 2 * nx));

      xTrajectory.push(vecClone(x));
      lambdaTrajectory.push(vecClone(lambda));
    }

    // Set up next segment start
    if (s < nSegments - 1) {
      xStart = new Float64Array(vars.subarray((s + 1) * nx, (s + 2) * nx));
      lambdaStart = vecClone(lambda);
    }
  }

  // Final control at terminal time
  const tFinal = config.T;
  const xFinal = xTrajectory[xTrajectory.length - 1]!;
  const lambdaFinal = lambdaTrajectory[lambdaTrajectory.length - 1]!;
  const uFinal = config.controlOptimality(xFinal, lambdaFinal, tFinal);
  uTrajectory.push(vecClone(uFinal));

  return {
    xTrajectory,
    uTrajectory,
    lambdaTrajectory,
    cost: totalCost,
    converged,
    iterations,
  };
}

// ---------------------------------------------------------------------------
// Linear least squares solver for Newton step (handles square and tall systems)
// ---------------------------------------------------------------------------

/**
 * Solve J * delta = rhs in the least-squares sense using normal equations.
 * For square systems this reduces to standard solve.
 */
function solveLinearLeastSquares(
  J: Float64Array,
  rhs: Float64Array,
  nRows: number,
  nCols: number,
): Float64Array {
  // Form normal equations: J^T J delta = J^T rhs
  const JtJ = new Float64Array(nCols * nCols);
  const Jtb = new Float64Array(nCols);

  for (let i = 0; i < nCols; i++) {
    for (let j = 0; j < nCols; j++) {
      let sum = 0;
      for (let k = 0; k < nRows; k++) {
        sum += J[k * nCols + i]! * J[k * nCols + j]!;
      }
      JtJ[i * nCols + j] = sum;
    }
    let sum = 0;
    for (let k = 0; k < nRows; k++) {
      sum += J[k * nCols + i]! * rhs[k]!;
    }
    Jtb[i] = sum;
  }

  // Add small regularization for numerical stability
  for (let i = 0; i < nCols; i++) {
    JtJ[i * nCols + i] = JtJ[i * nCols + i]! + 1e-10;
  }

  return solveLinearSystem(JtJ, Jtb, nCols);
}

// ---------------------------------------------------------------------------
// Dense linear system solver (Gauss elimination with partial pivoting)
// ---------------------------------------------------------------------------

/**
 * Solve A * x = b for x, where A is stored as row-major Float64Array
 * of size n x n.
 */
function solveLinearSystem(
  A: Float64Array,
  b: Float64Array,
  n: number,
): Float64Array {
  const aug = new Float64Array(n * (n + 1));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      aug[i * (n + 1) + j] = A[i * n + j]!;
    }
    aug[i * (n + 1) + n] = b[i]!;
  }

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    let maxVal = Math.abs(aug[col * (n + 1) + col]!);
    for (let row = col + 1; row < n; row++) {
      const val = Math.abs(aug[row * (n + 1) + col]!);
      if (val > maxVal) {
        maxVal = val;
        maxRow = row;
      }
    }

    if (maxRow !== col) {
      for (let j = 0; j <= n; j++) {
        const tmp = aug[col * (n + 1) + j]!;
        aug[col * (n + 1) + j] = aug[maxRow * (n + 1) + j]!;
        aug[maxRow * (n + 1) + j] = tmp;
      }
    }

    const pivot = aug[col * (n + 1) + col]!;
    if (Math.abs(pivot) < 1e-15) {
      return new Float64Array(n);
    }

    for (let row = col + 1; row < n; row++) {
      const factor = aug[row * (n + 1) + col]! / pivot;
      for (let j = col; j <= n; j++) {
        aug[row * (n + 1) + j] = aug[row * (n + 1) + j]! - factor * aug[col * (n + 1) + j]!;
      }
    }
  }

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
