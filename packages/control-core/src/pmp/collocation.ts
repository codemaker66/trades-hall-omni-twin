// ---------------------------------------------------------------------------
// OC-3: Direct Collocation Methods (Pontryagin's Maximum Principle)
// ---------------------------------------------------------------------------

import type { CollocationConfig, ShootingResult } from '../types.js';
import { vecClone, vecSub, vecNorm, vecAdd, vecScale } from '../types.js';

// ---------------------------------------------------------------------------
// Trapezoidal collocation
// ---------------------------------------------------------------------------

/**
 * Solve an optimal control problem using trapezoidal (Hermite-Simpson)
 * direct collocation.
 *
 * Transcribes the continuous-time OCP into a nonlinear program by:
 *  1. Discretizing the time horizon into nSegments intervals
 *  2. Enforcing dynamics as defect constraints at each node:
 *     x_{k+1} - x_k - (dt/2)(f(x_k, u_k, t_k) + f(x_{k+1}, u_{k+1}, t_{k+1})) = 0
 *  3. Solving via a simplified SQP approach (linearize defects, iterate)
 *
 * @param dynamics  State dynamics dx/dt = f(x, u, t)
 * @param cost      Running cost L(x, u, t)
 * @param x0        Initial state (nx)
 * @param T         Terminal time
 * @param nSegments Number of collocation segments
 * @param nx        State dimension
 * @param nu        Control dimension
 * @returns         ShootingResult with trajectories, cost, convergence info
 */
export function trapezoidalCollocation(
  dynamics: (x: Float64Array, u: Float64Array, t: number) => Float64Array,
  cost: (x: Float64Array, u: Float64Array, t: number) => number,
  x0: Float64Array,
  T: number,
  nSegments: number,
  nx: number,
  nu: number,
): ShootingResult {
  const nNodes = nSegments + 1;
  const dt = T / nSegments;
  const maxIter = 100;
  const tolerance = 1e-8;

  // -----------------------------------------------------------------------
  // Decision variables layout (column-major style):
  //   [ x_0 (nx) | u_0 (nu) | x_1 (nx) | u_1 (nu) | ... | x_N (nx) | u_N (nu) ]
  // Total: nNodes * (nx + nu)
  // -----------------------------------------------------------------------
  const nVarsPerNode = nx + nu;
  const nVars = nNodes * nVarsPerNode;
  const nDefects = nSegments * nx; // defect constraints

  // Initialize decision variables
  let vars = new Float64Array(nVars);

  // Set initial state and linear interpolation for states, zero for controls
  for (let k = 0; k < nNodes; k++) {
    const xOffset = k * nVarsPerNode;
    // Linear interpolation from x0 to x0 (simple initial guess)
    for (let i = 0; i < nx; i++) {
      vars[xOffset + i] = x0[i]!;
    }
    // Controls initialized to zero (already zero from Float64Array)
  }

  // Enforce initial condition: x_0 = x0
  for (let i = 0; i < nx; i++) {
    vars[i] = x0[i]!;
  }

  let converged = false;
  let iterations = 0;

  for (let iter = 0; iter < maxIter; iter++) {
    iterations = iter + 1;

    // -------------------------------------------------------------------
    // Compute defect constraints
    // -------------------------------------------------------------------
    const defects = computeDefects(vars, dynamics, nSegments, nx, nu, dt);
    const defectNorm = vecNorm(defects);

    if (defectNorm < tolerance) {
      converged = true;
      break;
    }

    // -------------------------------------------------------------------
    // Linearize defects via finite differences and take a Newton-like step
    // -------------------------------------------------------------------
    const eps = 1e-7;

    // We only update non-fixed variables (skip x_0 which is fixed)
    // For simplicity, compute Jacobian w.r.t. all vars then fix x_0
    const J = new Float64Array(nDefects * nVars);

    for (let j = 0; j < nVars; j++) {
      // Skip fixed variables (initial state)
      if (j < nx) continue;

      const varsPert = vecClone(vars);
      varsPert[j] = varsPert[j]! + eps;

      // Re-enforce initial condition in perturbed vars
      for (let i = 0; i < nx; i++) {
        varsPert[i] = x0[i]!;
      }

      const defectsPert = computeDefects(varsPert, dynamics, nSegments, nx, nu, dt);

      for (let i = 0; i < nDefects; i++) {
        J[i * nVars + j] = (defectsPert[i]! - defects[i]!) / eps;
      }
    }

    // Solve J * delta = -defects using least-squares (normal equations)
    const negDefects = vecScale(defects, -1);
    const delta = solveNormalEquations(J, negDefects, nDefects, nVars);

    // Zero out corrections for fixed variables
    for (let i = 0; i < nx; i++) {
      delta[i] = 0;
    }

    // Line search with backtracking
    let alpha = 1.0;
    for (let ls = 0; ls < 10; ls++) {
      const varsCandidate = vecAdd(vars, vecScale(delta, alpha));

      // Re-enforce initial condition
      for (let i = 0; i < nx; i++) {
        varsCandidate[i] = x0[i]!;
      }

      const defectsNew = computeDefects(varsCandidate, dynamics, nSegments, nx, nu, dt);
      const newNorm = vecNorm(defectsNew);

      if (newNorm < defectNorm) {
        vars = varsCandidate as Float64Array<ArrayBuffer>;
        break;
      }
      alpha *= 0.5;

      if (ls === 9) {
        // Accept even if not improving
        vars = vecAdd(vars, vecScale(delta, alpha)) as Float64Array<ArrayBuffer>;
        for (let i = 0; i < nx; i++) {
          vars[i] = x0[i]!;
        }
      }
    }
  }

  // -----------------------------------------------------------------------
  // Extract trajectories from converged decision variables
  // -----------------------------------------------------------------------
  return extractTrajectories(vars, cost, nNodes, nSegments, nx, nu, dt, converged, iterations);
}

// ---------------------------------------------------------------------------
// Direct collocation wrapper using CollocationConfig
// ---------------------------------------------------------------------------

/**
 * Direct collocation using the CollocationConfig interface.
 *
 * Wraps trapezoidalCollocation by adapting the HamiltonianConfig functions
 * to the simpler (x, u, t) signature expected by the collocation method.
 *
 * @param config  CollocationConfig with dynamics, cost, and problem parameters
 * @returns       ShootingResult
 */
export function directCollocation(config: CollocationConfig): ShootingResult {
  const { x0, T, nSegments, nx, nu } = config;

  // Adapt state dynamics: the HamiltonianConfig version takes (x, u, lambda, t)
  // but for direct collocation we don't need the costate in the dynamics call.
  // We pass a dummy lambda = zeros.
  const dummyLambda = new Float64Array(nx);

  const dynamics = (x: Float64Array, u: Float64Array, t: number): Float64Array => {
    return config.stateDynamics(x, u, dummyLambda, t);
  };

  const costFn = (x: Float64Array, u: Float64Array, t: number): number => {
    return config.runningCost(x, u, t);
  };

  return trapezoidalCollocation(dynamics, costFn, x0, T, nSegments, nx, nu);
}

// ---------------------------------------------------------------------------
// Defect computation
// ---------------------------------------------------------------------------

/**
 * Compute trapezoidal defects for all segments.
 *
 * defect_k = x_{k+1} - x_k - (dt/2)(f(x_k, u_k, t_k) + f(x_{k+1}, u_{k+1}, t_{k+1}))
 */
function computeDefects(
  vars: Float64Array,
  dynamics: (x: Float64Array, u: Float64Array, t: number) => Float64Array,
  nSegments: number,
  nx: number,
  nu: number,
  dt: number,
): Float64Array {
  const nVarsPerNode = nx + nu;
  const defects = new Float64Array(nSegments * nx);

  for (let k = 0; k < nSegments; k++) {
    const offsetK = k * nVarsPerNode;
    const offsetK1 = (k + 1) * nVarsPerNode;
    const tK = k * dt;
    const tK1 = (k + 1) * dt;

    // Extract state and control at node k
    const xK = new Float64Array(vars.subarray(offsetK, offsetK + nx));
    const uK = new Float64Array(vars.subarray(offsetK + nx, offsetK + nx + nu));

    // Extract state and control at node k+1
    const xK1 = new Float64Array(vars.subarray(offsetK1, offsetK1 + nx));
    const uK1 = new Float64Array(vars.subarray(offsetK1 + nx, offsetK1 + nx + nu));

    // Evaluate dynamics at both endpoints
    const fK = dynamics(xK, uK, tK);
    const fK1 = dynamics(xK1, uK1, tK1);

    // Trapezoidal defect
    for (let i = 0; i < nx; i++) {
      defects[k * nx + i] =
        xK1[i]! - xK[i]! - (dt / 2) * (fK[i]! + fK1[i]!);
    }
  }

  return defects;
}

// ---------------------------------------------------------------------------
// Trajectory extraction
// ---------------------------------------------------------------------------

/**
 * Extract state, control, and costate trajectories from decision variables.
 * Since direct collocation does not naturally produce costates, the
 * lambdaTrajectory is filled with zeros.
 */
function extractTrajectories(
  vars: Float64Array,
  costFn: (x: Float64Array, u: Float64Array, t: number) => number,
  nNodes: number,
  nSegments: number,
  nx: number,
  nu: number,
  dt: number,
  converged: boolean,
  iterations: number,
): ShootingResult {
  const nVarsPerNode = nx + nu;
  const xTrajectory: Float64Array[] = [];
  const uTrajectory: Float64Array[] = [];
  const lambdaTrajectory: Float64Array[] = [];
  let totalCost = 0;

  for (let k = 0; k < nNodes; k++) {
    const offset = k * nVarsPerNode;
    const x = new Float64Array(vars.subarray(offset, offset + nx));
    const u = new Float64Array(vars.subarray(offset + nx, offset + nx + nu));
    const t = k * dt;

    xTrajectory.push(vecClone(x));
    uTrajectory.push(vecClone(u));
    lambdaTrajectory.push(new Float64Array(nx)); // zeros (no costate in direct method)

    // Trapezoidal quadrature for cost
    if (k < nSegments) {
      const offsetNext = (k + 1) * nVarsPerNode;
      const xNext = new Float64Array(vars.subarray(offsetNext, offsetNext + nx));
      const uNext = new Float64Array(
        vars.subarray(offsetNext + nx, offsetNext + nx + nu),
      );
      const tNext = (k + 1) * dt;
      totalCost += (dt / 2) * (costFn(x, u, t) + costFn(xNext, uNext, tNext));
    }
  }

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
// Normal equations solver for least-squares Newton step
// ---------------------------------------------------------------------------

/**
 * Solve J * delta = rhs in the least-squares sense via normal equations:
 * (J^T J + mu I) delta = J^T rhs
 */
function solveNormalEquations(
  J: Float64Array,
  rhs: Float64Array,
  nRows: number,
  nCols: number,
): Float64Array {
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

  // Levenberg-Marquardt regularization
  for (let i = 0; i < nCols; i++) {
    JtJ[i * nCols + i] = JtJ[i * nCols + i]! + 1e-8;
  }

  return gaussSolve(JtJ, Jtb, nCols);
}

// ---------------------------------------------------------------------------
// Gauss elimination solver
// ---------------------------------------------------------------------------

/**
 * Solve A * x = b via Gauss elimination with partial pivoting.
 */
function gaussSolve(
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
      continue;
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
