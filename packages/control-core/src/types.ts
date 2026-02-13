// ---------------------------------------------------------------------------
// @omni-twin/control-core — Optimal Control Types
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// PRNG (mulberry32, shared pattern across packages)
// ---------------------------------------------------------------------------

/** Seedable PRNG function. */
export type PRNG = () => number;

/** Create a seedable PRNG using the mulberry32 algorithm. */
export function createPRNG(seed: number): PRNG {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// Matrix type + operations
// ---------------------------------------------------------------------------

/** Row-major matrix stored as flat Float64Array. */
export interface Matrix {
  data: Float64Array;
  rows: number;
  cols: number;
}

/** Create a zero-filled matrix. */
export function createMatrix(rows: number, cols: number): Matrix {
  return { data: new Float64Array(rows * cols), rows, cols };
}

/** Get element at (i, j) — row-major access. */
export function matGet(m: Matrix, i: number, j: number): number {
  return m.data[i * m.cols + j]!;
}

/** Set element at (i, j) — row-major access. */
export function matSet(m: Matrix, i: number, j: number, val: number): void {
  m.data[i * m.cols + j] = val;
}

/** Matrix-matrix multiply: C = A * B. */
export function matMul(A: Matrix, B: Matrix): Matrix {
  const C = createMatrix(A.rows, B.cols);
  for (let i = 0; i < A.rows; i++) {
    for (let j = 0; j < B.cols; j++) {
      let sum = 0;
      for (let k = 0; k < A.cols; k++) {
        sum += matGet(A, i, k) * matGet(B, k, j);
      }
      matSet(C, i, j, sum);
    }
  }
  return C;
}

/** Matrix-vector multiply: y = A * x. */
export function matVecMul(A: Matrix, x: Float64Array): Float64Array {
  const y = new Float64Array(A.rows);
  for (let i = 0; i < A.rows; i++) {
    let sum = 0;
    for (let j = 0; j < A.cols; j++) {
      sum += matGet(A, i, j) * x[j]!;
    }
    y[i] = sum;
  }
  return y;
}

/** Matrix transpose. */
export function matTranspose(A: Matrix): Matrix {
  const T = createMatrix(A.cols, A.rows);
  for (let i = 0; i < A.rows; i++) {
    for (let j = 0; j < A.cols; j++) {
      matSet(T, j, i, matGet(A, i, j));
    }
  }
  return T;
}

/** Element-wise matrix addition: C = A + B. */
export function matAdd(A: Matrix, B: Matrix): Matrix {
  const C = createMatrix(A.rows, A.cols);
  for (let i = 0; i < A.data.length; i++) {
    C.data[i] = A.data[i]! + B.data[i]!;
  }
  return C;
}

/** Element-wise matrix subtraction: C = A - B. */
export function matSub(A: Matrix, B: Matrix): Matrix {
  const C = createMatrix(A.rows, A.cols);
  for (let i = 0; i < A.data.length; i++) {
    C.data[i] = A.data[i]! - B.data[i]!;
  }
  return C;
}

/** Scale matrix: C = s * A. */
export function matScale(A: Matrix, s: number): Matrix {
  const C = createMatrix(A.rows, A.cols);
  for (let i = 0; i < A.data.length; i++) {
    C.data[i] = A.data[i]! * s;
  }
  return C;
}

/** Create n x n identity matrix. */
export function matIdentity(n: number): Matrix {
  const I = createMatrix(n, n);
  for (let i = 0; i < n; i++) {
    matSet(I, i, i, 1);
  }
  return I;
}

/** Invert a small matrix via Gauss-Jordan elimination with partial pivoting. */
export function matInvert(A: Matrix): Matrix {
  const n = A.rows;
  // Augmented matrix [A | I]
  const aug = createMatrix(n, 2 * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      matSet(aug, i, j, matGet(A, i, j));
    }
    matSet(aug, i, n + i, 1);
  }

  for (let col = 0; col < n; col++) {
    // Partial pivoting
    let maxRow = col;
    let maxVal = Math.abs(matGet(aug, col, col));
    for (let row = col + 1; row < n; row++) {
      const val = Math.abs(matGet(aug, row, col));
      if (val > maxVal) {
        maxVal = val;
        maxRow = row;
      }
    }
    if (maxRow !== col) {
      for (let j = 0; j < 2 * n; j++) {
        const tmp = matGet(aug, col, j);
        matSet(aug, col, j, matGet(aug, maxRow, j));
        matSet(aug, maxRow, j, tmp);
      }
    }

    const pivot = matGet(aug, col, col);
    if (Math.abs(pivot) < 1e-15) {
      throw new Error('Matrix is singular or near-singular');
    }

    // Scale pivot row
    for (let j = 0; j < 2 * n; j++) {
      matSet(aug, col, j, matGet(aug, col, j) / pivot);
    }

    // Eliminate column
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = matGet(aug, row, col);
      for (let j = 0; j < 2 * n; j++) {
        matSet(aug, row, j, matGet(aug, row, j) - factor * matGet(aug, col, j));
      }
    }
  }

  // Extract inverse
  const inv = createMatrix(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      matSet(inv, i, j, matGet(aug, i, n + j));
    }
  }
  return inv;
}

/** Diagonal matrix from vector. */
export function matDiag(v: Float64Array): Matrix {
  const n = v.length;
  const D = createMatrix(n, n);
  for (let i = 0; i < n; i++) {
    matSet(D, i, i, v[i]!);
  }
  return D;
}

/** Deep copy of a matrix. */
export function matClone(A: Matrix): Matrix {
  return { data: new Float64Array(A.data), rows: A.rows, cols: A.cols };
}

/** Sum of diagonal elements. */
export function matTrace(A: Matrix): number {
  const n = Math.min(A.rows, A.cols);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += matGet(A, i, i);
  }
  return sum;
}

/** Matrix norm: Frobenius or infinity norm. */
export function matNorm(A: Matrix, type: 'frobenius' | 'inf'): number {
  if (type === 'frobenius') {
    let sum = 0;
    for (let i = 0; i < A.data.length; i++) {
      sum += A.data[i]! * A.data[i]!;
    }
    return Math.sqrt(sum);
  }
  // Infinity norm: max absolute row sum
  let maxRowSum = 0;
  for (let i = 0; i < A.rows; i++) {
    let rowSum = 0;
    for (let j = 0; j < A.cols; j++) {
      rowSum += Math.abs(matGet(A, i, j));
    }
    if (rowSum > maxRowSum) {
      maxRowSum = rowSum;
    }
  }
  return maxRowSum;
}

/** Solve Ax = b via LU decomposition with partial pivoting. */
export function matSolve(A: Matrix, b: Float64Array): Float64Array {
  const n = A.rows;
  // Build augmented [A | b] working copy
  const L = createMatrix(n, n);
  const U = matClone(A);
  const perm = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    perm[i] = i;
  }

  // LU factorization with partial pivoting
  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxRow = col;
    let maxVal = Math.abs(matGet(U, col, col));
    for (let row = col + 1; row < n; row++) {
      const val = Math.abs(matGet(U, row, col));
      if (val > maxVal) {
        maxVal = val;
        maxRow = row;
      }
    }
    if (Math.abs(maxVal) < 1e-15) {
      throw new Error('Matrix is singular or near-singular in matSolve');
    }

    // Swap rows in U and L, update permutation
    if (maxRow !== col) {
      const tmpPerm = perm[col]!;
      perm[col] = perm[maxRow]!;
      perm[maxRow] = tmpPerm;
      // Swap rows in U
      for (let j = 0; j < n; j++) {
        const tmp = matGet(U, col, j);
        matSet(U, col, j, matGet(U, maxRow, j));
        matSet(U, maxRow, j, tmp);
      }
      // Swap rows in L (only columns before col)
      for (let j = 0; j < col; j++) {
        const tmp = matGet(L, col, j);
        matSet(L, col, j, matGet(L, maxRow, j));
        matSet(L, maxRow, j, tmp);
      }
    }

    matSet(L, col, col, 1);

    // Eliminate below pivot
    for (let row = col + 1; row < n; row++) {
      const factor = matGet(U, row, col) / matGet(U, col, col);
      matSet(L, row, col, factor);
      for (let j = col; j < n; j++) {
        matSet(U, row, j, matGet(U, row, j) - factor * matGet(U, col, j));
      }
    }
  }

  // Permute b
  const pb = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    pb[i] = b[perm[i]!]!;
  }

  // Forward substitution: L y = pb
  const y = matSolveLower(L, pb);

  // Back substitution: U x = y
  const x = matSolveUpper(U, y);

  return x;
}

/** Cholesky decomposition: returns lower triangular L where A = L L^T. A must be positive definite. */
export function matCholesky(A: Matrix): Matrix {
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
        if (diag <= 0) {
          throw new Error('Matrix is not positive definite');
        }
        matSet(L, i, j, Math.sqrt(diag));
      } else {
        matSet(L, i, j, (matGet(A, i, j) - sum) / matGet(L, j, j));
      }
    }
  }

  return L;
}

/** Forward substitution: solve L x = b where L is lower triangular. */
export function matSolveLower(L: Matrix, b: Float64Array): Float64Array {
  const n = L.rows;
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < i; j++) {
      sum += matGet(L, i, j) * x[j]!;
    }
    x[i] = (b[i]! - sum) / matGet(L, i, i);
  }
  return x;
}

/** Back substitution: solve U x = b where U is upper triangular. */
export function matSolveUpper(U: Matrix, b: Float64Array): Float64Array {
  const n = U.rows;
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++) {
      sum += matGet(U, i, j) * x[j]!;
    }
    x[i] = (b[i]! - sum) / matGet(U, i, i);
  }
  return x;
}

/** Wrap an existing Float64Array as a Matrix (copies the data). */
export function arrayToMatrix(data: Float64Array, rows: number, cols: number): Matrix {
  return { data: new Float64Array(data), rows, cols };
}

// ---------------------------------------------------------------------------
// Vector operations
// ---------------------------------------------------------------------------

/** Element-wise vector addition: c = a + b. */
export function vecAdd(a: Float64Array, b: Float64Array): Float64Array {
  const c = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    c[i] = a[i]! + b[i]!;
  }
  return c;
}

/** Element-wise vector subtraction: c = a - b. */
export function vecSub(a: Float64Array, b: Float64Array): Float64Array {
  const c = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    c[i] = a[i]! - b[i]!;
  }
  return c;
}

/** Scale vector: c = s * a. */
export function vecScale(a: Float64Array, s: number): Float64Array {
  const c = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    c[i] = a[i]! * s;
  }
  return c;
}

/** Inner product: sum(a_i * b_i). */
export function vecDot(a: Float64Array, b: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i]! * b[i]!;
  }
  return sum;
}

/** L2 (Euclidean) norm. */
export function vecNorm(a: Float64Array): number {
  return Math.sqrt(vecDot(a, a));
}

/** Deep copy a Float64Array. */
export function vecClone(a: Float64Array): Float64Array {
  return new Float64Array(a);
}

// ---------------------------------------------------------------------------
// OC-1: LQR / LQG Types
// ---------------------------------------------------------------------------

/** Linear-Quadratic Regulator configuration. */
export interface LQRConfig {
  /** State transition matrix (nx x nx, row-major). */
  A: Float64Array;
  /** Input matrix (nx x nu, row-major). */
  B: Float64Array;
  /** State cost matrix (nx x nx, row-major). */
  Q: Float64Array;
  /** Control cost matrix (nu x nu, row-major). */
  R: Float64Array;
  /** Cross-coupling matrix (nx x nu, row-major, optional). */
  N?: Float64Array;
  /** State dimension. */
  nx: number;
  /** Control dimension. */
  nu: number;
}

/** LQR solution result. */
export interface LQRResult {
  /** Optimal gain matrix K (nu x nx, row-major). */
  K: Float64Array;
  /** Solution to the DARE / CARE (nx x nx, row-major). */
  P: Float64Array;
  /** Closed-loop eigenvalues [re0, im0, re1, im1, ...]. */
  eigenvalues: Float64Array;
}

/** Configuration for automatic Q/R tuning from engineering tolerances. */
export interface QRTuningConfig {
  /** State tolerances (nx). */
  tolerances: Float64Array;
  /** State weights (nx). */
  weights: Float64Array;
  /** Control tolerances (nu). */
  controlTolerances: Float64Array;
  /** Control weights (nu). */
  controlWeights: Float64Array;
}

/** Time-varying LQR configuration for finite-horizon problems. */
export interface TimeVaryingLQRConfig {
  /** State transition matrices, one per time step. */
  As: Float64Array[];
  /** Input matrices, one per time step. */
  Bs: Float64Array[];
  /** State cost matrices, one per time step. */
  Qs: Float64Array[];
  /** Control cost matrices, one per time step. */
  Rs: Float64Array[];
  /** Terminal cost matrix (nx x nx, row-major). */
  Qf: Float64Array;
  /** State dimension. */
  nx: number;
  /** Control dimension. */
  nu: number;
  /** Planning horizon (number of steps). */
  horizon: number;
}

/** Time-varying LQR result with gain and cost-to-go at each step. */
export interface TimeVaryingLQRResult {
  /** Gain matrices K_t, one per time step. */
  Ks: Float64Array[];
  /** Cost-to-go matrices P_t, one per time step. */
  Ps: Float64Array[];
}

/** Tracking LQR configuration with output reference. */
export interface TrackingLQRConfig extends LQRConfig {
  /** Output matrix (ny x nx, row-major). */
  C: Float64Array;
  /** Output dimension. */
  ny: number;
}

/** Tracking LQR result with augmented and feedforward gains. */
export interface TrackingLQRResult extends LQRResult {
  /** Augmented gain (nu x (nx + ny), row-major). */
  Kaug: Float64Array;
  /** Feedforward gain (nu x ny, row-major). */
  Kff: Float64Array;
}

/** Linear-Quadratic-Gaussian configuration (LQR + Kalman filter). */
export interface LQGConfig extends LQRConfig {
  /** Output/observation matrix (ny x nx, row-major). */
  C: Float64Array;
  /** Process noise covariance (nx x nx, row-major). */
  Qn: Float64Array;
  /** Measurement noise covariance (ny x ny, row-major). */
  Rn: Float64Array;
  /** Output dimension. */
  ny: number;
}

/** LQG result combining optimal regulator and estimator. */
export interface LQGResult {
  /** LQR solution. */
  lqr: LQRResult;
  /** Kalman filter gain (nx x ny, row-major). */
  L: Float64Array;
  /** Filter error covariance (nx x nx, row-major). */
  Pf: Float64Array;
}

/** Running state for the LQG estimator. */
export interface LQGState {
  /** State estimate (nx). */
  xHat: Float64Array;
  /** Estimation error covariance (nx x nx, row-major). */
  P: Float64Array;
}

// ---------------------------------------------------------------------------
// OC-2: Model Predictive Control (MPC)
// ---------------------------------------------------------------------------

/** Standard linear MPC configuration. */
export interface MPCConfig {
  /** State transition matrix (nx x nx, row-major). */
  A: Float64Array;
  /** Input matrix (nx x nu, row-major). */
  B: Float64Array;
  /** State cost matrix (nx x nx, row-major). */
  Q: Float64Array;
  /** Control cost matrix (nu x nu, row-major). */
  R: Float64Array;
  /** Terminal cost matrix (nx x nx, row-major, optional). */
  Qf?: Float64Array;
  /** State dimension. */
  nx: number;
  /** Control dimension. */
  nu: number;
  /** Prediction horizon (number of steps). */
  horizon: number;
  /** Lower bound on control inputs (nu, optional). */
  uMin?: Float64Array;
  /** Upper bound on control inputs (nu, optional). */
  uMax?: Float64Array;
  /** Lower bound on states (nx, optional). */
  xMin?: Float64Array;
  /** Upper bound on states (nx, optional). */
  xMax?: Float64Array;
  /** Max control rate of change (nu, optional). */
  duMax?: Float64Array;
}

/** MPC solution result. */
export interface MPCResult {
  /** Optimal first control action (nu). */
  uOptimal: Float64Array;
  /** Full optimal control sequence over horizon. */
  uSequence: Float64Array[];
  /** Predicted state trajectory over horizon. */
  xPredicted: Float64Array[];
  /** Optimal cost value. */
  cost: number;
  /** Number of solver iterations. */
  iterations: number;
  /** Solver status. */
  status: 'optimal' | 'infeasible' | 'max_iter';
}

/** Economic MPC with demand-driven objective. */
export interface EconomicMPCConfig extends MPCConfig {
  /** Demand function: maps (price, state) to demand vector. */
  demandFn: (price: Float64Array, state: Float64Array) => Float64Array;
  /** Cost function: maps (state, control) to scalar running cost. */
  costFn: (state: Float64Array, control: Float64Array) => number;
}

/** Nonlinear MPC with general dynamics and costs. */
export interface NonlinearMPCConfig extends MPCConfig {
  /** Nonlinear dynamics: x_{k+1} = f(x_k, u_k). */
  dynamicsFn: (x: Float64Array, u: Float64Array) => Float64Array;
  /** Stage cost l(x, u). */
  costStageFn: (x: Float64Array, u: Float64Array) => number;
  /** Terminal cost V_f(x). */
  costTerminalFn: (x: Float64Array) => number;
  /** Max sequential quadratic programming iterations. */
  maxSQPIterations: number;
  /** Convergence tolerance. */
  tolerance: number;
}

/** Tube MPC for robust control under bounded disturbances. */
export interface TubeMPCConfig extends MPCConfig {
  /** Bound on additive disturbance (nx). */
  disturbanceBound: Float64Array;
  /** Ancillary controller gain (nu x nx, row-major). */
  tubeK: Float64Array;
}

/** Stochastic MPC with scenario-based optimization. */
export interface StochasticMPCConfig extends MPCConfig {
  /** Number of scenarios to consider. */
  nScenarios: number;
  /** Acceptable constraint violation probability. */
  chanceConstraintEpsilon: number;
}

/** A polyhedral region in explicit MPC lookup table. */
export interface ExplicitMPCRegion {
  /** Half-space matrix H_x (nConstraints x nx, row-major). */
  Hx: Float64Array;
  /** Half-space vector h_x (nConstraints). */
  hx: Float64Array;
  /** Affine feedback matrix F_x (nu x nx, row-major). */
  Fx: Float64Array;
  /** Affine feedback offset g_x (nu). */
  gx: Float64Array;
  /** Number of half-space constraints defining this region. */
  nConstraints: number;
}

/** Lookup table of explicit MPC regions. */
export interface ExplicitMPCTable {
  /** Polyhedral critical regions. */
  regions: ExplicitMPCRegion[];
  /** State dimension. */
  nx: number;
  /** Control dimension. */
  nu: number;
}

// ---------------------------------------------------------------------------
// OC-3: Pontryagin's Maximum Principle (PMP)
// ---------------------------------------------------------------------------

/** Hamiltonian system configuration for optimal control. */
export interface HamiltonianConfig {
  /** State dynamics: dx/dt = f(x, u, lambda, t). */
  stateDynamics: (x: Float64Array, u: Float64Array, lambda: Float64Array, t: number) => Float64Array;
  /** Costate dynamics: dlambda/dt = -dH/dx. */
  costateDynamics: (x: Float64Array, u: Float64Array, lambda: Float64Array, t: number) => Float64Array;
  /** Running cost: L(x, u, t). */
  runningCost: (x: Float64Array, u: Float64Array, t: number) => number;
  /** Control optimality condition: u*(x, lambda, t) from dH/du = 0. */
  controlOptimality: (x: Float64Array, lambda: Float64Array, t: number) => Float64Array;
  /** State dimension. */
  nx: number;
  /** Control dimension. */
  nu: number;
}

/** Shooting method configuration for boundary value problems. */
export interface ShootingConfig extends HamiltonianConfig {
  /** Initial state (nx). */
  x0: Float64Array;
  /** Terminal time. */
  T: number;
  /** Number of integration steps. */
  nSteps: number;
  /** Convergence tolerance for boundary conditions. */
  tolerance: number;
  /** Maximum shooting iterations. */
  maxIter: number;
}

/** Result from a shooting method solve. */
export interface ShootingResult {
  /** State trajectory at each time step. */
  xTrajectory: Float64Array[];
  /** Optimal control trajectory. */
  uTrajectory: Float64Array[];
  /** Costate trajectory. */
  lambdaTrajectory: Float64Array[];
  /** Total cost. */
  cost: number;
  /** Whether the solver converged. */
  converged: boolean;
  /** Number of iterations used. */
  iterations: number;
}

/** Direct collocation configuration. */
export interface CollocationConfig extends HamiltonianConfig {
  /** Initial state (nx). */
  x0: Float64Array;
  /** Terminal time. */
  T: number;
  /** Number of collocation segments. */
  nSegments: number;
  /** Polynomial order per segment (e.g., 3 for cubic). */
  collocationOrder: number;
  /** Convergence tolerance. */
  tolerance: number;
  /** Maximum iterations. */
  maxIter: number;
}

/** Result from bang-bang control analysis. */
export interface BangBangResult {
  /** Times at which control switches. */
  switchingTimes: Float64Array;
  /** Control level between each switch. */
  controlLevels: Float64Array;
  /** Switching function values over time. */
  switchingFunction: Float64Array;
}

// ---------------------------------------------------------------------------
// OC-4: Dynamic Programming
// ---------------------------------------------------------------------------

/** Discrete Bellman equation configuration. */
export interface BellmanConfig {
  /** Number of discrete states. */
  nStates: number;
  /** Number of discrete actions. */
  nActions: number;
  /** Transition model: returns list of (nextState, probability) for (s, a). */
  transitions: (s: number, a: number) => Array<{ nextState: number; probability: number }>;
  /** Reward function: r(s, a). */
  reward: (s: number, a: number) => number;
  /** Discount factor gamma in [0, 1]. */
  discount: number;
  /** Finite horizon (undefined = infinite horizon). */
  horizon?: number;
}

/** Result of value iteration / policy iteration. */
export interface BellmanResult {
  /** Optimal value function (nStates). */
  valueFunction: Float64Array;
  /** Optimal policy (nStates). */
  policy: Int32Array;
  /** Number of iterations to converge. */
  iterations: number;
}

/** Hamilton-Jacobi-Bellman PDE configuration for continuous state spaces. */
export interface HJBConfig {
  /** Lower bound of computational grid per dimension (nx). */
  gridMin: Float64Array;
  /** Upper bound of computational grid per dimension (nx). */
  gridMax: Float64Array;
  /** Number of grid points per dimension (nx). */
  gridN: Int32Array;
  /** Time step for PDE integration. */
  dt: number;
  /** Continuous dynamics: dx/dt = f(x, u). */
  dynamics: (x: Float64Array, u: Float64Array) => Float64Array;
  /** Running cost: l(x, u). */
  runningCost: (x: Float64Array, u: Float64Array) => number;
  /** Terminal cost: phi(x). */
  terminalCost: (x: Float64Array) => number;
  /** Discrete set of admissible controls. */
  controlSet: Float64Array[];
  /** Terminal time. */
  T: number;
  /** State dimension. */
  nx: number;
}

/** HJB PDE solution result. */
export interface HJBResult {
  /** Value function on grid (flattened, row-major). */
  valueGrid: Float64Array;
  /** Optimal control index on grid (flattened). */
  policyGrid: Int32Array;
  /** Grid dimensions. */
  gridDims: Int32Array;
}

/** Approximate dynamic programming with basis function approximation. */
export interface ApproxDPConfig {
  /** Number of basis features. */
  nFeatures: number;
  /** Number of sample transitions per iteration. */
  nSamples: number;
  /** Feature function: phi(s) -> feature vector. */
  featureFn: (s: Float64Array) => Float64Array;
  /** Sample next state: s' ~ P(. | s, a). */
  transitionSample: (s: Float64Array, a: Float64Array, rng: PRNG) => Float64Array;
  /** Reward function: r(s, a). */
  rewardFn: (s: Float64Array, a: Float64Array) => number;
  /** Discount factor. */
  discount: number;
  /** Max iterations. */
  maxIter: number;
  /** Convergence tolerance. */
  tolerance: number;
}

/** Approximate DP result. */
export interface ApproxDPResult {
  /** Learned weight vector (nFeatures). */
  weights: Float64Array;
  /** Number of iterations. */
  iterations: number;
  /** Whether weights converged. */
  converged: boolean;
}

/** Bid-price control for revenue management. */
export interface BidPriceConfig {
  /** Number of resource types. */
  nResources: number;
  /** Number of product types. */
  nProducts: number;
  /** Capacity per resource (nResources). */
  resourceCapacities: Float64Array;
  /** Incidence matrix (nResources x nProducts, row-major): which resources each product uses. */
  incidenceMatrix: Float64Array;
  /** Revenue per product (nProducts). */
  revenues: Float64Array;
  /** Mean demand per product (nProducts). */
  demandMeans: Float64Array;
}

/** Bid-price control result. */
export interface BidPriceResult {
  /** Dual prices / bid prices per resource (nResources). */
  bidPrices: Float64Array;
  /** Optimal allocations per product (nProducts). */
  allocations: Float64Array;
  /** Optimal expected revenue. */
  optimalRevenue: number;
}

/** Optimal stopping configuration (e.g., secretary problem variants). */
export interface OptimalStoppingConfig {
  /** Capacity (how many arrivals can be accepted). */
  capacity: number;
  /** Decision horizon (total periods). */
  horizon: number;
  /** Arrival probability at time t. */
  arrivalProb: (t: number) => number;
  /** Value distribution sample at time t. */
  valueDistribution: (t: number, rng: PRNG) => number;
  /** Discount factor per period. */
  discount: number;
}

/** Optimal stopping result. */
export interface OptimalStoppingResult {
  /** Acceptance thresholds per period (horizon). */
  thresholds: Float64Array;
  /** Value function per period (horizon). */
  valueFunction: Float64Array;
}

// ---------------------------------------------------------------------------
// OC-5: Reinforcement Learning
// ---------------------------------------------------------------------------

/** Base RL configuration. */
export interface RLConfig {
  /** State dimension. */
  stateDim: number;
  /** Action dimension. */
  actionDim: number;
  /** Hidden layer dimension for neural networks. */
  hiddenDim: number;
  /** Learning rate. */
  lr: number;
  /** Discount factor. */
  gamma: number;
  /** Mini-batch size. */
  batchSize: number;
}

/** Soft Actor-Critic configuration. */
export interface SACConfig extends RLConfig {
  /** Soft target update rate. */
  tau: number;
  /** Entropy coefficient (temperature). */
  alphaEntropy: number;
  /** Replay buffer capacity. */
  replayCapacity: number;
}

/** Proximal Policy Optimization configuration. */
export interface PPOConfig extends RLConfig {
  /** Clipping parameter for surrogate objective. */
  clipEpsilon: number;
  /** Number of optimization epochs per batch. */
  epochsPerBatch: number;
  /** GAE lambda for advantage estimation. */
  gaeLambda: number;
  /** Entropy bonus coefficient. */
  entropyCoeff: number;
}

/** Offline RL configuration (learning from fixed datasets). */
export interface OfflineRLConfig extends RLConfig {
  /** Offline RL algorithm variant. */
  method: 'cql' | 'iql' | 'dt';
  /** CQL conservatism alpha (for method='cql'). */
  cqlAlpha?: number;
  /** IQL expectile tau (for method='iql'). */
  iqlTau?: number;
  /** Decision Transformer sequence length (for method='dt'). */
  dtSeqLen?: number;
}

/** Safe / constrained RL configuration. */
export interface SafeRLConfig extends RLConfig {
  /** Number of constraint dimensions. */
  constraintDim: number;
  /** Per-constraint violation thresholds (constraintDim). */
  constraintThresholds: Float64Array;
  /** Lagrangian multiplier learning rate. */
  lagrangianLR: number;
}

/** Reward shaping weights for venue operations. */
export interface RewardConfig {
  /** Weight on revenue objective. */
  revenueWeight: number;
  /** Penalty for exceeding safe occupancy. */
  overcrowdingPenalty: number;
  /** Safe occupancy level (density threshold). */
  overcrowdingSafe: number;
  /** Penalty for price instability. */
  priceStabilityPenalty: number;
  /** Penalty for queue length. */
  queuePenalty: number;
  /** Maximum acceptable queue. */
  queueMax: number;
  /** Penalty for customer churn. */
  churnPenalty: number;
}

/** MLP weight storage for inference without a framework. */
export interface MLPWeights {
  /** Layer parameters from input to output. */
  layers: Array<{
    /** Weight matrix (outDim x inDim, row-major). */
    weight: Float64Array;
    /** Bias vector (outDim). */
    bias: Float64Array;
    /** Input dimension. */
    inDim: number;
    /** Output dimension. */
    outDim: number;
  }>;
  /** Activation function between layers. */
  activation: 'relu' | 'tanh';
}

/** Experience replay buffer for off-policy RL. */
export interface ReplayBuffer {
  /** Stored state vectors. */
  states: Float64Array[];
  /** Stored action vectors. */
  actions: Float64Array[];
  /** Stored next-state vectors. */
  nextStates: Float64Array[];
  /** Stored scalar rewards. */
  rewards: Float64Array;
  /** Terminal flags (1 = done, 0 = not done). */
  dones: Uint8Array;
  /** Current number of stored transitions. */
  size: number;
  /** Maximum buffer capacity. */
  capacity: number;
  /** Circular write pointer. */
  ptr: number;
}

// ---------------------------------------------------------------------------
// OC-6: Nonlinear Control
// ---------------------------------------------------------------------------

/** Feedback linearization configuration. */
export interface FeedbackLinConfig {
  /** Drift vector field f(x). */
  f: (x: Float64Array) => Float64Array;
  /** Input vector field g(x). */
  g: (x: Float64Array) => Float64Array;
  /** Output function h(x). */
  h: (x: Float64Array) => number;
  /** State dimension. */
  nx: number;
  /** Relative degree of the system. */
  relativeDegree: number;
}

/** Feedback linearization result. */
export interface FeedbackLinResult {
  /** Computed control law: u = alpha(x) + beta(x) * v. */
  controlLaw: (x: Float64Array, v: number) => number;
  /** Lie derivatives L_f^k h(x) for each order. */
  lieDerivatives: Float64Array[];
  /** Verified relative degree. */
  relativeDegree: number;
}

/** Backstepping controller configuration. */
export interface BacksteppingConfig {
  /** Number of integrator stages. */
  nStages: number;
  /** Stage dynamics: dx_i/dt = f_i(x, u). */
  dynamics: Array<(x: Float64Array, u: number) => number>;
  /** Lyapunov-based gain for each stage (nStages). */
  lyapunovGains: Float64Array;
}

/** Sliding mode control configuration. */
export interface SlidingModeConfig {
  /** Sliding surface function: sigma(x). */
  slidingSurface: (x: Float64Array) => number;
  /** Gradient of sliding surface: d(sigma)/dx. */
  surfaceGradient: (x: Float64Array) => Float64Array;
  /** Reaching gain for sign(sigma) term. */
  reachingGain: number;
  /** Boundary layer width for chattering reduction. */
  boundaryLayerWidth: number;
  /** State dimension. */
  nx: number;
}

/** Adaptive control configuration. */
export interface AdaptiveControlConfig {
  /** Reference model matrices. */
  modelRef: {
    /** State matrix (nx x nx, row-major). */
    A: Float64Array;
    /** Input matrix (nx x nu, row-major). */
    B: Float64Array;
    /** State dimension. */
    nx: number;
    /** Control dimension. */
    nu: number;
  };
  /** Adaptation gain matrix (nParams x nParams, row-major). */
  adaptationGain: Float64Array;
  /** Adaptive control method. */
  method: 'mrac' | 'l1';
}

/** Running state for an adaptive controller. */
export interface AdaptiveControlState {
  /** Current parameter estimate. */
  thetaHat: Float64Array;
  /** Reference model state. */
  xRef: Float64Array;
}

// ---------------------------------------------------------------------------
// OC-7: Multi-Agent Control
// ---------------------------------------------------------------------------

/** Decentralized MPC with coupled constraints. */
export interface DecentralizedMPCConfig {
  /** Number of agents. */
  nAgents: number;
  /** Per-agent MPC configurations. */
  agentConfigs: MPCConfig[];
  /** Coupling matrix encoding inter-agent constraints (nAgents x nAgents, row-major). */
  couplingMatrix: Float64Array;
  /** ADMM consensus penalty weight. */
  consensusWeight: number;
  /** Max ADMM consensus iterations. */
  maxConsensusIter: number;
}

/** Nash equilibrium computation configuration. */
export interface NashEquilibriumConfig {
  /** Number of players. */
  nPlayers: number;
  /** Payoff function for each player: (all actions) -> scalar. */
  payoffFns: Array<(actions: Float64Array[]) => number>;
  /** Action dimension per player. */
  actionDims: number[];
  /** Action bounds per player. */
  actionBounds: Array<{ min: Float64Array; max: Float64Array }>;
  /** Convergence tolerance. */
  tolerance: number;
  /** Max iterations. */
  maxIter: number;
}

/** Nash equilibrium result. */
export interface NashResult {
  /** Equilibrium action profile. */
  equilibriumActions: Float64Array[];
  /** Payoff at equilibrium for each player (nPlayers). */
  payoffs: Float64Array;
  /** Whether the algorithm converged. */
  converged: boolean;
  /** Number of iterations. */
  iterations: number;
}

/** Stackelberg game configuration with leader-follower structure. */
export interface StackelbergConfig {
  /** Leader's payoff function. */
  leaderPayoff: (leaderAction: Float64Array, followerActions: Float64Array[]) => number;
  /** Follower payoff functions. */
  followerPayoffs: Array<(leaderAction: Float64Array, followerAction: Float64Array) => number>;
  /** Number of followers. */
  nFollowers: number;
  /** Dimension of leader's action space. */
  leaderActionDim: number;
  /** Dimension of each follower's action space. */
  followerActionDim: number;
}

/** Mean-field game configuration for large-population approximations. */
export interface MeanFieldConfig {
  /** Approximate number of agents (for discretization). */
  nAgentsApprox: number;
  /** Agent dynamics: dx/dt = f(x, u, distribution). */
  agentDynamics: (x: Float64Array, u: Float64Array, distribution: Float64Array) => Float64Array;
  /** Per-agent cost function. */
  costFn: (x: Float64Array, u: Float64Array, distribution: Float64Array) => number;
  /** State dimension. */
  nx: number;
  /** Control dimension. */
  nu: number;
}

/** Multi-Agent PPO configuration. */
export interface MAPPOConfig {
  /** Number of agents. */
  nAgents: number;
  /** Observation dimension per agent. */
  obsPerAgent: number;
  /** Action dimension per agent. */
  actionPerAgent: number;
  /** Centralized critic input dimension. */
  centralStateDim: number;
  /** Critic hidden layer dimension. */
  criticHiddenDim: number;
  /** Actor hidden layer dimension. */
  actorHiddenDim: number;
  /** Learning rate. */
  lr: number;
  /** PPO clip epsilon. */
  clipEpsilon: number;
}

/** QMIX mixing network configuration. */
export interface QMIXConfig {
  /** Number of agents. */
  nAgents: number;
  /** Observation dimension per agent. */
  obsPerAgent: number;
  /** Number of discrete actions per agent. */
  nActions: number;
  /** Hidden dimension of the mixing network. */
  mixingHiddenDim: number;
  /** Hidden dimension of each agent's Q-network. */
  agentHiddenDim: number;
}

// ---------------------------------------------------------------------------
// OC-8: Crowd Control
// ---------------------------------------------------------------------------

/** Hughes model for macroscopic crowd flow on a grid. */
export interface HughesModelConfig {
  /** Grid points in x. */
  gridNx: number;
  /** Grid points in y. */
  gridNy: number;
  /** Grid spacing (meters). */
  dx: number;
  /** Maximum pedestrian density (ped/m^2). */
  maxDensity: number;
  /** Speed-density relationship. */
  speedFn: (density: number) => number;
  /** Exit locations on the grid. */
  exits: Array<{ x: number; y: number }>;
  /** Time step (seconds). */
  dt: number;
}

/** Hughes model state at a given time. */
export interface HughesModelState {
  /** Density field (gridNx * gridNy, row-major). */
  density: Float64Array;
  /** Eikonal potential field (gridNx * gridNy). */
  potential: Float64Array;
  /** Velocity field x-component (gridNx * gridNy). */
  velocityX: Float64Array;
  /** Velocity field y-component (gridNx * gridNy). */
  velocityY: Float64Array;
}

/** Social force model configuration for microscopic pedestrian simulation. */
export interface SocialForceConfig {
  /** Desired walking speed (m/s). */
  desiredSpeed: number;
  /** Relaxation time to desired velocity (s). */
  relaxationTime: number;
  /** Magnitude of social repulsion force (N). */
  socialMagnitude: number;
  /** Exponential range of social force (m). */
  socialRange: number;
  /** Magnitude of wall repulsion force (N). */
  wallMagnitude: number;
  /** Exponential range of wall force (m). */
  wallRange: number;
  /** Simulation time step (s). */
  dt: number;
}

/** State of a single pedestrian agent. */
export interface SocialForceAgent {
  /** Position x (m). */
  x: number;
  /** Position y (m). */
  y: number;
  /** Velocity x (m/s). */
  vx: number;
  /** Velocity y (m/s). */
  vy: number;
  /** Goal position x (m). */
  goalX: number;
  /** Goal position y (m). */
  goalY: number;
  /** Body radius (m). */
  radius: number;
}

/** Evacuation MPC configuration for zone-to-exit routing. */
export interface EvacuationMPCConfig {
  /** Number of building zones. */
  nZones: number;
  /** Number of exits. */
  nExits: number;
  /** Capacity per zone (nZones). */
  zoneCapacities: Float64Array;
  /** Capacity per exit (nExits). */
  exitCapacities: Float64Array;
  /** Travel time matrix (nZones x nExits, row-major). */
  travelTimes: Float64Array;
  /** MPC prediction horizon (steps). */
  horizon: number;
}

/** Density constraint specification for crowd control. */
export interface DensityConstraint {
  /** Maximum allowable density (ped/m^2). */
  maxDensity: number;
  /** Constraint enforcement method. */
  enforcementMethod: 'hard' | 'penalty';
  /** Penalty weight (only used when enforcementMethod = 'penalty'). */
  penaltyWeight: number;
}

// ---------------------------------------------------------------------------
// OC-9: Stochastic Optimal Control
// ---------------------------------------------------------------------------

/** Risk-sensitive control configuration (exponential cost). */
export interface RiskSensitiveConfig {
  /** Risk sensitivity parameter (theta > 0 = risk-averse, theta < 0 = risk-seeking). */
  theta: number;
  /** Base cost function c(x, u). */
  baseCostFn: (state: Float64Array, action: Float64Array) => number;
}

/** Conditional Value-at-Risk configuration. */
export interface CVaRConfig {
  /** Confidence level alpha in (0, 1). */
  alpha: number;
  /** Number of scenarios for sample approximation. */
  nScenarios: number;
  /** Scenario probabilities (nScenarios, optional — uniform if omitted). */
  scenarioProbs?: Float64Array;
}

/** CVaR optimization result. */
export interface CVaRResult {
  /** Conditional Value-at-Risk. */
  cvar: number;
  /** Value-at-Risk (alpha-quantile). */
  var: number;
  /** Optimal action. */
  optimalAction: Float64Array;
  /** Cost under each scenario (nScenarios). */
  scenarioCosts: Float64Array;
}

/** Distributionally robust optimization configuration. */
export interface DROConfig {
  /** Number of observed samples. */
  nSamples: number;
  /** Wasserstein ball radius. */
  epsilon: number;
  /** Cost function: c(action, scenario). */
  costFn: (action: Float64Array, scenario: Float64Array) => number;
}

/** Chance constraint configuration. */
export interface ChanceConstraintConfig {
  /** Maximum acceptable constraint violation probability. */
  violationProb: number;
  /** Number of scenarios for sample approximation. */
  nScenarios: number;
  /** Constraint function: g(state, action, scenario) <= 0 for feasibility. */
  constraintFn: (state: Float64Array, action: Float64Array, scenario: Float64Array) => number;
}

/** Node in a scenario tree for multi-stage stochastic programming. */
export interface ScenarioTreeNode {
  /** State at this node (nx). */
  state: Float64Array;
  /** Probability of reaching this node. */
  probability: number;
  /** Indices of child nodes. */
  children: number[];
  /** Index of parent node (-1 for root). */
  parent: number;
  /** Stage (time step) of this node. */
  stage: number;
}

/** Scenario-based MPC configuration. */
export interface ScenarioMPCConfig extends MPCConfig {
  /** Scenario tree structure. */
  scenarioTree: ScenarioTreeNode[];
  /** Total number of scenarios (leaf paths). */
  nScenarios: number;
  /** Number of decision stages. */
  nStages: number;
}

// ---------------------------------------------------------------------------
// OC-10: Experiment Design / Dual Control
// ---------------------------------------------------------------------------

/** Dual control configuration for simultaneous learning and control. */
export interface DualControlConfig {
  /** Prior mean of uncertain parameters (nx). */
  priorMean: Float64Array;
  /** Prior covariance of uncertain parameters (nx x nx, row-major). */
  priorCov: Float64Array;
  /** Parameter dimension. */
  nx: number;
  /** Weight on exploration vs exploitation. */
  explorationWeight: number;
}

/** Dual control result. */
export interface DualControlResult {
  /** Selected action. */
  action: Float64Array;
  /** Exploration component of the action. */
  explorationComponent: Float64Array;
  /** Exploitation component of the action. */
  exploitationComponent: Float64Array;
  /** Information gain from this action. */
  informationGain: number;
}

/** Information-Directed Sampling configuration. */
export interface IDSConfig {
  /** Number of arms / actions. */
  nArms: number;
  /** Beta prior shape alpha per arm (nArms). */
  priorAlpha: Float64Array;
  /** Beta prior shape beta per arm (nArms). */
  priorBeta: Float64Array;
}

/** Information-Directed Sampling result. */
export interface IDSResult {
  /** Sampling probabilities per arm (nArms). */
  armProbabilities: Float64Array;
  /** Expected regret per arm (nArms). */
  expectedRegret: Float64Array;
  /** Information gain per arm (nArms). */
  informationGain: Float64Array;
  /** IDS ratio (regret^2 / information) per arm (nArms). */
  idsRatio: Float64Array;
}

/** Active learning MPC configuration. */
export interface ActiveLearningMPCConfig extends MPCConfig {
  /** Weight on information value in the MPC objective. */
  informationValueWeight: number;
  /** Current uncertainty in model parameters (nParams). */
  parameterUncertainty: Float64Array;
}

/** Thompson sampling for dynamic pricing. */
export interface ThompsonPricingConfig {
  /** Discrete price grid (nPrices). */
  priceGrid: Float64Array;
  /** Beta prior alpha per price point (nPrices). */
  priorAlpha: Float64Array;
  /** Beta prior beta per price point (nPrices). */
  priorBeta: Float64Array;
  /** Allowable price range. */
  priceBounds: { min: number; max: number };
  /** Maximum price change per period. */
  rateLimit: number;
}

/** Safe Bayesian optimization configuration. */
export interface SafeBOConfig {
  /** Input bounds per dimension. */
  bounds: Array<{ min: number; max: number }>;
  /** Safety threshold: f(x) >= threshold is safe. */
  safetyThreshold: number;
  /** GP kernel lengthscale. */
  kernelLengthscale: number;
  /** GP kernel variance (signal variance). */
  kernelVariance: number;
  /** GP observation noise variance. */
  noiseVariance: number;
  /** UCB exploration parameter. */
  beta: number;
}

/** Safe Bayesian optimization result. */
export interface SafeBOResult {
  /** Next evaluation point. */
  nextPoint: Float64Array;
  /** Expected improvement at the point. */
  expectedImprovement: number;
  /** Probability that the point is safe. */
  safetyProbability: number;
  /** Whether the point meets the safety threshold. */
  isSafe: boolean;
}

// ---------------------------------------------------------------------------
// OC-11: Control Architecture
// ---------------------------------------------------------------------------

/** Hierarchical control loop timing configuration. */
export interface ControlLoopConfig {
  /** Sensor read period (ms). */
  sensingPeriodMs: number;
  /** State estimation period (ms). */
  estimationPeriodMs: number;
  /** Decision / optimization period (ms). */
  decisionPeriodMs: number;
  /** Actuation / command publish period (ms). */
  actuationPeriodMs: number;
}

/** Running state of the control loop. */
export interface ControlLoopState {
  /** Current wall-clock timestamp (ms). */
  timestamp: number;
  /** Latest sensor readings keyed by sensor name. */
  sensorReadings: Map<string, Float64Array>;
  /** Current state estimate (nx). */
  stateEstimate: Float64Array;
  /** Current control action being applied (nu). */
  currentAction: Float64Array;
  /** Previous control action (nu). */
  previousAction: Float64Array;
}

/** Sample rate configuration for different control subsystems. */
export interface SampleRateConfig {
  /** Pricing update rate (seconds). */
  pricing: number;
  /** Staffing adjustment rate (seconds). */
  staffing: number;
  /** Marketing decision rate (seconds). */
  marketing: number;
  /** Crowd operations rate (seconds). */
  crowdOps: number;
}

/** Multi-sensor fusion / estimation configuration. */
export interface MultiSensorEstimateConfig {
  /** Sensor descriptions. */
  sensors: Array<{
    /** Sensor identifier. */
    name: string;
    /** Observation matrix H_i (dimZ x nx, row-major). */
    H: Float64Array;
    /** Measurement noise covariance R_i (dimZ x dimZ, row-major). */
    R: Float64Array;
    /** Measurement dimension. */
    dimZ: number;
  }>;
  /** State transition matrix F (nx x nx, row-major). */
  F: Float64Array;
  /** Process noise covariance Q (nx x nx, row-major). */
  Q: Float64Array;
  /** State dimension. */
  nx: number;
}

/** Fault tolerance / graceful degradation configuration. */
export interface FaultToleranceConfig {
  /** Sensor data timeout before declaring fault (ms). */
  sensorTimeoutMs: number;
  /** Fallback policy on sensor fault. */
  fallbackPolicy: 'hold' | 'safe' | 'degraded';
  /** Max consecutive missed readings before triggering fallback. */
  maxMissedReadings: number;
  /** Degraded-mode controller gains (nu x nx, row-major). */
  degradedGains: Float64Array;
}

// ---------------------------------------------------------------------------
// OC-12: Edge Deployment
// ---------------------------------------------------------------------------

/** Simple linear feedback controller for edge devices. */
export interface LinearFeedbackConfig {
  /** Gain matrix K (nu x nx, row-major). */
  K: Float64Array;
  /** State dimension. */
  nx: number;
  /** Control dimension. */
  nu: number;
  /** Lower bound on control (nu, optional). */
  uMin?: Float64Array;
  /** Upper bound on control (nu, optional). */
  uMax?: Float64Array;
}

/** Explicit MPC lookup for edge deployment. */
export interface ExplicitMPCLookup {
  /** Pre-computed critical region table. */
  table: ExplicitMPCTable;
}

/** Neural network policy for edge inference. */
export interface NeuralPolicyConfig {
  /** Pre-trained MLP weights. */
  weights: MLPWeights;
  /** State normalization mean (stateDim). */
  stateNormMean: Float64Array;
  /** State normalization std dev (stateDim). */
  stateNormStd: Float64Array;
  /** Action scale (actionDim). */
  actionScale: Float64Array;
  /** Action bias (actionDim). */
  actionBias: Float64Array;
}

/** Constraint checker for safety enforcement on edge. */
export interface ConstraintCheckerConfig {
  /** Lower bound on control (nu). */
  uMin: Float64Array;
  /** Upper bound on control (nu). */
  uMax: Float64Array;
  /** Lower bound on state (nx). */
  xMin: Float64Array;
  /** Upper bound on state (nx). */
  xMax: Float64Array;
  /** Max control rate of change (nu, optional). */
  duMax?: Float64Array;
}

/** Result of constraint checking and action clipping. */
export interface ConstraintCheckResult {
  /** Whether the original action was feasible. */
  feasible: boolean;
  /** Action after clipping to constraints. */
  clippedAction: Float64Array;
  /** Human-readable descriptions of violations. */
  violations: string[];
}
