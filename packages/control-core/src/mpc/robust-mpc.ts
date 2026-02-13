// ---------------------------------------------------------------------------
// OC-2: Tube-based Robust Model Predictive Control
// ---------------------------------------------------------------------------
//
// Tube MPC handles bounded additive disturbances by:
//   1. Computing a Robust Positively Invariant (RPI) tube around the nominal
//      trajectory, which bounds the deviation due to worst-case disturbances.
//   2. Tightening the original constraints by the tube size so that the
//      nominal trajectory (without disturbances) satisfies tighter constraints.
//   3. Solving a standard (nominal) MPC on the tightened problem.
//   4. Applying an ancillary feedback controller K to reject the actual
//      deviation from the nominal trajectory.
//
// The control law is: u = u_nom + K (x - x_nom)
// ---------------------------------------------------------------------------

import type { MPCConfig, MPCResult, TubeMPCConfig } from '../types.js';
import {
  arrayToMatrix,
  matGet,
  matMul,
  matSub,
  matVecMul,
  vecAdd,
  vecClone,
  vecNorm,
  vecSub,
} from '../types.js';

import { solveLinearMPC } from './linear-mpc.js';

// ---------------------------------------------------------------------------
// Invariant tube computation
// ---------------------------------------------------------------------------

/**
 * Approximate the minimal Robust Positively Invariant (mRPI) set as a
 * scaled disturbance bound under the closed-loop dynamics (A - B K).
 *
 * The RPI set is the smallest set Z such that (A-BK) Z + W subset Z,
 * where W = {w : |w_i| <= distBound_i}. For box disturbances and
 * stable (A-BK), the mRPI can be approximated by summing the
 * propagation of the disturbance bound through the closed-loop system:
 *
 *   tube_i = sum_{k=0}^{M} |(A-BK)^k|_abs * distBound
 *
 * where M is chosen so that the series has approximately converged.
 *
 * @param A           State transition matrix (nx x nx, row-major)
 * @param B           Input matrix (nx x nu, row-major)
 * @param K           Ancillary controller gain (nu x nx, row-major)
 * @param distBound   Element-wise disturbance bound (nx)
 * @param nx          State dimension
 * @param nu          Control dimension
 * @returns tube      Element-wise tube half-width (nx)
 */
export function computeInvariantTube(
  A: Float64Array,
  B: Float64Array,
  K: Float64Array,
  distBound: Float64Array,
  nx: number,
  nu: number,
): Float64Array {
  const mA = arrayToMatrix(A, nx, nx);
  const mB = arrayToMatrix(B, nx, nu);
  const mK = arrayToMatrix(K, nu, nx);

  // Closed-loop matrix: Acl = A - B K
  const Acl = matSub(mA, matMul(mB, mK));

  // Approximate mRPI via geometric series of |Acl^k| * distBound
  // We iterate until the contribution is negligible or up to maxIter.
  const maxIter = 50;
  const convergenceTol = 1e-10;

  const tube = new Float64Array(nx);

  // Current power matrix |Acl|^k (element-wise absolute values)
  // Start with identity (k=0)
  const absPow = new Float64Array(nx * nx);
  for (let i = 0; i < nx; i++) {
    absPow[i * nx + i] = 1.0;
  }

  for (let k = 0; k < maxIter; k++) {
    // Contribution at step k: |Acl^k| * distBound
    const contrib = new Float64Array(nx);
    for (let i = 0; i < nx; i++) {
      let sum = 0;
      for (let j = 0; j < nx; j++) {
        sum += absPow[i * nx + j]! * distBound[j]!;
      }
      contrib[i] = sum;
    }

    // Accumulate
    let maxContrib = 0;
    for (let i = 0; i < nx; i++) {
      tube[i] = tube[i]! + contrib[i]!;
      if (contrib[i]! > maxContrib) {
        maxContrib = contrib[i]!;
      }
    }

    // Check convergence
    if (maxContrib < convergenceTol) {
      break;
    }

    // Update power: absPow = |absPow * Acl| (element-wise abs)
    const newPow = new Float64Array(nx * nx);
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < nx; j++) {
        let sum = 0;
        for (let p = 0; p < nx; p++) {
          sum += absPow[i * nx + p]! * Math.abs(matGet(Acl, p, j));
        }
        newPow[i * nx + j] = sum;
      }
    }
    absPow.set(newPow);
  }

  return tube;
}

// ---------------------------------------------------------------------------
// Tube MPC solver
// ---------------------------------------------------------------------------

/**
 * Solve a tube-based robust MPC problem.
 *
 * The approach:
 *   1. Compute the invariant tube (RPI set approximation)
 *   2. Tighten state and control constraints by the tube size
 *   3. Solve the nominal (tightened) MPC problem
 *   4. Apply ancillary feedback: u = u_nom + K (x - x_nom)
 *
 * @param config  Tube MPC configuration (extends MPCConfig with disturbanceBound and tubeK)
 * @param x0      Current state (nx)
 * @returns MPCResult with the robust control action
 */
export function solveTubeMPC(
  config: TubeMPCConfig,
  x0: Float64Array,
): MPCResult {
  const { nx, nu, horizon, disturbanceBound, tubeK } = config;

  // Step 1: Compute invariant tube
  const tube = computeInvariantTube(
    config.A,
    config.B,
    tubeK,
    disturbanceBound,
    nx,
    nu,
  );

  // Step 2: Tighten constraints
  // Tightened state bounds: [xMin + tube, xMax - tube]
  let xMinTight: Float64Array | undefined;
  let xMaxTight: Float64Array | undefined;
  if (config.xMin) {
    xMinTight = new Float64Array(nx);
    for (let i = 0; i < nx; i++) {
      xMinTight[i] = config.xMin[i]! + tube[i]!;
    }
  }
  if (config.xMax) {
    xMaxTight = new Float64Array(nx);
    for (let i = 0; i < nx; i++) {
      xMaxTight[i] = config.xMax[i]! - tube[i]!;
    }
  }

  // Tightened control bounds: account for ancillary controller range
  // u = u_nom + K * e, where |e_i| <= tube_i
  // => |K * e|_i <= sum_j |K_{i,j}| * tube_j
  const mK = arrayToMatrix(tubeK, nu, nx);
  const controlTube = new Float64Array(nu);
  for (let i = 0; i < nu; i++) {
    let sum = 0;
    for (let j = 0; j < nx; j++) {
      sum += Math.abs(matGet(mK, i, j)) * tube[j]!;
    }
    controlTube[i] = sum;
  }

  let uMinTight: Float64Array | undefined;
  let uMaxTight: Float64Array | undefined;
  if (config.uMin) {
    uMinTight = new Float64Array(nu);
    for (let i = 0; i < nu; i++) {
      uMinTight[i] = config.uMin[i]! + controlTube[i]!;
    }
  }
  if (config.uMax) {
    uMaxTight = new Float64Array(nu);
    for (let i = 0; i < nu; i++) {
      uMaxTight[i] = config.uMax[i]! - controlTube[i]!;
    }
  }

  // Step 3: Solve nominal MPC with tightened constraints
  const nominalConfig: MPCConfig = {
    A: config.A,
    B: config.B,
    Q: config.Q,
    R: config.R,
    Qf: config.Qf,
    nx,
    nu,
    horizon,
    uMin: uMinTight,
    uMax: uMaxTight,
    xMin: xMinTight,
    xMax: xMaxTight,
    duMax: config.duMax,
  };

  const nominalResult = solveLinearMPC(nominalConfig, x0);

  // Step 4: Apply ancillary feedback
  // The nominal trajectory starts at x0 (same as actual), so for the
  // first step the deviation is zero and u = u_nom.
  // However, in a real implementation, x0_nominal might differ from x0_actual
  // due to past disturbances. Here we compute the correction:
  //   u = u_nom + K * (x0 - x0_nominal)
  // Since we solve from x0, the nominal x0 equals x0, so correction = 0
  // at the first step. The correction becomes relevant when this is called
  // in a receding-horizon loop.

  const x0Nom = nominalResult.xPredicted[0]!;
  const deviation = vecSub(x0, x0Nom);
  const correction = matVecMul(mK, deviation);

  const uRobust = vecAdd(nominalResult.uOptimal, correction);

  // Clamp to original (untightened) bounds
  for (let i = 0; i < nu; i++) {
    if (config.uMin) {
      uRobust[i] = Math.max(uRobust[i]!, config.uMin[i]!);
    }
    if (config.uMax) {
      uRobust[i] = Math.min(uRobust[i]!, config.uMax[i]!);
    }
  }

  return {
    uOptimal: uRobust,
    uSequence: nominalResult.uSequence,
    xPredicted: nominalResult.xPredicted,
    cost: nominalResult.cost,
    iterations: nominalResult.iterations,
    status: nominalResult.status,
  };
}
