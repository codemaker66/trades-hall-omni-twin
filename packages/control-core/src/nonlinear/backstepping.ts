// ---------------------------------------------------------------------------
// OC-6: Backstepping Control for Strict-Feedback Systems
// ---------------------------------------------------------------------------

import type { BacksteppingConfig } from '../types.js';

// ---------------------------------------------------------------------------
// Backstepping Control
// ---------------------------------------------------------------------------

/**
 * Compute the backstepping control input for a strict-feedback system.
 *
 * Strict-feedback form:
 *   dx_1/dt = f_1(x_1) + x_2
 *   dx_2/dt = f_2(x_1, x_2) + x_3
 *   ...
 *   dx_n/dt = f_n(x_1, ..., x_n) + u
 *
 * The algorithm recursively designs virtual controls alpha_i at each stage,
 * stabilising each subsystem using Lyapunov gains. The final stage yields
 * the actual control input u.
 *
 * Stage i:
 *   z_i = x_i - alpha_{i-1}       (tracking error for virtual control)
 *   alpha_i = -k_i * z_i + dalpha_{i-1}/dt - f_i(x, 0)
 *
 * For stage 1:
 *   z_1 = x_1
 *   alpha_1 = -k_1 * z_1 - f_1(x, 0)
 *
 * The actual control u is the output of the final stage.
 *
 * @param config  Backstepping configuration
 * @param x       Current state vector (nStages)
 * @returns       Actual control input u
 */
export function backsteppingControl(
  config: BacksteppingConfig,
  x: Float64Array,
): number {
  const { nStages, dynamics, lyapunovGains } = config;

  // Tracking errors z_i and virtual controls alpha_i
  const z = new Float64Array(nStages);
  const alpha = new Float64Array(nStages);

  // Stage 1: z_1 = x_1, alpha_1 = -k_1 * x_1 - f_1(x, 0)
  z[0] = x[0]!;
  const f1 = dynamics[0]!(x, 0);
  alpha[0] = -lyapunovGains[0]! * z[0]! - f1;

  // Stages 2 through nStages
  for (let i = 1; i < nStages; i++) {
    // Tracking error: z_i = x_i - alpha_{i-1}
    z[i] = x[i]! - alpha[i - 1]!;

    // Stage dynamics f_i(x, 0)
    const fi = dynamics[i]!(x, 0);

    // Approximate dalpha_{i-1}/dx numerically to get dalpha_{i-1}/dt
    // dalpha_{i-1}/dt = sum_j (dalpha_{i-1}/dx_j) * dx_j/dt
    // For simplicity, we use the recursive structure:
    // alpha_i = -k_i * z_i - f_i + dalpha_{i-1}/dt
    // where dalpha_{i-1}/dt is estimated from the dynamics of previous states.

    // Compute dalpha_{i-1}/dt by chain rule through previous stage errors.
    // For a strict-feedback system, dalpha_{i-1}/dt depends on
    // all state derivatives dx_1/dt ... dx_{i-1}/dt.
    // We approximate dalpha_{i-1}/dt using the known dynamics and the
    // previous virtual control.
    let dAlphaDt = 0;
    for (let j = 0; j < i; j++) {
      // dx_j/dt = f_j(x, 0) + x_{j+1} (for strict-feedback form)
      const fj = dynamics[j]!(x, 0);
      const xNext = j + 1 < nStages ? x[j + 1]! : 0;
      const dxjDt = fj + xNext;

      // dalpha_{i-1}/dx_j: numerical partial derivative
      const eps = 1e-7;
      const xp = new Float64Array(x);
      const xm = new Float64Array(x);
      xp[j] = x[j]! + eps;
      xm[j] = x[j]! - eps;

      // Recompute alpha_{i-1} with perturbed states
      const alphaPlus = computeAlphaAtStage(config, xp, i - 1);
      const alphaMinus = computeAlphaAtStage(config, xm, i - 1);
      const dAlphaDxj = (alphaPlus - alphaMinus) / (2 * eps);

      dAlphaDt += dAlphaDxj * dxjDt;
    }

    // Virtual control for stage i
    // Ensures dV_i/dt < 0 by choosing:
    //   alpha_i = -k_i * z_i - f_i + dalpha_{i-1}/dt - z_{i-1}
    // The -z_{i-1} term couples this stage's CLF with the previous one.
    alpha[i] = -lyapunovGains[i]! * z[i]! - fi + dAlphaDt - z[i - 1]!;
  }

  // The actual control is the virtual control from the last stage
  // u = alpha_{nStages-1} + f_{nStages-1}(x, 0) + adjustment
  // In the strict-feedback design, the last alpha IS the control.
  return alpha[nStages - 1]!;
}

/**
 * Recompute the virtual control alpha at a specific stage for a given state.
 * Used internally by backsteppingControl for numerical differentiation.
 *
 * @param config  Backstepping configuration
 * @param x       State vector
 * @param stage   Stage index (0-based)
 * @returns       Virtual control alpha at the given stage
 */
function computeAlphaAtStage(
  config: BacksteppingConfig,
  x: Float64Array,
  stage: number,
): number {
  const { dynamics, lyapunovGains } = config;

  const z = new Float64Array(stage + 1);
  const alpha = new Float64Array(stage + 1);

  // Stage 1
  z[0] = x[0]!;
  const f1 = dynamics[0]!(x, 0);
  alpha[0] = -lyapunovGains[0]! * z[0]! - f1;

  // Subsequent stages up to the requested one
  for (let i = 1; i <= stage; i++) {
    z[i] = x[i]! - alpha[i - 1]!;
    const fi = dynamics[i]!(x, 0);

    // Simplified computation (without full chain rule) for partial
    // derivative purposes. This avoids infinite recursion in the
    // numerical differentiation and is consistent as the outer loop
    // handles the full chain rule.
    alpha[i] = -lyapunovGains[i]! * z[i]! - fi - z[i - 1]!;
  }

  return alpha[stage]!;
}

// ---------------------------------------------------------------------------
// Backstepping Lyapunov Function
// ---------------------------------------------------------------------------

/**
 * Evaluate the composite Lyapunov function for the backstepping design.
 *
 * V(x) = sum_{i=1}^{n} (1/2) * z_i^2
 *
 * where z_i is the tracking error at stage i. A positive-definite V
 * that decreases along trajectories certifies stability.
 *
 * @param config  Backstepping configuration
 * @param x       Current state vector
 * @returns       Value of the composite CLF V(x)
 */
export function backsteppingLyapunov(
  config: BacksteppingConfig,
  x: Float64Array,
): number {
  const { nStages, dynamics, lyapunovGains } = config;

  const z = new Float64Array(nStages);
  const alpha = new Float64Array(nStages);

  // Stage 1
  z[0] = x[0]!;
  const f1 = dynamics[0]!(x, 0);
  alpha[0] = -lyapunovGains[0]! * z[0]! - f1;

  // Subsequent stages
  for (let i = 1; i < nStages; i++) {
    z[i] = x[i]! - alpha[i - 1]!;
    const fi = dynamics[i]!(x, 0);
    alpha[i] = -lyapunovGains[i]! * z[i]! - fi - z[i - 1]!;
  }

  // V = sum (1/2) z_i^2
  let V = 0;
  for (let i = 0; i < nStages; i++) {
    V += 0.5 * z[i]! * z[i]!;
  }
  return V;
}
