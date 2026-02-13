// ---------------------------------------------------------------------------
// OC-3: Hamiltonian Evaluation & ODE Integration (Pontryagin's Maximum Principle)
// ---------------------------------------------------------------------------

import type { HamiltonianConfig } from '../types.js';
import { vecAdd, vecScale, vecDot, vecClone } from '../types.js';

// ---------------------------------------------------------------------------
// Hamiltonian evaluation
// ---------------------------------------------------------------------------

/**
 * Evaluate the Hamiltonian H = L(x, u, t) + lambda^T f(x, u, lambda, t).
 *
 * @param config  Hamiltonian system configuration
 * @param x       State vector (nx)
 * @param u       Control vector (nu)
 * @param lambda  Costate vector (nx)
 * @param t       Current time
 * @returns       Scalar Hamiltonian value
 */
export function evaluateHamiltonian(
  config: HamiltonianConfig,
  x: Float64Array,
  u: Float64Array,
  lambda: Float64Array,
  t: number,
): number {
  const L = config.runningCost(x, u, t);
  const f = config.stateDynamics(x, u, lambda, t);
  const lambdaTf = vecDot(lambda, f);
  return L + lambdaTf;
}

// ---------------------------------------------------------------------------
// RK4 single step
// ---------------------------------------------------------------------------

/**
 * Perform a single classic 4th-order Runge-Kutta step.
 *
 * @param f   Right-hand side dy/dt = f(y, t)
 * @param y   Current state vector
 * @param t   Current time
 * @param dt  Time step size
 * @returns   State vector at t + dt
 */
export function rk4Step(
  f: (y: Float64Array, t: number) => Float64Array,
  y: Float64Array,
  t: number,
  dt: number,
): Float64Array {
  const k1 = f(y, t);
  const k2 = f(vecAdd(y, vecScale(k1, dt / 2)), t + dt / 2);
  const k3 = f(vecAdd(y, vecScale(k2, dt / 2)), t + dt / 2);
  const k4 = f(vecAdd(y, vecScale(k3, dt)), t + dt);

  // y_{n+1} = y_n + (dt/6)(k1 + 2*k2 + 2*k3 + k4)
  const sum = vecAdd(
    vecAdd(k1, vecScale(k2, 2)),
    vecAdd(vecScale(k3, 2), k4),
  );
  return vecAdd(y, vecScale(sum, dt / 6));
}

// ---------------------------------------------------------------------------
// ODE integration over an interval
// ---------------------------------------------------------------------------

/**
 * Integrate an ODE system dy/dt = f(y, t) from t0 to t1 using nSteps RK4 steps.
 *
 * @param f       Right-hand side function
 * @param y0      Initial state vector
 * @param t0      Start time
 * @param t1      End time
 * @param nSteps  Number of RK4 steps
 * @returns       Trajectory array of length nSteps + 1 (includes initial state)
 */
export function integrateODE(
  f: (y: Float64Array, t: number) => Float64Array,
  y0: Float64Array,
  t0: number,
  t1: number,
  nSteps: number,
): Float64Array[] {
  const dt = (t1 - t0) / nSteps;
  const trajectory: Float64Array[] = [vecClone(y0)];

  let y = vecClone(y0);
  let t = t0;

  for (let step = 0; step < nSteps; step++) {
    y = rk4Step(f, y, t, dt);
    t = t0 + (step + 1) * dt;
    trajectory.push(vecClone(y));
  }

  return trajectory;
}
