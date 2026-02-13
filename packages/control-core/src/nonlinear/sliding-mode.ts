// ---------------------------------------------------------------------------
// OC-6: Sliding Mode Control
// ---------------------------------------------------------------------------

import type { SlidingModeConfig } from '../types.js';
import { vecClone, vecDot } from '../types.js';

// ---------------------------------------------------------------------------
// Saturation Function
// ---------------------------------------------------------------------------

/**
 * Saturation function: sat(s) = clamp(s, -1, 1).
 *
 * Used in place of sign(s) to reduce chattering in the boundary layer.
 */
function sat(s: number): number {
  if (s > 1) return 1;
  if (s < -1) return -1;
  return s;
}

// ---------------------------------------------------------------------------
// Evaluate Sliding Surface
// ---------------------------------------------------------------------------

/**
 * Evaluate the sliding surface sigma(x) at the current state.
 *
 * @param config  Sliding mode configuration
 * @param x       Current state vector
 * @returns       Scalar value of the sliding surface sigma(x)
 */
export function evaluateSurface(
  config: SlidingModeConfig,
  x: Float64Array,
): number {
  return config.slidingSurface(x);
}

// ---------------------------------------------------------------------------
// Sliding Mode Control Law
// ---------------------------------------------------------------------------

/**
 * Compute the sliding mode control input.
 *
 * The control is composed of two parts:
 *
 * 1. Equivalent control u_eq: keeps the state on the sliding surface
 *    u_eq = -(dσ/dx . g(x))^{-1} * (dσ/dx . f(x))
 *
 * 2. Switching control u_sw: drives the state toward the sliding surface
 *    u_sw = -eta * sat(σ / φ)
 *
 * where:
 *   σ = sigma(x) is the sliding surface value
 *   dσ/dx is the surface gradient
 *   eta is the reaching gain
 *   φ is the boundary layer width (for chattering reduction)
 *   sat is the saturation function
 *
 * Total control: u = u_eq + u_sw
 *
 * @param config  Sliding mode configuration
 * @param f       Drift dynamics f(x)
 * @param g       Input dynamics g(x)
 * @param x       Current state vector
 * @returns       Scalar control input u
 */
export function slidingModeControl(
  config: SlidingModeConfig,
  f: (x: Float64Array) => Float64Array,
  g: (x: Float64Array) => Float64Array,
  x: Float64Array,
): number {
  const { surfaceGradient, reachingGain, boundaryLayerWidth } = config;

  // Evaluate surface and its gradient
  const sigma = evaluateSurface(config, x);
  const dSigmaDx = surfaceGradient(x);

  // Evaluate dynamics at current state
  const fVal = f(x);
  const gVal = g(x);

  // dσ/dx . f(x)
  const dSigmaF = vecDot(dSigmaDx, fVal);

  // dσ/dx . g(x)
  const dSigmaG = vecDot(dSigmaDx, gVal);

  // Guard: if dσ/dx . g(x) is near zero, the sliding surface and input
  // are not properly matched. Fall back to switching-only control.
  let uEq = 0;
  if (Math.abs(dSigmaG) > 1e-12) {
    uEq = -dSigmaF / dSigmaG;
  }

  // Switching control with boundary layer for chattering reduction
  const phi = boundaryLayerWidth > 0 ? boundaryLayerWidth : 1e-6;
  const uSw = -reachingGain * sat(sigma / phi);

  return uEq + uSw;
}

// ---------------------------------------------------------------------------
// Reaching Condition Check
// ---------------------------------------------------------------------------

/**
 * Check the reaching condition: sigma * (d sigma/dt) < 0.
 *
 * This ensures the state is moving toward the sliding surface.
 *
 * d sigma/dt = dσ/dx . (f(x) + g(x) * u)
 *
 * @param config  Sliding mode configuration
 * @param f       Drift dynamics f(x)
 * @param g       Input dynamics g(x)
 * @param x       Current state vector
 * @param u       Applied control input
 * @returns       True if the reaching condition is satisfied
 */
export function checkReachingCondition(
  config: SlidingModeConfig,
  f: (x: Float64Array) => Float64Array,
  g: (x: Float64Array) => Float64Array,
  x: Float64Array,
  u: number,
): boolean {
  const { surfaceGradient } = config;

  const sigma = evaluateSurface(config, x);
  const dSigmaDx = surfaceGradient(x);
  const fVal = f(x);
  const gVal = g(x);

  // dx/dt = f(x) + g(x) * u
  const nx = x.length;
  const xDot = vecClone(fVal);
  for (let i = 0; i < nx; i++) {
    xDot[i] = xDot[i]! + gVal[i]! * u;
  }

  // d sigma/dt = dσ/dx . dx/dt
  const sigmaDot = vecDot(dSigmaDx, xDot);

  // Reaching condition: sigma * sigmaDot < 0
  // (or sigma is already approximately zero)
  if (Math.abs(sigma) < 1e-12) {
    return true; // Already on the surface
  }

  return sigma * sigmaDot < 0;
}
