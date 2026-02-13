// ---------------------------------------------------------------------------
// OC-12: Constraint Checker for Safety Enforcement on Edge
// ---------------------------------------------------------------------------

import type {
  ConstraintCheckerConfig,
  ConstraintCheckResult,
} from '../types.js';

/**
 * Element-wise clamp a control vector to [uMin, uMax].
 *
 * @param u     Control vector (nu).
 * @param uMin  Lower bounds (nu).
 * @param uMax  Upper bounds (nu).
 * @returns     Clamped control vector (nu).
 */
export function clipAction(
  u: Float64Array,
  uMin: Float64Array,
  uMax: Float64Array,
): Float64Array {
  const clipped = new Float64Array(u.length);
  for (let i = 0; i < u.length; i++) {
    const lo = uMin[i]!;
    const hi = uMax[i]!;
    const val = u[i]!;
    if (val < lo) {
      clipped[i] = lo;
    } else if (val > hi) {
      clipped[i] = hi;
    } else {
      clipped[i] = val;
    }
  }
  return clipped;
}

/**
 * Check and enforce a rate-of-change constraint on the control action.
 * Clamps du = u - uPrev to [-duMax, duMax] and returns the adjusted u.
 *
 * @param u      Proposed control vector (nu).
 * @param uPrev  Previous control vector (nu).
 * @param duMax  Maximum absolute change per element (nu).
 * @returns      Rate-constrained control vector (nu).
 */
export function checkRateConstraint(
  u: Float64Array,
  uPrev: Float64Array,
  duMax: Float64Array,
): Float64Array {
  const adjusted = new Float64Array(u.length);
  for (let i = 0; i < u.length; i++) {
    const du = u[i]! - uPrev[i]!;
    const maxDu = duMax[i]!;
    if (du > maxDu) {
      adjusted[i] = uPrev[i]! + maxDu;
    } else if (du < -maxDu) {
      adjusted[i] = uPrev[i]! - maxDu;
    } else {
      adjusted[i] = u[i]!;
    }
  }
  return adjusted;
}

/**
 * Run all constraint checks on a proposed control action and state, collecting
 * violation descriptions and returning a clipped (feasible) action.
 *
 * Checks performed:
 *   1. State bounds: x in [xMin, xMax] (informational only, not clipped)
 *   2. Action bounds: u clamped to [uMin, uMax]
 *   3. Rate constraint: du clamped to [-duMax, duMax] (if uPrev and duMax provided)
 *
 * @param config  Constraint checker configuration with bounds.
 * @param u       Proposed control action (nu).
 * @param x       Current state vector (nx).
 * @param uPrev   Previous control action (nu, optional for rate constraint).
 * @returns       Result with feasibility flag, clipped action, and violations.
 */
export function checkConstraints(
  config: ConstraintCheckerConfig,
  u: Float64Array,
  x: Float64Array,
  uPrev?: Float64Array,
): ConstraintCheckResult {
  const { uMin, uMax, xMin, xMax, duMax } = config;
  const violations: string[] = [];

  // 1. Check state bounds (informational)
  for (let i = 0; i < x.length; i++) {
    const xi = x[i]!;
    if (xi < xMin[i]!) {
      violations.push(
        `State x[${i}] = ${xi.toFixed(4)} below lower bound ${xMin[i]!.toFixed(4)}`,
      );
    }
    if (xi > xMax[i]!) {
      violations.push(
        `State x[${i}] = ${xi.toFixed(4)} above upper bound ${xMax[i]!.toFixed(4)}`,
      );
    }
  }

  // 2. Clip action to bounds
  let clipped = clipAction(u, uMin, uMax);

  // Check if any action element was clipped
  for (let i = 0; i < u.length; i++) {
    const original = u[i]!;
    if (original < uMin[i]!) {
      violations.push(
        `Action u[${i}] = ${original.toFixed(4)} clipped to lower bound ${uMin[i]!.toFixed(4)}`,
      );
    }
    if (original > uMax[i]!) {
      violations.push(
        `Action u[${i}] = ${original.toFixed(4)} clipped to upper bound ${uMax[i]!.toFixed(4)}`,
      );
    }
  }

  // 3. Rate constraint (if applicable)
  if (uPrev !== undefined && duMax !== undefined) {
    const beforeRate = clipped;
    clipped = checkRateConstraint(clipped, uPrev, duMax);

    for (let i = 0; i < clipped.length; i++) {
      if (beforeRate[i]! !== clipped[i]!) {
        const du = beforeRate[i]! - uPrev[i]!;
        violations.push(
          `Action u[${i}] rate |du| = ${Math.abs(du).toFixed(4)} exceeds duMax = ${duMax[i]!.toFixed(4)}`,
        );
      }
    }
  }

  return {
    feasible: violations.length === 0,
    clippedAction: clipped,
    violations,
  };
}
