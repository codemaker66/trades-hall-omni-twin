// ---------------------------------------------------------------------------
// OC-12  Edge/Browser Deployment -- Explicit MPC Lookup
// ---------------------------------------------------------------------------

import type { ExplicitMPCTable } from '../types.js';

// ---------------------------------------------------------------------------
// isInsidePolytope
// ---------------------------------------------------------------------------

/**
 * Check whether a state vector x lies inside a polyhedral region defined by
 * the half-space representation H x <= h.
 *
 * @param H             Half-space matrix (nConstraints x nx, row-major flat).
 * @param h             Half-space RHS vector (nConstraints).
 * @param x             State vector (nx).
 * @param nConstraints  Number of half-space constraints.
 * @param nx            State dimension.
 * @returns             True if x satisfies all constraints Hx <= h.
 */
export function isInsidePolytope(
  H: Float64Array,
  h: Float64Array,
  x: Float64Array,
  nConstraints: number,
  nx: number,
): boolean {
  for (let i = 0; i < nConstraints; i++) {
    let dot = 0;
    for (let j = 0; j < nx; j++) {
      dot += H[i * nx + j]! * x[j]!;
    }
    // Allow a small numerical tolerance
    if (dot > h[i]! + 1e-12) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// explicitMPCQuery
// ---------------------------------------------------------------------------

/**
 * Query an explicit MPC lookup table to find the optimal control for a given
 * state. Iterates over the pre-computed critical regions and returns the
 * affine control law u = F x + g for the first region containing x.
 *
 * @param table  Explicit MPC table with polyhedral critical regions.
 * @param x      Current state vector (nx).
 * @returns      Optimal control vector (nu), or null if no region contains x.
 */
export function explicitMPCQuery(
  table: ExplicitMPCTable,
  x: Float64Array,
): Float64Array | null {
  const { regions, nx, nu } = table;

  for (let r = 0; r < regions.length; r++) {
    const region = regions[r]!;

    if (isInsidePolytope(region.Hx, region.hx, x, region.nConstraints, nx)) {
      // Compute u = F_x * x + g_x
      const u = new Float64Array(nu);
      for (let i = 0; i < nu; i++) {
        let sum = 0;
        for (let j = 0; j < nx; j++) {
          sum += region.Fx[i * nx + j]! * x[j]!;
        }
        u[i] = sum + region.gx[i]!;
      }
      return u;
    }
  }

  return null;
}
