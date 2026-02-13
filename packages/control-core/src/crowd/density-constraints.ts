// ---------------------------------------------------------------------------
// OC-8  Crowd Flow & Evacuation -- Density Constraints
// ---------------------------------------------------------------------------

import type { DensityConstraint } from '../types.js';

// ---------------------------------------------------------------------------
// Constraint Checking
// ---------------------------------------------------------------------------

/**
 * Check whether a density field satisfies a set of density constraints.
 *
 * For each cell i, the effective density is density[i] / areas[i].
 * A constraint is violated if the effective density exceeds `maxDensity`
 * for any of the provided constraints.
 *
 * Returns:
 *   feasible:   true if no violations exist
 *   violations: indices of cells where a constraint is violated
 */
export function checkDensityConstraints(
  density: Float64Array,
  constraints: DensityConstraint[],
  areas: Float64Array,
): { feasible: boolean; violations: number[] } {
  const violations: number[] = [];
  const n = density.length;

  for (let i = 0; i < n; i++) {
    const area = areas[i]!;
    if (area <= 1e-12) continue;

    const effectiveDensity = density[i]! / area;

    for (let c = 0; c < constraints.length; c++) {
      const constraint = constraints[c]!;

      if (effectiveDensity > constraint.maxDensity) {
        violations.push(i);
        break; // only record cell once even if it violates multiple constraints
      }
    }
  }

  return { feasible: violations.length === 0, violations };
}

// ---------------------------------------------------------------------------
// Density Projection
// ---------------------------------------------------------------------------

/**
 * Project a density field so that no cell exceeds `maxDensity`.
 *
 * Excess density is clamped and redistributed to immediate neighbours
 * via a single diffusion pass.  If neighbours are also at capacity the
 * excess is simply removed (mass is not conserved in pathological cases).
 *
 * The input is treated as a 1-D array.  For 2-D grids the caller should
 * provide the total number of cells; neighbour connectivity assumes a
 * 1-D chain (left/right neighbours).  For a full 2-D projection, call
 * with the grid flattened in row-major order and apply this function to
 * each row, or use the Hughes model's built-in conservation step.
 *
 * Returns a new Float64Array (does not mutate the input).
 */
export function projectDensity(
  density: Float64Array,
  maxDensity: number,
): Float64Array {
  const n = density.length;
  const result = new Float64Array(density);

  // First pass: identify cells that exceed the cap and accumulate excess
  const excess = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    if (result[i]! > maxDensity) {
      excess[i] = result[i]! - maxDensity;
      result[i] = maxDensity;
    }
  }

  // Second pass: redistribute excess to neighbours (simple diffusion)
  for (let i = 0; i < n; i++) {
    const ex = excess[i]!;
    if (ex <= 1e-12) continue;

    // Count available neighbours (those below maxDensity)
    let nNeighbours = 0;
    const headroom: number[] = [];

    if (i > 0) {
      const hr = maxDensity - result[i - 1]!;
      if (hr > 1e-12) {
        nNeighbours++;
        headroom.push(hr);
      } else {
        headroom.push(0);
      }
    } else {
      headroom.push(0);
    }

    if (i < n - 1) {
      const hr = maxDensity - result[i + 1]!;
      if (hr > 1e-12) {
        nNeighbours++;
        headroom.push(hr);
      } else {
        headroom.push(0);
      }
    } else {
      headroom.push(0);
    }

    if (nNeighbours === 0) continue;

    // Distribute excess equally, but clamped by each neighbour's headroom
    const share = ex / nNeighbours;

    if (i > 0 && headroom[0]! > 1e-12) {
      const transfer = Math.min(share, headroom[0]!);
      result[i - 1] = result[i - 1]! + transfer;
    }

    if (i < n - 1 && headroom[1]! > 1e-12) {
      const transfer = Math.min(share, headroom[1]!);
      result[i + 1] = result[i + 1]! + transfer;
    }
  }

  // Final clamp: ensure nothing crept above maxDensity due to double-redistribution
  for (let i = 0; i < n; i++) {
    if (result[i]! > maxDensity) {
      result[i] = maxDensity;
    }
    if (result[i]! < 0) {
      result[i] = 0;
    }
  }

  return result;
}
