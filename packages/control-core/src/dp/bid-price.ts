// ---------------------------------------------------------------------------
// OC-4  Dynamic Programming -- Bid-Price Control for Revenue Management
// ---------------------------------------------------------------------------

import type { BidPriceConfig, BidPriceResult } from '../types.js';

// ---------------------------------------------------------------------------
// simplexLP
// ---------------------------------------------------------------------------

/**
 * Solve a small linear program via the simplex method with slack variables:
 *
 *   max  c^T x
 *   s.t. A x <= b,  x >= 0
 *
 * @param c   Objective coefficients (nVars).
 * @param A   Constraint matrix (nConstraints x nVars, row-major Float64Array).
 * @param b   Right-hand side (nConstraints).  Must be non-negative for a
 *            feasible starting basis of slack variables.
 * @param nVars  Number of decision variables.
 * @param nConstraints  Number of inequality constraints.
 * @returns  Optimal x (nVars) or `null` if the LP is unbounded.
 */
export function simplexLP(
  c: Float64Array,
  A: Float64Array,
  b: Float64Array,
  nVars: number,
  nConstraints: number,
): Float64Array | null {
  // Tableau layout (nConstraints+1 rows x totalCols cols):
  //   constraint row i:  [ A_i | I_i | b_i ]
  //   objective  row  :  [ -c  |  0  |  0  ]
  const totalVars = nVars + nConstraints; // original + slack
  const totalCols = totalVars + 1;        // + RHS column
  const nRows = nConstraints + 1;         // constraints + objective row
  const objRow = nConstraints;

  const tab = new Float64Array(nRows * totalCols);

  // Fill constraint rows
  for (let i = 0; i < nConstraints; i++) {
    for (let j = 0; j < nVars; j++) {
      tab[i * totalCols + j] = A[i * nVars + j]!;
    }
    // Slack variable (identity column)
    tab[i * totalCols + nVars + i] = 1;
    // RHS
    tab[i * totalCols + totalVars] = b[i]!;
  }

  // Objective row: negate c for maximisation tableau convention
  for (let j = 0; j < nVars; j++) {
    tab[objRow * totalCols + j] = -c[j]!;
  }

  // Basis tracking: initially all slacks are basic
  const basis = new Int32Array(nConstraints);
  for (let i = 0; i < nConstraints; i++) {
    basis[i] = nVars + i;
  }

  // Pivoting
  const maxPivots = 10 * (nVars + nConstraints);
  for (let iter = 0; iter < maxPivots; iter++) {
    // --- Entering variable: most negative reduced cost in objective row ---
    let pivotCol = -1;
    let minVal = -1e-12;
    for (let j = 0; j < totalVars; j++) {
      const rc = tab[objRow * totalCols + j]!;
      if (rc < minVal) {
        minVal = rc;
        pivotCol = j;
      }
    }
    if (pivotCol === -1) break; // optimal

    // --- Leaving variable: minimum ratio test ---
    let pivotRow = -1;
    let minRatio = Infinity;
    for (let i = 0; i < nConstraints; i++) {
      const aij = tab[i * totalCols + pivotCol]!;
      if (aij > 1e-12) {
        const ratio = tab[i * totalCols + totalVars]! / aij;
        if (ratio < minRatio) {
          minRatio = ratio;
          pivotRow = i;
        }
      }
    }
    if (pivotRow === -1) return null; // unbounded

    // --- Pivot operation ---
    const pivotElem = tab[pivotRow * totalCols + pivotCol]!;

    // Scale pivot row
    for (let j = 0; j < totalCols; j++) {
      tab[pivotRow * totalCols + j] = tab[pivotRow * totalCols + j]! / pivotElem;
    }

    // Eliminate pivot column from all other rows
    for (let i = 0; i < nRows; i++) {
      if (i === pivotRow) continue;
      const factor = tab[i * totalCols + pivotCol]!;
      if (Math.abs(factor) < 1e-15) continue;
      for (let j = 0; j < totalCols; j++) {
        tab[i * totalCols + j] = tab[i * totalCols + j]! - factor * tab[pivotRow * totalCols + j]!;
      }
    }

    basis[pivotRow] = pivotCol;
  }

  // Extract primal solution
  const x = new Float64Array(nVars);
  for (let i = 0; i < nConstraints; i++) {
    const bv = basis[i]!;
    if (bv < nVars) {
      x[bv] = tab[i * totalCols + totalVars]!;
    }
  }

  return x;
}

// ---------------------------------------------------------------------------
// solveBidPriceLP
// ---------------------------------------------------------------------------

/**
 * Solve the deterministic LP relaxation for network revenue management
 * and extract bid prices (shadow prices on resource capacity constraints).
 *
 * Primal LP:
 *
 *   max  revenue^T x
 *   s.t. incidence * x <= capacities   (resource constraints)
 *        0 <= x <= demandMeans          (demand upper bounds)
 *
 * Bid prices are estimated via finite-difference sensitivity analysis on
 * the resource capacity constraints.
 */
export function solveBidPriceLP(config: BidPriceConfig): BidPriceResult {
  const {
    nResources,
    nProducts,
    resourceCapacities,
    incidenceMatrix,
    revenues,
    demandMeans,
  } = config;

  // Constraints: nResources capacity + nProducts demand bounds
  const nConstraints = nResources + nProducts;

  // Build constraint matrix A (nConstraints x nProducts) and RHS b
  const Amat = new Float64Array(nConstraints * nProducts);
  const bvec = new Float64Array(nConstraints);

  // Resource capacity rows: incidence * x <= capacities
  for (let i = 0; i < nResources; i++) {
    for (let j = 0; j < nProducts; j++) {
      Amat[i * nProducts + j] = incidenceMatrix[i * nProducts + j]!;
    }
    bvec[i] = resourceCapacities[i]!;
  }

  // Demand bound rows: x_j <= d_j
  for (let j = 0; j < nProducts; j++) {
    Amat[(nResources + j) * nProducts + j] = 1;
    bvec[nResources + j] = demandMeans[j]!;
  }

  // Solve primal
  const allocations = simplexLP(revenues, Amat, bvec, nProducts, nConstraints);

  if (allocations === null) {
    return {
      bidPrices: new Float64Array(nResources),
      allocations: new Float64Array(nProducts),
      optimalRevenue: 0,
    };
  }

  // Base objective
  let baseRevenue = 0;
  for (let j = 0; j < nProducts; j++) {
    baseRevenue += revenues[j]! * allocations[j]!;
  }

  // Bid prices via finite-difference perturbation of each resource capacity
  const bidPrices = new Float64Array(nResources);
  const eps = 1e-4;

  for (let i = 0; i < nResources; i++) {
    const bPerturbed = new Float64Array(bvec);
    bPerturbed[i] = bvec[i]! + eps;

    const pertAlloc = simplexLP(revenues, Amat, bPerturbed, nProducts, nConstraints);
    if (pertAlloc !== null) {
      let pertRev = 0;
      for (let j = 0; j < nProducts; j++) {
        pertRev += revenues[j]! * pertAlloc[j]!;
      }
      bidPrices[i] = Math.max(0, (pertRev - baseRevenue) / eps);
    }
  }

  return { bidPrices, allocations, optimalRevenue: baseRevenue };
}

// ---------------------------------------------------------------------------
// bidPriceAcceptReject
// ---------------------------------------------------------------------------

/**
 * Accept/reject decision for a single booking request using bid prices.
 *
 * Accept iff the offered revenue meets or exceeds the displacement cost:
 *
 *   accept  <=>  revenue >= sum_i { bidPrices[i] * incidence[i] }
 *
 * @param bidPrices   Dual values per resource (nResources).
 * @param incidence   Resource consumption vector for this product (nResources).
 * @param revenue     Revenue from accepting this request.
 * @param nResources  Number of resources.
 */
export function bidPriceAcceptReject(
  bidPrices: Float64Array,
  incidence: Float64Array,
  revenue: number,
  nResources: number,
): boolean {
  let displacementCost = 0;
  for (let i = 0; i < nResources; i++) {
    displacementCost += bidPrices[i]! * incidence[i]!;
  }
  return revenue >= displacementCost;
}
