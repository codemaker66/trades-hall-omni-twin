// ---------------------------------------------------------------------------
// OC-2: Explicit Model Predictive Control
// ---------------------------------------------------------------------------
//
// Explicit MPC pre-computes the optimal control law as a piecewise affine
// function of the state. The state space is partitioned into polyhedral
// critical regions, each with an associated affine feedback law:
//
//   u(x) = F_r x + g_r   if H_r x <= h_r  (region r)
//
// At runtime, the lookup simply tests which region contains the current
// state and evaluates the affine law -- no QP solve needed. This is
// suitable for small problems deployed on edge devices with limited
// computation.
//
// The buildExplicitMPCTable function constructs the table by solving
// the MPC problem at a grid of sample points and fitting affine laws
// to each region (simplified multi-parametric QP approximation).
// ---------------------------------------------------------------------------

import type {
  ExplicitMPCRegion,
  ExplicitMPCTable,
  MPCConfig,
  MPCResult,
} from '../types.js';
import {
  arrayToMatrix,
  createMatrix,
  matGet,
  matInvert,
  matMul,
  matSet,
  matTranspose,
  matVecMul,
  vecClone,
} from '../types.js';

import { solveLinearMPC } from './linear-mpc.js';

// ---------------------------------------------------------------------------
// Explicit MPC lookup
// ---------------------------------------------------------------------------

/**
 * Look up the optimal control for state x from a pre-computed explicit
 * MPC table. Iterates over regions and returns the affine control law
 * for the first region that contains x.
 *
 * @param table  Pre-computed explicit MPC table
 * @param x      Current state (nx)
 * @returns Optimal control (nu) or null if no region contains x
 */
export function explicitMPCLookup(
  table: ExplicitMPCTable,
  x: Float64Array,
): Float64Array | null {
  const { regions, nx, nu } = table;

  for (let r = 0; r < regions.length; r++) {
    const region = regions[r]!;
    if (isInsideRegion(region, x, nx)) {
      // Compute u = Fx * x + gx
      const Fx = arrayToMatrix(region.Fx, nu, nx);
      const u = matVecMul(Fx, x);
      for (let i = 0; i < nu; i++) {
        u[i] = u[i]! + region.gx[i]!;
      }
      return u;
    }
  }

  return null;
}

/**
 * Check whether state x is inside the polyhedral region defined by
 * Hx * x <= hx.
 */
function isInsideRegion(
  region: ExplicitMPCRegion,
  x: Float64Array,
  nx: number,
): boolean {
  const { Hx, hx, nConstraints } = region;
  const HxMat = arrayToMatrix(Hx, nConstraints, nx);

  for (let i = 0; i < nConstraints; i++) {
    let sum = 0;
    for (let j = 0; j < nx; j++) {
      sum += matGet(HxMat, i, j) * x[j]!;
    }
    if (sum > hx[i]! + 1e-10) {
      return false;
    }
  }

  return true;
}

// ---------------------------------------------------------------------------
// Build explicit MPC table
// ---------------------------------------------------------------------------

/**
 * Build an explicit MPC lookup table by solving the MPC problem at a
 * grid of sample states and fitting piecewise affine control laws.
 *
 * This is a simplified approach to multi-parametric QP: rather than
 * performing exact critical region enumeration, we:
 *   1. Create a grid of sample points across the state space
 *   2. Solve the MPC at each grid point to get (x, u*) pairs
 *   3. Cluster nearby solutions and fit affine laws via least-squares
 *   4. Define polyhedral regions as Voronoi-like cells around cluster centres
 *
 * For small problems (nx <= 3, nu <= 2), this produces a practical
 * approximation.
 *
 * @param config  MPC configuration (must have xMin/xMax for grid bounds)
 * @returns ExplicitMPCTable with polyhedral regions and affine laws
 */
export function buildExplicitMPCTable(config: MPCConfig): ExplicitMPCTable {
  const { nx, nu } = config;

  // Determine grid bounds
  const lower = new Float64Array(nx);
  const upper = new Float64Array(nx);
  for (let i = 0; i < nx; i++) {
    lower[i] = config.xMin ? config.xMin[i]! : -5;
    upper[i] = config.xMax ? config.xMax[i]! : 5;
  }

  // Grid resolution per dimension (keep manageable for small nx)
  const nPerDim = Math.max(3, Math.min(7, Math.ceil(Math.pow(200, 1 / nx))));

  // Generate grid points
  const gridPoints: Float64Array[] = [];
  generateGrid(lower, upper, nPerDim, nx, 0, new Float64Array(nx), gridPoints);

  // Solve MPC at each grid point
  const solutions: { x: Float64Array; u: Float64Array }[] = [];
  for (const xSample of gridPoints) {
    try {
      const result = solveLinearMPC(config, xSample);
      if (result.status !== 'infeasible') {
        solutions.push({ x: vecClone(xSample), u: vecClone(result.uOptimal) });
      }
    } catch {
      // Skip infeasible points
    }
  }

  if (solutions.length === 0) {
    return { regions: [], nx, nu };
  }

  // Cluster solutions into regions using k-means on the active set
  // (simplified: cluster by proximity of the control response)
  const nRegions = Math.min(
    Math.max(2, Math.floor(solutions.length / Math.max(nx + 1, 3))),
    20,
  );

  const clusters = kMeansClusters(solutions, nRegions, nx, nu);

  // Fit affine law to each cluster and define bounding region
  const regions: ExplicitMPCRegion[] = [];

  for (const cluster of clusters) {
    if (cluster.length < nx + 1) {
      // Not enough points for affine fit; skip or use constant
      if (cluster.length > 0) {
        // Constant control: u = mean(u_i)
        const uMean = new Float64Array(nu);
        for (const pt of cluster) {
          for (let i = 0; i < nu; i++) {
            uMean[i] = uMean[i]! + pt.u[i]! / cluster.length;
          }
        }

        // Define a large bounding box as region
        const region = createBoundingRegion(cluster, nx, nu, null, uMean);
        regions.push(region);
      }
      continue;
    }

    // Fit affine law: u = Fx * x + gx via least-squares
    // Build [X | 1] matrix and U matrix
    const nPts = cluster.length;
    const Xaug = createMatrix(nPts, nx + 1);
    const Umat = createMatrix(nPts, nu);

    for (let p = 0; p < nPts; p++) {
      const pt = cluster[p]!;
      for (let j = 0; j < nx; j++) {
        matSet(Xaug, p, j, pt.x[j]!);
      }
      matSet(Xaug, p, nx, 1); // bias column
      for (let j = 0; j < nu; j++) {
        matSet(Umat, p, j, pt.u[j]!);
      }
    }

    // Least-squares: theta = (X^T X)^{-1} X^T U
    const Xt = matTranspose(Xaug);
    const XtX = matMul(Xt, Xaug);

    // Regularise for numerical stability
    for (let i = 0; i < nx + 1; i++) {
      matSet(XtX, i, i, matGet(XtX, i, i) + 1e-8);
    }

    let theta: ReturnType<typeof matMul>;
    try {
      const XtXinv = matInvert(XtX);
      const XtU = matMul(Xt, Umat);
      theta = matMul(XtXinv, XtU);
    } catch {
      // Singular: use constant
      const uMean = new Float64Array(nu);
      for (const pt of cluster) {
        for (let i = 0; i < nu; i++) {
          uMean[i] = uMean[i]! + pt.u[i]! / cluster.length;
        }
      }
      const region = createBoundingRegion(cluster, nx, nu, null, uMean);
      regions.push(region);
      continue;
    }

    // Extract Fx (nu x nx) and gx (nu)
    const Fx = new Float64Array(nu * nx);
    const gx = new Float64Array(nu);
    for (let i = 0; i < nu; i++) {
      for (let j = 0; j < nx; j++) {
        // theta is (nx+1) x nu, row j col i
        Fx[i * nx + j] = matGet(theta, j, i);
      }
      gx[i] = matGet(theta, nx, i);
    }

    const region = createBoundingRegion(cluster, nx, nu, Fx, gx);
    regions.push(region);
  }

  return { regions, nx, nu };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Recursively generate a Cartesian grid of points. */
function generateGrid(
  lower: Float64Array,
  upper: Float64Array,
  nPerDim: number,
  nx: number,
  dim: number,
  current: Float64Array,
  result: Float64Array[],
): void {
  if (dim === nx) {
    result.push(vecClone(current));
    return;
  }
  for (let i = 0; i < nPerDim; i++) {
    const t = nPerDim > 1 ? i / (nPerDim - 1) : 0.5;
    current[dim] = lower[dim]! + t * (upper[dim]! - lower[dim]!);
    generateGrid(lower, upper, nPerDim, nx, dim + 1, current, result);
  }
}

/** Simple k-means clustering on (x, u) pairs. */
function kMeansClusters(
  solutions: { x: Float64Array; u: Float64Array }[],
  k: number,
  nx: number,
  nu: number,
): { x: Float64Array; u: Float64Array }[][] {
  const n = solutions.length;
  if (n <= k) {
    // Each point is its own cluster
    return solutions.map((s) => [s]);
  }

  // Initialise centroids: pick k evenly spaced solutions
  const centroids: Float64Array[] = [];
  for (let i = 0; i < k; i++) {
    const idx = Math.floor((i * n) / k);
    const s = solutions[idx]!;
    const c = new Float64Array(nx + nu);
    for (let j = 0; j < nx; j++) c[j] = s.x[j]!;
    for (let j = 0; j < nu; j++) c[nx + j] = s.u[j]!;
    centroids.push(c);
  }

  const assignments = new Int32Array(n);
  const maxIter = 30;

  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false;

    // Assign each point to nearest centroid
    for (let p = 0; p < n; p++) {
      const s = solutions[p]!;
      let bestDist = Infinity;
      let bestC = 0;
      for (let c = 0; c < k; c++) {
        let dist = 0;
        for (let j = 0; j < nx; j++) {
          const d = s.x[j]! - centroids[c]![j]!;
          dist += d * d;
        }
        for (let j = 0; j < nu; j++) {
          const d = s.u[j]! - centroids[c]![nx + j]!;
          dist += d * d;
        }
        if (dist < bestDist) {
          bestDist = dist;
          bestC = c;
        }
      }
      if (assignments[p] !== bestC) {
        assignments[p] = bestC;
        changed = true;
      }
    }

    if (!changed) break;

    // Update centroids
    const counts = new Float64Array(k);
    for (let c = 0; c < k; c++) {
      centroids[c]!.fill(0);
    }
    for (let p = 0; p < n; p++) {
      const c = assignments[p]!;
      const s = solutions[p]!;
      counts[c] = counts[c]! + 1;
      for (let j = 0; j < nx; j++) {
        centroids[c]![j] = centroids[c]![j]! + s.x[j]!;
      }
      for (let j = 0; j < nu; j++) {
        centroids[c]![nx + j] = centroids[c]![nx + j]! + s.u[j]!;
      }
    }
    for (let c = 0; c < k; c++) {
      if (counts[c]! > 0) {
        for (let j = 0; j < nx + nu; j++) {
          centroids[c]![j] = centroids[c]![j]! / counts[c]!;
        }
      }
    }
  }

  // Build cluster arrays
  const clusters: { x: Float64Array; u: Float64Array }[][] = [];
  for (let c = 0; c < k; c++) {
    clusters.push([]);
  }
  for (let p = 0; p < n; p++) {
    clusters[assignments[p]!]!.push(solutions[p]!);
  }

  // Remove empty clusters
  return clusters.filter((c) => c.length > 0);
}

/**
 * Create an ExplicitMPCRegion from a cluster of (x, u) pairs.
 * The region is defined as a bounding box (in half-space form)
 * around the cluster's state points, with margin.
 */
function createBoundingRegion(
  cluster: { x: Float64Array; u: Float64Array }[],
  nx: number,
  nu: number,
  Fx: Float64Array | null,
  gx: Float64Array,
): ExplicitMPCRegion {
  // Find bounding box of cluster states
  const xMin = new Float64Array(nx).fill(Infinity);
  const xMax = new Float64Array(nx).fill(-Infinity);

  for (const pt of cluster) {
    for (let i = 0; i < nx; i++) {
      if (pt.x[i]! < xMin[i]!) xMin[i] = pt.x[i]!;
      if (pt.x[i]! > xMax[i]!) xMax[i] = pt.x[i]!;
    }
  }

  // Add margin (10% of range or minimum 0.1)
  for (let i = 0; i < nx; i++) {
    const range = xMax[i]! - xMin[i]!;
    const margin = Math.max(range * 0.1, 0.1);
    xMin[i] = xMin[i]! - margin;
    xMax[i] = xMax[i]! + margin;
  }

  // Half-space representation: 2*nx constraints (box)
  // x_i <= xMax_i  and  -x_i <= -xMin_i
  const nConstraints = 2 * nx;
  const Hx = new Float64Array(nConstraints * nx);
  const hx = new Float64Array(nConstraints);

  for (let i = 0; i < nx; i++) {
    // x_i <= xMax_i
    Hx[i * nx + i] = 1;
    hx[i] = xMax[i]!;

    // -x_i <= -xMin_i
    Hx[(nx + i) * nx + i] = -1;
    hx[nx + i] = -xMin[i]!;
  }

  return {
    Hx,
    hx,
    Fx: Fx ?? new Float64Array(nu * nx),
    gx,
    nConstraints,
  };
}
