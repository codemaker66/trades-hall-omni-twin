// ---------------------------------------------------------------------------
// OC-4  Dynamic Programming -- Hamilton-Jacobi-Bellman PDE Solver
// ---------------------------------------------------------------------------

import type { HJBConfig, HJBResult } from '../types.js';

// ---------------------------------------------------------------------------
// Grid helpers
// ---------------------------------------------------------------------------

/**
 * Compute the total number of grid cells from per-dimension counts.
 */
function totalGridSize(gridN: Int32Array): number {
  let size = 1;
  for (let d = 0; d < gridN.length; d++) {
    size *= gridN[d]!;
  }
  return size;
}

/**
 * Convert a multi-dimensional index array to a flat (row-major) index.
 */
function multiToFlat(indices: Int32Array, gridN: Int32Array): number {
  let flat = 0;
  let stride = 1;
  for (let d = gridN.length - 1; d >= 0; d--) {
    flat += indices[d]! * stride;
    stride *= gridN[d]!;
  }
  return flat;
}

/**
 * Convert a flat index to multi-dimensional indices (row-major).
 */
function flatToMulti(flat: number, gridN: Int32Array): Int32Array {
  const nx = gridN.length;
  const indices = new Int32Array(nx);
  let rem = flat;
  for (let d = nx - 1; d >= 0; d--) {
    const n = gridN[d]!;
    indices[d] = rem % n;
    rem = Math.floor(rem / n);
  }
  return indices;
}

/**
 * Map a grid multi-index to the physical coordinate.
 */
function gridToCoord(
  indices: Int32Array,
  gridMin: Float64Array,
  gridMax: Float64Array,
  gridN: Int32Array,
): Float64Array {
  const nx = gridN.length;
  const x = new Float64Array(nx);
  for (let d = 0; d < nx; d++) {
    const n = gridN[d]!;
    const lo = gridMin[d]!;
    const hi = gridMax[d]!;
    x[d] = n > 1 ? lo + indices[d]! * ((hi - lo) / (n - 1)) : lo;
  }
  return x;
}

/**
 * Grid spacing for each dimension.
 */
function gridSpacing(
  gridMin: Float64Array,
  gridMax: Float64Array,
  gridN: Int32Array,
): Float64Array {
  const nx = gridN.length;
  const dx = new Float64Array(nx);
  for (let d = 0; d < nx; d++) {
    const n = gridN[d]!;
    dx[d] = n > 1 ? (gridMax[d]! - gridMin[d]!) / (n - 1) : 1;
  }
  return dx;
}

// ---------------------------------------------------------------------------
// solveHJB
// ---------------------------------------------------------------------------

/**
 * Solve the HJB PDE on a regular grid via upwind finite differences
 * integrated backward in time.
 *
 * -dV/dt + min_u { l(x,u) + (dV/dx) . f(x,u) } = 0
 *
 * with terminal condition V(T, x) = phi(x).
 */
export function solveHJB(config: HJBConfig): HJBResult {
  const {
    gridMin,
    gridMax,
    gridN,
    dt,
    dynamics,
    runningCost,
    terminalCost,
    controlSet,
    T,
    nx,
  } = config;

  const nTotal = totalGridSize(gridN);
  const dx = gridSpacing(gridMin, gridMax, gridN);
  const nSteps = Math.max(1, Math.round(T / dt));

  // Stride per dimension for flat indexing
  const strides = new Int32Array(nx);
  strides[nx - 1] = 1;
  for (let d = nx - 2; d >= 0; d--) {
    strides[d] = strides[d + 1]! * gridN[d + 1]!;
  }

  // Initialise value grid with terminal cost
  let valueGrid = new Float64Array(nTotal);
  const policyGrid = new Int32Array(nTotal);

  for (let idx = 0; idx < nTotal; idx++) {
    const mi = flatToMulti(idx, gridN);
    const x = gridToCoord(mi, gridMin, gridMax, gridN);
    valueGrid[idx] = terminalCost(x);
  }

  // Backward time integration
  for (let step = 0; step < nSteps; step++) {
    const Vnew = new Float64Array(nTotal);

    for (let idx = 0; idx < nTotal; idx++) {
      const mi = flatToMulti(idx, gridN);
      const x = gridToCoord(mi, gridMin, gridMax, gridN);

      let bestHamiltonian = Infinity;
      let bestUidx = 0;

      for (let ui = 0; ui < controlSet.length; ui++) {
        const u = controlSet[ui]!;
        const f = dynamics(x, u);
        const l = runningCost(x, u);

        // Upwind finite difference for each dimension
        let gradDotF = 0;
        for (let d = 0; d < nx; d++) {
          const fd = f[d]!;
          const currIdx = mi[d]!;
          const Nd = gridN[d]!;

          let dVdx: number;
          if (fd >= 0) {
            // Forward difference (upwind for positive velocity)
            if (currIdx < Nd - 1) {
              const fwdFlat = idx + strides[d]!;
              dVdx = (valueGrid[fwdFlat]! - valueGrid[idx]!) / dx[d]!;
            } else {
              // At boundary, use backward
              const bwdFlat = idx - strides[d]!;
              dVdx = (valueGrid[idx]! - valueGrid[bwdFlat]!) / dx[d]!;
            }
          } else {
            // Backward difference (upwind for negative velocity)
            if (currIdx > 0) {
              const bwdFlat = idx - strides[d]!;
              dVdx = (valueGrid[idx]! - valueGrid[bwdFlat]!) / dx[d]!;
            } else {
              // At boundary, use forward
              const fwdFlat = idx + strides[d]!;
              dVdx = (valueGrid[fwdFlat]! - valueGrid[idx]!) / dx[d]!;
            }
          }

          gradDotF += dVdx * fd;
        }

        const hamiltonian = l + gradDotF;
        if (hamiltonian < bestHamiltonian) {
          bestHamiltonian = hamiltonian;
          bestUidx = ui;
        }
      }

      // Euler backward: V(t) = V(t+dt) + dt * min_u H
      Vnew[idx] = valueGrid[idx]! + dt * bestHamiltonian;
      policyGrid[idx] = bestUidx;
    }

    valueGrid = Vnew;
  }

  return {
    valueGrid,
    policyGrid,
    gridDims: new Int32Array(gridN),
  };
}

// ---------------------------------------------------------------------------
// queryHJBValue
// ---------------------------------------------------------------------------

/**
 * Interpolate the HJB value function at an arbitrary continuous point
 * using multilinear (n-dimensional bilinear) interpolation.
 */
export function queryHJBValue(
  result: HJBResult,
  config: HJBConfig,
  x: Float64Array,
): number {
  const { gridMin, gridMax, gridN, nx } = config;
  const { valueGrid, gridDims } = result;

  const dx = gridSpacing(gridMin, gridMax, gridN);

  // Compute fractional grid coordinates
  const frac = new Float64Array(nx);
  const lo = new Int32Array(nx);

  for (let d = 0; d < nx; d++) {
    const coord = (x[d]! - gridMin[d]!) / dx[d]!;
    const clamped = Math.max(0, Math.min(coord, gridN[d]! - 1));
    lo[d] = Math.min(Math.floor(clamped), gridN[d]! - 2);
    if (lo[d]! < 0) lo[d] = 0;
    frac[d] = clamped - lo[d]!;
  }

  // Multilinear interpolation over 2^nx corners
  const nCorners = 1 << nx;
  let value = 0;

  for (let c = 0; c < nCorners; c++) {
    const corner = new Int32Array(nx);
    let weight = 1;

    for (let d = 0; d < nx; d++) {
      const bit = (c >> d) & 1;
      corner[d] = lo[d]! + bit;
      // Clamp to valid range
      if (corner[d]! >= gridDims[d]!) {
        corner[d] = gridDims[d]! - 1;
      }
      weight *= bit === 1 ? frac[d]! : 1 - frac[d]!;
    }

    const flatIdx = multiToFlat(corner, gridDims);
    value += weight * valueGrid[flatIdx]!;
  }

  return value;
}
