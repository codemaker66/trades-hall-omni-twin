// ---------------------------------------------------------------------------
// OC-8  Crowd Flow & Evacuation -- Hughes Macroscopic Model
// ---------------------------------------------------------------------------

import type { HughesModelConfig, HughesModelState } from '../types.js';

// ---------------------------------------------------------------------------
// State Initialisation
// ---------------------------------------------------------------------------

/**
 * Create the initial Hughes model state from a density field.
 *
 * Computes the eikonal potential from the initial density and derives
 * the velocity field from its gradient.
 */
export function createHughesState(
  config: HughesModelConfig,
  initialDensity: Float64Array,
): HughesModelState {
  const n = config.gridNx * config.gridNy;
  const density = new Float64Array(initialDensity);
  const potential = solveEikonal(config, density);
  const velocityX = new Float64Array(n);
  const velocityY = new Float64Array(n);

  computeVelocity(config, density, potential, velocityX, velocityY);

  return { density, potential, velocityX, velocityY };
}

// ---------------------------------------------------------------------------
// Eikonal Solver (Fast Sweeping Method)
// ---------------------------------------------------------------------------

/**
 * Solve the eikonal equation on the grid using iterative Gauss-Seidel
 * sweeping in all four diagonal directions.
 *
 * The cost function is 1 / speed(density), so high-density regions have
 * higher travel cost.  Exit cells are initialised to zero potential;
 * all other cells start at +Infinity and are relaxed downward.
 */
export function solveEikonal(
  config: HughesModelConfig,
  density: Float64Array,
): Float64Array {
  const { gridNx, gridNy, dx, exits, speedFn } = config;
  const n = gridNx * gridNy;
  const potential = new Float64Array(n);
  potential.fill(Infinity);

  // Initialise exit cells to zero
  for (const exit of exits) {
    const idx = exit.y * gridNx + exit.x;
    potential[idx] = 0;
  }

  const maxSweeps = 20;

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let changed = false;

    // Four diagonal sweep orders
    for (let dir = 0; dir < 4; dir++) {
      const iStart = dir < 2 ? 0 : gridNy - 1;
      const iEnd = dir < 2 ? gridNy : -1;
      const iStep = dir < 2 ? 1 : -1;
      const jStart = dir % 2 === 0 ? 0 : gridNx - 1;
      const jEnd = dir % 2 === 0 ? gridNx : -1;
      const jStep = dir % 2 === 0 ? 1 : -1;

      for (let i = iStart; i !== iEnd; i += iStep) {
        for (let j = jStart; j !== jEnd; j += jStep) {
          const idx = i * gridNx + j;

          // Skip exit cells
          if (potential[idx]! === 0) continue;

          const rho = density[idx]!;
          const speed = speedFn(rho);
          if (speed <= 1e-12) continue; // stagnant cell

          const cost = dx / speed;

          // Gather smallest x-neighbor and y-neighbor potentials
          let ux = Infinity;
          if (j > 0) ux = Math.min(ux, potential[i * gridNx + (j - 1)]!);
          if (j < gridNx - 1) ux = Math.min(ux, potential[i * gridNx + (j + 1)]!);

          let uy = Infinity;
          if (i > 0) uy = Math.min(uy, potential[(i - 1) * gridNx + j]!);
          if (i < gridNy - 1) uy = Math.min(uy, potential[(i + 1) * gridNx + j]!);

          let candidate: number;

          if (!isFinite(ux) && !isFinite(uy)) {
            continue;
          } else if (!isFinite(ux)) {
            candidate = uy + cost;
          } else if (!isFinite(uy)) {
            candidate = ux + cost;
          } else {
            // 2D upwind: solve (u - ux)^2 + (u - uy)^2 = cost^2
            const uMin = Math.min(ux, uy);
            const uMax = Math.max(ux, uy);

            if (uMax - uMin >= cost) {
              // 1D update from the smaller neighbor
              candidate = uMin + cost;
            } else {
              // Quadratic 2D update
              const diff = ux - uy;
              const disc = 2 * cost * cost - diff * diff;
              candidate = (ux + uy + Math.sqrt(disc)) / 2;
            }
          }

          if (candidate < potential[idx]! - 1e-12) {
            potential[idx] = candidate;
            changed = true;
          }
        }
      }
    }

    if (!changed) break;
  }

  return potential;
}

// ---------------------------------------------------------------------------
// Velocity from potential gradient
// ---------------------------------------------------------------------------

/**
 * Compute velocity field v = -speed(rho) * grad(phi) / |grad(phi)|.
 * Writes into the provided velocityX / velocityY arrays in-place.
 */
function computeVelocity(
  config: HughesModelConfig,
  density: Float64Array,
  potential: Float64Array,
  velocityX: Float64Array,
  velocityY: Float64Array,
): void {
  const { gridNx, gridNy, dx, speedFn } = config;

  for (let i = 0; i < gridNy; i++) {
    for (let j = 0; j < gridNx; j++) {
      const idx = i * gridNx + j;
      const speed = speedFn(density[idx]!);

      // Central differences with one-sided at boundaries
      let dPhiDx: number;
      if (j === 0) {
        dPhiDx = (potential[i * gridNx + (j + 1)]! - potential[idx]!) / dx;
      } else if (j === gridNx - 1) {
        dPhiDx = (potential[idx]! - potential[i * gridNx + (j - 1)]!) / dx;
      } else {
        dPhiDx =
          (potential[i * gridNx + (j + 1)]! - potential[i * gridNx + (j - 1)]!) /
          (2 * dx);
      }

      let dPhiDy: number;
      if (i === 0) {
        dPhiDy = (potential[(i + 1) * gridNx + j]! - potential[idx]!) / dx;
      } else if (i === gridNy - 1) {
        dPhiDy = (potential[idx]! - potential[(i - 1) * gridNx + j]!) / dx;
      } else {
        dPhiDy =
          (potential[(i + 1) * gridNx + j]! - potential[(i - 1) * gridNx + j]!) /
          (2 * dx);
      }

      const gradMag = Math.sqrt(dPhiDx * dPhiDx + dPhiDy * dPhiDy);
      if (gradMag > 1e-12) {
        velocityX[idx] = -speed * dPhiDx / gradMag;
        velocityY[idx] = -speed * dPhiDy / gradMag;
      } else {
        velocityX[idx] = 0;
        velocityY[idx] = 0;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Time Step
// ---------------------------------------------------------------------------

/**
 * Advance the Hughes model by one time step:
 *   1. Solve the eikonal equation for the current density.
 *   2. Compute velocity from the negative gradient of the potential.
 *   3. Update density via the conservation law with upwind finite volume.
 *
 * Returns a new state (does not mutate the input).
 */
export function hughesStep(
  config: HughesModelConfig,
  state: HughesModelState,
): HughesModelState {
  const { gridNx, gridNy, dx, dt, maxDensity } = config;
  const n = gridNx * gridNy;

  // (1) Solve eikonal
  const potential = solveEikonal(config, state.density);

  // (2) Compute velocity
  const velocityX = new Float64Array(n);
  const velocityY = new Float64Array(n);
  computeVelocity(config, state.density, potential, velocityX, velocityY);

  // (3) Update density via upwind finite volume conservation law:
  //     d(rho)/dt + div(rho * v) = 0
  const density = new Float64Array(state.density);

  for (let i = 0; i < gridNy; i++) {
    for (let j = 0; j < gridNx; j++) {
      const idx = i * gridNx + j;

      // Upwind fluxes in x-direction
      let fluxXLeft = 0;
      let fluxXRight = 0;

      if (j > 0) {
        const idxL = i * gridNx + (j - 1);
        const vxFace = (velocityX[idx]! + velocityX[idxL]!) * 0.5;
        // Upwind: use density from upstream side
        fluxXLeft = vxFace >= 0
          ? vxFace * state.density[idxL]!
          : vxFace * state.density[idx]!;
      }

      if (j < gridNx - 1) {
        const idxR = i * gridNx + (j + 1);
        const vxFace = (velocityX[idx]! + velocityX[idxR]!) * 0.5;
        fluxXRight = vxFace >= 0
          ? vxFace * state.density[idx]!
          : vxFace * state.density[idxR]!;
      }

      // Upwind fluxes in y-direction
      let fluxYBottom = 0;
      let fluxYTop = 0;

      if (i > 0) {
        const idxB = (i - 1) * gridNx + j;
        const vyFace = (velocityY[idx]! + velocityY[idxB]!) * 0.5;
        fluxYBottom = vyFace >= 0
          ? vyFace * state.density[idxB]!
          : vyFace * state.density[idx]!;
      }

      if (i < gridNy - 1) {
        const idxT = (i + 1) * gridNx + j;
        const vyFace = (velocityY[idx]! + velocityY[idxT]!) * 0.5;
        fluxYTop = vyFace >= 0
          ? vyFace * state.density[idx]!
          : vyFace * state.density[idxT]!;
      }

      // Conservation: rho_new = rho_old - dt/dx * (flux_out - flux_in)
      const divFlux =
        (fluxXRight - fluxXLeft) / dx + (fluxYTop - fluxYBottom) / dx;

      density[idx] = state.density[idx]! - dt * divFlux;

      // Clamp density to physical bounds
      if (density[idx]! < 0) density[idx] = 0;
      if (density[idx]! > maxDensity) density[idx] = maxDensity;
    }
  }

  return { density, potential, velocityX, velocityY };
}

// ---------------------------------------------------------------------------
// Egress Time Estimation
// ---------------------------------------------------------------------------

/**
 * Estimate total egress time: the maximum time for all pedestrians to
 * reach an exit, approximated by the maximum potential value weighted
 * by density across occupied cells.
 *
 * Returns 0 when the domain is empty.
 */
export function hughesEgressTime(
  config: HughesModelConfig,
  state: HughesModelState,
): number {
  const n = config.gridNx * config.gridNy;
  let maxTime = 0;

  for (let idx = 0; idx < n; idx++) {
    const rho = state.density[idx]!;
    if (rho < 1e-12) continue;

    const phi = state.potential[idx]!;
    if (!isFinite(phi)) continue;

    // The potential represents travel time to the exit.
    // Account for density-induced queuing: add rho / maxDensity * phi as delay.
    const speed = config.speedFn(rho);
    const queueDelay = speed > 1e-12 ? (rho / config.maxDensity) * phi : phi;
    const cellEgressTime = phi + queueDelay;

    if (cellEgressTime > maxTime) {
      maxTime = cellEgressTime;
    }
  }

  return maxTime;
}
