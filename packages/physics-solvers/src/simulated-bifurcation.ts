/**
 * PS-4: Simulated Bifurcation Algorithm
 *
 * Implements the Toshiba Simulated Bifurcation (SB) algorithm for
 * combinatorial optimization on Ising models.
 *
 * Based on: Goto, Tatsumura & Dixon, "Combinatorial optimization by simulating
 * adiabatic bifurcations in nonlinear Hamiltonian systems",
 * Science Advances 5(4), eaav2372 (2019).
 *
 * The algorithm simulates N coupled nonlinear oscillators whose dynamics
 * undergo a pitchfork bifurcation as a pump parameter p(t) increases.
 * After bifurcation, each oscillator settles into one of two wells,
 * encoding a binary spin value.
 *
 * Equations of motion:
 *   dx_i/dt = y_i
 *   dy_i/dt = -(1 - p(t)) * x_i - K * x_i^3 + xi_0 * (sum_j J_ij * f(x_j) + h_i)
 *
 * Variants:
 *   - Ballistic SB (bSB): f(x) = x        (smooth, good for structured problems)
 *   - Discrete SB  (dSB): f(x) = sign(x)  (quasi-quantum tunneling, best overall)
 */

import type { IsingModel, SBConfig, QUBOMatrix, PRNG } from './types.js'
import { SBVariant, createPRNG } from './types.js'
import { quboToIsing, spinsToBinary } from './qubo.js'

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Compute the maximum absolute value of the coupling matrix.
 * Used to normalize the coupling strength xi_0.
 */
function maxAbsCoupling(model: IsingModel): number {
  const { n, couplings } = model
  let maxVal = 0
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const absVal = Math.abs(couplings[i * n + j]!)
      if (absVal > maxVal) {
        maxVal = absVal
      }
    }
  }
  return maxVal
}

/**
 * Initialize position and momentum arrays with small random perturbations.
 * Values are drawn uniformly from [-amplitude, +amplitude].
 */
function initializeOscillators(
  n: number,
  rng: PRNG,
  amplitude: number,
): { x: Float64Array; y: Float64Array } {
  const x = new Float64Array(n)
  const y = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    x[i] = (rng.random() * 2 - 1) * amplitude
    y[i] = (rng.random() * 2 - 1) * amplitude
  }
  return { x, y }
}

/**
 * Compute the coupling term for each spin:
 *   coupling_i = h_i + sum_j J_ij * f(x_j)
 *
 * For Ballistic SB, f(x_j) = x_j.
 * For Discrete SB,  f(x_j) = sign(x_j).
 */
function computeCoupling(
  model: IsingModel,
  x: Float64Array,
  variant: SBVariant,
  out: Float64Array,
): void {
  const { n, couplings, field } = model

  // Start with the external field
  for (let i = 0; i < n; i++) {
    out[i] = field[i]!
  }

  if (variant === SBVariant.Ballistic) {
    // f(x_j) = x_j — standard matrix-vector product J * x
    for (let i = 0; i < n; i++) {
      let sum = out[i]!
      const rowOffset = i * n
      for (let j = 0; j < n; j++) {
        sum += couplings[rowOffset + j]! * x[j]!
      }
      out[i] = sum
    }
  } else {
    // Discrete: f(x_j) = sign(x_j)
    // Pre-compute signs to avoid repeated branching in the inner loop
    const signs = new Float64Array(n)
    for (let j = 0; j < n; j++) {
      const xj = x[j]!
      signs[j] = xj >= 0 ? 1.0 : -1.0
    }
    for (let i = 0; i < n; i++) {
      let sum = out[i]!
      const rowOffset = i * n
      for (let j = 0; j < n; j++) {
        sum += couplings[rowOffset + j]! * signs[j]!
      }
      out[i] = sum
    }
  }
}

// ---------------------------------------------------------------------------
// Main SB solver
// ---------------------------------------------------------------------------

/**
 * Solve an Ising model using the Simulated Bifurcation algorithm.
 *
 * @param model  - Ising model with coupling matrix J and external field h
 * @param config - SB hyperparameters (variant, nSteps, dt, pumpRate, kerr, seed)
 * @returns Spin configuration as Int8Array with values in {-1, +1}
 */
export function simulatedBifurcation(
  model: IsingModel,
  config: SBConfig,
): Int8Array {
  const { n } = model
  const { variant, nSteps, dt, pumpRate, kerr } = config
  const rng = createPRNG(config.seed ?? 42)

  // --- Compute normalized coupling strength ---
  // xi_0 = 0.7 / max(|J_ij|)  to keep dynamics stable
  const maxJ = maxAbsCoupling(model)
  const xi0 = maxJ > 0 ? 0.7 / maxJ : 0.7

  // --- Initialize oscillators with small random perturbations ---
  const initAmplitude = 0.01
  const { x, y } = initializeOscillators(n, rng, initAmplitude)

  // --- Pre-allocate coupling buffer ---
  const coupling = new Float64Array(n)

  // --- Time evolution via symplectic Euler integration ---
  for (let step = 0; step < nSteps; step++) {
    // Linearly increasing pump: p(t) = pumpRate * t
    // where t goes from 0 to 1 over the simulation
    const t = step / nSteps
    const p = pumpRate * t

    // Compute coupling_i = h_i + sum_j J_ij * f(x_j)
    computeCoupling(model, x, variant, coupling)

    // Symplectic Euler update (momentum first, then position)
    for (let i = 0; i < n; i++) {
      const xi = x[i]!
      const yi = y[i]!

      // dy_i/dt = -(1 - p) * x_i - K * x_i^3 + xi_0 * coupling_i
      const detuning = -(1 - p) * xi
      const nonlinearity = -kerr * xi * xi * xi
      const drive = xi0 * coupling[i]!

      const yNew = yi + dt * (detuning + nonlinearity + drive)

      // dx_i/dt = y_i (using updated y)
      const xNew = xi + dt * yNew

      y[i] = yNew
      x[i] = xNew
    }
  }

  // --- Read out spins from sign of oscillator positions ---
  const spins = new Int8Array(n)
  for (let i = 0; i < n; i++) {
    spins[i] = x[i]! >= 0 ? 1 : -1
  }

  return spins
}

// ---------------------------------------------------------------------------
// QUBO wrapper
// ---------------------------------------------------------------------------

/**
 * Solve a QUBO problem using Simulated Bifurcation.
 *
 * Converts the QUBO matrix to an Ising model, runs SB, then maps
 * the Ising spins back to binary variables.
 *
 * @param qubo   - Upper-triangular QUBO matrix
 * @param config - SB hyperparameters
 * @returns Binary solution as Uint8Array with values in {0, 1}
 */
export function simulatedBifurcationQUBO(
  qubo: QUBOMatrix,
  config: SBConfig,
): Uint8Array {
  const ising = quboToIsing(qubo)
  const spins = simulatedBifurcation(ising, config)
  return spinsToBinary(spins)
}
