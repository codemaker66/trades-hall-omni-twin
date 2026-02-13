/**
 * PS-1: Simulated Annealing Engine
 *
 * Implements SA with three cooling schedules (Geometric, Lam-Delosme, Huang),
 * systematic reheating, Metropolis acceptance, and full telemetry tracking.
 */

import type { SAConfig, SAResult, EnergyFunction, NeighborFunction, PRNG } from './types.js'
import { CoolingSchedule, createPRNG } from './types.js'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Rolling window size for Lam-Delosme and Huang adaptive schedules */
const WINDOW_SIZE = 100

/** Lam-Delosme target acceptance ratio (Lam & Delosme, DAC 1988) */
const LAM_TARGET_ACCEPTANCE = 0.44

/** Lam-Delosme adjustment rate per window */
const LAM_ADJUSTMENT_RATE = 0.01

/** Huang lambda parameter controlling cooling aggressiveness */
const HUANG_LAMBDA = 0.7

/** Minimum sigma to avoid division by zero in Huang schedule */
const HUANG_MIN_SIGMA = 1e-12

// ---------------------------------------------------------------------------
// Cooling schedule implementations
// ---------------------------------------------------------------------------

function coolGeometric(temp: number, alpha: number): number {
  return temp * alpha
}

/**
 * Lam-Delosme adaptive cooling.
 *
 * Adjusts temperature to maintain ~44% acceptance ratio.
 * When acceptance is too high, the landscape is being explored too freely
 * and temperature is reduced. When too low, temperature is raised.
 */
function coolLamDelosme(temp: number, acceptanceRatio: number): number {
  if (acceptanceRatio > LAM_TARGET_ACCEPTANCE) {
    // Too many accepts — cool faster
    return temp * (1 - LAM_ADJUSTMENT_RATE * (acceptanceRatio - LAM_TARGET_ACCEPTANCE))
  }
  // Too few accepts — cool slower (or warm slightly)
  return temp * (1 + LAM_ADJUSTMENT_RATE * (LAM_TARGET_ACCEPTANCE - acceptanceRatio))
}

/**
 * Huang adaptive cooling (Huang, Romeo & Sangiovanni-Vincentelli, ICCAD 1986).
 *
 * T_{k+1} = T_k * exp(-T_k * lambda / sigma_k)
 *
 * Cools faster when the energy landscape is flat (low sigma) and
 * slower near phase transitions (high sigma).
 */
function coolHuang(temp: number, sigma: number): number {
  const safeSigma = Math.max(sigma, HUANG_MIN_SIGMA)
  return temp * Math.exp(-temp * HUANG_LAMBDA / safeSigma)
}

// ---------------------------------------------------------------------------
// Window statistics tracker
// ---------------------------------------------------------------------------

interface WindowStats {
  energySum: number
  energySqSum: number
  acceptCount: number
  windowCount: number
}

function createWindowStats(): WindowStats {
  return { energySum: 0, energySqSum: 0, acceptCount: 0, windowCount: 0 }
}

function recordEnergy(stats: WindowStats, energy: number, accepted: boolean): void {
  stats.energySum += energy
  stats.energySqSum += energy * energy
  stats.windowCount++
  if (accepted) {
    stats.acceptCount++
  }
}

function getAcceptanceRatio(stats: WindowStats): number {
  return stats.windowCount > 0 ? stats.acceptCount / stats.windowCount : 0.5
}

function getEnergySigma(stats: WindowStats): number {
  if (stats.windowCount < 2) return HUANG_MIN_SIGMA
  const mean = stats.energySum / stats.windowCount
  const variance = stats.energySqSum / stats.windowCount - mean * mean
  // Variance can be slightly negative due to floating-point; clamp to 0
  return Math.sqrt(Math.max(0, variance))
}

function resetWindowStats(stats: WindowStats): void {
  stats.energySum = 0
  stats.energySqSum = 0
  stats.acceptCount = 0
  stats.windowCount = 0
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Copy a Float64Array into a fresh Float64Array backed by a plain ArrayBuffer */
function copyState(src: Float64Array): Float64Array {
  const dst = new Float64Array(src.length)
  dst.set(src)
  return dst
}

// ---------------------------------------------------------------------------
// Core SA implementation
// ---------------------------------------------------------------------------

/**
 * Run simulated annealing on a given initial state.
 *
 * @param initialState  - Starting configuration as Float64Array
 * @param config        - SA hyperparameters (schedule, temps, iterations, reheating)
 * @param energyFn      - Evaluates the energy (cost) of a state
 * @param neighborFn    - Generates a neighboring state from the current one
 * @returns SAResult with best solution found, telemetry, and full history
 */
export function simulatedAnnealing(
  initialState: Float64Array,
  config: SAConfig,
  energyFn: EnergyFunction,
  neighborFn: NeighborFunction,
): SAResult {
  const rng: PRNG = createPRNG(config.seed ?? 42)

  // --- State tracking ---
  let currentState = copyState(initialState)
  let currentEnergy = energyFn(currentState)

  let bestState = copyState(initialState)
  let bestEnergy = currentEnergy

  let temp = config.initialTemp

  // --- Telemetry ---
  const energyHistory = new Float64Array(config.maxIterations)
  const tempHistory = new Float64Array(config.maxIterations)
  let totalAccepts = 0
  let totalRejects = 0
  let totalReheats = 0

  // --- Adaptive window for Lam-Delosme and Huang ---
  const windowStats = createWindowStats()

  let iterations = 0

  for (let i = 0; i < config.maxIterations; i++) {
    // --- Termination check ---
    if (temp < config.finalTemp) break

    // --- Generate candidate ---
    const candidateState = neighborFn(currentState, rng)
    const candidateEnergy = energyFn(candidateState)
    const deltaE = candidateEnergy - currentEnergy

    // --- Metropolis acceptance criterion ---
    let accepted = false
    if (deltaE <= 0) {
      // Always accept improvements (lower energy)
      accepted = true
    } else {
      // Accept worse solutions with probability exp(-deltaE / T)
      const probability = Math.exp(-deltaE / temp)
      accepted = rng.random() < probability
    }

    if (accepted) {
      currentState = copyState(candidateState)
      currentEnergy = candidateEnergy
      totalAccepts++
    } else {
      totalRejects++
    }

    // --- Track best solution ---
    if (currentEnergy < bestEnergy) {
      bestEnergy = currentEnergy
      bestState = copyState(currentState)
    }

    // --- Record telemetry ---
    energyHistory[i] = currentEnergy
    tempHistory[i] = temp

    // --- Update window statistics (for adaptive schedules) ---
    recordEnergy(windowStats, currentEnergy, accepted)

    iterations = i + 1

    // --- Cooling step ---
    if (windowStats.windowCount >= WINDOW_SIZE) {
      // Apply adaptive cooling once per window
      switch (config.cooling) {
        case CoolingSchedule.Geometric:
          // Geometric cooling applied per-window for consistency
          temp = coolGeometric(temp, config.alpha)
          break
        case CoolingSchedule.LamDelosme:
          temp = coolLamDelosme(temp, getAcceptanceRatio(windowStats))
          break
        case CoolingSchedule.Huang:
          temp = coolHuang(temp, getEnergySigma(windowStats))
          break
      }
      resetWindowStats(windowStats)
    } else if (config.cooling === CoolingSchedule.Geometric) {
      // Geometric schedule cools every iteration for smooth decay
      temp = coolGeometric(temp, config.alpha)
    }

    // --- Systematic reheating ---
    if (
      config.reheatInterval > 0 &&
      iterations % config.reheatInterval === 0
    ) {
      temp = config.initialTemp * config.reheatTempFraction
      totalReheats++
    }
  }

  // --- Trim history arrays to actual iteration count ---
  // Use .slice() to produce a clean Float64Array<ArrayBuffer> (avoids SharedArrayBuffer type mismatch)
  const trimmedEnergyHistory: Float64Array = new Float64Array(iterations)
  trimmedEnergyHistory.set(energyHistory.subarray(0, iterations))
  const trimmedTempHistory: Float64Array = new Float64Array(iterations)
  trimmedTempHistory.set(tempHistory.subarray(0, iterations))

  return {
    bestEnergy,
    bestState,
    iterations,
    accepts: totalAccepts,
    rejects: totalRejects,
    reheats: totalReheats,
    energyHistory: trimmedEnergyHistory,
    tempHistory: trimmedTempHistory,
  }
}
