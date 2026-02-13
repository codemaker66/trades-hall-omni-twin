// ---------------------------------------------------------------------------
// HPC-3: Workers — Parallel Tempering Coordinator
// ---------------------------------------------------------------------------
// Parallel tempering (replica exchange) distributes MCMC replicas across
// workers at different temperatures. Adjacent replicas periodically propose
// swaps via the Metropolis-Hastings criterion, enabling global exploration
// while maintaining detailed balance.
// ---------------------------------------------------------------------------

import type { PRNG } from '../types.js';

// ---------------------------------------------------------------------------
// Temperature ladder
// ---------------------------------------------------------------------------

/**
 * Generate a geometric temperature ladder from `tMin` to `tMax`.
 *
 * Geometric spacing ensures roughly equal swap acceptance rates between
 * adjacent replicas for systems where energy scales linearly with temperature.
 *
 *   T_i = tMin * (tMax / tMin) ^ (i / (n - 1))
 *
 * For a single replica, returns `[tMin]`.
 */
export function generateTemperatureLadder(
  numReplicas: number,
  tMin: number,
  tMax: number,
): Float64Array {
  if (numReplicas < 1) {
    throw new RangeError('numReplicas must be at least 1');
  }
  if (tMin <= 0 || tMax <= 0) {
    throw new RangeError('Temperatures must be positive');
  }
  if (tMin > tMax) {
    throw new RangeError('tMin must be <= tMax');
  }

  const temperatures = new Float64Array(numReplicas);
  if (numReplicas === 1) {
    temperatures[0] = tMin;
    return temperatures;
  }

  const ratio = tMax / tMin;
  for (let i = 0; i < numReplicas; i++) {
    temperatures[i] = tMin * Math.pow(ratio, i / (numReplicas - 1));
  }
  return temperatures;
}

// ---------------------------------------------------------------------------
// Replica swap
// ---------------------------------------------------------------------------

/**
 * Metropolis-Hastings swap criterion for parallel tempering.
 *
 * Two adjacent replicas i (temperature T_i) and j (temperature T_j) propose
 * a configuration swap. The acceptance probability is:
 *
 *   alpha = min(1, exp(delta))
 *   where delta = (1/T_i - 1/T_j) * (E_j - E_i)
 *
 * Note: the sign convention assumes we want to swap when the lower-temperature
 * replica would benefit from the higher-energy configuration, weighted by the
 * inverse-temperature difference.
 *
 * Returns true if the swap should be accepted.
 */
export function shouldSwapReplicas(
  energyI: number,
  energyJ: number,
  tempI: number,
  tempJ: number,
  rng: PRNG,
): boolean {
  const delta = (1 / tempI - 1 / tempJ) * (energyJ - energyI);

  // Always accept if delta >= 0 (energetically favorable)
  if (delta >= 0) {
    return true;
  }

  // Accept with probability exp(delta)
  return rng() < Math.exp(delta);
}

// ---------------------------------------------------------------------------
// Acceptance rate analysis
// ---------------------------------------------------------------------------

/**
 * Compute the average swap acceptance rate across all adjacent replica pairs.
 *
 * Uses the deterministic acceptance probability (without RNG):
 *   p_accept = min(1, exp((1/T_i - 1/T_{i+1}) * (E_{i+1} - E_i)))
 *
 * For N replicas there are N-1 adjacent pairs. Returns 0 if fewer than
 * 2 replicas.
 */
export function computeSwapAcceptanceRate(
  energies: Float64Array,
  temperatures: Float64Array,
): number {
  const n = Math.min(energies.length, temperatures.length);
  if (n < 2) return 0;

  let totalAcceptance = 0;
  for (let i = 0; i < n - 1; i++) {
    const eI = energies[i]!;
    const eJ = energies[i + 1]!;
    const tI = temperatures[i]!;
    const tJ = temperatures[i + 1]!;

    const delta = (1 / tI - 1 / tJ) * (eJ - eI);
    totalAcceptance += Math.min(1, Math.exp(delta));
  }

  return totalAcceptance / (n - 1);
}

// ---------------------------------------------------------------------------
// Temperature optimization
// ---------------------------------------------------------------------------

/**
 * Adjust temperatures to drive acceptance rates toward a target rate.
 *
 * Uses a simple proportional feedback rule: if the acceptance rate for a
 * pair (i, i+1) is below the target, move T_{i+1} closer to T_i (reduce
 * spacing); if above, move T_{i+1} further away (increase spacing).
 *
 * The first and last temperatures (tMin, tMax) are held fixed.
 *
 * The adjustment factor for interior temperature T_i is:
 *   T_i *= 1 + alpha * (acceptanceRate_{i-1} - targetRate)
 * where alpha is a small learning rate (0.1).
 *
 * Returns a new Float64Array of optimized temperatures.
 */
export function optimizeTemperatures(
  acceptanceRates: Float64Array,
  temperatures: Float64Array,
  targetRate: number,
): Float64Array {
  const n = temperatures.length;
  const optimized = new Float64Array(n);

  if (n < 2) {
    if (n === 1) optimized[0] = temperatures[0]!;
    return optimized;
  }

  // Pin endpoints
  optimized[0] = temperatures[0]!;
  optimized[n - 1] = temperatures[n - 1]!;

  const alpha = 0.1; // learning rate

  for (let i = 1; i < n - 1; i++) {
    // Use the acceptance rate of the pair (i-1, i) to adjust T_i
    const rate = acceptanceRates[i - 1] ?? targetRate;
    const adjustment = 1 + alpha * (rate - targetRate);
    let newTemp = temperatures[i]! * adjustment;

    // Clamp between neighbors to maintain monotonicity
    const lo = optimized[i - 1]!;
    const hi = temperatures[n - 1]!;
    newTemp = Math.max(lo + 1e-10, Math.min(hi - 1e-10, newTemp));
    optimized[i] = newTemp;
  }

  return optimized;
}

// ---------------------------------------------------------------------------
// Worker assignment
// ---------------------------------------------------------------------------

/**
 * Distribute replicas across workers as evenly as possible.
 *
 * Returns an array of length `numWorkers` where each element is an array
 * of replica indices assigned to that worker. Replicas are distributed
 * round-robin to balance load.
 *
 * Example: 7 replicas, 3 workers -> [[0,3,6], [1,4], [2,5]]
 */
export function assignReplicasToWorkers(
  numReplicas: number,
  numWorkers: number,
): Array<number[]> {
  if (numWorkers < 1) {
    throw new RangeError('numWorkers must be at least 1');
  }

  const assignment: Array<number[]> = [];
  for (let w = 0; w < numWorkers; w++) {
    assignment.push([]);
  }

  for (let r = 0; r < numReplicas; r++) {
    assignment[r % numWorkers]!.push(r);
  }

  return assignment;
}
