// ---------------------------------------------------------------------------
// OC-8  Crowd Flow & Evacuation -- Evacuation MPC
// ---------------------------------------------------------------------------

import type { EvacuationMPCConfig } from '../types.js';

// ---------------------------------------------------------------------------
// Greedy Evacuation
// ---------------------------------------------------------------------------

/**
 * Simple nearest-exit assignment respecting exit capacity.
 *
 * For each zone, assigns people to the exit with the shortest travel time
 * that still has remaining capacity.  Returns a nZones * nExits row-major
 * matrix of fractional assignments (fraction of people from zone i sent
 * to exit j).
 */
export function greedyEvacuation(
  config: EvacuationMPCConfig,
  occupancy: Float64Array,
): Float64Array {
  const { nZones, nExits, exitCapacities, travelTimes } = config;
  const assignment = new Float64Array(nZones * nExits);
  const remainingCapacity = new Float64Array(exitCapacities);

  // Build list of (zone, bestExit) pairs sorted by shortest travel time
  // so zones with the shortest available path are assigned first.
  const zoneOrder: Array<{ zone: number; bestTime: number }> = [];
  for (let z = 0; z < nZones; z++) {
    let bestTime = Infinity;
    for (let e = 0; e < nExits; e++) {
      const tt = travelTimes[z * nExits + e]!;
      if (tt < bestTime) bestTime = tt;
    }
    zoneOrder.push({ zone: z, bestTime });
  }
  zoneOrder.sort((a, b) => a.bestTime - b.bestTime);

  for (const { zone: z } of zoneOrder) {
    const occ = occupancy[z]!;
    if (occ <= 0) continue;

    // Sort exits by travel time from this zone
    const exits: Array<{ exit: number; time: number }> = [];
    for (let e = 0; e < nExits; e++) {
      exits.push({ exit: e, time: travelTimes[z * nExits + e]! });
    }
    exits.sort((a, b) => a.time - b.time);

    let remaining = occ;

    for (const { exit: e } of exits) {
      if (remaining <= 0) break;
      const cap = remainingCapacity[e]!;
      if (cap <= 0) continue;

      const send = Math.min(remaining, cap);
      assignment[z * nExits + e] = send / occ;
      remainingCapacity[e] = cap - send;
      remaining -= send;
    }

    // If there is still remaining occupancy (all exits at capacity),
    // distribute proportionally to closest exits anyway
    if (remaining > 1e-12) {
      const closest = exits[0]!;
      assignment[z * nExits + closest.exit] =
        (assignment[z * nExits + closest.exit]! * occ + remaining) / occ;
    }
  }

  return assignment;
}

// ---------------------------------------------------------------------------
// MPC-based Evacuation Optimiser
// ---------------------------------------------------------------------------

/**
 * Solve the evacuation routing problem using a simplified iterative MPC.
 *
 * Objective: minimise the maximum (travel_time * flow) across all zone-exit
 * pairs, subject to:
 *   - Each zone's occupants must be fully assigned to exits (sum_j a_ij = 1)
 *   - Exit flow does not exceed capacity (sum_i occ_i * a_ij <= cap_j)
 *
 * Approach: iterative redistribution.  Start from greedy assignment, then
 * for each MPC horizon step, identify the bottleneck zone-exit pair and
 * redistribute flow to less-loaded exits.
 *
 * Returns:
 *   assignment: nZones * nExits row-major matrix of fractional assignments
 *   totalTime:  estimated total evacuation time (max over zone-exit weighted flows)
 */
export function solveEvacuationMPC(
  config: EvacuationMPCConfig,
  currentOccupancy: Float64Array,
): { assignment: Float64Array; totalTime: number } {
  const { nZones, nExits, exitCapacities, travelTimes, horizon } = config;

  // Initialise from greedy solution
  const assignment = greedyEvacuation(config, currentOccupancy);

  // Iterative improvement over MPC horizon
  for (let iter = 0; iter < horizon; iter++) {
    // Compute exit loads and find bottleneck
    const exitLoads = new Float64Array(nExits);
    let bottleneckTime = -Infinity;
    let bottleneckZone = 0;
    let bottleneckExit = 0;

    for (let z = 0; z < nZones; z++) {
      const occ = currentOccupancy[z]!;
      if (occ <= 0) continue;

      for (let e = 0; e < nExits; e++) {
        const flow = occ * assignment[z * nExits + e]!;
        exitLoads[e] = exitLoads[e]! + flow;

        const time = travelTimes[z * nExits + e]! * flow;
        if (time > bottleneckTime) {
          bottleneckTime = time;
          bottleneckZone = z;
          bottleneckExit = e;
        }
      }
    }

    // Try to redistribute some flow from bottleneck exit to a less-loaded exit
    const bOcc = currentOccupancy[bottleneckZone]!;
    if (bOcc <= 0) continue;

    const currentFrac = assignment[bottleneckZone * nExits + bottleneckExit]!;
    if (currentFrac < 1e-12) continue;

    // Find the best alternative exit for this zone
    let bestAlt = -1;
    let bestAltMetric = Infinity;

    for (let e = 0; e < nExits; e++) {
      if (e === bottleneckExit) continue;

      const altLoad = exitLoads[e]!;
      const altCap = exitCapacities[e]!;

      // Skip if exit is already over capacity
      if (altLoad >= altCap) continue;

      const altTime = travelTimes[bottleneckZone * nExits + e]!;
      // Metric: travel time adjusted by load ratio
      const metric = altTime * (1 + altLoad / Math.max(altCap, 1e-12));
      if (metric < bestAltMetric) {
        bestAltMetric = metric;
        bestAlt = e;
      }
    }

    if (bestAlt < 0) continue;

    // Transfer a fraction of the bottleneck flow
    const transferFrac = Math.min(currentFrac * 0.5, 0.2);
    assignment[bottleneckZone * nExits + bottleneckExit] = currentFrac - transferFrac;
    assignment[bottleneckZone * nExits + bestAlt] =
      assignment[bottleneckZone * nExits + bestAlt]! + transferFrac;
  }

  // Normalise rows to ensure they sum to 1
  for (let z = 0; z < nZones; z++) {
    let rowSum = 0;
    for (let e = 0; e < nExits; e++) {
      rowSum += assignment[z * nExits + e]!;
    }
    if (rowSum > 1e-12) {
      for (let e = 0; e < nExits; e++) {
        assignment[z * nExits + e] = assignment[z * nExits + e]! / rowSum;
      }
    }
  }

  // Compute total evacuation time estimate
  let totalTime = 0;
  for (let z = 0; z < nZones; z++) {
    const occ = currentOccupancy[z]!;
    if (occ <= 0) continue;

    for (let e = 0; e < nExits; e++) {
      const flow = occ * assignment[z * nExits + e]!;
      const time = travelTimes[z * nExits + e]! * flow;
      if (time > totalTime) totalTime = time;
    }
  }

  return { assignment, totalTime };
}
