// ---------------------------------------------------------------------------
// SP-8: CO2-Based Occupancy Estimation
// ---------------------------------------------------------------------------
// Mass balance: N = Q·(C_indoor - C_outdoor)/G
//   G ≈ 0.005 L/s per person (at rest)
//   NDIR sensors (Sensirion SCD41): T90 = 30-120s, ±30ppm
// Suitable for event-level analysis, not real-time headcount.

import type { CO2Config, OccupancyEstimate } from '../types.js';

const DEFAULT_CO2_CONFIG: CO2Config = {
  outdoorCO2: 420,         // ppm (2024 global average)
  ventilationRate: 8.5,    // L/s per person (ASHRAE 62.1 default)
  generationRate: 0.005,   // L/s per person at rest (~0.3 L/min)
};

/**
 * Steady-state occupancy estimate from CO2 level.
 * N = Q · (C_indoor - C_outdoor) / G
 *
 * @param indoorCO2 Indoor CO2 concentration (ppm)
 * @param ventilationRateTotal Total ventilation rate (L/s) for the space
 * @param config CO2 model parameters
 */
export function estimateOccupancyFromCO2(
  indoorCO2: number,
  ventilationRateTotal: number,
  config: CO2Config = DEFAULT_CO2_CONFIG,
): OccupancyEstimate {
  const deltaC = indoorCO2 - config.outdoorCO2;

  if (deltaC <= 0) {
    return { count: 0, uncertainty: 0, lower: 0, upper: 0 };
  }

  // Convert ppm to L/L: 1 ppm = 1e-6
  const deltaCRatio = deltaC * 1e-6;

  // N = Q_total · ΔC / G
  const count = (ventilationRateTotal * deltaCRatio) / config.generationRate;

  // Uncertainty from sensor noise (±30 ppm) and generation rate variability (±20%)
  const sensorUncertainty = (30 * 1e-6 * ventilationRateTotal) / config.generationRate;
  const genRateUncertainty = count * 0.2;
  const totalUncertainty = Math.sqrt(sensorUncertainty * sensorUncertainty + genRateUncertainty * genRateUncertainty);

  return {
    count: Math.max(0, Math.round(count)),
    uncertainty: totalUncertainty,
    lower: Math.max(0, Math.round(count - 1.96 * totalUncertainty)),
    upper: Math.round(count + 1.96 * totalUncertainty),
  };
}

/**
 * Dynamic CO2 occupancy model using exponential approach.
 * Accounts for sensor response lag (T90).
 *
 * dC/dt = G·N/V - Q/V·(C - C_outdoor)
 *
 * Discretized: C(t+1) = C(t) + dt·[G·N/V - Q/V·(C(t) - C_outdoor)]
 */
export function dynamicCO2Model(
  timeSeriesCO2: Float64Array,
  ventilationRate: number,
  roomVolume: number,
  dt: number = 60, // seconds between samples
  config: CO2Config = DEFAULT_CO2_CONFIG,
): Float64Array {
  const N = timeSeriesCO2.length;
  const estimates = new Float64Array(N);

  for (let t = 0; t < N; t++) {
    if (t === 0) {
      // Initial estimate from steady-state
      const est = estimateOccupancyFromCO2(timeSeriesCO2[0]!, ventilationRate, config);
      estimates[0] = est.count;
      continue;
    }

    // Rate of CO2 change
    const dCdt = (timeSeriesCO2[t]! - timeSeriesCO2[t - 1]!) / dt * 1e-6; // ppm/s → L/L/s

    // N = [V·dC/dt + Q·(C - C_out)] / G
    const deltaC = (timeSeriesCO2[t]! - config.outdoorCO2) * 1e-6;
    const occupancy = (roomVolume * dCdt + ventilationRate * deltaC) / config.generationRate;
    estimates[t] = Math.max(0, Math.round(occupancy));
  }

  return estimates;
}
