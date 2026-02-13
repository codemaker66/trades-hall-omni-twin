// ---------------------------------------------------------------------------
// SP-8: Particle Filter for Bayesian Occupancy Estimation
// ---------------------------------------------------------------------------
// Maintain weighted particles {(N_t^i, w_t^i)}
// Predict: N_t^i ~ p(N_t | N_{t-1}^i)
// Update:  w_t^i ∝ p(z_t | N_t^i)
// Systematic resampling when N_eff < N_p/2

import type { ParticleFilterConfig, ParticleState, OccupancyEstimate, PRNG } from '../types.js';
import { createPRNG } from '../types.js';

const DEFAULT_CONFIG: ParticleFilterConfig = {
  nParticles: 500,
  maxOccupancy: 300,
  processNoise: 5,
  measurementNoise: 10,
};

/**
 * Initialize particle filter state.
 */
export function initParticleFilter(
  initialEstimate: number = 50,
  config: ParticleFilterConfig = DEFAULT_CONFIG,
  seed: number = 42,
): ParticleState {
  const rng = createPRNG(seed);
  const { nParticles } = config;

  const particles = new Float64Array(nParticles);
  const weights = new Float64Array(nParticles);

  // Initialize particles around initial estimate
  for (let i = 0; i < nParticles; i++) {
    particles[i] = Math.max(0, initialEstimate + gaussianNoise(rng) * config.processNoise * 3);
    weights[i] = 1 / nParticles;
  }

  return {
    particles,
    weights,
    estimate: computeEstimate(particles, weights, nParticles),
    effectiveSampleSize: nParticles,
  };
}

/** Box-Muller Gaussian noise. */
function gaussianNoise(rng: PRNG): number {
  const u1 = Math.max(1e-10, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/** Gaussian likelihood: p(z | x) */
function gaussianLikelihood(z: number, x: number, sigma: number): number {
  const diff = z - x;
  return Math.exp(-0.5 * (diff * diff) / (sigma * sigma));
}

/** Compute weighted estimate from particles. */
function computeEstimate(
  particles: Float64Array,
  weights: Float64Array,
  n: number,
): OccupancyEstimate {
  let mean = 0;
  for (let i = 0; i < n; i++) {
    mean += weights[i]! * particles[i]!;
  }

  let variance = 0;
  for (let i = 0; i < n; i++) {
    const diff = particles[i]! - mean;
    variance += weights[i]! * diff * diff;
  }
  const std = Math.sqrt(variance);

  return {
    count: Math.round(mean),
    uncertainty: std,
    lower: Math.max(0, Math.round(mean - 1.96 * std)),
    upper: Math.round(mean + 1.96 * std),
  };
}

/**
 * Systematic resampling: proportional to weights with even spacing.
 */
function systematicResample(
  particles: Float64Array,
  weights: Float64Array,
  n: number,
  rng: PRNG,
): Float64Array {
  const newParticles = new Float64Array(n);
  const cumWeights = new Float64Array(n);
  cumWeights[0] = weights[0]!;
  for (let i = 1; i < n; i++) {
    cumWeights[i] = cumWeights[i - 1]! + weights[i]!;
  }

  const step = 1 / n;
  let u = rng() * step;
  let j = 0;

  for (let i = 0; i < n; i++) {
    while (j < n - 1 && u > cumWeights[j]!) {
      j++;
    }
    newParticles[i] = particles[j]!;
    u += step;
  }

  return newParticles;
}

/**
 * Particle filter update step.
 *
 * @param state Current particle state
 * @param measurement Observed occupancy measurement (e.g., from CO2 or WiFi CSI)
 * @param config Filter configuration
 * @param rng PRNG for reproducibility
 */
export function particleFilterStep(
  state: ParticleState,
  measurement: number,
  config: ParticleFilterConfig = DEFAULT_CONFIG,
  rng?: PRNG,
): ParticleState {
  const { nParticles, maxOccupancy, processNoise, measurementNoise } = config;
  const random = rng ?? createPRNG(Date.now());

  const newParticles = new Float64Array(nParticles);
  const newWeights = new Float64Array(nParticles);

  // Predict: add process noise
  for (let i = 0; i < nParticles; i++) {
    newParticles[i] = Math.max(0, Math.min(maxOccupancy,
      state.particles[i]! + gaussianNoise(random) * processNoise,
    ));
  }

  // Update: compute likelihood weights
  let weightSum = 0;
  for (let i = 0; i < nParticles; i++) {
    newWeights[i] = gaussianLikelihood(measurement, newParticles[i]!, measurementNoise);
    weightSum += newWeights[i]!;
  }

  // Normalize weights
  if (weightSum > 0) {
    for (let i = 0; i < nParticles; i++) {
      newWeights[i] = newWeights[i]! / weightSum;
    }
  } else {
    // All weights zero → uniform
    for (let i = 0; i < nParticles; i++) {
      newWeights[i] = 1 / nParticles;
    }
  }

  // Effective sample size
  let sumW2 = 0;
  for (let i = 0; i < nParticles; i++) {
    sumW2 += newWeights[i]! * newWeights[i]!;
  }
  const nEff = sumW2 > 0 ? 1 / sumW2 : 0;

  // Resample if effective sample size drops below threshold
  let particles: Float64Array;
  let weights: Float64Array;

  if (nEff < nParticles / 2) {
    particles = systematicResample(newParticles, newWeights, nParticles, random);
    weights = new Float64Array(nParticles);
    weights.fill(1 / nParticles);
  } else {
    particles = newParticles;
    weights = newWeights;
  }

  return {
    particles,
    weights,
    estimate: computeEstimate(particles, weights, nParticles),
    effectiveSampleSize: nEff,
  };
}

/**
 * Run particle filter over a sequence of measurements.
 */
export function particleFilterBatch(
  measurements: Float64Array,
  config: ParticleFilterConfig = DEFAULT_CONFIG,
  initialEstimate: number = 50,
  seed: number = 42,
): OccupancyEstimate[] {
  const rng = createPRNG(seed);
  let state = initParticleFilter(initialEstimate, config, seed);
  const estimates: OccupancyEstimate[] = [];

  for (let t = 0; t < measurements.length; t++) {
    state = particleFilterStep(state, measurements[t]!, config, rng);
    estimates.push(state.estimate);
  }

  return estimates;
}
