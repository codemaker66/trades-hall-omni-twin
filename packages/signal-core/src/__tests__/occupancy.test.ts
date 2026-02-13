// ---------------------------------------------------------------------------
// SP-8: Occupancy Sensing Tests
// ---------------------------------------------------------------------------
import { describe, it, expect } from 'vitest';
import {
  estimateOccupancyFromCO2,
  dynamicCO2Model,
} from '../occupancy/co2-estimation.js';
import { initializeCrowd, socialForceStep, simulateCrowdFlow } from '../occupancy/crowd-flow.js';
import {
  initParticleFilter,
  particleFilterStep,
  particleFilterBatch,
} from '../occupancy/particle-filter.js';

describe('SP-8: Occupancy Sensing', () => {
  describe('CO2 Estimation', () => {
    it('estimates occupancy from steady-state CO2', () => {
      // estimateOccupancyFromCO2(indoorCO2, ventilationRateTotal, config?)
      const result = estimateOccupancyFromCO2(800, 100, {
        outdoorCO2: 400,
        ventilationRate: 10,
        generationRate: 0.005,
      });
      expect(result.count).toBeGreaterThan(0);
      expect(result.uncertainty).toBeGreaterThan(0);
    });

    it('higher CO2 means more occupants', () => {
      const config = { outdoorCO2: 400, ventilationRate: 10, generationRate: 0.005 };
      const low = estimateOccupancyFromCO2(600, 100, config);
      const high = estimateOccupancyFromCO2(1200, 100, config);
      expect(high.count).toBeGreaterThan(low.count);
    });

    it('returns zero for below-outdoor CO2', () => {
      const result = estimateOccupancyFromCO2(300, 100, {
        outdoorCO2: 400,
        ventilationRate: 10,
        generationRate: 0.005,
      });
      expect(result.count).toBe(0);
    });
  });

  describe('Dynamic CO2 Model', () => {
    it('estimates occupancy from CO2 time series', () => {
      // dynamicCO2Model(timeSeriesCO2, ventilationRate, roomVolume, dt?, config?)
      const n = 60;
      const co2 = new Float64Array(n);
      // Simulate gradually increasing CO2
      for (let i = 0; i < n; i++) co2[i] = 400 + i * 5;
      const estimates = dynamicCO2Model(co2, 10, 200, 60);
      expect(estimates.length).toBe(n);
    });
  });

  describe('Social Force Model', () => {
    it('initializes crowd agents', () => {
      // initializeCrowd(count, roomWidth, roomHeight, seed?)
      const agents = initializeCrowd(10, 20, 15, 42);
      expect(agents.length).toBe(10);
      for (const agent of agents) {
        expect(agent.x).toBeGreaterThanOrEqual(0);
        expect(agent.y).toBeGreaterThanOrEqual(0);
        expect(typeof agent.vx).toBe('number');
        expect(typeof agent.vy).toBe('number');
        expect(typeof agent.goalX).toBe('number');
        expect(typeof agent.goalY).toBe('number');
      }
    });

    it('simulates one step of crowd movement', () => {
      // socialForceStep(agents, dt?, roomWidth?, roomHeight?, config?)
      const agents = initializeCrowd(5, 20, 15, 42);
      const updated = socialForceStep(agents, 0.1, 20, 15);
      expect(updated.length).toBe(5);
    });

    it('full simulation runs for N steps', () => {
      // simulateCrowdFlow(agents, nSteps, dt?, roomWidth?, roomHeight?, config?)
      const agents = initializeCrowd(5, 20, 15, 42);
      const result = simulateCrowdFlow(agents, 10, 0.1, 20, 15);
      expect(result.length).toBe(11); // initial + 10 steps
      for (const frame of result) {
        expect(frame.length).toBe(5);
      }
    });
  });

  describe('Particle Filter', () => {
    it('initializes particles', () => {
      // initParticleFilter(initialEstimate?, config?, seed?)
      const pf = initParticleFilter(20, {
        nParticles: 100,
        maxOccupancy: 200,
        processNoise: 2,
        measurementNoise: 5,
      }, 42);
      expect(pf.particles.length).toBe(100);
      expect(pf.weights.length).toBe(100);
      expect(pf.estimate.count).toBeGreaterThanOrEqual(0);
    });

    it('converges to measurement', () => {
      const config = {
        nParticles: 200,
        maxOccupancy: 200,
        processNoise: 1,
        measurementNoise: 3,
      };
      let pf = initParticleFilter(10, config, 42);
      for (let i = 0; i < 20; i++) {
        pf = particleFilterStep(pf, 30, config);
      }
      // Estimate should converge toward measurement of 30
      expect(Math.abs(pf.estimate.count - 30)).toBeLessThan(15);
    });

    it('batch processing works', () => {
      const config = {
        nParticles: 100,
        maxOccupancy: 200,
        processNoise: 1,
        measurementNoise: 3,
      };
      const measurements = new Float64Array(10).fill(25);
      // particleFilterBatch(measurements, config?, initialEstimate?, seed?)
      const estimates = particleFilterBatch(measurements, config, 10, 42);
      expect(estimates.length).toBe(10);
      for (const est of estimates) {
        expect(typeof est.count).toBe('number');
        expect(typeof est.uncertainty).toBe('number');
      }
    });
  });
});
