// ---------------------------------------------------------------------------
// SP-3: Kalman Filtering Tests
// ---------------------------------------------------------------------------
import { describe, it, expect } from 'vitest';
import {
  createKalmanState,
  kalmanPredict,
  kalmanUpdate,
  kalmanStep,
  kalmanBatch,
  createDemandTracker,
} from '../kalman/kalman-filter.js';
import { generateSigmaPoints, ukfPredict, ukfUpdate, ukfStep } from '../kalman/ukf.js';
import { rtsSmooth } from '../kalman/rts-smoother.js';
import { createAdaptiveState, adaptiveKalmanStep } from '../kalman/adaptive.js';
import { MultiSensorFusion, createVenueFusion } from '../kalman/multi-sensor.js';
import type { KalmanConfig, KalmanState, UKFConfig, AdaptiveKalmanConfig } from '../types.js';

describe('SP-3: Kalman Filtering', () => {
  // Helper: build a 1D identity Kalman config for simple tests
  function make1DConfig(): KalmanConfig {
    return {
      F: new Float64Array([1]),
      H: new Float64Array([1]),
      Q: new Float64Array([0.1]),
      R: new Float64Array([1.0]),
      dimX: 1,
      dimZ: 1,
    };
  }

  describe('Linear Kalman Filter', () => {
    it('creates initial state correctly', () => {
      const state = createKalmanState(2);
      expect(state.x.length).toBe(2);
      expect(state.P.length).toBe(4); // 2x2 covariance as flat Float64Array
      expect(state.dim).toBe(2);
      // Default diagonal = 100
      expect(state.P[0]).toBe(100);
      expect(state.P[3]).toBe(100);
      // Off-diagonals = 0
      expect(state.P[1]).toBe(0);
      expect(state.P[2]).toBe(0);
    });

    it('creates initial state with custom values', () => {
      const x0 = new Float64Array([1, 2]);
      const P0 = new Float64Array([10, 0, 0, 10]);
      const state = createKalmanState(2, x0, P0);
      expect(state.x[0]).toBe(1);
      expect(state.x[1]).toBe(2);
      expect(state.P[0]).toBe(10);
      expect(state.P[3]).toBe(10);
    });

    it('predict increases uncertainty', () => {
      const config = make1DConfig();
      const state = createKalmanState(1, new Float64Array([0]), new Float64Array([1]));
      const pBefore = state.P[0]!;
      const predicted = kalmanPredict(state, config);
      // P_pred = F*P*F' + Q = 1*1*1 + 0.1 = 1.1
      expect(predicted.P[0]!).toBeGreaterThan(pBefore);
    });

    it('update reduces uncertainty', () => {
      const config = make1DConfig();
      const state = createKalmanState(1, new Float64Array([0]), new Float64Array([1]));
      const predicted = kalmanPredict(state, config);
      const pBefore = predicted.P[0]!;
      const result = kalmanUpdate(predicted, new Float64Array([5.0]), config);
      expect(result.state.P[0]!).toBeLessThan(pBefore);
    });

    it('kalmanUpdate returns innovation and Kalman gain', () => {
      const config = make1DConfig();
      const state = createKalmanState(1, new Float64Array([0]), new Float64Array([1]));
      const predicted = kalmanPredict(state, config);
      const result = kalmanUpdate(predicted, new Float64Array([5.0]), config);

      expect(result.innovation).toBeInstanceOf(Float64Array);
      expect(result.innovation.length).toBe(1);
      expect(result.kalmanGain).toBeInstanceOf(Float64Array);
      expect(result.kalmanGain.length).toBe(1); // dimX * dimZ = 1
    });

    it('kalmanStep performs predict + update', () => {
      const config = make1DConfig();
      const state = createKalmanState(1);
      const result = kalmanStep(state, new Float64Array([5.0]), config);
      expect(result.state.x.length).toBe(1);
      expect(result.innovation.length).toBe(1);
    });

    it('batch processing tracks constant signal', () => {
      const config: KalmanConfig = {
        F: new Float64Array([1]),
        H: new Float64Array([1]),
        Q: new Float64Array([0.01]),
        R: new Float64Array([1.0]),
        dimX: 1,
        dimZ: 1,
      };
      const measurements: Float64Array[] = [];
      for (let i = 0; i < 50; i++) {
        measurements.push(new Float64Array([5 + Math.random() * 0.5 - 0.25]));
      }
      const { states, predictions, innovations } = kalmanBatch(measurements, config);
      expect(states.length).toBe(50);
      expect(predictions.length).toBe(50);
      expect(innovations.length).toBe(50);
      // Final estimate should be close to 5
      const finalX = states[states.length - 1]!.x[0]!;
      expect(Math.abs(finalX - 5)).toBeLessThan(1);
    });
  });

  describe('Demand Tracker', () => {
    it('creates 3-state tracker', () => {
      const tracker = createDemandTracker();
      expect(tracker.state.x.length).toBe(3);
      expect(tracker.config.dimX).toBe(3);
      expect(tracker.config.dimZ).toBe(3);
    });

    it('tracks increasing demand via update()', () => {
      const tracker = createDemandTracker();
      let estimate;
      for (let i = 0; i < 20; i++) {
        estimate = tracker.update(50 + i * 2, 10 + i, 3 + i * 0.2);
      }
      // demandLevel should be positive
      expect(estimate!.demandLevel).toBeGreaterThan(0);
      // uncertainty should be a finite number
      expect(Number.isFinite(estimate!.uncertainty)).toBe(true);
    });

    it('returns DemandEstimate with all fields', () => {
      const tracker = createDemandTracker();
      const estimate = tracker.update(100, 20, 5);
      expect(typeof estimate.demandLevel).toBe('number');
      expect(typeof estimate.demandVelocity).toBe('number');
      expect(typeof estimate.seasonal).toBe('number');
      expect(typeof estimate.uncertainty).toBe('number');
    });
  });

  describe('UKF', () => {
    it('generates correct number of sigma points', () => {
      const n = 3;
      const x = new Float64Array(n);
      // Identity covariance as flat array
      const P = new Float64Array(n * n);
      for (let i = 0; i < n; i++) P[i * n + i] = 1;

      const { sigmaPoints, weightsMean, weightsCov } = generateSigmaPoints(
        x, P, 1e-3, 2, 0,
      );
      expect(sigmaPoints.length).toBe(2 * n + 1);
      expect(weightsMean.length).toBe(2 * n + 1);
      expect(weightsCov.length).toBe(2 * n + 1);
    });

    it('sigma point weights sum approximately to 1', () => {
      const n = 2;
      const x = new Float64Array(n);
      const P = new Float64Array([1, 0, 0, 1]);

      const { weightsMean } = generateSigmaPoints(x, P, 1e-3, 2, 0);
      let sum = 0;
      for (let i = 0; i < weightsMean.length; i++) sum += weightsMean[i]!;
      expect(Math.abs(sum - 1)).toBeLessThan(1e-6);
    });

    it('UKF step converges on measurement', () => {
      const n = 1;
      const ukfConfig: UKFConfig = {
        F: new Float64Array([1]),
        H: new Float64Array([1]),
        Q: new Float64Array([0.01]),
        R: new Float64Array([0.5]),
        dimX: 1,
        dimZ: 1,
        alpha: 1e-3,
        beta: 2,
        kappa: 0,
        stateTransitionFn: (x: Float64Array) => new Float64Array(x),
        observationFn: (x: Float64Array) => new Float64Array(x),
      };
      let state: KalmanState = createKalmanState(1);
      for (let i = 0; i < 30; i++) {
        const result = ukfStep(state, new Float64Array([10.0]), ukfConfig);
        state = result.state;
      }
      expect(Math.abs(state.x[0]! - 10)).toBeLessThan(1);
    });

    it('ukfPredict and ukfUpdate work independently', () => {
      const ukfConfig: UKFConfig = {
        F: new Float64Array([1]),
        H: new Float64Array([1]),
        Q: new Float64Array([0.1]),
        R: new Float64Array([1.0]),
        dimX: 1,
        dimZ: 1,
        alpha: 1e-3,
        beta: 2,
        kappa: 0,
        stateTransitionFn: (x: Float64Array) => new Float64Array(x),
        observationFn: (x: Float64Array) => new Float64Array(x),
      };
      const state = createKalmanState(1);
      const predicted = ukfPredict(state, ukfConfig);
      expect(predicted.x.length).toBe(1);
      expect(predicted.P.length).toBe(1); // 1x1

      const { state: updated, innovation } = ukfUpdate(predicted, new Float64Array([5]), ukfConfig);
      expect(updated.x.length).toBe(1);
      expect(innovation.length).toBe(1);
    });
  });

  describe('RTS Smoother', () => {
    it('reduces variance compared to filter', () => {
      const config: KalmanConfig = {
        F: new Float64Array([1]),
        H: new Float64Array([1]),
        Q: new Float64Array([0.1]),
        R: new Float64Array([1.0]),
        dimX: 1,
        dimZ: 1,
      };
      const measurements: Float64Array[] = [];
      for (let i = 0; i < 20; i++) {
        measurements.push(new Float64Array([5 + Math.sin(i / 3)]));
      }
      const { states: filteredStates, predictions } = kalmanBatch(measurements, config);
      const smoothed = rtsSmooth(filteredStates, predictions, config);

      expect(smoothed.smoothedStates.length).toBe(filteredStates.length);
      expect(smoothed.smoothedCovariances.length).toBe(filteredStates.length);

      // Smoother should have lower or equal variance at interior points
      const midIdx = 10;
      expect(smoothed.smoothedCovariances[midIdx]![0]!).toBeLessThanOrEqual(
        filteredStates[midIdx]!.P[0]! + 1e-10
      );
    });

    it('smoothed states are Float64Arrays', () => {
      const config: KalmanConfig = {
        F: new Float64Array([1]),
        H: new Float64Array([1]),
        Q: new Float64Array([0.1]),
        R: new Float64Array([1.0]),
        dimX: 1,
        dimZ: 1,
      };
      const measurements = [new Float64Array([1]), new Float64Array([2]), new Float64Array([3])];
      const { states, predictions } = kalmanBatch(measurements, config);
      const smoothed = rtsSmooth(states, predictions, config);
      for (const s of smoothed.smoothedStates) {
        expect(s).toBeInstanceOf(Float64Array);
      }
      for (const c of smoothed.smoothedCovariances) {
        expect(c).toBeInstanceOf(Float64Array);
      }
    });
  });

  describe('Adaptive Kalman', () => {
    it('adapts measurement noise estimate', () => {
      const config: AdaptiveKalmanConfig = {
        F: new Float64Array([1]),
        H: new Float64Array([1]),
        Q: new Float64Array([0.01]),
        R: new Float64Array([1.0]),
        dimX: 1,
        dimZ: 1,
        forgettingFactor: 0.95,
      };
      let state = createAdaptiveState(config);
      for (let i = 0; i < 30; i++) {
        const result = adaptiveKalmanStep(state, new Float64Array([5 + Math.random() * 0.1]), config);
        state = result.state;
      }
      // estimatedR is a Float64Array; check the first element is positive
      expect(state.estimatedR[0]!).toBeGreaterThan(0);
    });

    it('createAdaptiveState with initial state', () => {
      const config: AdaptiveKalmanConfig = {
        F: new Float64Array([1]),
        H: new Float64Array([1]),
        Q: new Float64Array([0.01]),
        R: new Float64Array([1.0]),
        dimX: 1,
        dimZ: 1,
        forgettingFactor: 0.98,
      };
      const state = createAdaptiveState(config, new Float64Array([42]));
      expect(state.x[0]).toBe(42);
      expect(state.innovationCov).toBeInstanceOf(Float64Array);
      expect(state.estimatedR).toBeInstanceOf(Float64Array);
    });
  });

  describe('Multi-Sensor Fusion', () => {
    it('fuses multiple sensors', () => {
      const dimX = 1;
      const sensors = [
        { name: 'sensor_a', observationRow: new Float64Array([1]), noiseVariance: 1.0 },
        { name: 'sensor_b', observationRow: new Float64Array([1]), noiseVariance: 0.5 },
      ];
      const F = new Float64Array([1]);
      const Q = new Float64Array([0.01]);
      const fusion = new MultiSensorFusion(dimX, sensors, F, Q);
      fusion.predict();
      fusion.updateSensor(0, 10);
      fusion.updateSensor(1, 11);
      const state = fusion.getState();
      // Fused estimate should be between 0 and ~12, pulled toward measurements
      expect(state.x[0]!).toBeGreaterThan(0);
    });

    it('updateAll handles null measurements', () => {
      const dimX = 1;
      const sensors = [
        { name: 'sensor_a', observationRow: new Float64Array([1]), noiseVariance: 1.0 },
        { name: 'sensor_b', observationRow: new Float64Array([1]), noiseVariance: 0.5 },
      ];
      const F = new Float64Array([1]);
      const Q = new Float64Array([0.01]);
      const fusion = new MultiSensorFusion(dimX, sensors, F, Q);
      fusion.predict();
      // Only sensor_b provides data
      fusion.updateAll([null, 7.5]);
      const state = fusion.getState();
      expect(Number.isFinite(state.x[0]!)).toBe(true);
    });

    it('venue fusion creates 3-state system', () => {
      const fusion = createVenueFusion();
      const state = fusion.getState();
      expect(state.x.length).toBe(3);
      expect(state.dim).toBe(3);
    });

    it('venue fusion getEstimate returns DemandEstimate', () => {
      const fusion = createVenueFusion();
      fusion.predict();
      fusion.updateAll([200, 30, 8]);
      const est = fusion.getEstimate();
      expect(typeof est.demandLevel).toBe('number');
      expect(typeof est.demandVelocity).toBe('number');
      expect(typeof est.seasonal).toBe('number');
      expect(typeof est.uncertainty).toBe('number');
    });
  });
});
