import { describe, it, expect } from 'vitest';
import {
  createControlLoop,
  controlLoopStep,
  createMultiRateScheduler,
  getActiveSubsystems,
  multiSensorEstimate,
  movingHorizonEstimate,
  createFaultTolerantController,
} from '../architecture/index.js';
import type {
  ControlLoopConfig,
  SampleRateConfig,
  MultiSensorEstimateConfig,
  FaultToleranceConfig,
} from '../types.js';
import { matTrace, matIdentity } from '../types.js';

// ---------------------------------------------------------------------------
// Helpers -- small test fixtures
// ---------------------------------------------------------------------------

/** Identity-like 2x2 flat array. */
const eye2 = Float64Array.from([1, 0, 0, 1]);

/** Simple 2-state, 1-input timing config. */
const loopConfig: ControlLoopConfig = {
  sensingPeriodMs: 10,
  estimationPeriodMs: 10,
  decisionPeriodMs: 10,
  actuationPeriodMs: 10,
};

// ---------------------------------------------------------------------------
// OC-11: Control Loop
// ---------------------------------------------------------------------------

describe('OC-11: Real-Time Control Architecture', () => {
  // -----------------------------------------------------------------------
  // createControlLoop
  // -----------------------------------------------------------------------
  describe('createControlLoop', () => {
    it('initialises state with correct dimensions', () => {
      const nx = 2;
      const nu = 1;
      const state = createControlLoop(loopConfig, nx, nu);

      expect(state.timestamp).toBe(0);
      expect(state.stateEstimate.length).toBe(nx);
      expect(state.currentAction.length).toBe(nu);
      expect(state.previousAction.length).toBe(nu);
      expect(state.sensorReadings.size).toBe(0);

      // All vectors should be zero-initialised
      for (let i = 0; i < nx; i++) {
        expect(state.stateEstimate[i]!).toBe(0);
      }
      for (let i = 0; i < nu; i++) {
        expect(state.currentAction[i]!).toBe(0);
        expect(state.previousAction[i]!).toBe(0);
      }
    });
  });

  // -----------------------------------------------------------------------
  // controlLoopStep
  // -----------------------------------------------------------------------
  describe('controlLoopStep', () => {
    it('advances timestamp by the minimum period', () => {
      const nx = 2;
      const nu = 1;
      const state = createControlLoop(loopConfig, nx, nu);

      const identity = (_x: Float64Array) => new Float64Array(nu);
      const passthrough = (xHat: Float64Array, _r: Map<string, Float64Array>) => xHat;

      const next = controlLoopStep(state, loopConfig, identity, passthrough);

      // min period = 10ms, starting from 0 => timestamp should be 10
      expect(next.timestamp).toBe(10);

      // Step again
      const next2 = controlLoopStep(next, loopConfig, identity, passthrough);
      expect(next2.timestamp).toBe(20);
    });

    it('runs controller and updates currentAction when decision fires', () => {
      const nx = 2;
      const nu = 1;
      const state = createControlLoop(loopConfig, nx, nu);

      // Controller always returns [42]
      const controller = (_x: Float64Array) => Float64Array.from([42]);
      const passthrough = (xHat: Float64Array, _r: Map<string, Float64Array>) => xHat;

      const next = controlLoopStep(state, loopConfig, controller, passthrough);

      // At t=10, decision fires (10 % 10 = 0) so currentAction should be [42]
      expect(next.currentAction[0]!).toBe(42);
    });
  });

  // -----------------------------------------------------------------------
  // Multi-rate scheduler
  // -----------------------------------------------------------------------
  describe('createMultiRateScheduler', () => {
    it('fires subsystems at correct times', () => {
      const config: SampleRateConfig = {
        pricing: 300,   // every 300s
        staffing: 900,  // every 900s
        marketing: 600, // every 600s
        crowdOps: 60,   // every 60s
      };

      const scheduler = createMultiRateScheduler(config);

      // At t=0, all subsystems fire (0 is a multiple of every period)
      const atZero = scheduler.schedule(0);
      expect(atZero).toContain('pricing');
      expect(atZero).toContain('staffing');
      expect(atZero).toContain('marketing');
      expect(atZero).toContain('crowdOps');

      // At t=60, only crowdOps fires
      const at60 = scheduler.schedule(60);
      expect(at60).toContain('crowdOps');
      expect(at60).not.toContain('pricing');
      expect(at60).not.toContain('staffing');

      // At t=300, pricing and crowdOps fire
      const at300 = scheduler.schedule(300);
      expect(at300).toContain('pricing');
      expect(at300).toContain('crowdOps');
      expect(at300).not.toContain('staffing');

      // At t=1800, all fire (LCM of 300, 900, 600, 60)
      const at1800 = scheduler.schedule(1800);
      expect(at1800).toContain('pricing');
      expect(at1800).toContain('staffing');
      expect(at1800).toContain('marketing');
      expect(at1800).toContain('crowdOps');
    });
  });

  // -----------------------------------------------------------------------
  // getActiveSubsystems
  // -----------------------------------------------------------------------
  describe('getActiveSubsystems', () => {
    it('returns correct subsystems for various time values', () => {
      const config: SampleRateConfig = {
        pricing: 100,
        staffing: 200,
        marketing: 300,
        crowdOps: 50,
      };

      // t=0 => all fire
      const at0 = getActiveSubsystems(config, 0);
      expect(at0).toHaveLength(4);

      // t=50 => only crowdOps (50 % 50 = 0, 50 % 100 = 50)
      const at50 = getActiveSubsystems(config, 50);
      expect(at50).toEqual(['crowdOps']);

      // t=100 => pricing + crowdOps
      const at100 = getActiveSubsystems(config, 100);
      expect(at100).toContain('pricing');
      expect(at100).toContain('crowdOps');
      expect(at100).not.toContain('staffing');
      expect(at100).not.toContain('marketing');

      // t=600 => all fire (LCM boundary)
      const at600 = getActiveSubsystems(config, 600);
      expect(at600).toHaveLength(4);

      // t=73 => none fire
      const at73 = getActiveSubsystems(config, 73);
      expect(at73).toHaveLength(0);
    });
  });

  // -----------------------------------------------------------------------
  // multiSensorEstimate
  // -----------------------------------------------------------------------
  describe('multiSensorEstimate', () => {
    it('fusing two sensors reduces uncertainty (trace of P decreases)', () => {
      const nx = 2;

      // F = identity (static model)
      const F = Float64Array.from([1, 0, 0, 1]);
      // Small process noise
      const Q = Float64Array.from([0.01, 0, 0, 0.01]);

      // Two sensors each observe the full state with noisy measurements
      const sensorConfig: MultiSensorEstimateConfig = {
        sensors: [
          {
            name: 'cam',
            H: Float64Array.from([1, 0, 0, 1]),
            R: Float64Array.from([1, 0, 0, 1]),
            dimZ: 2,
          },
          {
            name: 'lidar',
            H: Float64Array.from([1, 0, 0, 1]),
            R: Float64Array.from([0.5, 0, 0, 0.5]),
            dimZ: 2,
          },
        ],
        F,
        Q,
        nx,
      };

      // Prior: large uncertainty
      const xHat = Float64Array.from([0, 0]);
      const P0 = Float64Array.from([10, 0, 0, 10]);
      const traceP0 = 20;

      // Measurements close to [1, 2]
      const measurements = new Map<string, Float64Array>();
      measurements.set('cam', Float64Array.from([1.1, 2.1]));
      measurements.set('lidar', Float64Array.from([0.9, 1.9]));

      const result = multiSensorEstimate(sensorConfig, xHat, P0, measurements);

      // After fusing two sensors, posterior covariance trace should be < prior
      const I = matIdentity(nx);
      const Pmat = { data: result.P, rows: nx, cols: nx };
      const tracePosterior = matTrace(Pmat);

      expect(tracePosterior).toBeLessThan(traceP0);

      // State estimate should be near [1, 2]
      expect(result.xHat[0]!).toBeCloseTo(1, 0);
      expect(result.xHat[1]!).toBeCloseTo(2, 0);
    });
  });

  // -----------------------------------------------------------------------
  // movingHorizonEstimate
  // -----------------------------------------------------------------------
  describe('movingHorizonEstimate', () => {
    it('returns state estimate of correct dimension', () => {
      const nx = 2;
      const nz = 2;

      // F = identity, H = identity
      const F = Float64Array.from([1, 0, 0, 1]);
      const H = Float64Array.from([1, 0, 0, 1]);

      // 3 measurements all near [5, 3]
      const measurements = [
        Float64Array.from([5.1, 3.1]),
        Float64Array.from([4.9, 2.9]),
        Float64Array.from([5.0, 3.0]),
      ];

      const est = movingHorizonEstimate(F, H, nx, nz, 3, measurements);

      expect(est.length).toBe(nx);

      // Should be close to [5, 3] given consistent measurements
      expect(est[0]!).toBeCloseTo(5, 0);
      expect(est[1]!).toBeCloseTo(3, 0);
    });

    it('returns zero vector when no measurements provided', () => {
      const nx = 2;
      const nz = 2;
      const F = Float64Array.from([1, 0, 0, 1]);
      const H = Float64Array.from([1, 0, 0, 1]);

      const est = movingHorizonEstimate(F, H, nx, nz, 5, []);

      expect(est.length).toBe(nx);
      expect(est[0]!).toBe(0);
      expect(est[1]!).toBe(0);
    });
  });

  // -----------------------------------------------------------------------
  // Fault-tolerant controller
  // -----------------------------------------------------------------------
  describe('createFaultTolerantController', () => {
    const nx = 2;
    const nu = 1;

    // Nominal controller: u = -[1, 2] * x
    const nominalController = (x: Float64Array): Float64Array => {
      return Float64Array.from([-(x[0]! + 2 * x[1]!)]);
    };

    it('uses nominal controller when all sensors are healthy', () => {
      const config: FaultToleranceConfig = {
        sensorTimeoutMs: 1000,
        fallbackPolicy: 'degraded',
        maxMissedReadings: 3,
        degradedGains: Float64Array.from([0.5, 0.5]),
      };

      const controller = createFaultTolerantController(
        config,
        nominalController,
        nx,
        nu,
      );

      const x = Float64Array.from([1, 2]);
      const healthyStatus = new Map<string, { lastUpdate: number; missedCount: number }>();
      healthyStatus.set('cam', { lastUpdate: 990, missedCount: 0 });
      healthyStatus.set('lidar', { lastUpdate: 995, missedCount: 0 });

      const u = controller(x, healthyStatus, 1000);

      // Nominal: -(1 + 2*2) = -5
      expect(u[0]!).toBeCloseTo(-5, 5);
    });

    it('switches to degraded mode when a sensor fails', () => {
      const config: FaultToleranceConfig = {
        sensorTimeoutMs: 100,
        fallbackPolicy: 'degraded',
        maxMissedReadings: 3,
        degradedGains: Float64Array.from([0.5, 0.5]), // nu=1, nx=2
      };

      const controller = createFaultTolerantController(
        config,
        nominalController,
        nx,
        nu,
      );

      const x = Float64Array.from([2, 4]);
      const faultedStatus = new Map<string, { lastUpdate: number; missedCount: number }>();
      faultedStatus.set('cam', { lastUpdate: 0, missedCount: 10 }); // faulted
      faultedStatus.set('lidar', { lastUpdate: 995, missedCount: 0 });

      const u = controller(x, faultedStatus, 1000);

      // Degraded: u = -K_deg * x = -(0.5*2 + 0.5*4) = -3
      expect(u[0]!).toBeCloseTo(-3, 5);
    });

    it('holds previous action when fallback policy is hold', () => {
      const config: FaultToleranceConfig = {
        sensorTimeoutMs: 100,
        fallbackPolicy: 'hold',
        maxMissedReadings: 3,
        degradedGains: Float64Array.from([0.5, 0.5]),
      };

      const controller = createFaultTolerantController(
        config,
        nominalController,
        nx,
        nu,
      );

      const x = Float64Array.from([1, 1]);
      const healthy = new Map<string, { lastUpdate: number; missedCount: number }>();
      healthy.set('cam', { lastUpdate: 95, missedCount: 0 });

      // First call -- healthy, nominal fires: u = -(1 + 2*1) = -3
      const u1 = controller(x, healthy, 100);
      expect(u1[0]!).toBeCloseTo(-3, 5);

      // Second call -- faulted, should hold the previous action (-3)
      const faulted = new Map<string, { lastUpdate: number; missedCount: number }>();
      faulted.set('cam', { lastUpdate: 0, missedCount: 10 });
      const u2 = controller(Float64Array.from([99, 99]), faulted, 200);
      expect(u2[0]!).toBeCloseTo(-3, 5);
    });

    it('returns zeros when fallback policy is safe', () => {
      const config: FaultToleranceConfig = {
        sensorTimeoutMs: 100,
        fallbackPolicy: 'safe',
        maxMissedReadings: 3,
        degradedGains: Float64Array.from([0.5, 0.5]),
      };

      const controller = createFaultTolerantController(
        config,
        nominalController,
        nx,
        nu,
      );

      const faulted = new Map<string, { lastUpdate: number; missedCount: number }>();
      faulted.set('cam', { lastUpdate: 0, missedCount: 10 });

      const u = controller(Float64Array.from([5, 5]), faulted, 1000);
      expect(u[0]!).toBe(0);
    });
  });
});
