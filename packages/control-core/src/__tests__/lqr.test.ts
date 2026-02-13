import { describe, it, expect } from 'vitest';
import {
  solveDARE,
  computeLQRGain,
  discreteLQR,
  simulateLQR,
  inverseTolerance,
  venueDefaultQR,
  timeVaryingLQR,
  simulateTimeVaryingLQR,
  trackingLQR,
  simulateTracking,
  designLQG,
  createLQGState,
  lqgStep,
} from '../lqr/index.js';
import {
  arrayToMatrix,
  matGet,
  matVecMul,
  vecNorm,
} from '../types.js';
import type {
  LQRConfig,
  TimeVaryingLQRConfig,
  TrackingLQRConfig,
  LQGConfig,
} from '../types.js';

// ---------------------------------------------------------------------------
// Helper: build a stable 2-state discrete-time system
//
//   A = [[0.9, 0.1],    B = [[0],
//        [0.0, 0.8]]         [1]]
//
//   Q = I_2,   R = [1]
// ---------------------------------------------------------------------------

function make2x1Config(): LQRConfig {
  return {
    A: new Float64Array([0.9, 0.1, 0.0, 0.8]),
    B: new Float64Array([0, 1]),
    Q: new Float64Array([1, 0, 0, 1]),
    R: new Float64Array([1]),
    nx: 2,
    nu: 1,
  };
}

// ---------------------------------------------------------------------------
// 1. DARE convergence and positive semi-definiteness
// ---------------------------------------------------------------------------

describe('LQR / DARE', () => {
  it('DARE converges for stable 2-state system and P is positive semi-definite', () => {
    const cfg = make2x1Config();
    const result = solveDARE(cfg);

    // P should exist and have correct length
    expect(result.P.length).toBe(cfg.nx * cfg.nx);

    // Positive semi-definiteness proxy: diagonal elements non-negative
    const P = arrayToMatrix(result.P, cfg.nx, cfg.nx);
    for (let i = 0; i < cfg.nx; i++) {
      expect(matGet(P, i, i)).toBeGreaterThanOrEqual(0);
    }

    // Symmetry: P[i][j] ~ P[j][i]
    for (let i = 0; i < cfg.nx; i++) {
      for (let j = i + 1; j < cfg.nx; j++) {
        expect(matGet(P, i, j)).toBeCloseTo(matGet(P, j, i), 10);
      }
    }

    // Check via quadratic form: x^T P x >= 0 for a random x
    const x = new Float64Array([1, -2]);
    const Px = matVecMul(P, x);
    let xTPx = 0;
    for (let i = 0; i < cfg.nx; i++) {
      xTPx += x[i]! * Px[i]!;
    }
    expect(xTPx).toBeGreaterThanOrEqual(-1e-10);
  });

  // ---------------------------------------------------------------------------
  // 2. LQR gain K stabilises the closed-loop
  // ---------------------------------------------------------------------------

  it('LQR gain K stabilises the closed-loop (state decays to zero)', () => {
    const cfg = make2x1Config();
    const result = discreteLQR(cfg);

    // Simulate 50 steps starting from a non-zero initial condition
    const x0 = new Float64Array([5, -3]);
    const traj = simulateLQR(cfg.A, cfg.B, result.K, x0, 50, cfg.nx, cfg.nu);

    // The state norm should decay
    const normFirst = vecNorm(traj[0]!);
    const normLast = vecNorm(traj[50]!);

    expect(normLast).toBeLessThan(normFirst * 0.01);

    // Also check via eigenvalue magnitudes from result
    const eigs = result.eigenvalues;
    for (let i = 0; i < cfg.nx; i++) {
      const re = eigs[2 * i]!;
      const im = eigs[2 * i + 1]!;
      const mag = Math.sqrt(re * re + im * im);
      expect(mag).toBeLessThan(1);
    }
  });

  // ---------------------------------------------------------------------------
  // 3. inverseTolerance produces correct diagonal Q, R
  // ---------------------------------------------------------------------------

  it('inverseTolerance produces diagonal Q, R with correct reciprocals', () => {
    const tol = new Float64Array([2, 5]);
    const w = new Float64Array([1, 1]);
    const ctol = new Float64Array([10]);
    const cw = new Float64Array([1]);

    const { Q, R } = inverseTolerance({
      tolerances: tol,
      weights: w,
      controlTolerances: ctol,
      controlWeights: cw,
    });

    // Q should be 2x2 diagonal: diag(1/4, 1/25)
    const Qm = arrayToMatrix(Q, 2, 2);
    expect(matGet(Qm, 0, 0)).toBeCloseTo(1 / 4, 10);
    expect(matGet(Qm, 1, 1)).toBeCloseTo(1 / 25, 10);
    expect(matGet(Qm, 0, 1)).toBeCloseTo(0, 10);
    expect(matGet(Qm, 1, 0)).toBeCloseTo(0, 10);

    // R should be 1x1: [1/100]
    const Rm = arrayToMatrix(R, 1, 1);
    expect(matGet(Rm, 0, 0)).toBeCloseTo(1 / 100, 10);
  });

  // ---------------------------------------------------------------------------
  // 4. venueDefaultQR returns sensible defaults
  // ---------------------------------------------------------------------------

  it('venueDefaultQR returns sensible defaults for 4-state 3-control system', () => {
    const { Q, R } = venueDefaultQR(4, 3);

    // Should be 4x4 and 3x3
    expect(Q.length).toBe(16);
    expect(R.length).toBe(9);

    // All diagonal entries should be positive
    const Qm = arrayToMatrix(Q, 4, 4);
    const Rm = arrayToMatrix(R, 3, 3);
    for (let i = 0; i < 4; i++) {
      expect(matGet(Qm, i, i)).toBeGreaterThan(0);
    }
    for (let i = 0; i < 3; i++) {
      expect(matGet(Rm, i, i)).toBeGreaterThan(0);
    }
  });

  // ---------------------------------------------------------------------------
  // 5. Time-varying LQR: gains approach infinite-horizon at long horizon
  // ---------------------------------------------------------------------------

  it('time-varying LQR gains approach infinite-horizon LQR at long horizon', () => {
    const cfg = make2x1Config();

    // Solve infinite-horizon LQR
    const infResult = discreteLQR(cfg);

    // Build time-varying config with constant matrices and long horizon
    const horizon = 100;
    const tvCfg: TimeVaryingLQRConfig = {
      As: Array.from({ length: horizon }, () => new Float64Array(cfg.A)),
      Bs: Array.from({ length: horizon }, () => new Float64Array(cfg.B)),
      Qs: Array.from({ length: horizon }, () => new Float64Array(cfg.Q)),
      Rs: Array.from({ length: horizon }, () => new Float64Array(cfg.R)),
      Qf: new Float64Array(cfg.Q),
      nx: cfg.nx,
      nu: cfg.nu,
      horizon,
    };

    const tvResult = timeVaryingLQR(tvCfg);

    // Early gains (near t=0) should be close to infinite-horizon K
    const K0 = tvResult.Ks[0]!;
    for (let i = 0; i < cfg.nu * cfg.nx; i++) {
      expect(K0[i]!).toBeCloseTo(infResult.K[i]!, 2);
    }
  });

  // ---------------------------------------------------------------------------
  // 6. Tracking LQR: output converges to reference
  // ---------------------------------------------------------------------------

  it('tracking LQR: output converges to reference', () => {
    const cfg = make2x1Config();

    const trackCfg: TrackingLQRConfig = {
      ...cfg,
      C: new Float64Array([1, 0]), // y = x[0]
      ny: 1,
    };

    const result = trackingLQR(trackCfg);

    // Simulate tracking a reference of y = 2.0
    const x0 = new Float64Array([0, 0]);
    const nSteps = 100;
    const reference = Array.from({ length: nSteps }, () => new Float64Array([2.0]));

    const traj = simulateTracking(
      cfg.A, cfg.B, result.K, result.Kff, trackCfg.C,
      x0, reference, cfg.nx, cfg.nu, trackCfg.ny,
    );

    // After many steps the first state (output) should be near the reference
    const xFinal = traj[nSteps]!;
    const yFinal = xFinal[0]!; // C = [1, 0], so y = x[0]
    // With integral action the output should approach the reference
    // Allow generous tolerance as tracking is iterative
    expect(Math.abs(yFinal - 2.0)).toBeLessThan(1.0);
  });

  // ---------------------------------------------------------------------------
  // 7. LQG separation principle: designLQG produces both K and L gains
  // ---------------------------------------------------------------------------

  it('LQG separation principle: designLQG produces both K and L gains', () => {
    const cfg = make2x1Config();

    const lqgCfg: LQGConfig = {
      ...cfg,
      C: new Float64Array([1, 0]), // observe first state
      Qn: new Float64Array([0.01, 0, 0, 0.01]), // process noise cov
      Rn: new Float64Array([0.1]), // measurement noise cov
      ny: 1,
    };

    const result = designLQG(lqgCfg);

    // LQR gain K should exist and have correct size (nu x nx)
    expect(result.lqr.K.length).toBe(cfg.nu * cfg.nx);

    // Kalman gain L should exist and have correct size (nx x ny)
    expect(result.L.length).toBe(cfg.nx * lqgCfg.ny);

    // Filter error covariance Pf should be positive semi-definite
    const Pf = arrayToMatrix(result.Pf, cfg.nx, cfg.nx);
    for (let i = 0; i < cfg.nx; i++) {
      expect(matGet(Pf, i, i)).toBeGreaterThanOrEqual(0);
    }
  });

  // ---------------------------------------------------------------------------
  // 8. LQG step: state estimate converges toward true state
  // ---------------------------------------------------------------------------

  it('LQG step: state estimate converges toward true state', () => {
    const cfg = make2x1Config();

    const lqgCfg: LQGConfig = {
      ...cfg,
      C: new Float64Array([1, 0]),
      Qn: new Float64Array([0.01, 0, 0, 0.01]),
      Rn: new Float64Array([0.1]),
      ny: 1,
    };

    const result = designLQG(lqgCfg);
    let state = createLQGState(lqgCfg);

    // True state
    const mA = arrayToMatrix(cfg.A, cfg.nx, cfg.nx);
    const mB = arrayToMatrix(cfg.B, cfg.nx, cfg.nu);
    let xTrue = new Float64Array([3, -1]);

    // Run 30 steps; feed the true measurement y = C * xTrue
    for (let t = 0; t < 30; t++) {
      const y = new Float64Array([xTrue[0]!]); // C = [1, 0]
      const step = lqgStep(state, y, lqgCfg, result);
      state = step.nextState;

      // Apply control to true system: x_{t+1} = A * xTrue + B * u
      const Ax = matVecMul(mA, xTrue);
      const Bu = matVecMul(mB, step.u);
      const xNext = new Float64Array(cfg.nx);
      for (let i = 0; i < cfg.nx; i++) {
        xNext[i] = Ax[i]! + Bu[i]!;
      }
      xTrue = xNext;
    }

    // After 30 steps, the estimate should be close to the true state
    const estErr = vecNorm(
      new Float64Array([
        state.xHat[0]! - xTrue[0]!,
        state.xHat[1]! - xTrue[1]!,
      ]),
    );
    // Both should be near zero (system is being controlled to origin)
    const trueNorm = vecNorm(xTrue);
    expect(trueNorm).toBeLessThan(1.0);
    expect(estErr).toBeLessThan(1.0);
  });
});
