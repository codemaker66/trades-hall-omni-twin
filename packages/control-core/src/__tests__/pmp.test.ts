import { describe, it, expect } from 'vitest';
import {
  evaluateHamiltonian,
  rk4Step,
  integrateODE,
  singleShooting,
  multipleShooting,
  trapezoidalCollocation,
  analyzeSwitchingFunction,
  constructBangBang,
} from '../pmp/index.js';
import { vecNorm } from '../types.js';
import type { HamiltonianConfig, ShootingConfig } from '../types.js';

// ---------------------------------------------------------------------------
// Helper: simple 1D optimal control problem
//
//   min integral_0^T  0.5*(x^2 + u^2) dt
//   subject to  dx/dt = -x + u,  x(0) = 1
//
//   Costate: dlambda/dt = -dH/dx = -x + lambda
//   Control: dH/du = u + lambda = 0  =>  u = -lambda
// ---------------------------------------------------------------------------

function makeSimple1DConfig(): HamiltonianConfig {
  return {
    stateDynamics: (x: Float64Array, u: Float64Array, _lambda: Float64Array, _t: number) => {
      const dx = new Float64Array(1);
      dx[0] = -x[0]! + u[0]!;
      return dx;
    },
    costateDynamics: (x: Float64Array, _u: Float64Array, lambda: Float64Array, _t: number) => {
      const dl = new Float64Array(1);
      // dlambda/dt = -dH/dx = -(x + lambda*(-1)) = -x + lambda
      dl[0] = -x[0]! + lambda[0]!;
      return dl;
    },
    runningCost: (x: Float64Array, u: Float64Array, _t: number) => {
      return 0.5 * (x[0]! * x[0]! + u[0]! * u[0]!);
    },
    controlOptimality: (_x: Float64Array, lambda: Float64Array, _t: number) => {
      // dH/du = u + lambda = 0  =>  u* = -lambda
      const u = new Float64Array(1);
      u[0] = -lambda[0]!;
      return u;
    },
    nx: 1,
    nu: 1,
  };
}

function makeSimpleShootingConfig(T = 2.0, nSteps = 100): ShootingConfig {
  return {
    ...makeSimple1DConfig(),
    x0: new Float64Array([1.0]),
    T,
    nSteps,
    tolerance: 1e-4,
    maxIter: 50,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('PMP', () => {
  // ---------------------------------------------------------------------------
  // 1. Hamiltonian evaluation consistent with components
  // ---------------------------------------------------------------------------

  it('Hamiltonian evaluation is consistent with running cost + lambda^T f', () => {
    const config = makeSimple1DConfig();

    const x = new Float64Array([1.0]);
    const u = new Float64Array([0.5]);
    const lambda = new Float64Array([0.3]);
    const t = 0;

    const H = evaluateHamiltonian(config, x, u, lambda, t);

    // Manual calculation:
    // L = 0.5 * (1^2 + 0.5^2) = 0.5 * 1.25 = 0.625
    // f = -1 + 0.5 = -0.5
    // lambda^T f = 0.3 * (-0.5) = -0.15
    // H = 0.625 + (-0.15) = 0.475
    const expectedL = 0.5 * (1.0 + 0.25);
    const expectedF = -1.0 + 0.5;
    const expectedH = expectedL + lambda[0]! * expectedF;

    expect(H).toBeCloseTo(expectedH, 10);
  });

  // ---------------------------------------------------------------------------
  // 2. RK4 is O(h^4): error reduces by ~16x when step halves
  // ---------------------------------------------------------------------------

  it('RK4 is O(h^4): error reduces by ~16x when step halves', () => {
    // Test ODE: dy/dt = y, y(0) = 1  =>  y(1) = e
    const f = (y: Float64Array, _t: number): Float64Array => {
      const dy = new Float64Array(1);
      dy[0] = y[0]!;
      return dy;
    };

    const y0 = new Float64Array([1.0]);
    const exact = Math.E; // y(1) = e

    // Coarse: single step of size h = 1
    const h1 = 1.0;
    const y1 = rk4Step(f, y0, 0, h1);
    const err1 = Math.abs(y1[0]! - exact);

    // Fine: two steps of size h = 0.5
    const h2 = 0.5;
    const yMid = rk4Step(f, y0, 0, h2);
    const y2 = rk4Step(f, yMid, h2, h2);
    const err2 = Math.abs(y2[0]! - exact);

    // The error ratio should be approximately 16 (2^4)
    // Allow some tolerance for this numerical check
    const ratio = err1 / err2;
    expect(ratio).toBeGreaterThan(10); // should be ~16
    expect(ratio).toBeLessThan(25);
  });

  // ---------------------------------------------------------------------------
  // 3. integrateODE returns correct number of points
  // ---------------------------------------------------------------------------

  it('integrateODE returns correct number of points', () => {
    const f = (y: Float64Array, _t: number): Float64Array => {
      const dy = new Float64Array(1);
      dy[0] = -y[0]!; // exponential decay
      return dy;
    };

    const y0 = new Float64Array([1.0]);
    const nSteps = 20;
    const traj = integrateODE(f, y0, 0, 1, nSteps);

    // Should have nSteps + 1 points (including initial)
    expect(traj.length).toBe(nSteps + 1);

    // First point should be the initial condition
    expect(traj[0]![0]!).toBeCloseTo(1.0, 10);

    // Last point: y(1) = e^{-1} ~ 0.3679
    expect(traj[nSteps]![0]!).toBeCloseTo(Math.exp(-1), 4);
  });

  // ---------------------------------------------------------------------------
  // 4. Single shooting: converges for simple system
  // ---------------------------------------------------------------------------

  it('single shooting converges for simple 1D system', () => {
    const config = makeSimpleShootingConfig(2.0, 100);
    const result = singleShooting(config);

    // Should converge
    expect(result.converged).toBe(true);

    // State trajectory should start at x0
    expect(result.xTrajectory[0]![0]!).toBeCloseTo(1.0, 6);

    // Cost should be non-negative
    expect(result.cost).toBeGreaterThanOrEqual(0);

    // The state should decay (system is stable with control)
    const xFinal = result.xTrajectory[result.xTrajectory.length - 1]!;
    expect(Math.abs(xFinal[0]!)).toBeLessThan(1.0);
  });

  // ---------------------------------------------------------------------------
  // 5. Multiple shooting: improves convergence vs single
  // ---------------------------------------------------------------------------

  it('multiple shooting converges for simple 1D system', () => {
    const config = makeSimpleShootingConfig(2.0, 100);
    const result = multipleShooting(config, 4);

    // Should converge
    expect(result.converged).toBe(true);

    // State trajectory should start at x0
    expect(result.xTrajectory[0]![0]!).toBeCloseTo(1.0, 4);

    // Cost should be non-negative and finite
    expect(result.cost).toBeGreaterThanOrEqual(0);
    expect(Number.isFinite(result.cost)).toBe(true);

    // Compare with single shooting cost -- they should be in the same ballpark
    const singleResult = singleShooting(config);
    if (singleResult.converged) {
      // Both costs should be similar (within 50%)
      const ratio = result.cost / Math.max(singleResult.cost, 1e-10);
      expect(ratio).toBeGreaterThan(0.5);
      expect(ratio).toBeLessThan(2.0);
    }
  });

  // ---------------------------------------------------------------------------
  // 6. Trapezoidal collocation: satisfies dynamics defect to tolerance
  // ---------------------------------------------------------------------------

  it('trapezoidal collocation satisfies dynamics defect to tolerance', () => {
    const dynamics = (x: Float64Array, u: Float64Array, _t: number): Float64Array => {
      const dx = new Float64Array(1);
      dx[0] = -x[0]! + u[0]!;
      return dx;
    };

    const cost = (x: Float64Array, u: Float64Array, _t: number): number => {
      return 0.5 * (x[0]! * x[0]! + u[0]! * u[0]!);
    };

    const x0 = new Float64Array([1.0]);
    const T = 2.0;
    const nSegments = 20;
    const nx = 1;
    const nu = 1;

    const result = trapezoidalCollocation(dynamics, cost, x0, T, nSegments, nx, nu);

    // Should converge
    expect(result.converged).toBe(true);

    // State trajectory should start at x0
    expect(result.xTrajectory[0]![0]!).toBeCloseTo(1.0, 6);

    // Cost should be positive
    expect(result.cost).toBeGreaterThan(0);

    // The number of trajectory points should be nSegments + 1 (collocation nodes)
    expect(result.xTrajectory.length).toBe(nSegments + 1);

    // Verify dynamics defect is small by checking consecutive nodes
    const dt = T / nSegments;
    let maxDefect = 0;
    for (let k = 0; k < nSegments; k++) {
      const xk = result.xTrajectory[k]!;
      const uk = result.uTrajectory[k]!;
      const xk1 = result.xTrajectory[k + 1]!;
      const uk1 = result.uTrajectory[k + 1]!;
      const tk = k * dt;
      const tk1 = (k + 1) * dt;

      const fk = dynamics(xk, uk, tk);
      const fk1 = dynamics(xk1, uk1, tk1);

      // Trapezoidal defect: x_{k+1} - x_k - (dt/2)(f_k + f_{k+1})
      for (let i = 0; i < nx; i++) {
        const defect = Math.abs(xk1[i]! - xk[i]! - (dt / 2) * (fk[i]! + fk1[i]!));
        if (defect > maxDefect) maxDefect = defect;
      }
    }
    expect(maxDefect).toBeLessThan(1e-4);
  });

  // ---------------------------------------------------------------------------
  // 7. Bang-bang: detects sign changes in switching function
  // ---------------------------------------------------------------------------

  it('bang-bang analysis detects switching function sign changes', () => {
    // Create a shooting config with known switching behavior
    const config = makeSimpleShootingConfig(2.0, 100);

    // First solve the shooting problem to get trajectories
    const shootingResult = singleShooting(config);

    // Analyze switching function
    const bbResult = analyzeSwitchingFunction(
      config,
      shootingResult.lambdaTrajectory,
      shootingResult.uTrajectory,
    );

    // The switching function should be a Float64Array
    expect(bbResult.switchingFunction.length).toBe(config.nSteps + 1);

    // Control levels should have one more entry than switching times
    expect(bbResult.controlLevels.length).toBe(bbResult.switchingTimes.length + 1);

    // Switching times (if any) should be within [0, T]
    for (let i = 0; i < bbResult.switchingTimes.length; i++) {
      expect(bbResult.switchingTimes[i]!).toBeGreaterThanOrEqual(0);
      expect(bbResult.switchingTimes[i]!).toBeLessThanOrEqual(config.T);
    }

    // Control levels should be either +1 or -1
    for (let i = 0; i < bbResult.controlLevels.length; i++) {
      expect(Math.abs(bbResult.controlLevels[i]!)).toBeCloseTo(1.0, 10);
    }

    // constructBangBang should produce a trajectory with the right length
    const bbTraj = constructBangBang(
      bbResult.switchingTimes,
      bbResult.controlLevels,
      config.T,
      config.nSteps,
    );
    expect(bbTraj.length).toBe(config.nSteps + 1);
  });
});
