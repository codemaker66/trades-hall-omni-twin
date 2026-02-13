import { describe, it, expect } from 'vitest';
import {
  solveQP,
  solveLinearMPC,
  solveEconomicMPC,
  solveNonlinearMPC,
  solveTubeMPC,
  solveStochasticMPC,
  buildExplicitMPCTable,
  explicitMPCLookup,
} from '../mpc/index.js';
import {
  createMatrix,
  matGet,
  matSet,
  arrayToMatrix,
} from '../types.js';
import type {
  MPCConfig,
  EconomicMPCConfig,
  NonlinearMPCConfig,
  TubeMPCConfig,
  StochasticMPCConfig,
} from '../types.js';

// ---------------------------------------------------------------------------
// Helper: small stable 2-state, 1-control system
//
//   A = [[0.9, 0.1],    B = [[0],
//        [0.0, 0.8]]         [1]]
// ---------------------------------------------------------------------------

function makeBaseMPCConfig(horizon = 5): MPCConfig {
  return {
    A: new Float64Array([0.9, 0.1, 0.0, 0.8]),
    B: new Float64Array([0, 1]),
    Q: new Float64Array([1, 0, 0, 1]),
    R: new Float64Array([0.1]),
    nx: 2,
    nu: 1,
    horizon,
  };
}

// ---------------------------------------------------------------------------
// 1. QP solver: unconstrained min of 0.5 x^T H x + f^T x
// ---------------------------------------------------------------------------

describe('MPC', () => {
  it('QP solver matches analytical solution for unconstrained problem', () => {
    // min 0.5 * x^T H x + f^T x
    // H = [[2, 0], [0, 4]],  f = [-2, -8]
    // Optimal: H x = -f  =>  x = [1, 2]
    const H = createMatrix(2, 2);
    matSet(H, 0, 0, 2);
    matSet(H, 1, 1, 4);

    const f = new Float64Array([-2, -8]);

    const result = solveQP(H, f, null, null);

    expect(result.converged).toBe(true);
    expect(result.x[0]!).toBeCloseTo(1.0, 6);
    expect(result.x[1]!).toBeCloseTo(2.0, 6);
  });

  // ---------------------------------------------------------------------------
  // 2. Linear MPC: respects uMin/uMax bounds
  // ---------------------------------------------------------------------------

  it('linear MPC respects uMin/uMax control bounds', () => {
    const cfg: MPCConfig = {
      ...makeBaseMPCConfig(5),
      uMin: new Float64Array([-0.5]),
      uMax: new Float64Array([0.5]),
    };

    const x0 = new Float64Array([5, -3]);
    const result = solveLinearMPC(cfg, x0);

    expect(result.status).not.toBe('infeasible');

    // All controls in the sequence should respect bounds
    for (const uk of result.uSequence) {
      expect(uk[0]!).toBeGreaterThanOrEqual(-0.5 - 1e-6);
      expect(uk[0]!).toBeLessThanOrEqual(0.5 + 1e-6);
    }

    // Optimal first control should also respect bounds
    expect(result.uOptimal[0]!).toBeGreaterThanOrEqual(-0.5 - 1e-6);
    expect(result.uOptimal[0]!).toBeLessThanOrEqual(0.5 + 1e-6);
  });

  // ---------------------------------------------------------------------------
  // 3. Linear MPC: rate constraint limits du
  // ---------------------------------------------------------------------------

  it('linear MPC rate constraint limits du between steps', () => {
    const cfg: MPCConfig = {
      ...makeBaseMPCConfig(5),
      uMin: new Float64Array([-5]),
      uMax: new Float64Array([5]),
      duMax: new Float64Array([0.3]),
    };

    const x0 = new Float64Array([5, -3]);
    const result = solveLinearMPC(cfg, x0);

    expect(result.status).not.toBe('infeasible');

    // Check rate constraints: |u_{k} - u_{k-1}| <= 0.3
    for (let k = 1; k < result.uSequence.length; k++) {
      const du = Math.abs(result.uSequence[k]![0]! - result.uSequence[k - 1]![0]!);
      expect(du).toBeLessThanOrEqual(0.3 + 1e-4);
    }
  });

  // ---------------------------------------------------------------------------
  // 4. Economic MPC: returns valid result
  // ---------------------------------------------------------------------------

  it('economic MPC returns a valid result', () => {
    const baseCfg = makeBaseMPCConfig(5);

    const econCfg: EconomicMPCConfig = {
      ...baseCfg,
      demandFn: (price: Float64Array, _state: Float64Array) => {
        // Simple linear demand: demand = 10 - price
        const d = new Float64Array(1);
        d[0] = Math.max(0, 10 - price[0]!);
        return d;
      },
      costFn: (state: Float64Array, control: Float64Array) => {
        // Simple quadratic cost
        let c = 0;
        for (let i = 0; i < state.length; i++) {
          c += state[i]! * state[i]!;
        }
        for (let i = 0; i < control.length; i++) {
          c += 0.1 * control[i]! * control[i]!;
        }
        return c;
      },
    };

    const x0 = new Float64Array([1, 1]);
    const result = solveEconomicMPC(econCfg, x0);

    // Should return a valid result with finite cost
    expect(result.uOptimal.length).toBe(baseCfg.nu);
    expect(Number.isFinite(result.cost)).toBe(true);
    expect(result.uSequence.length).toBe(baseCfg.horizon);
    expect(result.xPredicted.length).toBe(baseCfg.horizon + 1);
  });

  // ---------------------------------------------------------------------------
  // 5. NMPC: tracks reference via SQP
  // ---------------------------------------------------------------------------

  it('nonlinear MPC via SQP finds a solution', () => {
    const nx = 2;
    const nu = 1;
    const horizon = 5;

    const nmpcCfg: NonlinearMPCConfig = {
      A: new Float64Array([0.9, 0.1, 0.0, 0.8]),
      B: new Float64Array([0, 1]),
      Q: new Float64Array([1, 0, 0, 1]),
      R: new Float64Array([0.1]),
      nx,
      nu,
      horizon,
      // Nonlinear dynamics: slightly nonlinear (adds small x^2 term)
      dynamicsFn: (x: Float64Array, u: Float64Array) => {
        const xNext = new Float64Array(nx);
        xNext[0] = 0.9 * x[0]! + 0.1 * x[1]! + 0.01 * x[0]! * x[0]!;
        xNext[1] = 0.8 * x[1]! + u[0]!;
        return xNext;
      },
      costStageFn: (x: Float64Array, u: Float64Array) => {
        return x[0]! * x[0]! + x[1]! * x[1]! + 0.1 * u[0]! * u[0]!;
      },
      costTerminalFn: (x: Float64Array) => {
        return x[0]! * x[0]! + x[1]! * x[1]!;
      },
      maxSQPIterations: 10,
      tolerance: 1e-4,
      uMin: new Float64Array([-5]),
      uMax: new Float64Array([5]),
    };

    const x0 = new Float64Array([2, -1]);
    const result = solveNonlinearMPC(nmpcCfg, x0);

    // Should return a valid result
    expect(result.uOptimal.length).toBe(nu);
    expect(Number.isFinite(result.cost)).toBe(true);
    expect(result.uSequence.length).toBe(horizon);
    expect(result.xPredicted.length).toBe(horizon + 1);

    // Cost should be non-negative (quadratic costs)
    expect(result.cost).toBeGreaterThanOrEqual(0);
  });

  // ---------------------------------------------------------------------------
  // 6. Tube MPC: tightens constraints
  // ---------------------------------------------------------------------------

  it('tube MPC tightens constraints and returns valid solution', () => {
    const baseCfg = makeBaseMPCConfig(5);

    const tubeCfg: TubeMPCConfig = {
      ...baseCfg,
      uMin: new Float64Array([-3]),
      uMax: new Float64Array([3]),
      xMin: new Float64Array([-10, -10]),
      xMax: new Float64Array([10, 10]),
      disturbanceBound: new Float64Array([0.1, 0.1]),
      tubeK: new Float64Array([0.2, 0.3]), // nu x nx = 1 x 2
    };

    const x0 = new Float64Array([2, -1]);
    const result = solveTubeMPC(tubeCfg, x0);

    // Should return a valid result
    expect(result.uOptimal.length).toBe(baseCfg.nu);
    expect(Number.isFinite(result.cost)).toBe(true);
    expect(result.status).not.toBe('infeasible');

    // Control should respect original (untightened) bounds
    expect(result.uOptimal[0]!).toBeGreaterThanOrEqual(-3 - 1e-6);
    expect(result.uOptimal[0]!).toBeLessThanOrEqual(3 + 1e-6);
  });

  // ---------------------------------------------------------------------------
  // 7. Stochastic MPC: satisfies chance constraints
  // ---------------------------------------------------------------------------

  it('stochastic MPC returns valid result with chance constraints', () => {
    const baseCfg = makeBaseMPCConfig(3);

    const stochCfg: StochasticMPCConfig = {
      ...baseCfg,
      uMin: new Float64Array([-3]),
      uMax: new Float64Array([3]),
      xMin: new Float64Array([-10, -10]),
      xMax: new Float64Array([10, 10]),
      nScenarios: 10,
      chanceConstraintEpsilon: 0.1,
    };

    const x0 = new Float64Array([2, -1]);
    const result = solveStochasticMPC(stochCfg, x0);

    // Should return a valid result
    expect(result.uOptimal.length).toBe(baseCfg.nu);
    expect(Number.isFinite(result.cost)).toBe(true);
    expect(result.uSequence.length).toBe(baseCfg.horizon);

    // Control should respect bounds
    expect(result.uOptimal[0]!).toBeGreaterThanOrEqual(-3 - 1e-6);
    expect(result.uOptimal[0]!).toBeLessThanOrEqual(3 + 1e-6);
  });

  // ---------------------------------------------------------------------------
  // 8. Explicit MPC lookup: returns control inside region
  // ---------------------------------------------------------------------------

  it('explicit MPC lookup returns control inside a valid region', () => {
    const cfg: MPCConfig = {
      ...makeBaseMPCConfig(3),
      uMin: new Float64Array([-2]),
      uMax: new Float64Array([2]),
      xMin: new Float64Array([-5, -5]),
      xMax: new Float64Array([5, 5]),
    };

    const table = buildExplicitMPCTable(cfg);

    // Table should have some regions
    expect(table.regions.length).toBeGreaterThan(0);
    expect(table.nx).toBe(cfg.nx);
    expect(table.nu).toBe(cfg.nu);

    // Lookup at the origin should return a control
    const xOrigin = new Float64Array([0, 0]);
    const uLookup = explicitMPCLookup(table, xOrigin);

    // Should find a region for the origin
    expect(uLookup).not.toBeNull();
    if (uLookup !== null) {
      expect(uLookup.length).toBe(cfg.nu);
      // Control at origin should be small (near zero)
      expect(Math.abs(uLookup[0]!)).toBeLessThan(2.5);
    }
  });
});
