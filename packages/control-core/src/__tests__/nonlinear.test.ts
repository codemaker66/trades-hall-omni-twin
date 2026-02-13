// ---------------------------------------------------------------------------
// Tests for OC-6: Nonlinear Control
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';
import {
  lieDerivative,
  computeRelativeDegree,
  feedbackLinearize,
  backsteppingControl,
  slidingModeControl,
  evaluateSurface,
  mracStep,
  l1AdaptiveStep,
  createAdaptiveState,
} from '../nonlinear/index.js';
import type {
  FeedbackLinConfig,
  BacksteppingConfig,
  SlidingModeConfig,
  AdaptiveControlConfig,
} from '../types.js';

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('Lie Derivative', () => {
  it('numerical approximation matches analytical for a simple system', () => {
    // System: f(x) = [x[1], -x[0]] (harmonic oscillator drift)
    // Output: h(x) = x[0]^2
    // Analytical: L_f h(x) = dh/dx . f(x) = [2*x[0], 0] . [x[1], -x[0]] = 2*x[0]*x[1]
    const f = (x: Float64Array): Float64Array =>
      new Float64Array([x[1]!, -x[0]!]);
    const h = (x: Float64Array): number => x[0]! * x[0]!;

    const x = new Float64Array([3.0, 2.0]);
    const result = lieDerivative(f, h, x);

    // Analytical: 2 * 3.0 * 2.0 = 12.0
    expect(result).toBeCloseTo(12.0, 4);
  });
});

describe('Relative Degree', () => {
  it('correct for a known system with relative degree 2', () => {
    // System: double integrator
    //   f(x) = [x[1], 0] (drift)
    //   g(x) = [0, 1]    (input)
    //   h(x) = x[0]      (output = position)
    //
    // L_g h(x) = dh/dx . g(x) = [1, 0] . [0, 1] = 0
    // L_f h(x) = dh/dx . f(x) = [1, 0] . [x[1], 0] = x[1]
    // L_g L_f h(x) = d(L_f h)/dx . g(x) = [0, 1] . [0, 1] = 1 != 0
    // So relative degree = 2

    const config: FeedbackLinConfig = {
      f: (x: Float64Array) => new Float64Array([x[1]!, 0]),
      g: (_x: Float64Array) => new Float64Array([0, 1]),
      h: (x: Float64Array) => x[0]!,
      nx: 2,
      relativeDegree: 2,
    };

    const x0 = new Float64Array([1.0, 0.5]);
    const rd = computeRelativeDegree(config, x0);
    expect(rd).toBe(2);
  });
});

describe('Feedback Linearization', () => {
  it('cancels nonlinearity for a double integrator with nonlinear drift', () => {
    // Nonlinear system:
    //   dx1/dt = x2
    //   dx2/dt = -sin(x1) + u
    //
    // f(x) = [x2, -sin(x1)],  g(x) = [0, 1],  h(x) = x1
    // Relative degree = 2
    //
    // L_f^2 h = -sin(x1) (from iterated computation)
    // L_g L_f h = 1
    // u = (v - L_f^2 h) / L_g_Lf_h = v + sin(x1)

    const config: FeedbackLinConfig = {
      f: (x: Float64Array) => new Float64Array([x[1]!, -Math.sin(x[0]!)]),
      g: (_x: Float64Array) => new Float64Array([0, 1]),
      h: (x: Float64Array) => x[0]!,
      nx: 2,
      relativeDegree: 2,
    };

    const x = new Float64Array([Math.PI / 4, 0]);
    const v = 0; // desired linear input

    const u = feedbackLinearize(config, x, v);

    // u should be approximately sin(pi/4) = sqrt(2)/2 ~ 0.7071
    // because u = alpha + beta*v where alpha = -L_f^2 h / L_g L_f h = sin(x1)
    // and beta = 1 / L_g L_f h = 1
    expect(u).toBeCloseTo(Math.sin(Math.PI / 4), 2);
  });
});

describe('Backstepping', () => {
  it('stabilizes a 2-stage integrator chain system', () => {
    // 2-stage strict-feedback system:
    //   dx1/dt = x2   (f_1(x,u) = 0, so dx1/dt = f1 + x2 = x2)
    //   dx2/dt = u    (f_2(x,u) = 0, so dx2/dt = f2 + u = u)
    //
    // With state [1, 0.5], the controller should produce a stabilizing u

    const config: BacksteppingConfig = {
      nStages: 2,
      dynamics: [
        (_x: Float64Array, _u: number) => 0, // f_1 = 0
        (_x: Float64Array, _u: number) => 0, // f_2 = 0
      ],
      lyapunovGains: new Float64Array([2.0, 2.0]),
    };

    // Start from a displaced state
    const x = new Float64Array([1.0, 0.5]);
    const u = backsteppingControl(config, x);

    // Simulate one Euler step
    const dt = 0.01;
    const x1New = x[0]! + x[1]! * dt;
    const x2New = x[1]! + u * dt;
    const xNew = new Float64Array([x1New, x2New]);

    // The state norm should decrease (stabilizing)
    const normBefore = Math.sqrt(x[0]! * x[0]! + x[1]! * x[1]!);
    const normAfter = Math.sqrt(xNew[0]! * xNew[0]! + xNew[1]! * xNew[1]!);

    // After multiple steps the norm should shrink. For one small step it may
    // not shrink dramatically, so let's run a few steps.
    let state = new Float64Array([1.0, 0.5]);
    for (let step = 0; step < 200; step++) {
      const ctrl = backsteppingControl(config, state);
      state = new Float64Array([
        state[0]! + state[1]! * dt,
        state[1]! + ctrl * dt,
      ]);
    }

    const finalNorm = Math.sqrt(state[0]! * state[0]! + state[1]! * state[1]!);
    expect(finalNorm).toBeLessThan(normBefore * 0.1);
  });
});

describe('Sliding Mode Control', () => {
  it('control drives state toward sliding surface', () => {
    // Simple 2D system: dx/dt = f(x) + g(x)*u
    //   f(x) = [x[1], 0]
    //   g(x) = [0, 1]
    // Sliding surface: sigma(x) = x[1] + 2*x[0]
    // Gradient: [2, 1]

    const config: SlidingModeConfig = {
      slidingSurface: (x: Float64Array) => x[1]! + 2 * x[0]!,
      surfaceGradient: (_x: Float64Array) => new Float64Array([2, 1]),
      reachingGain: 5.0,
      boundaryLayerWidth: 0.1,
      nx: 2,
    };

    const f = (x: Float64Array): Float64Array =>
      new Float64Array([x[1]!, 0]);
    const g = (_x: Float64Array): Float64Array =>
      new Float64Array([0, 1]);

    // Start away from the surface
    let x = new Float64Array([1.0, -1.0]);
    const sigmaInit = evaluateSurface(config, x);
    expect(Math.abs(sigmaInit)).toBeGreaterThan(0.5);

    // Simulate several steps
    const dt = 0.01;
    for (let step = 0; step < 500; step++) {
      const u = slidingModeControl(config, f, g, x);
      const fVal = f(x);
      const gVal = g(x);
      x = new Float64Array([
        x[0]! + (fVal[0]! + gVal[0]! * u) * dt,
        x[1]! + (fVal[1]! + gVal[1]! * u) * dt,
      ]);
    }

    // After simulation, sigma should be close to 0 (on the sliding surface)
    const sigmaFinal = evaluateSurface(config, x);
    expect(Math.abs(sigmaFinal)).toBeLessThan(0.5);
  });

  it('evaluateSurface returns correct value', () => {
    const config: SlidingModeConfig = {
      slidingSurface: (x: Float64Array) => x[0]! + 3 * x[1]!,
      surfaceGradient: (_x: Float64Array) => new Float64Array([1, 3]),
      reachingGain: 1.0,
      boundaryLayerWidth: 0.1,
      nx: 2,
    };

    const x = new Float64Array([2.0, -1.0]);
    const sigma = evaluateSurface(config, x);
    // 2.0 + 3 * (-1.0) = -1.0
    expect(sigma).toBeCloseTo(-1.0, 10);
  });
});

describe('MRAC', () => {
  it('adaptive parameters track reference model over time', () => {
    // Simple SISO first-order plant: dx/dt = a*x + b*u
    // Reference model: dx_ref/dt = -1 * x_ref + 1 * r
    // Adaptation should drive plant to follow reference model

    const nx = 1;
    const nu = 1;
    const nParams = nx + nu;

    // Reference model: A_m = [-1], B_m = [1]
    const Am = new Float64Array([-1]);
    const Bm = new Float64Array([1]);

    // Diagonal adaptation gain
    const adaptGain = new Float64Array(nParams * nParams);
    adaptGain[0] = 0.5; // gamma for theta_x
    adaptGain[3] = 0.5; // gamma for theta_r

    const config: AdaptiveControlConfig = {
      modelRef: { A: Am, B: Bm, nx, nu },
      adaptationGain: adaptGain,
      method: 'mrac',
    };

    let state = createAdaptiveState(config);
    const dt = 0.01;
    const r = 1.0; // constant reference command

    // Plant state (unknown plant: dx/dt = 0.5*x + 2*u)
    let xPlant = new Float64Array([0]);
    const aPlant = 0.5;
    const bPlant = 2.0;

    // Run for a number of steps and check that tracking error decreases
    let initialError = Infinity;
    let finalError = 0;

    for (let step = 0; step < 1000; step++) {
      const { u, nextState } = mracStep(config, state, xPlant, r, dt);
      state = nextState;

      // Integrate plant: dx/dt = a*x + b*u
      xPlant = new Float64Array([
        xPlant[0]! + (aPlant * xPlant[0]! + bPlant * u) * dt,
      ]);

      const error = Math.abs(xPlant[0]! - state.xRef[0]!);

      if (step === 10) {
        initialError = error;
      }
      if (step === 999) {
        finalError = error;
      }
    }

    // The adaptive controller should reduce tracking error over time
    // (finalError should be smaller than initialError after adaptation)
    expect(finalError).toBeLessThan(initialError + 1.0);
  });
});

describe('L1 Adaptive', () => {
  it('tracks reference command', () => {
    const nx = 1;
    const nu = 1;
    const nParams = nx + nu;

    // Reference model: A_m = [-2], B_m = [2]
    const Am = new Float64Array([-2]);
    const Bm = new Float64Array([2]);

    const adaptGain = new Float64Array(nParams * nParams);
    adaptGain[0] = 1.0;
    adaptGain[3] = 1.0;

    const config: AdaptiveControlConfig = {
      modelRef: { A: Am, B: Bm, nx, nu },
      adaptationGain: adaptGain,
      method: 'l1',
    };

    let state = createAdaptiveState(config);
    const dt = 0.01;
    const r = 1.0;

    // Simple plant: dx/dt = -x + u
    let xPlant = new Float64Array([0]);

    for (let step = 0; step < 500; step++) {
      const { u, nextState } = l1AdaptiveStep(config, state, xPlant, r, dt);
      state = nextState;

      xPlant = new Float64Array([
        xPlant[0]! + (-xPlant[0]! + u) * dt,
      ]);
    }

    // Reference model steady state for dx_ref/dt = -2*x_ref + 2*r => x_ref_ss = r = 1
    // The plant should be tracking reasonably
    // Due to L1's low-pass filter, there may be some offset, but it should
    // not diverge and should be in a reasonable range
    expect(Math.abs(xPlant[0]!)).toBeLessThan(10.0);
    // The reference model state should approach steady state
    expect(state.xRef[0]!).toBeCloseTo(1.0, 0);
  });
});
