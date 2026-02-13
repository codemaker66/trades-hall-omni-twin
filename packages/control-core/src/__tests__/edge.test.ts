import { describe, it, expect } from 'vitest';
import {
  linearFeedback,
  loadFeedbackGain,
  explicitMPCQuery,
  isInsidePolytope,
  neuralPolicyForward,
  mlpLayer,
  checkConstraints,
  clipAction,
  checkRateConstraint,
} from '../edge/index.js';
import type {
  LinearFeedbackConfig,
  ExplicitMPCTable,
  ExplicitMPCRegion,
  NeuralPolicyConfig,
  MLPWeights,
  ConstraintCheckerConfig,
} from '../types.js';

// ---------------------------------------------------------------------------
// OC-12: Edge Deployment
// ---------------------------------------------------------------------------

describe('OC-12: Edge Deployment', () => {
  // -----------------------------------------------------------------------
  // linearFeedback
  // -----------------------------------------------------------------------
  describe('linearFeedback', () => {
    it('produces output of correct dimension (nu)', () => {
      const nx = 2;
      const nu = 1;
      // K = [1, 2]  (1x2)
      const config: LinearFeedbackConfig = {
        K: Float64Array.from([1, 2]),
        nx,
        nu,
      };

      const x = Float64Array.from([3, 4]);
      const u = linearFeedback(config, x);

      expect(u.length).toBe(nu);
      // u = -K * x = -(1*3 + 2*4) = -11
      expect(u[0]!).toBeCloseTo(-11, 10);
    });

    it('clips output to [uMin, uMax] bounds', () => {
      const nx = 2;
      const nu = 2;
      // K = [[10, 0], [0, 10]]  (2x2) -- large gains will push output out of bounds
      const config: LinearFeedbackConfig = {
        K: Float64Array.from([10, 0, 0, 10]),
        nx,
        nu,
        uMin: Float64Array.from([-5, -5]),
        uMax: Float64Array.from([5, 5]),
      };

      const x = Float64Array.from([3, -3]);
      const u = linearFeedback(config, x);

      // u_raw = -K*x = [-30, 30]
      // Clipped: [-5, 5]
      expect(u[0]!).toBe(-5);
      expect(u[1]!).toBe(5);
    });

    it('computes feedback relative to a non-zero reference', () => {
      const nx = 2;
      const nu = 1;
      const config: LinearFeedbackConfig = {
        K: Float64Array.from([1, 1]),
        nx,
        nu,
      };

      const x = Float64Array.from([5, 5]);
      const xRef = Float64Array.from([3, 3]);
      const u = linearFeedback(config, x, xRef);

      // e = x - xRef = [2, 2], u = -K*e = -(2+2) = -4
      expect(u[0]!).toBeCloseTo(-4, 10);
    });
  });

  // -----------------------------------------------------------------------
  // loadFeedbackGain
  // -----------------------------------------------------------------------
  describe('loadFeedbackGain', () => {
    it('round-trips through JSON correctly', () => {
      const json = {
        K: [
          [1.5, -0.3],
          [0.0, 2.7],
        ],
        nx: 2,
        nu: 2,
      };

      const config = loadFeedbackGain(json);

      expect(config.nx).toBe(2);
      expect(config.nu).toBe(2);
      expect(config.K.length).toBe(4);

      // Verify row-major layout: K[0,0]=1.5, K[0,1]=-0.3, K[1,0]=0, K[1,1]=2.7
      expect(config.K[0]!).toBeCloseTo(1.5, 10);
      expect(config.K[1]!).toBeCloseTo(-0.3, 10);
      expect(config.K[2]!).toBeCloseTo(0.0, 10);
      expect(config.K[3]!).toBeCloseTo(2.7, 10);

      // Feeding it back into linearFeedback should work
      const x = Float64Array.from([1, 0]);
      const u = linearFeedback(config, x);
      // u = -K*x = [-(1.5*1 + (-0.3)*0), -(0*1 + 2.7*0)] = [-1.5, 0]
      expect(u[0]!).toBeCloseTo(-1.5, 10);
      expect(u[1]!).toBeCloseTo(0, 10);
    });
  });

  // -----------------------------------------------------------------------
  // explicitMPCQuery & isInsidePolytope
  // -----------------------------------------------------------------------
  describe('isInsidePolytope', () => {
    it('returns true for a point inside and false for a point outside', () => {
      // Define a square [-1,1] x [-1,1] via 4 half-plane constraints:
      //   x0 <= 1,  -x0 <= 1,  x1 <= 1,  -x1 <= 1
      const H = Float64Array.from([
        1, 0,   // x0 <= 1
        -1, 0,  // -x0 <= 1
        0, 1,   // x1 <= 1
        0, -1,  // -x1 <= 1
      ]);
      const h = Float64Array.from([1, 1, 1, 1]);
      const nConstraints = 4;
      const nx = 2;

      // Point inside
      expect(isInsidePolytope(H, h, Float64Array.from([0.5, 0.5]), nConstraints, nx)).toBe(true);
      expect(isInsidePolytope(H, h, Float64Array.from([0, 0]), nConstraints, nx)).toBe(true);

      // Point outside
      expect(isInsidePolytope(H, h, Float64Array.from([2, 0]), nConstraints, nx)).toBe(false);
      expect(isInsidePolytope(H, h, Float64Array.from([0, -1.5]), nConstraints, nx)).toBe(false);

      // Point on boundary (should be inside due to <= with tolerance)
      expect(isInsidePolytope(H, h, Float64Array.from([1, 1]), nConstraints, nx)).toBe(true);
    });
  });

  describe('explicitMPCQuery', () => {
    it('finds the correct region and returns affine control', () => {
      const nx = 2;
      const nu = 1;

      // Region 1: x0 >= 0 AND x1 >= 0  (first quadrant)
      //   -x0 <= 0, -x1 <= 0
      const region1: ExplicitMPCRegion = {
        Hx: Float64Array.from([-1, 0, 0, -1]),
        hx: Float64Array.from([0, 0]),
        Fx: Float64Array.from([1, 0.5]),  // u = 1*x0 + 0.5*x1 + 0.1
        gx: Float64Array.from([0.1]),
        nConstraints: 2,
      };

      // Region 2: x0 < 0 AND x1 >= 0  (second quadrant)
      //   x0 <= 0, -x1 <= 0
      const region2: ExplicitMPCRegion = {
        Hx: Float64Array.from([1, 0, 0, -1]),
        hx: Float64Array.from([0, 0]),
        Fx: Float64Array.from([-1, 0.5]),  // u = -1*x0 + 0.5*x1 + 0.2
        gx: Float64Array.from([0.2]),
        nConstraints: 2,
      };

      const table: ExplicitMPCTable = {
        regions: [region1, region2],
        nx,
        nu,
      };

      // Query point in first quadrant
      const x1 = Float64Array.from([2, 3]);
      const u1 = explicitMPCQuery(table, x1);
      expect(u1).not.toBeNull();
      // u = 1*2 + 0.5*3 + 0.1 = 3.6
      expect(u1![0]!).toBeCloseTo(3.6, 10);

      // Query point in second quadrant
      const x2 = Float64Array.from([-1, 2]);
      const u2 = explicitMPCQuery(table, x2);
      expect(u2).not.toBeNull();
      // u = -1*(-1) + 0.5*2 + 0.2 = 2.2
      expect(u2![0]!).toBeCloseTo(2.2, 10);

      // Query point not in any region (fourth quadrant: x0 > 0, x1 < 0)
      const x3 = Float64Array.from([1, -1]);
      const u3 = explicitMPCQuery(table, x3);
      expect(u3).toBeNull();
    });
  });

  // -----------------------------------------------------------------------
  // neuralPolicyForward & mlpLayer
  // -----------------------------------------------------------------------
  describe('mlpLayer', () => {
    it('relu activation zeroes out negative pre-activations', () => {
      // Single layer: 2 -> 2 with relu
      // W = [[1, 0], [0, 1]], b = [-1, 1]
      const W = Float64Array.from([1, 0, 0, 1]);
      const b = Float64Array.from([-1, 1]);
      const input = Float64Array.from([0.5, -2]);

      const out = mlpLayer(W, b, input, 2, 2, 'relu');

      // Pre-activation: [1*0.5 + 0*(-2) + (-1), 0*0.5 + 1*(-2) + 1] = [-0.5, -1]
      // After relu: [0, 0]
      expect(out[0]!).toBe(0);
      expect(out[1]!).toBe(0);
    });

    it('linear activation passes through raw values', () => {
      const W = Float64Array.from([2, 3]);
      const b = Float64Array.from([1]);
      const input = Float64Array.from([1, 1]);

      const out = mlpLayer(W, b, input, 2, 1, 'linear');

      // Pre-activation: 2*1 + 3*1 + 1 = 6
      expect(out[0]!).toBeCloseTo(6, 10);
    });
  });

  describe('neuralPolicyForward', () => {
    it('produces output of correct dimensions with a 2-layer MLP', () => {
      const stateDim = 2;
      const hiddenDim = 4;
      const actionDim = 1;

      // Layer 1: 2 -> 4 (relu)
      const w1 = new Float64Array(hiddenDim * stateDim);
      const b1 = new Float64Array(hiddenDim);
      // Set up a simple pattern: each hidden unit looks at one feature
      // W1 = [[1, 0], [0, 1], [1, 1], [-1, 1]]
      w1.set([1, 0, 0, 1, 1, 1, -1, 1]);
      b1.set([0, 0, 0, 0]);

      // Layer 2: 4 -> 1 (linear, final layer)
      const w2 = new Float64Array(actionDim * hiddenDim);
      const b2 = new Float64Array(actionDim);
      // W2 = [[1, 1, 1, 1]], b2 = [0]
      w2.set([1, 1, 1, 1]);
      b2.set([0]);

      const weights: MLPWeights = {
        layers: [
          { weight: w1, bias: b1, inDim: stateDim, outDim: hiddenDim },
          { weight: w2, bias: b2, inDim: hiddenDim, outDim: actionDim },
        ],
        activation: 'relu',
      };

      const config: NeuralPolicyConfig = {
        weights,
        stateNormMean: Float64Array.from([0, 0]),
        stateNormStd: Float64Array.from([1, 1]),
        actionScale: Float64Array.from([1]),
        actionBias: Float64Array.from([0]),
      };

      const state = Float64Array.from([1, 2]);
      const u = neuralPolicyForward(config, state);

      expect(u.length).toBe(actionDim);

      // Manual computation:
      // Normalized: z = [1, 2]
      // Layer 1 pre-act: [1*1+0*2, 0*1+1*2, 1*1+1*2, -1*1+1*2] = [1, 2, 3, 1]
      // After relu: [1, 2, 3, 1]
      // Layer 2 (linear): 1+2+3+1 = 7
      // De-normalized: 1*7 + 0 = 7
      expect(u[0]!).toBeCloseTo(7, 10);
    });

    it('applies state normalisation and action denormalisation', () => {
      const stateDim = 2;
      const actionDim = 1;

      // Single-layer identity-like network: 2 -> 1 linear
      const w = Float64Array.from([1, 1]);
      const b = Float64Array.from([0]);

      const weights: MLPWeights = {
        layers: [
          { weight: w, bias: b, inDim: stateDim, outDim: actionDim },
        ],
        activation: 'relu', // irrelevant since last layer is always linear
      };

      const config: NeuralPolicyConfig = {
        weights,
        stateNormMean: Float64Array.from([10, 20]),
        stateNormStd: Float64Array.from([2, 5]),
        actionScale: Float64Array.from([3]),
        actionBias: Float64Array.from([100]),
      };

      const state = Float64Array.from([12, 30]);
      const u = neuralPolicyForward(config, state);

      // Normalized: z = [(12-10)/2, (30-20)/5] = [1, 2]
      // Linear layer: 1*1 + 1*2 + 0 = 3
      // Denorm: 3 * 3 + 100 = 109
      expect(u[0]!).toBeCloseTo(109, 10);
    });
  });

  // -----------------------------------------------------------------------
  // checkConstraints / clipAction / checkRateConstraint
  // -----------------------------------------------------------------------
  describe('clipAction', () => {
    it('clamps values to [uMin, uMax]', () => {
      const u = Float64Array.from([10, -10, 3]);
      const uMin = Float64Array.from([-5, -5, -5]);
      const uMax = Float64Array.from([5, 5, 5]);

      const clipped = clipAction(u, uMin, uMax);

      expect(clipped[0]!).toBe(5);
      expect(clipped[1]!).toBe(-5);
      expect(clipped[2]!).toBe(3); // within bounds, unchanged
    });
  });

  describe('checkRateConstraint', () => {
    it('limits rate of change of control action', () => {
      const u = Float64Array.from([10, -5]);
      const uPrev = Float64Array.from([2, 0]);
      const duMax = Float64Array.from([3, 2]);

      const adjusted = checkRateConstraint(u, uPrev, duMax);

      // du[0] = 10 - 2 = 8 > 3 => adjusted = 2 + 3 = 5
      expect(adjusted[0]!).toBe(5);

      // du[1] = -5 - 0 = -5 < -2 => adjusted = 0 - 2 = -2
      expect(adjusted[1]!).toBe(-2);
    });

    it('leaves action unchanged when rate is within limits', () => {
      const u = Float64Array.from([3, 1]);
      const uPrev = Float64Array.from([2, 0]);
      const duMax = Float64Array.from([5, 5]);

      const adjusted = checkRateConstraint(u, uPrev, duMax);

      expect(adjusted[0]!).toBe(3);
      expect(adjusted[1]!).toBe(1);
    });
  });

  describe('checkConstraints', () => {
    it('detects violations and clips action', () => {
      const config: ConstraintCheckerConfig = {
        uMin: Float64Array.from([-2]),
        uMax: Float64Array.from([2]),
        xMin: Float64Array.from([-10, -10]),
        xMax: Float64Array.from([10, 10]),
      };

      // Action exceeds upper bound
      const u = Float64Array.from([5]);
      const x = Float64Array.from([0, 0]);

      const result = checkConstraints(config, u, x);

      expect(result.feasible).toBe(false);
      expect(result.clippedAction[0]!).toBe(2);
      expect(result.violations.length).toBeGreaterThan(0);
      expect(result.violations.some((v) => v.includes('clipped'))).toBe(true);
    });

    it('reports no violations when everything is in bounds', () => {
      const config: ConstraintCheckerConfig = {
        uMin: Float64Array.from([-10]),
        uMax: Float64Array.from([10]),
        xMin: Float64Array.from([-10, -10]),
        xMax: Float64Array.from([10, 10]),
      };

      const u = Float64Array.from([1]);
      const x = Float64Array.from([0, 0]);

      const result = checkConstraints(config, u, x);

      expect(result.feasible).toBe(true);
      expect(result.violations).toHaveLength(0);
      expect(result.clippedAction[0]!).toBe(1);
    });

    it('reports state bound violations as informational', () => {
      const config: ConstraintCheckerConfig = {
        uMin: Float64Array.from([-10]),
        uMax: Float64Array.from([10]),
        xMin: Float64Array.from([0, 0]),
        xMax: Float64Array.from([5, 5]),
      };

      // State out of bounds, action in bounds
      const u = Float64Array.from([1]);
      const x = Float64Array.from([-1, 6]);

      const result = checkConstraints(config, u, x);

      expect(result.feasible).toBe(false);
      // Should detect state violations
      expect(result.violations.some((v) => v.includes('below lower bound'))).toBe(true);
      expect(result.violations.some((v) => v.includes('above upper bound'))).toBe(true);
      // Action itself was not clipped
      expect(result.clippedAction[0]!).toBe(1);
    });

    it('applies rate constraint when uPrev is provided', () => {
      const config: ConstraintCheckerConfig = {
        uMin: Float64Array.from([-100]),
        uMax: Float64Array.from([100]),
        xMin: Float64Array.from([-100, -100]),
        xMax: Float64Array.from([100, 100]),
        duMax: Float64Array.from([1]),
      };

      const u = Float64Array.from([10]);
      const x = Float64Array.from([0, 0]);
      const uPrev = Float64Array.from([5]);

      const result = checkConstraints(config, u, x, uPrev);

      // du = 10 - 5 = 5 > 1, should be rate-limited to 5 + 1 = 6
      expect(result.feasible).toBe(false);
      expect(result.clippedAction[0]!).toBe(6);
      expect(result.violations.some((v) => v.includes('rate'))).toBe(true);
    });
  });
});
