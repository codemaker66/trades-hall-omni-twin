// ---------------------------------------------------------------------------
// SP-9: Compressed Sensing Tests
// ---------------------------------------------------------------------------
import { describe, it, expect } from 'vitest';
import { omp, recoverDemandCurve } from '../compressed-sensing/omp.js';
import { fista } from '../compressed-sensing/fista.js';
import { matrixCompletionALS } from '../compressed-sensing/matrix-completion.js';

describe('SP-9: Compressed Sensing', () => {
  describe('OMP', () => {
    it('recovers sparse signal', () => {
      // Create a simple sparse recovery problem
      // x = [3, 0, 0, 5, 0] with measurement y = A*x
      const m = 4; // measurements
      const n = 5; // signal length
      const A = new Float64Array([
        1, 0, 0, 0, 1,
        0, 1, 0, 1, 0,
        1, 1, 0, 0, 0,
        0, 0, 1, 1, 0,
      ]);
      const xTrue = new Float64Array([3, 0, 0, 5, 0]);
      // y = A * x
      const y = new Float64Array(m);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          y[i] = (y[i] ?? 0) + A[i * n + j]! * xTrue[j]!;
        }
      }

      // omp(A, y, M, N, config)
      const result = omp(A, y, m, n, { nComponents: 2 });
      expect(result.coefficients.length).toBe(n);
      expect(result.support.length).toBeLessThanOrEqual(2);
      // Should recover approximately correct values
      expect(Math.abs(result.coefficients[0]! - 3)).toBeLessThan(1);
      expect(Math.abs(result.coefficients[3]! - 5)).toBeLessThan(1);
    });

    it('returns SparseRecoveryResult with correct fields', () => {
      const m = 3;
      const n = 4;
      const A = new Float64Array(m * n);
      for (let i = 0; i < m * n; i++) A[i] = Math.random();
      const y = new Float64Array([1, 2, 3]);
      const result = omp(A, y, m, n, { nComponents: 2 });
      expect(result.signal).toBeInstanceOf(Float64Array);
      expect(result.coefficients).toBeInstanceOf(Float64Array);
      expect(Array.isArray(result.support)).toBe(true);
      expect(typeof result.residualNorm).toBe('number');
    });
  });

  describe('Demand curve recovery', () => {
    it('recovers from sparse samples', () => {
      const n = 64;
      // Create sparse sampled demand curve
      const fullCurve = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        fullCurve[i] = 50 + 20 * Math.sin(2 * Math.PI * i / n);
      }
      // Sample only 20 points
      const sampleIndices: number[] = Array.from({ length: 20 }, (_, i) => Math.floor(i * n / 20));
      const samples = new Float64Array(sampleIndices.map(i => fullCurve[i]!));

      // recoverDemandCurve(observedDays, observedValues, totalDays, nComponents)
      const result = recoverDemandCurve(sampleIndices, samples, n, 8);
      expect(result.signal.length).toBe(n);
      // Recovered values should be finite
      for (let i = 0; i < n; i++) {
        expect(Number.isFinite(result.signal[i])).toBe(true);
      }
    });
  });

  describe('FISTA', () => {
    it('solves L1-regularized problem', () => {
      const m = 10;
      const n = 5;
      const A = new Float64Array(m * n);
      for (let i = 0; i < m * n; i++) A[i] = Math.random();
      const xTrue = new Float64Array([1, 0, 0, 2, 0]);
      const b = new Float64Array(m);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          b[i] = (b[i] ?? 0) + A[i * n + j]! * xTrue[j]!;
        }
      }

      // fista(A, y, M, N, config)
      const result = fista(A, b, m, n, {
        stepSize: 0.01,
        lambda: 0.01,
        maxIter: 1000,
        tolerance: 1e-8,
      });
      expect(result.coefficients.length).toBe(n);
      expect(result.signal).toBeInstanceOf(Float64Array);
    });
  });

  describe('Matrix Completion (ALS)', () => {
    it('completes partially observed matrix', () => {
      const rows = 5;
      const cols = 5;
      // Low-rank matrix: rank 1
      const full = new Float64Array(rows * cols);
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          full[i * cols + j] = (i + 1) * (j + 1);
        }
      }
      // Build observed entries array
      const observed: Array<{ row: number; col: number; value: number }> = [];
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          if (Math.random() < 0.6) {
            observed.push({ row: i, col: j, value: full[i * cols + j]! });
          }
        }
      }

      // matrixCompletionALS(observed, config, rank?)
      const result = matrixCompletionALS(observed, {
        nRows: rows,
        nCols: cols,
        lambda: 0.01,
        maxIter: 50,
        tolerance: 1e-6,
      }, 2);
      expect(result.completed.length).toBe(rows * cols);
      expect(typeof result.rank).toBe('number');
      expect(typeof result.residual).toBe('number');
      // Completed values should be finite
      for (let i = 0; i < result.completed.length; i++) {
        expect(Number.isFinite(result.completed[i])).toBe(true);
      }
    });
  });
});
