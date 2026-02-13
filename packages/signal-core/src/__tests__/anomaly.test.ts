// ---------------------------------------------------------------------------
// SP-6: Anomaly Detection Ensemble Tests
// ---------------------------------------------------------------------------
import { describe, it, expect } from 'vitest';
import { spectralResidual } from '../anomaly/spectral-residual.js';
import { matrixProfile, matrixProfileAnomalies } from '../anomaly/matrix-profile.js';
import { cusum, adaptiveCUSUM } from '../anomaly/cusum.js';
import { pelt, binarySegmentation } from '../anomaly/pelt.js';
import { stlDecompose, stlAnomalies } from '../anomaly/stl.js';
import { ensembleAnomalyDetection } from '../anomaly/ensemble.js';
import { createPRNG } from '../types.js';

describe('SP-6: Anomaly Detection Ensemble', () => {
  describe('Spectral Residual', () => {
    it('detects spike anomaly', () => {
      const n = 256;
      const x = new Float64Array(n);
      for (let i = 0; i < n; i++) x[i] = Math.sin(2 * Math.PI * i / 32);
      // Insert anomaly
      x[100] = 10;
      x[101] = 10;

      const result = spectralResidual(x);
      expect(result.scores.length).toBe(n);
      expect(result.anomalies.length).toBe(n);
      expect(result.method).toBe('spectral-residual');
      // Score at anomaly should be higher than average
      let sum = 0;
      for (let i = 0; i < n; i++) sum += result.scores[i]!;
      const avg = sum / n;
      expect(result.scores[100]!).toBeGreaterThan(avg);
    });
  });

  describe('Matrix Profile', () => {
    it('computes profile with correct length', () => {
      const n = 200;
      const x = new Float64Array(n);
      for (let i = 0; i < n; i++) x[i] = Math.sin(2 * Math.PI * i / 20);
      const m = 20;
      const result = matrixProfile(x, m);
      expect(result.profile.length).toBe(n - m + 1);
      expect(result.profileIndex.length).toBe(n - m + 1);
      expect(result.windowSize).toBe(m);
    });

    it('detects anomalous subsequences', () => {
      const n = 200;
      const x = new Float64Array(n);
      for (let i = 0; i < n; i++) x[i] = Math.sin(2 * Math.PI * i / 20);
      // Insert anomalous subsequence
      for (let i = 100; i < 120; i++) x[i] = 5;

      const result = matrixProfileAnomalies(x, 20, 2);
      expect(result.anomalies.length).toBeGreaterThan(0);
      expect(result.method).toBe('matrix-profile');
    });
  });

  describe('CUSUM', () => {
    it('detects mean shift', () => {
      const n = 200;
      const x = new Float64Array(n);
      for (let i = 0; i < 100; i++) x[i] = 0;
      for (let i = 100; i < n; i++) x[i] = 5;

      const result = cusum(x, { k: 0.5, h: 3 });
      expect(result.scores.length).toBe(n);
      expect(result.anomalies.length).toBe(n);
      expect(result.method).toBe('cusum');
      // At least some anomalies should be detected after the mean shift
      const hasAnomaly = result.anomalies.some(v => v);
      expect(hasAnomaly).toBe(true);
    });

    it('no false alarms on constant signal', () => {
      const n = 100;
      const x = new Float64Array(n).fill(5);
      const result = cusum(x, { k: 0.5, h: 5 });
      // Constant signal should have no anomalies
      const alarmCount = result.anomalies.filter(v => v).length;
      expect(alarmCount).toBe(0);
    });
  });

  describe('Adaptive CUSUM', () => {
    it('adapts to changing mean', () => {
      const n = 200;
      const x = new Float64Array(n);
      for (let i = 0; i < n; i++) x[i] = i < 100 ? 10 : 15;
      const result = adaptiveCUSUM(x, 30, 0.5, 3);
      expect(result.scores.length).toBe(n);
      expect(result.anomalies.length).toBe(n);
      expect(result.method).toBe('adaptive-cusum');
      // At least some anomalies should be detected around the transition
      const hasAnomaly = result.anomalies.some(v => v);
      expect(hasAnomaly).toBe(true);
    });
  });

  describe('PELT', () => {
    it('detects changepoints in mean', () => {
      const n = 300;
      const rng = createPRNG(99);
      const x = new Float64Array(n);
      for (let i = 0; i < 100; i++) x[i] = 0 + rng() * 0.1;
      for (let i = 100; i < 200; i++) x[i] = 5 + rng() * 0.1;
      for (let i = 200; i < 300; i++) x[i] = 2 + rng() * 0.1;

      const result = pelt(x, 10);
      expect(result.changepoints.length).toBeGreaterThanOrEqual(1);
      expect(result.penalty).toBe(10);
    });
  });

  describe('Binary Segmentation', () => {
    it('detects changepoints', () => {
      const n = 200;
      const x = new Float64Array(n);
      for (let i = 0; i < 100; i++) x[i] = 0;
      for (let i = 100; i < 200; i++) x[i] = 5;

      const result = binarySegmentation(x, 5, 10);
      expect(result.changepoints.length).toBeGreaterThanOrEqual(1);
      expect(result.penalty).toBe(5);
    });
  });

  describe('STL Decomposition', () => {
    it('decomposes into trend, seasonal, remainder', () => {
      const period = 20;
      const n = period * 5;
      const x = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        x[i] = 0.1 * i + 5 * Math.sin(2 * Math.PI * i / period) + Math.random() * 0.5;
      }

      const result = stlDecompose(x, period);
      expect(result.trend.length).toBe(n);
      expect(result.seasonal.length).toBe(n);
      expect(result.remainder.length).toBe(n);

      // Components should sum to original (approximately)
      for (let i = 0; i < n; i++) {
        const sum = result.trend[i]! + result.seasonal[i]! + result.remainder[i]!;
        expect(sum).toBeCloseTo(x[i]!, 6);
      }
    });

    it('detects STL anomalies', () => {
      const period = 20;
      const n = period * 5;
      const x = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        x[i] = 5 * Math.sin(2 * Math.PI * i / period);
      }
      x[50] = 50; // anomaly

      const result = stlAnomalies(x, period, 3);
      expect(result.anomalies.length).toBe(n);
      expect(result.method).toBe('stl-remainder');
      // The anomaly at index 50 should be flagged
      expect(result.anomalies[50]).toBe(true);
    });
  });

  describe('Ensemble Anomaly Detection', () => {
    it('combines multiple detectors', () => {
      const period = 20;
      const n = period * 10;
      const x = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        x[i] = 5 * Math.sin(2 * Math.PI * i / period);
      }
      x[100] = 50; // clear anomaly

      const result = ensembleAnomalyDetection(x, period, period, 2);

      expect(result.consensus.length).toBe(n);
      expect(result.voteCount.length).toBe(n);
      // Each sub-detector result should be present
      expect(result.stl.method).toBe('stl-remainder');
      expect(result.matrixProfile.method).toBe('matrix-profile');
      expect(result.cusum.method).toBe('cusum');
      expect(result.spectralResidual.method).toBe('spectral-residual');
      // The anomaly at index 100 should have votes
      expect(result.voteCount[100]!).toBeGreaterThan(0);
    });
  });
});
