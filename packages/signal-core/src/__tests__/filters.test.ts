// ---------------------------------------------------------------------------
// SP-4: Digital Filter Preprocessing Tests
// ---------------------------------------------------------------------------
import { describe, it, expect } from 'vitest';
import { designButterworth, sosfilt, sosfiltfilt } from '../filters/butterworth.js';
import { sgCoefficients, savitzkyGolayFilter } from '../filters/savitzky-golay.js';
import { medianFilter, weightedMedianFilter } from '../filters/median.js';
import { preprocessBookings } from '../filters/preprocessing.js';

function sinWave(freq: number, sr: number, n: number): Float64Array {
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.sin(2 * Math.PI * freq * i / sr);
  return x;
}

describe('SP-4: Digital Filter Preprocessing', () => {
  describe('Butterworth design', () => {
    it('designs lowpass SOS sections', () => {
      const sections = designButterworth({
        order: 4,
        cutoff: 10,
        fs: 100,
        type: 'lowpass',
      });
      expect(sections.length).toBe(2); // order 4 = 2 biquads
      for (const s of sections) {
        // SOSSection is a 6-element tuple [b0, b1, b2, a0, a1, a2]
        expect(s.length).toBe(6);
      }
    });

    it('designs highpass SOS sections', () => {
      const sections = designButterworth({
        order: 2,
        cutoff: 20,
        fs: 100,
        type: 'highpass',
      });
      expect(sections.length).toBe(1); // order 2 = 1 biquad
    });
  });

  describe('sosfilt', () => {
    it('filters a signal', () => {
      const sections = designButterworth({
        order: 2,
        cutoff: 10,
        fs: 100,
        type: 'lowpass',
      });
      const x = sinWave(5, 100, 200);
      const y = sosfilt(sections, x);
      expect(y.length).toBe(x.length);
      // Low-frequency signal should mostly pass through
      let energy = 0;
      for (let i = 50; i < y.length; i++) energy += y[i]! ** 2;
      expect(energy).toBeGreaterThan(0);
    });
  });

  describe('sosfiltfilt', () => {
    it('produces zero-phase output', () => {
      const sections = designButterworth({
        order: 2,
        cutoff: 10,
        fs: 100,
        type: 'lowpass',
      });
      const n = 200;
      const x = sinWave(5, 100, n);
      const y = sosfiltfilt(sections, x);
      expect(y.length).toBe(n);
    });
  });

  describe('Savitzky-Golay', () => {
    it('computes correct coefficient length', () => {
      const coeffs = sgCoefficients(5, 2);
      expect(coeffs.length).toBe(5);
    });

    it('smooths noisy data', () => {
      const n = 100;
      const clean = new Float64Array(n);
      const noisy = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        clean[i] = Math.sin(2 * Math.PI * i / 20);
        noisy[i] = clean[i]! + (Math.random() - 0.5) * 0.3;
      }
      const smoothed = savitzkyGolayFilter(noisy, { windowLength: 7, polyOrder: 3 });
      expect(smoothed.length).toBe(n);

      // Smoothed should be closer to clean than noisy (in middle portion)
      let errNoisy = 0, errSmoothed = 0;
      for (let i = 10; i < n - 10; i++) {
        errNoisy += (noisy[i]! - clean[i]!) ** 2;
        errSmoothed += (smoothed[i]! - clean[i]!) ** 2;
      }
      expect(errSmoothed).toBeLessThan(errNoisy);
    });
  });

  describe('Median filter', () => {
    it('removes impulse noise', () => {
      const x = new Float64Array([1, 1, 1, 100, 1, 1, 1]);
      const y = medianFilter(x, 3);
      expect(y.length).toBe(7);
      expect(y[3]).toBe(1); // Spike removed
    });

    it('preserves edges', () => {
      const x = new Float64Array([0, 0, 0, 5, 5, 5, 5]);
      const y = medianFilter(x, 3);
      // Edge transition preserved
      expect(y[2]).toBe(0);
      expect(y[4]).toBe(5);
    });
  });

  describe('Weighted median filter', () => {
    it('applies triangular weights', () => {
      const x = new Float64Array([1, 2, 3, 4, 5]);
      const y = weightedMedianFilter(x, 3);
      expect(y.length).toBe(5);
    });
  });

  describe('preprocessBookings', () => {
    it('produces expected output fields', () => {
      const n = 128;
      const x = new Float64Array(n);
      for (let i = 0; i < n; i++) x[i] = 50 + 20 * Math.sin(2 * Math.PI * i / 30) + Math.random() * 5;
      const result = preprocessBookings(x, 1);
      expect(result.cleaned.length).toBe(n);
      expect(result.trend.length).toBe(n);
      expect(result.velocity.length).toBe(n);
      expect(result.acceleration.length).toBe(n);
    });
  });
});
