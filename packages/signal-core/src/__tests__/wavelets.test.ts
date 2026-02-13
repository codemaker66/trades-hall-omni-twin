// ---------------------------------------------------------------------------
// SP-2: Wavelet Multi-Resolution Tests
// ---------------------------------------------------------------------------
import { describe, it, expect } from 'vitest';
import {
  getScalingFilter,
  getWaveletFilter,
  dwtDecompose,
  dwtReconstruct,
  maxLevel,
} from '../wavelets/dwt.js';
import { modwtDecompose, modwtReconstruct, modwtMRA } from '../wavelets/modwt.js';
import { waveletDenoise } from '../wavelets/denoising.js';
import { multiscaleForecast } from '../wavelets/multiscale-forecast.js';
import { createPRNG } from '../types.js';

function sinWave(freq: number, sr: number, n: number): Float64Array {
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.sin(2 * Math.PI * freq * i / sr);
  return x;
}

describe('SP-2: Wavelet Multi-Resolution', () => {
  describe('DWT filters', () => {
    it('Haar scaling filter is correct', () => {
      const h = getScalingFilter('haar');
      expect(h.length).toBe(2);
      expect(h[0]).toBeCloseTo(1 / Math.SQRT2, 10);
      expect(h[1]).toBeCloseTo(1 / Math.SQRT2, 10);
    });

    it('wavelet filter is QMF of scaling filter', () => {
      const h = getScalingFilter('db4');
      const g = getWaveletFilter('db4');
      expect(g.length).toBe(h.length);
      // QMF: g[i] = (-1)^i * h[L-1-i]
      for (let i = 0; i < h.length; i++) {
        const expected = ((i % 2 === 0) ? 1 : -1) * h[h.length - 1 - i]!;
        expect(g[i]).toBeCloseTo(expected, 10);
      }
    });
  });

  describe('maxLevel', () => {
    it('computes correct max decomposition level', () => {
      expect(maxLevel(256, 'haar')).toBe(8); // log2(256/1) = 8
      expect(maxLevel(256, 'db4')).toBe(5);  // floor(log2(256/(8-1))) = floor(log2(36.57)) = 5
    });
  });

  describe('DWT decompose / reconstruct roundtrip', () => {
    it('reconstructs original signal (Haar)', () => {
      const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]);
      const result = dwtDecompose(x, 'haar', 3);
      const recovered = dwtReconstruct(result);
      for (let i = 0; i < x.length; i++) {
        expect(recovered[i]).toBeCloseTo(x[i]!, 8);
      }
    });

    it('reconstructs with db4', () => {
      const n = 64;
      const x = sinWave(4, 64, n);
      const result = dwtDecompose(x, 'db4', 3);
      expect(result.details.length).toBe(3);
      const recovered = dwtReconstruct(result);
      // db4 may have boundary effects; check middle
      for (let i = 8; i < n - 8; i++) {
        expect(recovered[i]).toBeCloseTo(x[i]!, 4);
      }
    });
  });

  describe('MODWT decompose / reconstruct roundtrip', () => {
    it('reconstructs original signal', () => {
      const n = 64;
      const x = sinWave(8, 64, n);
      const result = modwtDecompose(x, 'haar', 3);
      expect(result.details.length).toBe(3);
      const recovered = modwtReconstruct(result);
      for (let i = 0; i < n; i++) {
        expect(recovered[i]).toBeCloseTo(x[i]!, 6);
      }
    });
  });

  describe('MODWT MRA', () => {
    it('components sum to original', () => {
      const n = 64;
      const x = sinWave(8, 64, n);
      const mra = modwtMRA(x, 'haar', 3);
      // mra returns { details: Float64Array[], approximation: Float64Array }
      // 3 details + 1 approximation should sum to original
      expect(mra.details.length).toBe(3);
      const sum = new Float64Array(n);
      for (const d of mra.details) {
        for (let i = 0; i < n; i++) sum[i]! += d[i]!;
      }
      for (let i = 0; i < n; i++) sum[i]! += mra.approximation[i]!;
      for (let i = 0; i < n; i++) {
        expect(sum[i]).toBeCloseTo(x[i]!, 6);
      }
    });
  });

  describe('Wavelet denoising', () => {
    it('reduces noise in signal', () => {
      const n = 256;
      const rng = createPRNG(42);
      const clean = sinWave(4, 256, n);
      const noisy = new Float64Array(n);
      // Use substantial noise (σ ≈ 0.58) so denoising can demonstrate improvement
      for (let i = 0; i < n; i++) noisy[i] = clean[i]! + 2.0 * (rng() - 0.5);

      const denoised = waveletDenoise(noisy, {
        wavelet: 'db4',
        method: 'soft',
        thresholdRule: 'universal',
      });

      // Denoised should be closer to clean than noisy
      let errNoisy = 0, errDenoised = 0;
      for (let i = 0; i < n; i++) {
        errNoisy += (noisy[i]! - clean[i]!) ** 2;
        errDenoised += (denoised[i]! - clean[i]!) ** 2;
      }
      expect(errDenoised).toBeLessThan(errNoisy);
    });
  });

  describe('Multiscale forecast', () => {
    it('produces forecast of requested horizon', () => {
      const n = 128;
      const x = new Float64Array(n);
      for (let i = 0; i < n; i++) x[i] = Math.sin(2 * Math.PI * i / 32) + 0.01 * i;

      const horizon = 16;
      // multiscaleForecast(signal, horizon, wavelet, levels)
      const result = multiscaleForecast(x, horizon, 'haar', 3);
      expect(result.forecast.length).toBe(horizon);
      expect(result.componentForecasts.length).toBeGreaterThan(0);
      expect(result.levels).toBeGreaterThan(0);
      // Values should be finite
      for (let i = 0; i < horizon; i++) {
        expect(Number.isFinite(result.forecast[i])).toBe(true);
      }
    });
  });
});
