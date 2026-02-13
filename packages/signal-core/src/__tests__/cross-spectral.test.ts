// ---------------------------------------------------------------------------
// SP-5: Cross-Spectral Analysis Tests
// ---------------------------------------------------------------------------
import { describe, it, expect } from 'vitest';
import { coherence, estimateDelay } from '../cross-spectral/coherence.js';
import { realCepstrum, powerCepstrum } from '../cross-spectral/cepstrum.js';
import { spectralClusterVenues } from '../cross-spectral/spectral-clustering.js';
import { createPRNG } from '../types.js';

function sinWave(freq: number, sr: number, n: number): Float64Array {
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.sin(2 * Math.PI * freq * i / sr);
  return x;
}

describe('SP-5: Cross-Spectral Analysis', () => {
  describe('Coherence', () => {
    it('identical signals have coherence ≈ 1', () => {
      const n = 512;
      const x = sinWave(10, 128, n);
      // coherence(signalX, signalY, fs, nperseg, noverlap, windowType)
      const result = coherence(x, x, 128, 128, 64, 'hann');
      expect(result.frequencies.length).toBeGreaterThan(0);
      expect(result.coherence.length).toBe(result.frequencies.length);
      // Coherence should be ~1 everywhere (except DC)
      for (let k = 1; k < result.coherence.length; k++) {
        expect(result.coherence[k]!).toBeGreaterThan(0.95);
      }
    });

    it('uncorrelated signals have low coherence', () => {
      const n = 2048;
      const x = sinWave(10, 128, n);
      // Use noise-like signal for uncorrelated test
      const rng = createPRNG(42);
      const y = new Float64Array(n);
      for (let i = 0; i < n; i++) y[i] = rng() - 0.5;
      const result = coherence(x, y, 128, 256, 128, 'hann');
      // Average coherence should be low
      let avg = 0;
      for (let k = 0; k < result.coherence.length; k++) avg += result.coherence[k]!;
      avg /= result.coherence.length;
      expect(avg).toBeLessThan(0.5);
    });
  });

  describe('Delay estimation', () => {
    it('estimates delay from phase and frequency', () => {
      // estimateDelay(phase: number, frequency: number) → number
      const freq = 10;
      const phase = -Math.PI / 4; // some phase
      const delay = estimateDelay(phase, freq);
      // τ = -phase / (2πf)
      const expected = -phase / (2 * Math.PI * freq);
      expect(delay).toBeCloseTo(expected, 10);
    });

    it('returns 0 for zero frequency', () => {
      expect(estimateDelay(1.0, 0)).toBe(0);
    });
  });

  describe('Cepstrum', () => {
    it('real cepstrum returns CepstrumResult', () => {
      const x = sinWave(10, 128, 256);
      const result = realCepstrum(x, 128);
      expect(result.quefrencies.length).toBeGreaterThan(0);
      expect(result.cepstrum.length).toBe(result.quefrencies.length);
      expect(Array.isArray(result.dominantQuefrencies)).toBe(true);
    });

    it('power cepstrum returns CepstrumResult', () => {
      const x = sinWave(10, 128, 256);
      const result = powerCepstrum(x, 128);
      expect(result.quefrencies.length).toBeGreaterThan(0);
      expect(result.cepstrum.length).toBe(result.quefrencies.length);
      // Power cepstrum values should be non-negative
      for (let i = 0; i < result.cepstrum.length; i++) {
        expect(result.cepstrum[i]!).toBeGreaterThanOrEqual(0);
      }
    });
  });

  describe('Spectral Clustering', () => {
    it('clusters similar venues together', () => {
      const n = 128;
      const venues = [
        sinWave(10, 128, n),
        sinWave(10, 128, n),
        sinWave(50, 128, n),
        sinWave(50, 128, n),
      ];
      // spectralClusterVenues(signals, nClusters, fs, sigma)
      const clusters = spectralClusterVenues(venues, 2, 128, 0.1);
      expect(clusters.length).toBe(2);
      // Each cluster should have memberIndices
      const sizes = clusters.map(c => c.memberIndices.length).sort();
      expect(sizes).toEqual([2, 2]);
    });

    it('returns cluster with expected fields', () => {
      const n = 128;
      const venues = [sinWave(10, 128, n), sinWave(20, 128, n)];
      const clusters = spectralClusterVenues(venues, 2, 128);
      for (const c of clusters) {
        expect(typeof c.clusterId).toBe('number');
        expect(typeof c.label).toBe('string');
        expect(Array.isArray(c.memberIndices)).toBe(true);
        expect(c.centroidPSD).toBeInstanceOf(Float64Array);
      }
    });
  });
});
