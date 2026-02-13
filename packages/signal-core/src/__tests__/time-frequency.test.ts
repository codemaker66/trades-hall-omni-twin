// ---------------------------------------------------------------------------
// SP-10: Time-Frequency Methods Tests
// ---------------------------------------------------------------------------
import { describe, it, expect } from 'vitest';
import { emd, ceemdan } from '../time-frequency/emd.js';
import { vmd } from '../time-frequency/vmd.js';
import { synchrosqueezingTransform } from '../time-frequency/sst.js';
import { stockwellTransform } from '../time-frequency/stockwell.js';

function sinWave(freq: number, sr: number, n: number): Float64Array {
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.sin(2 * Math.PI * freq * i / sr);
  return x;
}

describe('SP-10: Time-Frequency Methods', () => {
  describe('EMD', () => {
    it('decomposes signal into IMFs', () => {
      const n = 256;
      const x = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        x[i] = Math.sin(2 * Math.PI * 5 * i / 256) + 0.5 * Math.sin(2 * Math.PI * 30 * i / 256);
      }
      const result = emd(x, 5, 100);
      expect(result.imfs.length).toBeGreaterThanOrEqual(1);
      expect(result.residue.length).toBe(n);

      // IMFs + residue should approximately sum to original
      const sum = new Float64Array(n);
      for (const imf of result.imfs) {
        for (let i = 0; i < n; i++) sum[i]! += imf.data[i]!;
      }
      for (let i = 0; i < n; i++) sum[i]! += result.residue[i]!;
      for (let i = 0; i < n; i++) {
        expect(sum[i]).toBeCloseTo(x[i]!, 4);
      }
    });

    it('returns nIterations', () => {
      const x = sinWave(10, 256, 256);
      const result = emd(x);
      expect(result.nIterations).toBeGreaterThanOrEqual(0);
    });
  });

  describe('CEEMDAN', () => {
    it('produces IMFs with noise assistance', () => {
      const n = 128;
      const x = sinWave(10, 128, n);
      // ceemdan(signal, nEnsembles, noiseStd, seed, maxIMFs)
      const result = ceemdan(x, 5, 0.1, 42, 3);
      expect(result.imfs.length).toBeGreaterThanOrEqual(1);
      expect(result.residue.length).toBe(n);
      expect(result.nIterations).toBeGreaterThanOrEqual(1);
    });
  });

  describe('VMD', () => {
    it('decomposes signal into modes', () => {
      const n = 256;
      const x = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        x[i] = Math.sin(2 * Math.PI * 10 * i / 256) + Math.sin(2 * Math.PI * 40 * i / 256);
      }
      const result = vmd(x, {
        nModes: 2,
        alpha: 2000,
        tau: 0,
        dc: false,
        tolerance: 1e-6,
        maxIter: 200,
      });
      expect(result.modes.length).toBe(2);
      for (const mode of result.modes) {
        expect(mode.length).toBe(n);
      }
      expect(result.centerFrequencies.length).toBe(2);
      expect(result.nIterations).toBeGreaterThan(0);
    });
  });

  describe('Synchrosqueezing Transform', () => {
    it('produces time-frequency representation', () => {
      const n = 128;
      const x = sinWave(10, 128, n);
      // synchrosqueezingTransform(signal, fs, nVoices, fMin, fMax)
      const result = synchrosqueezingTransform(x, 128, 32);
      expect(result.tfr.length).toBeGreaterThan(0);
      expect(result.frequencies.length).toBeGreaterThan(0);
      expect(result.times.length).toBe(n);
      expect(result.nFreqs).toBeGreaterThan(0);
      expect(result.nTimes).toBe(n);
    });
  });

  describe('Stockwell Transform', () => {
    it('produces time-frequency matrix', () => {
      const n = 64;
      const x = sinWave(8, 64, n);
      // stockwellTransform(signal, fs, fMin, fMax)
      const result = stockwellTransform(x, 64, 1, 32);
      expect(result.stransform.length).toBeGreaterThan(0);
      expect(result.frequencies.length).toBeGreaterThan(0);
      expect(result.times.length).toBe(n);
      expect(result.nFreqs).toBeGreaterThan(0);
      expect(result.nTimes).toBe(n);
    });
  });
});
