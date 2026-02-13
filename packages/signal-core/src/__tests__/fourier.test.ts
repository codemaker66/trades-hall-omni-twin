// ---------------------------------------------------------------------------
// SP-1: Fourier Analysis Tests
// ---------------------------------------------------------------------------
import { describe, it, expect } from 'vitest';
import {
  fft,
  ifft,
  magnitudeSpectrum,
  extractSeasonality,
  fftConvolve,
  applyWindow,
  nextPow2,
  detrend,
} from '../fourier/fft.js';
import { welchPSD, multitaperPSD } from '../fourier/welch.js';
import { stft, istft } from '../fourier/stft.js';
import {
  lowpassFilter,
  highpassFilter,
  bandpassFilter,
  bandstopFilter,
  extractPeriodic,
} from '../fourier/frequency-filter.js';

// ---- Helpers ----
function sinWave(freq: number, sr: number, n: number): Float64Array {
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.sin(2 * Math.PI * freq * i / sr);
  return x;
}

function almostEqual(a: number, b: number, tol = 1e-6): boolean {
  return Math.abs(a - b) < tol;
}

describe('SP-1: Fourier Analysis', () => {
  describe('nextPow2', () => {
    it('returns power of 2 for exact inputs', () => {
      expect(nextPow2(1)).toBe(1);
      expect(nextPow2(2)).toBe(2);
      expect(nextPow2(1024)).toBe(1024);
    });
    it('rounds up non-powers', () => {
      expect(nextPow2(3)).toBe(4);
      expect(nextPow2(5)).toBe(8);
      expect(nextPow2(1000)).toBe(1024);
    });
  });

  describe('applyWindow', () => {
    it('rectangular window is identity', () => {
      const x = new Float64Array([1, 2, 3, 4]);
      const w = applyWindow(x, 'rectangular');
      expect(Array.from(w)).toEqual([1, 2, 3, 4]);
    });
    it('hann window zeros endpoints', () => {
      const x = new Float64Array([1, 1, 1, 1]);
      const w = applyWindow(x, 'hann');
      expect(w[0]).toBeCloseTo(0, 10);
      expect(w[3]).toBeCloseTo(0, 10);
    });
  });

  describe('detrend', () => {
    it('removes linear trend', () => {
      const x = new Float64Array([0, 1, 2, 3, 4, 5, 6, 7]);
      const d = detrend(x);
      const mean = d.reduce((a, b) => a + b, 0) / d.length;
      expect(Math.abs(mean)).toBeLessThan(1e-10);
    });
  });

  describe('FFT / IFFT roundtrip', () => {
    it('recovers original signal', () => {
      const n = 64;
      const x = sinWave(10, 64, n);
      const { re, im } = fft(x, n);
      expect(re.length).toBe(n);
      expect(im.length).toBe(n);

      const recovered = ifft(re, im);
      for (let i = 0; i < n; i++) {
        expect(recovered[i]).toBeCloseTo(x[i]!, 8);
      }
    });

    it('handles DC signal', () => {
      const n = 16;
      const x = new Float64Array(n).fill(3.0);
      const { re, im } = fft(x, n);
      // DC bin should be 3*n, all others ~0
      expect(re[0]).toBeCloseTo(3 * n, 6);
      for (let k = 1; k < n; k++) {
        expect(Math.abs(re[k]!)).toBeLessThan(1e-10);
        expect(Math.abs(im[k]!)).toBeLessThan(1e-10);
      }
    });
  });

  describe('magnitudeSpectrum', () => {
    it('detects single frequency peak', () => {
      const n = 256;
      const sr = 256;
      const freq = 32;
      const x = sinWave(freq, sr, n);
      // magnitudeSpectrum(signal, fs, windowType) → SpectralResult
      const result = magnitudeSpectrum(x, sr);
      expect(result.frequencies.length).toBeGreaterThan(0);
      expect(result.magnitudes.length).toBe(result.frequencies.length);

      // Find peak bin
      let maxBin = 0;
      let maxVal = 0;
      for (let k = 1; k < result.magnitudes.length; k++) {
        if (result.magnitudes[k]! > maxVal) { maxVal = result.magnitudes[k]!; maxBin = k; }
      }
      // Peak frequency should be close to 32Hz
      expect(Math.abs(result.frequencies[maxBin]! - freq)).toBeLessThan(2);
    });
  });

  describe('extractSeasonality', () => {
    it('finds dominant periods', () => {
      const n = 1024;
      const sr = 256;
      const x = sinWave(16, sr, n);
      const result = extractSeasonality(x, sr);
      expect(result.dominantPeriods.length).toBeGreaterThan(0);
      // Period = 1 / 16Hz = 0.0625 seconds
      const topPeriod = result.dominantPeriods[0]!.period;
      expect(Math.abs(topPeriod - 1 / 16)).toBeLessThan(0.005);
    });
  });

  describe('fftConvolve', () => {
    it('convolves two simple signals', () => {
      const a = new Float64Array([1, 2, 3]);
      const b = new Float64Array([0, 1, 0.5]);
      const c = fftConvolve(a, b);
      // Expected: [0, 1, 2.5, 4, 1.5]
      expect(c.length).toBe(5);
      expect(c[0]).toBeCloseTo(0, 6);
      expect(c[1]).toBeCloseTo(1, 6);
      expect(c[2]).toBeCloseTo(2.5, 6);
      expect(c[3]).toBeCloseTo(4, 6);
      expect(c[4]).toBeCloseTo(1.5, 6);
    });
  });

  describe('Welch PSD', () => {
    it('returns correct number of bins', () => {
      const x = sinWave(10, 100, 1024);
      // welchPSD(signal, fs, nperseg, noverlap, windowType)
      const result = welchPSD(x, 100, 256, 128, 'hann');
      const nFreqs = Math.floor(nextPow2(256) / 2) + 1;
      expect(result.psd.length).toBe(nFreqs);
      expect(result.frequencies.length).toBe(nFreqs);
    });

    it('peak at signal frequency', () => {
      const sr = 256;
      const freq = 30;
      const x = sinWave(freq, sr, 2048);
      const result = welchPSD(x, sr, 256, 128, 'hann');
      let maxBin = 0;
      let maxVal = 0;
      for (let k = 1; k < result.psd.length; k++) {
        if (result.psd[k]! > maxVal) { maxVal = result.psd[k]!; maxBin = k; }
      }
      const peakFreq = result.frequencies[maxBin]!;
      expect(Math.abs(peakFreq - freq)).toBeLessThan(2);
    });
  });

  describe('Multitaper PSD', () => {
    it('returns correct number of bins', () => {
      const x = sinWave(10, 100, 256);
      // multitaperPSD(signal, fs, nw, nfft)
      const result = multitaperPSD(x, 100, 4);
      expect(result.psd.length).toBeGreaterThan(0);
      expect(result.frequencies.length).toBe(result.psd.length);
    });
  });

  describe('STFT / ISTFT roundtrip', () => {
    it('produces spectrogram from STFT', () => {
      const n = 512;
      const nperseg = 128;
      const hop = 64;
      const x = sinWave(20, 128, n);
      // stft(signal, fs, nperseg, noverlap, windowType)
      const result = stft(x, 128, nperseg, nperseg - hop, 'hann');
      expect(result.spectrogram.length).toBeGreaterThan(0);
      expect(result.times.length).toBe(result.nTimes);
      expect(result.frequencies.length).toBe(result.nFreqs);
      expect(result.nFreqs).toBe(Math.floor(nextPow2(nperseg) / 2) + 1);

      // Roundtrip via ISTFT
      const recovered = istft(result.spectrogram, result.nFreqs, result.nTimes, nperseg, hop, 'hann');
      expect(recovered.length).toBeGreaterThan(0);
    });
  });

  describe('Frequency-domain filters', () => {
    it('lowpass removes high frequency', () => {
      const sr = 256;
      const n = 512;
      const low = sinWave(10, sr, n);
      const high = sinWave(100, sr, n);
      const x = new Float64Array(n);
      for (let i = 0; i < n; i++) x[i] = low[i]! + high[i]!;

      const filtered = lowpassFilter(x, 50, sr);
      // Filtered should be closer to low-freq component
      let errLow = 0, errHigh = 0;
      for (let i = 0; i < n; i++) {
        errLow += (filtered[i]! - low[i]!) ** 2;
        errHigh += (filtered[i]! - high[i]!) ** 2;
      }
      expect(errLow).toBeLessThan(errHigh);
    });

    it('highpass removes low frequency', () => {
      const sr = 256;
      const n = 512;
      const low = sinWave(5, sr, n);
      const high = sinWave(80, sr, n);
      const x = new Float64Array(n);
      for (let i = 0; i < n; i++) x[i] = low[i]! + high[i]!;

      const filtered = highpassFilter(x, 40, sr);
      let errHigh = 0, errLow = 0;
      for (let i = 0; i < n; i++) {
        errHigh += (filtered[i]! - high[i]!) ** 2;
        errLow += (filtered[i]! - low[i]!) ** 2;
      }
      expect(errHigh).toBeLessThan(errLow);
    });

    it('bandpass preserves middle frequency', () => {
      const sr = 256;
      const n = 512;
      const mid = sinWave(40, sr, n);
      const filtered = bandpassFilter(mid, 30, 50, sr);
      // Should preserve most of the energy
      let energy = 0;
      for (let i = 0; i < n; i++) energy += filtered[i]! ** 2;
      expect(energy).toBeGreaterThan(0);
    });

    it('bandstop removes target frequency', () => {
      const sr = 256;
      const n = 512;
      const target = sinWave(40, sr, n);
      const other = sinWave(10, sr, n);
      const x = new Float64Array(n);
      for (let i = 0; i < n; i++) x[i] = target[i]! + other[i]!;

      const filtered = bandstopFilter(x, 35, 45, sr);
      // Should mostly remove the 40Hz component
      let errOther = 0;
      for (let i = 0; i < n; i++) errOther += (filtered[i]! - other[i]!) ** 2;
      let errTarget = 0;
      for (let i = 0; i < n; i++) errTarget += (filtered[i]! - target[i]!) ** 2;
      expect(errOther).toBeLessThan(errTarget);
    });

    it('extractPeriodic isolates target frequency', () => {
      const sr = 256;
      const n = 512;
      const x = sinWave(32, sr, n);
      const extracted = extractPeriodic(x, 1 / 32, 0.2, sr);
      let energy = 0;
      for (let i = 0; i < n; i++) energy += extracted[i]! ** 2;
      expect(energy).toBeGreaterThan(0);
    });
  });
});
