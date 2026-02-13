// ---------------------------------------------------------------------------
// SP-12: WASM/WebGPU DSP Tests
// ---------------------------------------------------------------------------
import { describe, it, expect } from 'vitest';
import {
  createFFTModule,
  FallbackFFT,
  createWebGPUFFTModule,
  STOCKHAM_FFT_WGSL,
} from '../wasm-gpu/index.js';

function sinWave(freq: number, sr: number, n: number): Float32Array {
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.sin(2 * Math.PI * freq * i / sr);
  return x;
}

describe('SP-12: WASM/WebGPU DSP', () => {
  describe('FallbackFFT', () => {
    it('computes magnitude spectrum', () => {
      const fft = new FallbackFFT();
      const x = sinWave(10, 128, 128);
      const mags = fft.magnitudeSpectrum(x, 128);
      expect(mags.length).toBe(64); // n/2
      // Should have a peak near bin 10
      let maxBin = 0;
      let maxVal = 0;
      for (let k = 1; k < mags.length; k++) {
        if (mags[k]! > maxVal) { maxVal = mags[k]!; maxBin = k; }
      }
      expect(maxBin).toBe(10);
    });

    it('forward/inverse roundtrip', () => {
      const fft = new FallbackFFT();
      const n = 64;
      const real = sinWave(8, 64, n);
      const imag = new Float32Array(n);
      const { real: fwdRe, imag: fwdIm } = fft.fftForward(real, imag, n);
      const { real: recRe } = fft.fftInverse(fwdRe, fwdIm, n);
      for (let i = 0; i < n; i++) {
        expect(recRe[i]).toBeCloseTo(real[i]!, 3);
      }
    });

    it('dispose is a no-op', () => {
      const fft = new FallbackFFT();
      fft.dispose(); // should not throw
    });
  });

  describe('createFFTModule', () => {
    it('returns a FallbackFFT instance', () => {
      const mod = createFFTModule({ size: 1024, useSIMD: true });
      expect(mod).toBeInstanceOf(FallbackFFT);
    });
  });

  describe('WebGPU FFT', () => {
    it('STOCKHAM_FFT_WGSL is valid string', () => {
      expect(typeof STOCKHAM_FFT_WGSL).toBe('string');
      expect(STOCKHAM_FFT_WGSL.length).toBeGreaterThan(100);
      // Contains key WGSL constructs
      expect(STOCKHAM_FFT_WGSL).toContain('@compute');
      expect(STOCKHAM_FFT_WGSL).toContain('@workgroup_size');
      expect(STOCKHAM_FFT_WGSL).toContain('vec2<f32>');
    });

    it('createWebGPUFFTModule returns module', () => {
      const mod = createWebGPUFFTModule({ size: 1024, useStockham: true });
      expect(typeof mod.isAvailable).toBe('function');
      expect(typeof mod.initialize).toBe('function');
      expect(typeof mod.dispose).toBe('function');
    });

    it('fftForward throws not-implemented', async () => {
      const mod = createWebGPUFFTModule({ size: 1024, useStockham: true });
      await expect(mod.fftForward(new Float32Array(10))).rejects.toThrow('WebGPU FFT not implemented');
    });

    it('fftInverse throws not-implemented', async () => {
      const mod = createWebGPUFFTModule({ size: 1024, useStockham: true });
      await expect(mod.fftInverse(new Float32Array(10))).rejects.toThrow('WebGPU FFT not implemented');
    });
  });
});
