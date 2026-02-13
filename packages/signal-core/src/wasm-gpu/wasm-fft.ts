// ---------------------------------------------------------------------------
// SP-12: WASM FFT Interface Types
// ---------------------------------------------------------------------------
// Type definitions for rustfft WASM module integration.
// The actual WASM module would be built from compute-wasm/src/fft.rs
// and loaded at runtime. This provides the TypeScript interface.

import type { WASMFFTConfig } from '../types.js';

/**
 * Interface for a WASM-backed FFT module.
 * Implementations would wrap a rustfft WASM instance.
 */
export interface WASMFFTModule {
  /** Compute forward FFT: real input → complex output magnitudes */
  magnitudeSpectrum(input: Float32Array, size: number): Float32Array;
  /** Compute forward FFT: complex input → complex output */
  fftForward(realInput: Float32Array, imagInput: Float32Array, size: number): { real: Float32Array; imag: Float32Array };
  /** Compute inverse FFT */
  fftInverse(realInput: Float32Array, imagInput: Float32Array, size: number): { real: Float32Array; imag: Float32Array };
  /** Free WASM memory */
  dispose(): void;
}

/**
 * Pure-TS fallback FFT that matches the WASM interface.
 * Used when WASM module is not available (SSR, test environments).
 */
export class FallbackFFT implements WASMFFTModule {
  magnitudeSpectrum(input: Float32Array, size: number): Float32Array {
    const re = new Float64Array(size);
    const im = new Float64Array(size);
    for (let i = 0; i < Math.min(input.length, size); i++) {
      re[i] = input[i]!;
    }

    // Bit-reversal permutation
    for (let i = 1, j = 0; i < size; i++) {
      let bit = size >> 1;
      while (j & bit) { j ^= bit; bit >>= 1; }
      j ^= bit;
      if (i < j) {
        let tmp = re[i]!; re[i] = re[j]!; re[j] = tmp;
        tmp = im[i]!; im[i] = im[j]!; im[j] = tmp;
      }
    }

    for (let len = 2; len <= size; len <<= 1) {
      const halfLen = len >> 1;
      const angle = -2 * Math.PI / len;
      const wRe = Math.cos(angle), wIm = Math.sin(angle);
      for (let i = 0; i < size; i += len) {
        let curRe = 1, curIm = 0;
        for (let j = 0; j < halfLen; j++) {
          const eIdx = i + j, oIdx = i + j + halfLen;
          const tRe = curRe * re[oIdx]! - curIm * im[oIdx]!;
          const tIm = curRe * im[oIdx]! + curIm * re[oIdx]!;
          re[oIdx] = re[eIdx]! - tRe;
          im[oIdx] = im[eIdx]! - tIm;
          re[eIdx] = re[eIdx]! + tRe;
          im[eIdx] = im[eIdx]! + tIm;
          const nextRe = curRe * wRe - curIm * wIm;
          curIm = curRe * wIm + curIm * wRe;
          curRe = nextRe;
        }
      }
    }

    const nPos = size >> 1;
    const result = new Float32Array(nPos);
    for (let k = 0; k < nPos; k++) {
      result[k] = Math.sqrt(re[k]! * re[k]! + im[k]! * im[k]!) as number;
    }
    return result;
  }

  fftForward(realInput: Float32Array, imagInput: Float32Array, size: number): { real: Float32Array; imag: Float32Array } {
    const re = new Float64Array(size);
    const im = new Float64Array(size);
    for (let i = 0; i < Math.min(realInput.length, size); i++) {
      re[i] = realInput[i]!;
      im[i] = imagInput[i]!;
    }

    // In-place FFT (same as above)
    for (let i = 1, j = 0; i < size; i++) {
      let bit = size >> 1;
      while (j & bit) { j ^= bit; bit >>= 1; }
      j ^= bit;
      if (i < j) {
        let tmp = re[i]!; re[i] = re[j]!; re[j] = tmp;
        tmp = im[i]!; im[i] = im[j]!; im[j] = tmp;
      }
    }
    for (let len = 2; len <= size; len <<= 1) {
      const halfLen = len >> 1;
      const angle = -2 * Math.PI / len;
      const wRe = Math.cos(angle), wIm = Math.sin(angle);
      for (let i = 0; i < size; i += len) {
        let curRe = 1, curIm = 0;
        for (let j = 0; j < halfLen; j++) {
          const eIdx = i + j, oIdx = i + j + halfLen;
          const tRe = curRe * re[oIdx]! - curIm * im[oIdx]!;
          const tIm = curRe * im[oIdx]! + curIm * re[oIdx]!;
          re[oIdx] = re[eIdx]! - tRe;
          im[oIdx] = im[eIdx]! - tIm;
          re[eIdx] = re[eIdx]! + tRe;
          im[eIdx] = im[eIdx]! + tIm;
          const nextRe = curRe * wRe - curIm * wIm;
          curIm = curRe * wIm + curIm * wRe;
          curRe = nextRe;
        }
      }
    }

    const real = new Float32Array(size);
    const imag = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      real[i] = re[i]! as number;
      imag[i] = im[i]! as number;
    }
    return { real, imag };
  }

  fftInverse(realInput: Float32Array, imagInput: Float32Array, size: number): { real: Float32Array; imag: Float32Array } {
    // IFFT via conjugate trick
    const conjImag = new Float32Array(size);
    for (let i = 0; i < size; i++) conjImag[i] = -imagInput[i]!;

    const { real, imag } = this.fftForward(realInput, conjImag, size);

    for (let i = 0; i < size; i++) {
      real[i] = real[i]! / size;
      imag[i] = -imag[i]! / size;
    }
    return { real, imag };
  }

  dispose(): void {
    // No-op for pure TS fallback
  }
}

/**
 * Create an FFT module (WASM with fallback to pure TS).
 */
export function createFFTModule(_config: WASMFFTConfig): WASMFFTModule {
  // In a real implementation, this would attempt to load the WASM module:
  // const wasmModule = await import('@omni-twin/compute-wasm');
  // For now, return the pure-TS fallback
  return new FallbackFFT();
}
