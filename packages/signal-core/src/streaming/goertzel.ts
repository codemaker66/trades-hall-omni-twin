// ---------------------------------------------------------------------------
// SP-11: Goertzel Algorithm — Single-Bin DFT via 2nd-Order IIR
// ---------------------------------------------------------------------------
// More efficient than full FFT when computing < log₂(N) bins.
// Tracks specific cycles: weekly (k=N/7), monthly (k=N/30).
// s[n] = x[n] + 2cos(2πk/N)·s[n-1] - s[n-2]
// X[k] = s[N-1] - e^{-j2πk/N}·s[N-2]

import type { GoertzelConfig } from '../types.js';

/**
 * Goertzel algorithm: compute single DFT bin.
 * O(N) with very low overhead — just 1 multiply and 2 adds per sample.
 *
 * @param signal Input block
 * @param targetFrequency Target frequency in Hz
 * @param sampleRate Sample rate
 * @returns Magnitude at target frequency
 */
export function goertzel(
  signal: Float64Array,
  targetFrequency: number,
  sampleRate: number,
): { magnitude: number; phase: number } {
  const N = signal.length;
  const k = Math.round((targetFrequency * N) / sampleRate);
  const omega = (2 * Math.PI * k) / N;
  const coeff = 2 * Math.cos(omega);

  let s1 = 0;
  let s2 = 0;

  for (let n = 0; n < N; n++) {
    const s0 = signal[n]! + coeff * s1 - s2;
    s2 = s1;
    s1 = s0;
  }

  // X[k] = s1 - e^{-jω}·s2
  const re = s1 - s2 * Math.cos(omega);
  const im = s2 * Math.sin(omega);
  const magnitude = Math.sqrt(re * re + im * im);
  const phase = Math.atan2(im, re);

  return { magnitude, phase };
}

/**
 * Streaming Goertzel processor.
 * Accumulates samples in blocks and computes the target bin when a block is full.
 */
export class StreamingGoertzel {
  private readonly blockSize: number;
  private readonly targetFrequency: number;
  private readonly sampleRate: number;
  private readonly omega: number;
  private readonly coeff: number;
  private buffer: Float64Array;
  private writePos: number;
  private s1: number;
  private s2: number;

  constructor(config: GoertzelConfig) {
    this.blockSize = config.blockSize;
    this.targetFrequency = config.targetFrequency;
    this.sampleRate = config.sampleRate;

    const k = Math.round((this.targetFrequency * this.blockSize) / this.sampleRate);
    this.omega = (2 * Math.PI * k) / this.blockSize;
    this.coeff = 2 * Math.cos(this.omega);

    this.buffer = new Float64Array(this.blockSize);
    this.writePos = 0;
    this.s1 = 0;
    this.s2 = 0;
  }

  /**
   * Push a single sample. Returns magnitude if block is complete, null otherwise.
   */
  push(sample: number): { magnitude: number; phase: number } | null {
    // Update Goertzel state
    const s0 = sample + this.coeff * this.s1 - this.s2;
    this.s2 = this.s1;
    this.s1 = s0;

    this.writePos++;

    if (this.writePos >= this.blockSize) {
      // Block complete — compute result
      const re = this.s1 - this.s2 * Math.cos(this.omega);
      const im = this.s2 * Math.sin(this.omega);
      const magnitude = Math.sqrt(re * re + im * im);
      const phase = Math.atan2(im, re);

      // Reset for next block
      this.s1 = 0;
      this.s2 = 0;
      this.writePos = 0;

      return { magnitude, phase };
    }

    return null;
  }

  /**
   * Reset the processor state.
   */
  reset(): void {
    this.s1 = 0;
    this.s2 = 0;
    this.writePos = 0;
  }
}

/**
 * Multi-frequency Goertzel: track multiple frequencies simultaneously.
 * More efficient than FFT when tracking < log₂(N) specific frequencies.
 */
export function multiGoertzel(
  signal: Float64Array,
  targetFrequencies: number[],
  sampleRate: number,
): Array<{ frequency: number; magnitude: number; phase: number }> {
  return targetFrequencies.map(freq => {
    const result = goertzel(signal, freq, sampleRate);
    return { frequency: freq, ...result };
  });
}
