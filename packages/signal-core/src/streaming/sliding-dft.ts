// ---------------------------------------------------------------------------
// SP-11: Sliding DFT — O(N) Recursive Bin Update
// ---------------------------------------------------------------------------
// X_k[n] = e^{j2πk/N} · (X_k[n-1] + x[n] - x[n-N])
// vs O(N log N) for full FFT recomputation per new sample.

import type { SlidingDFTConfig } from '../types.js';

/**
 * Sliding DFT processor.
 * Maintains frequency bins that update in O(1) per new sample.
 */
export class SlidingDFT {
  private readonly N: number;
  private readonly trackedBins: number[];
  private readonly buffer: Float64Array;
  private readonly binRe: Float64Array;
  private readonly binIm: Float64Array;
  private readonly twiddleRe: Float64Array;
  private readonly twiddleIm: Float64Array;
  private writePos: number;
  private sampleCount: number;

  constructor(config: SlidingDFTConfig) {
    this.N = config.windowSize;
    this.trackedBins = config.trackedBins ?? Array.from({ length: this.N }, (_, i) => i);

    this.buffer = new Float64Array(this.N);
    const nBins = this.trackedBins.length;
    this.binRe = new Float64Array(nBins);
    this.binIm = new Float64Array(nBins);

    // Precompute twiddle factors: e^{j2πk/N}
    this.twiddleRe = new Float64Array(nBins);
    this.twiddleIm = new Float64Array(nBins);
    for (let i = 0; i < nBins; i++) {
      const k = this.trackedBins[i]!;
      const angle = (2 * Math.PI * k) / this.N;
      this.twiddleRe[i] = Math.cos(angle);
      this.twiddleIm[i] = Math.sin(angle);
    }

    this.writePos = 0;
    this.sampleCount = 0;
  }

  /**
   * Push a new sample and update all tracked bins.
   * O(K) where K is number of tracked bins.
   */
  push(sample: number): void {
    const oldest = this.buffer[this.writePos]!;
    this.buffer[this.writePos] = sample;
    this.writePos = (this.writePos + 1) % this.N;
    this.sampleCount++;

    const diff = sample - oldest;

    for (let i = 0; i < this.trackedBins.length; i++) {
      // X_k[n] = twiddle · (X_k[n-1] + x[n] - x[n-N])
      const newRe = this.binRe[i]! + diff;
      const newIm = this.binIm[i]!;

      // Multiply by twiddle factor
      this.binRe[i] = this.twiddleRe[i]! * newRe - this.twiddleIm[i]! * newIm;
      this.binIm[i] = this.twiddleRe[i]! * newIm + this.twiddleIm[i]! * newRe;
    }
  }

  /**
   * Get magnitude at a specific tracked bin index.
   */
  getMagnitude(binIndex: number): number {
    const idx = this.trackedBins.indexOf(binIndex);
    if (idx < 0) return 0;
    return Math.sqrt(this.binRe[idx]! * this.binRe[idx]! + this.binIm[idx]! * this.binIm[idx]!);
  }

  /**
   * Get all tracked bin magnitudes.
   */
  getAllMagnitudes(): Float64Array {
    const mags = new Float64Array(this.trackedBins.length);
    for (let i = 0; i < this.trackedBins.length; i++) {
      mags[i] = Math.sqrt(this.binRe[i]! * this.binRe[i]! + this.binIm[i]! * this.binIm[i]!);
    }
    return mags;
  }

  /**
   * Get complex value at a tracked bin.
   */
  getBin(binIndex: number): { re: number; im: number } {
    const idx = this.trackedBins.indexOf(binIndex);
    if (idx < 0) return { re: 0, im: 0 };
    return { re: this.binRe[idx]!, im: this.binIm[idx]! };
  }

  /**
   * Check if the buffer is fully filled.
   */
  isReady(): boolean {
    return this.sampleCount >= this.N;
  }

  /**
   * Reset all state.
   */
  reset(): void {
    this.buffer.fill(0);
    this.binRe.fill(0);
    this.binIm.fill(0);
    this.writePos = 0;
    this.sampleCount = 0;
  }
}
