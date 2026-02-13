// ---------------------------------------------------------------------------
// SP-11: Stream Processor — Windowed Spectral Pipeline
// ---------------------------------------------------------------------------
// Architecture: input → ring buffer → windowed FFT → spectral output
// Suitable for real-time dashboard: ~10Hz update rate.

import type { StreamProcessorConfig, WindowFunction } from '../types.js';
import { RingBuffer } from './ring-buffer.js';
import { fft, applyWindow, nextPow2 } from '../fourier/fft.js';

export interface SpectralFrame {
  magnitudes: Float64Array;
  timestamp: number;
}

/**
 * Streaming spectral processor.
 * Buffers input samples, applies windowed FFT at hop intervals,
 * outputs spectral frames.
 */
export class StreamProcessor {
  private readonly config: StreamProcessorConfig;
  private readonly buffer: RingBuffer;
  private readonly nfft: number;
  private readonly nFreqs: number;
  private frameCount: number;
  private readonly pendingFrames: SpectralFrame[];

  constructor(config: StreamProcessorConfig) {
    this.config = config;
    this.nfft = nextPow2(config.fftSize);
    this.nFreqs = Math.floor(this.nfft / 2) + 1;
    this.buffer = new RingBuffer({ capacity: config.fftSize * 4 });
    this.frameCount = 0;
    this.pendingFrames = [];
  }

  /**
   * Push new samples into the processor.
   * Returns any completed spectral frames.
   */
  push(samples: Float64Array): SpectralFrame[] {
    this.buffer.write(samples);
    const frames: SpectralFrame[] = [];

    while (this.buffer.availableRead() >= this.config.fftSize) {
      const segment = this.buffer.peek(this.config.fftSize);
      const windowed = applyWindow(segment, this.config.window);
      const { re, im } = fft(windowed, this.nfft);

      const magnitudes = new Float64Array(this.nFreqs);
      for (let k = 0; k < this.nFreqs; k++) {
        magnitudes[k] = Math.sqrt(re[k]! * re[k]! + im[k]! * im[k]!);
      }

      const timestamp = this.frameCount * this.config.hopSize / this.config.sampleRate;
      frames.push({ magnitudes, timestamp });
      this.frameCount++;

      // Advance by hop size
      this.buffer.skip(this.config.hopSize);
    }

    return frames;
  }

  /**
   * Get frequency array for the spectral frames.
   */
  getFrequencies(): Float64Array {
    const freqs = new Float64Array(this.nFreqs);
    for (let k = 0; k < this.nFreqs; k++) {
      freqs[k] = (k * this.config.sampleRate) / this.nfft;
    }
    return freqs;
  }

  /**
   * Get number of frequency bins.
   */
  getNumFreqs(): number {
    return this.nFreqs;
  }

  /**
   * Reset processor state.
   */
  reset(): void {
    this.buffer.reset();
    this.frameCount = 0;
  }
}

/**
 * Exponential Moving Average filter for smoothing spectral frames.
 * Useful for reducing noise in real-time spectral display.
 */
export class SpectralSmoother {
  private readonly alpha: number;
  private state: Float64Array | null;

  constructor(alpha: number = 0.3) {
    this.alpha = alpha;
    this.state = null;
  }

  smooth(frame: Float64Array): Float64Array {
    if (!this.state) {
      this.state = new Float64Array(frame);
      return new Float64Array(frame);
    }

    const result = new Float64Array(frame.length);
    for (let i = 0; i < frame.length; i++) {
      this.state[i] = this.alpha * frame[i]! + (1 - this.alpha) * this.state[i]!;
      result[i] = this.state[i]!;
    }
    return result;
  }

  reset(): void {
    this.state = null;
  }
}
