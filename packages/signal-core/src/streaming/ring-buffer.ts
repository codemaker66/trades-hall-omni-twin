// ---------------------------------------------------------------------------
// SP-11: Ring Buffer — Lock-Free SPSC for Streaming
// ---------------------------------------------------------------------------
// Wait-free single-producer single-consumer ring buffer.
// Used for AudioWorklet communication and streaming FFT.
// Capacity must be power of 2 for efficient modular arithmetic.

import type { RingBufferConfig } from '../types.js';

/**
 * Single-Producer Single-Consumer (SPSC) ring buffer.
 * Lock-free for single-threaded use, wait-free design for AudioWorklet.
 */
export class RingBuffer {
  private readonly capacity: number;
  private readonly mask: number;
  private readonly channels: number;
  private readonly data: Float64Array[];
  private readPtr: number;
  private writePtr: number;

  constructor(config: RingBufferConfig) {
    // Round capacity up to next power of 2
    let cap = 1;
    while (cap < config.capacity) cap <<= 1;
    this.capacity = cap;
    this.mask = cap - 1;
    this.channels = config.channels ?? 1;
    this.data = [];
    for (let ch = 0; ch < this.channels; ch++) {
      this.data.push(new Float64Array(cap));
    }
    this.readPtr = 0;
    this.writePtr = 0;
  }

  /**
   * Number of samples available to read.
   */
  availableRead(): number {
    return (this.writePtr - this.readPtr + this.capacity) & this.mask;
  }

  /**
   * Number of samples that can be written.
   */
  availableWrite(): number {
    return this.capacity - 1 - this.availableRead();
  }

  /**
   * Write samples into the buffer (single channel).
   * Returns number of samples actually written.
   */
  write(data: Float64Array, channel: number = 0): number {
    const toWrite = Math.min(data.length, this.availableWrite());
    const buf = this.data[channel]!;

    for (let i = 0; i < toWrite; i++) {
      buf[(this.writePtr + i) & this.mask] = data[i]!;
    }
    this.writePtr = (this.writePtr + toWrite) & this.mask;
    return toWrite;
  }

  /**
   * Write interleaved multi-channel samples.
   * Data format: [ch0_s0, ch1_s0, ch0_s1, ch1_s1, ...]
   */
  writeInterleaved(data: Float64Array): number {
    const nSamples = Math.floor(data.length / this.channels);
    const toWrite = Math.min(nSamples, this.availableWrite());

    for (let i = 0; i < toWrite; i++) {
      for (let ch = 0; ch < this.channels; ch++) {
        this.data[ch]![(this.writePtr + i) & this.mask] = data[i * this.channels + ch]!;
      }
    }
    this.writePtr = (this.writePtr + toWrite) & this.mask;
    return toWrite;
  }

  /**
   * Read samples from the buffer (single channel).
   * Returns the samples read.
   */
  read(count: number, channel: number = 0): Float64Array {
    const toRead = Math.min(count, this.availableRead());
    const result = new Float64Array(toRead);
    const buf = this.data[channel]!;

    for (let i = 0; i < toRead; i++) {
      result[i] = buf[(this.readPtr + i) & this.mask]!;
    }
    this.readPtr = (this.readPtr + toRead) & this.mask;
    return result;
  }

  /**
   * Peek at samples without advancing read pointer.
   */
  peek(count: number, channel: number = 0): Float64Array {
    const toRead = Math.min(count, this.availableRead());
    const result = new Float64Array(toRead);
    const buf = this.data[channel]!;

    for (let i = 0; i < toRead; i++) {
      result[i] = buf[(this.readPtr + i) & this.mask]!;
    }
    return result;
  }

  /**
   * Skip (discard) samples from the read side.
   */
  skip(count: number): number {
    const toSkip = Math.min(count, this.availableRead());
    this.readPtr = (this.readPtr + toSkip) & this.mask;
    return toSkip;
  }

  /**
   * Reset the buffer (clear all data).
   */
  reset(): void {
    this.readPtr = 0;
    this.writePtr = 0;
    for (const buf of this.data) buf.fill(0);
  }

  /**
   * Get buffer capacity.
   */
  getCapacity(): number {
    return this.capacity;
  }
}
