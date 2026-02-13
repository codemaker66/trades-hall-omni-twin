// ---------------------------------------------------------------------------
// SP-11: Streaming Architecture Tests
// ---------------------------------------------------------------------------
import { describe, it, expect } from 'vitest';
import { SlidingDFT } from '../streaming/sliding-dft.js';
import { goertzel, StreamingGoertzel, multiGoertzel } from '../streaming/goertzel.js';
import { RingBuffer } from '../streaming/ring-buffer.js';
import { StreamProcessor, SpectralSmoother } from '../streaming/stream-processor.js';

function sinWave(freq: number, sr: number, n: number): Float64Array {
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.sin(2 * Math.PI * freq * i / sr);
  return x;
}

describe('SP-11: Streaming Architecture', () => {
  describe('SlidingDFT', () => {
    it('initializes and updates', () => {
      const sdft = new SlidingDFT({ windowSize: 32 });
      // Push a window of samples
      for (let i = 0; i < 32; i++) {
        sdft.push(Math.sin(2 * Math.PI * 16 * i / 128));
      }
      const spectrum = sdft.getAllMagnitudes();
      expect(spectrum.length).toBe(32);
    });

    it('detects frequency after initial fill', () => {
      const windowSize = 64;
      const sr = 256;
      const freq = 32; // bin = freq * windowSize / sr = 8
      const sdft = new SlidingDFT({ windowSize });

      // Fill the window
      for (let i = 0; i < windowSize; i++) {
        sdft.push(Math.sin(2 * Math.PI * freq * i / sr));
      }
      const spectrum = sdft.getAllMagnitudes();
      const expectedBin = Math.round(freq * windowSize / sr);
      let maxBin = 0;
      let maxVal = 0;
      for (let k = 1; k < spectrum.length; k++) {
        if (spectrum[k]! > maxVal) { maxVal = spectrum[k]!; maxBin = k; }
      }
      expect(maxBin).toBe(expectedBin);
    });

    it('reports ready after N samples', () => {
      const sdft = new SlidingDFT({ windowSize: 16 });
      expect(sdft.isReady()).toBe(false);
      for (let i = 0; i < 16; i++) sdft.push(0);
      expect(sdft.isReady()).toBe(true);
    });

    it('getMagnitude returns magnitude for tracked bin', () => {
      const sdft = new SlidingDFT({ windowSize: 32, trackedBins: [4, 8] });
      for (let i = 0; i < 32; i++) {
        sdft.push(Math.sin(2 * Math.PI * 4 * i / 32));
      }
      const mag = sdft.getMagnitude(4);
      expect(mag).toBeGreaterThan(0);
      // Untracked bin returns 0
      const untracked = sdft.getMagnitude(2);
      expect(untracked).toBe(0);
    });

    it('reset clears state', () => {
      const sdft = new SlidingDFT({ windowSize: 16 });
      for (let i = 0; i < 16; i++) sdft.push(1);
      expect(sdft.isReady()).toBe(true);
      sdft.reset();
      expect(sdft.isReady()).toBe(false);
    });
  });

  describe('Goertzel', () => {
    it('detects single frequency magnitude', () => {
      const sr = 256;
      const n = 256;
      const freq = 32;
      const x = sinWave(freq, sr, n);
      const result = goertzel(x, freq, sr);
      expect(result.magnitude).toBeGreaterThan(0);
    });

    it('target frequency has higher magnitude than off-target', () => {
      const sr = 256;
      const n = 256;
      const x = sinWave(32, sr, n);
      const resultTarget = goertzel(x, 32, sr);
      const resultOff = goertzel(x, 80, sr);
      expect(resultTarget.magnitude).toBeGreaterThan(resultOff.magnitude);
    });

    it('returns phase', () => {
      const sr = 256;
      const n = 256;
      const x = sinWave(32, sr, n);
      const result = goertzel(x, 32, sr);
      expect(typeof result.phase).toBe('number');
      expect(Number.isFinite(result.phase)).toBe(true);
    });

    it('StreamingGoertzel accumulates correctly', () => {
      const sr = 256;
      const freq = 32;
      const sg = new StreamingGoertzel({ targetFrequency: freq, sampleRate: sr, blockSize: 256 });
      const x = sinWave(freq, sr, 256);
      let lastResult: { magnitude: number; phase: number } | null = null;
      for (let i = 0; i < 256; i++) {
        const result = sg.push(x[i]!);
        if (result !== null) lastResult = result;
      }
      expect(lastResult).not.toBeNull();
      expect(lastResult!.magnitude).toBeGreaterThan(0);
    });

    it('multiGoertzel computes magnitudes for multiple frequencies', () => {
      const sr = 256;
      const n = 256;
      const x = sinWave(32, sr, n);
      const results = multiGoertzel(x, [16, 32, 64], sr);
      expect(results.length).toBe(3);
      // Each result has frequency, magnitude, phase
      expect(results[0]!.frequency).toBe(16);
      expect(results[1]!.frequency).toBe(32);
      expect(results[2]!.frequency).toBe(64);
      // 32Hz should be strongest
      expect(results[1]!.magnitude).toBeGreaterThan(results[0]!.magnitude);
      expect(results[1]!.magnitude).toBeGreaterThan(results[2]!.magnitude);
    });
  });

  describe('RingBuffer', () => {
    it('writes and reads correctly', () => {
      const rb = new RingBuffer({ capacity: 16 });
      const data = new Float64Array([1, 2, 3, 4, 5]);
      rb.write(data);
      expect(rb.availableRead()).toBe(5);
      const out = rb.read(3);
      expect(out.length).toBe(3);
      expect(out[0]).toBe(1);
      expect(out[1]).toBe(2);
      expect(out[2]).toBe(3);
      expect(rb.availableRead()).toBe(2);
    });

    it('wraps around correctly', () => {
      const rb = new RingBuffer({ capacity: 8 }); // actual 8
      const d1 = new Float64Array([1, 2, 3, 4, 5, 6]);
      rb.write(d1);
      rb.read(4); // consume 4
      const d2 = new Float64Array([7, 8, 9, 10]);
      rb.write(d2);
      expect(rb.availableRead()).toBe(6);
      const out = rb.read(6);
      expect(Array.from(out)).toEqual([5, 6, 7, 8, 9, 10]);
    });

    it('peek does not consume data', () => {
      const rb = new RingBuffer({ capacity: 16 });
      rb.write(new Float64Array([1, 2, 3, 4]));
      const peeked = rb.peek(2);
      expect(peeked[0]).toBe(1);
      expect(peeked[1]).toBe(2);
      expect(rb.availableRead()).toBe(4); // unchanged
    });

    it('skip advances read pointer', () => {
      const rb = new RingBuffer({ capacity: 16 });
      rb.write(new Float64Array([1, 2, 3, 4]));
      rb.skip(2);
      expect(rb.availableRead()).toBe(2);
      const out = rb.read(2);
      expect(out[0]).toBe(3);
      expect(out[1]).toBe(4);
    });

    it('reset clears buffer', () => {
      const rb = new RingBuffer({ capacity: 16 });
      rb.write(new Float64Array([1, 2, 3]));
      rb.reset();
      expect(rb.availableRead()).toBe(0);
    });
  });

  describe('StreamProcessor', () => {
    it('produces spectral frames', () => {
      const sp = new StreamProcessor({
        fftSize: 64,
        hopSize: 32,
        sampleRate: 128,
        window: 'hann',
      });
      const x = sinWave(16, 128, 256);
      const frames = sp.push(x);
      expect(frames.length).toBeGreaterThan(0);
      for (const frame of frames) {
        expect(frame.magnitudes.length).toBe(33); // 64/2 + 1
        expect(frame.timestamp).toBeGreaterThanOrEqual(0);
      }
    });

    it('getFrequencies returns correct array', () => {
      const sp = new StreamProcessor({
        fftSize: 64,
        hopSize: 32,
        sampleRate: 128,
        window: 'hann',
      });
      const freqs = sp.getFrequencies();
      expect(freqs.length).toBe(33);
      expect(freqs[0]).toBe(0);
      expect(freqs[freqs.length - 1]).toBeCloseTo(64, 1); // Nyquist
    });

    it('reset clears state', () => {
      const sp = new StreamProcessor({
        fftSize: 64,
        hopSize: 32,
        sampleRate: 128,
        window: 'hann',
      });
      sp.push(sinWave(10, 128, 128));
      sp.reset();
      // After reset, no frames until buffer fills again
      const frames = sp.push(new Float64Array(16));
      expect(frames.length).toBe(0);
    });
  });

  describe('SpectralSmoother', () => {
    it('smooths spectral frames', () => {
      const smoother = new SpectralSmoother(0.3);
      const frame1 = new Float64Array([10, 20, 30]);
      const frame2 = new Float64Array([20, 30, 40]);
      const s1 = smoother.smooth(frame1);
      expect(Array.from(s1)).toEqual([10, 20, 30]); // first frame = identity
      const s2 = smoother.smooth(frame2);
      // EMA: alpha * new + (1-alpha) * old
      expect(s2[0]).toBeCloseTo(0.3 * 20 + 0.7 * 10, 10);
      expect(s2[1]).toBeCloseTo(0.3 * 30 + 0.7 * 20, 10);
    });

    it('reset clears smoothing state', () => {
      const smoother = new SpectralSmoother(0.5);
      smoother.smooth(new Float64Array([10, 20]));
      smoother.reset();
      const s = smoother.smooth(new Float64Array([30, 40]));
      // After reset, first frame is identity again
      expect(s[0]).toBe(30);
      expect(s[1]).toBe(40);
    });
  });
});
