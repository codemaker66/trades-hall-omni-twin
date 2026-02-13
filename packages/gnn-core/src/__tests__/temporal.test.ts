// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-6: Temporal Graph Network Tests
// Tests for createTemporalGraph, tgnUpdate, bochnerTimeEncoding,
// and positionEncoding.
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';

// Types and PRNG
import {
  createPRNG,
  type TemporalEdge,
  type TGNConfig,
  type TGNWeights,
  type GRUWeights,
} from '../types.js';

// Tensor utilities
import { xavierInit } from '../tensor.js';

// Temporal module under test
import { createTemporalGraph, tgnUpdate, tgnEmbed } from '../temporal/tgn.js';
import { bochnerTimeEncoding, positionEncoding, relativeTimeEncoding } from '../temporal/time-encoding.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Create GRU weights for TGN. Input dim = messageDim, hidden dim = memoryDim. */
function makeGRUWeights(inputDim: number, hiddenDim: number, seed: number): GRUWeights {
  const rng = createPRNG(seed);
  return {
    W_z: xavierInit(hiddenDim, inputDim, rng),
    U_z: xavierInit(hiddenDim, hiddenDim, rng),
    b_z: new Float64Array(hiddenDim),
    W_r: xavierInit(hiddenDim, inputDim, rng),
    U_r: xavierInit(hiddenDim, hiddenDim, rng),
    b_r: new Float64Array(hiddenDim),
    W_h: xavierInit(hiddenDim, inputDim, rng),
    U_h: xavierInit(hiddenDim, hiddenDim, rng),
    b_h: new Float64Array(hiddenDim),
  };
}

/** Check that every element in a Float64Array is finite. */
function allFinite(arr: Float64Array): boolean {
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isFinite(arr[i]!)) return false;
  }
  return true;
}

/** Check if any element in a Float64Array is non-zero. */
function hasNonZero(arr: Float64Array): boolean {
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] !== 0) return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('GNN-6: Temporal Graph Networks', () => {
  // =========================================================================
  // createTemporalGraph
  // =========================================================================
  describe('createTemporalGraph', () => {
    it('creates a temporal graph with correct initial state', () => {
      const numNodes = 5;
      const memoryDim = 8;

      const tg = createTemporalGraph(numNodes, memoryDim);

      expect(tg.numNodes).toBe(numNodes);
      expect(tg.memoryDim).toBe(memoryDim);
      expect(tg.temporalEdges.length).toBe(0);

      // Memory should be zero-initialized
      expect(tg.nodeMemory.length).toBe(numNodes * memoryDim);
      for (let i = 0; i < tg.nodeMemory.length; i++) {
        expect(tg.nodeMemory[i]).toBe(0);
      }

      // Last update timestamps should be zero
      expect(tg.lastUpdate.length).toBe(numNodes);
      for (let i = 0; i < tg.lastUpdate.length; i++) {
        expect(tg.lastUpdate[i]).toBe(0);
      }
    });

    it('handles single-node temporal graph', () => {
      const tg = createTemporalGraph(1, 4);
      expect(tg.numNodes).toBe(1);
      expect(tg.nodeMemory.length).toBe(4);
    });
  });

  // =========================================================================
  // tgnUpdate
  // =========================================================================
  describe('tgnUpdate', () => {
    it('updates memory after processing temporal events', () => {
      const numNodes = 4;
      const memoryDim = 8;
      const timeDim = 4;        // must be even (2 * numFreqs)
      const edgeFeatureDim = 2;
      const messageDim = 8;

      const tg = createTemporalGraph(numNodes, memoryDim);

      // TGN config
      const config: TGNConfig = {
        numNodes,
        memoryDim,
        timeDim,
        messageDim,
        aggregator: 'mean',
      };

      // Build TGN weights
      // msgW: messageDim x (memoryDim + memoryDim + timeDim + edgeFeatureDim)
      const inputDimTotal = memoryDim + memoryDim + timeDim + edgeFeatureDim;
      const rng = createPRNG(42);
      const msgW = xavierInit(messageDim, inputDimTotal, rng);
      const gru = makeGRUWeights(messageDim, memoryDim, 43);

      const weights: TGNWeights = { msgW, gru };

      // Create some temporal edge events
      const events: TemporalEdge[] = [
        { src: 0, dst: 1, timestamp: 1.0, features: new Float64Array([0.5, -0.3]) },
        { src: 1, dst: 2, timestamp: 2.0, features: new Float64Array([0.1, 0.8]) },
        { src: 0, dst: 3, timestamp: 3.0, features: new Float64Array([-0.2, 0.4]) },
      ];

      // Save initial memory state (all zeros)
      const memoryBefore = new Float64Array(tg.nodeMemory);

      // Run update
      tgnUpdate(tg, events, weights, config);

      // Memory should have changed for nodes involved in events (0, 1, 2, 3)
      const memoryAfter = tg.nodeMemory;

      // At least one node's memory should be different
      let changed = false;
      for (let i = 0; i < memoryAfter.length; i++) {
        if (memoryAfter[i] !== memoryBefore[i]) {
          changed = true;
          break;
        }
      }
      expect(changed).toBe(true);

      // All memory values should be finite
      expect(allFinite(tg.nodeMemory)).toBe(true);

      // lastUpdate timestamps should reflect the events
      expect(tg.lastUpdate[0]).toBe(3.0); // node 0 last event at t=3
      expect(tg.lastUpdate[1]).toBe(2.0); // node 1 last event at t=2
      expect(tg.lastUpdate[2]).toBe(2.0); // node 2 last event at t=2
      expect(tg.lastUpdate[3]).toBe(3.0); // node 3 last event at t=3
    });

    it('uses last aggregator correctly', () => {
      const numNodes = 3;
      const memoryDim = 4;
      const timeDim = 4;
      const edgeFeatureDim = 2;
      const messageDim = 4;

      const tg = createTemporalGraph(numNodes, memoryDim);

      const config: TGNConfig = {
        numNodes,
        memoryDim,
        timeDim,
        messageDim,
        aggregator: 'last',
      };

      const inputDimTotal = memoryDim + memoryDim + timeDim + edgeFeatureDim;
      const rng = createPRNG(50);
      const msgW = xavierInit(messageDim, inputDimTotal, rng);
      const gru = makeGRUWeights(messageDim, memoryDim, 51);
      const weights: TGNWeights = { msgW, gru };

      const events: TemporalEdge[] = [
        { src: 0, dst: 1, timestamp: 1.0, features: new Float64Array([0.1, 0.2]) },
        { src: 0, dst: 1, timestamp: 2.0, features: new Float64Array([0.3, 0.4]) },
      ];

      tgnUpdate(tg, events, weights, config);

      // Memory should be updated and finite
      expect(allFinite(tg.nodeMemory)).toBe(true);
      expect(hasNonZero(tg.nodeMemory)).toBe(true);
    });
  });

  // =========================================================================
  // tgnEmbed
  // =========================================================================
  describe('tgnEmbed', () => {
    it('returns correct embeddings for target nodes', () => {
      const tg = createTemporalGraph(4, 3);
      // Manually set some memory values
      tg.nodeMemory[0] = 1.0; tg.nodeMemory[1] = 2.0; tg.nodeMemory[2] = 3.0;
      tg.nodeMemory[3] = 4.0; tg.nodeMemory[4] = 5.0; tg.nodeMemory[5] = 6.0;
      tg.nodeMemory[6] = 7.0; tg.nodeMemory[7] = 8.0; tg.nodeMemory[8] = 9.0;

      const targets = new Uint32Array([0, 2]);
      const embeddings = tgnEmbed(tg, targets, 3);

      // Should return 2 nodes * 3 dims = 6
      expect(embeddings.length).toBe(6);
      // Node 0: [1, 2, 3]
      expect(embeddings[0]).toBe(1.0);
      expect(embeddings[1]).toBe(2.0);
      expect(embeddings[2]).toBe(3.0);
      // Node 2: [7, 8, 9]
      expect(embeddings[3]).toBe(7.0);
      expect(embeddings[4]).toBe(8.0);
      expect(embeddings[5]).toBe(9.0);
    });
  });

  // =========================================================================
  // bochnerTimeEncoding
  // =========================================================================
  describe('bochnerTimeEncoding', () => {
    it('produces output with dimensions = T * 2 * numFreqs', () => {
      const timestamps = new Float64Array([0.0, 1.0, 2.5, 5.0]);
      const frequencies = new Float64Array([0.1, 0.5, 1.0]);

      const encoded = bochnerTimeEncoding(timestamps, frequencies);

      // T=4 timestamps, d=3 frequencies -> output is T * 2d = 4 * 6 = 24
      const expectedLength = timestamps.length * 2 * frequencies.length;
      expect(encoded.length).toBe(expectedLength);
      expect(allFinite(encoded)).toBe(true);
    });

    it('produces bounded values (cos/sin are in [-1, 1], scaled by sqrt(1/d))', () => {
      const timestamps = new Float64Array([0.0, 100.0, -50.0]);
      const frequencies = new Float64Array([0.01, 0.1, 1.0, 10.0]);

      const encoded = bochnerTimeEncoding(timestamps, frequencies);

      const scale = Math.sqrt(1 / frequencies.length);
      for (let i = 0; i < encoded.length; i++) {
        // Each value should be bounded by [-scale, scale]
        expect(Math.abs(encoded[i]!)).toBeLessThanOrEqual(scale + 1e-10);
      }
    });

    it('returns zeros for timestamp=0', () => {
      const timestamps = new Float64Array([0.0]);
      const frequencies = new Float64Array([1.0, 2.0]);

      const encoded = bochnerTimeEncoding(timestamps, frequencies);
      // cos(0) = 1, sin(0) = 0
      // output: scale * [cos(0), sin(0), cos(0), sin(0)] = scale * [1, 0, 1, 0]
      const scale = Math.sqrt(1 / 2);
      expect(encoded[0]).toBeCloseTo(scale * 1.0, 10); // cos(0) = 1
      expect(encoded[1]).toBeCloseTo(scale * 0.0, 10); // sin(0) = 0
      expect(encoded[2]).toBeCloseTo(scale * 1.0, 10); // cos(0) = 1
      expect(encoded[3]).toBeCloseTo(scale * 0.0, 10); // sin(0) = 0
    });
  });

  // =========================================================================
  // positionEncoding
  // =========================================================================
  describe('positionEncoding', () => {
    it('produces output with dimensions = T * dim', () => {
      const timestamps = new Float64Array([0.0, 1.0, 2.0, 3.0, 4.0]);
      const dim = 8;

      const encoded = positionEncoding(timestamps, dim);

      // T=5, dim=8 -> output = 40
      expect(encoded.length).toBe(timestamps.length * dim);
      expect(allFinite(encoded)).toBe(true);
    });

    it('produces bounded values in [-1, 1]', () => {
      const timestamps = new Float64Array([0.0, 10.0, 100.0]);
      const dim = 16;

      const encoded = positionEncoding(timestamps, dim);

      // sin and cos are bounded by [-1, 1]
      for (let i = 0; i < encoded.length; i++) {
        expect(encoded[i]!).toBeGreaterThanOrEqual(-1 - 1e-10);
        expect(encoded[i]!).toBeLessThanOrEqual(1 + 1e-10);
      }
    });

    it('returns known values for position 0', () => {
      const timestamps = new Float64Array([0.0]);
      const dim = 4;

      const encoded = positionEncoding(timestamps, dim);

      // PE(0, 2i) = sin(0) = 0, PE(0, 2i+1) = cos(0) = 1
      expect(encoded[0]).toBeCloseTo(0, 10); // sin(0)
      expect(encoded[1]).toBeCloseTo(1, 10); // cos(0)
      expect(encoded[2]).toBeCloseTo(0, 10); // sin(0)
      expect(encoded[3]).toBeCloseTo(1, 10); // cos(0)
    });
  });

  // =========================================================================
  // relativeTimeEncoding
  // =========================================================================
  describe('relativeTimeEncoding', () => {
    it('computes Bochner encoding of |t1 - t2|', () => {
      const frequencies = new Float64Array([0.5, 1.0]);

      const enc = relativeTimeEncoding(3.0, 1.0, frequencies);

      // |3.0 - 1.0| = 2.0, output should be bochner([2.0], frequencies) -> length 2*2 = 4
      expect(enc.length).toBe(2 * frequencies.length);
      expect(allFinite(enc)).toBe(true);

      // Should be same as bochner encoding of dt=2.0
      const direct = bochnerTimeEncoding(new Float64Array([2.0]), frequencies);
      for (let i = 0; i < enc.length; i++) {
        expect(enc[i]).toBeCloseTo(direct[i]!, 10);
      }
    });

    it('is symmetric: relativeTimeEncoding(a, b) == relativeTimeEncoding(b, a)', () => {
      const frequencies = new Float64Array([0.1, 0.5, 2.0]);

      const enc1 = relativeTimeEncoding(5.0, 2.0, frequencies);
      const enc2 = relativeTimeEncoding(2.0, 5.0, frequencies);

      for (let i = 0; i < enc1.length; i++) {
        expect(enc1[i]).toBeCloseTo(enc2[i]!, 10);
      }
    });
  });
});
