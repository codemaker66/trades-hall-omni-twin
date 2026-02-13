// ---------------------------------------------------------------------------
// HPC-6: Streaming Algorithms — Bloom Filter
// ---------------------------------------------------------------------------
// Probabilistic set membership test. False positives possible, false negatives
// are not. Pure TypeScript, zero external dependencies.
// ---------------------------------------------------------------------------

import type { BloomFilterConfig, BloomFilterState } from '../types.js';
import { murmurHash3_32 } from '../types.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LN2 = Math.LN2;
const LN2_SQ = LN2 * LN2;

// ---------------------------------------------------------------------------
// Create
// ---------------------------------------------------------------------------

/**
 * Create a Bloom filter with optimal size and hash count derived from
 * the expected number of items and desired false positive rate.
 *
 * Bit array size: m = ceil(-n * ln(p) / (ln2)^2)
 * Hash count:     k = ceil(m / n * ln2)
 *
 * @param config.expectedItems      Expected number of distinct items to insert.
 * @param config.falsePositiveRate  Desired FP rate (e.g. 0.01 for 1%).
 */
export function createBloomFilter(config: BloomFilterConfig): BloomFilterState {
  const n = Math.max(1, config.expectedItems);
  const p = Math.max(1e-15, Math.min(1 - 1e-15, config.falsePositiveRate));

  // Optimal bit count
  const m = Math.ceil((-n * Math.log(p)) / LN2_SQ);
  // Optimal hash count
  const k = Math.max(1, Math.ceil((m / n) * LN2));

  // Uint8Array for the bit array (1 bit per position, packed into bytes)
  const byteCount = Math.ceil(m / 8);

  return {
    bits: new Uint8Array(byteCount),
    numHashes: k,
    size: m,
    count: 0,
  };
}

// ---------------------------------------------------------------------------
// Internal: multi-hash using double-hashing scheme
// ---------------------------------------------------------------------------

/**
 * Generate the i-th hash position for `item` using the double-hashing technique:
 *   h_i(item) = (h1 + i * h2) mod m
 *
 * Where h1 and h2 are two independent MurmurHash3 evaluations with different seeds.
 * This avoids computing k independent hashes while maintaining good distribution.
 */
function hashPosition(item: string, i: number, size: number): number {
  const h1 = murmurHash3_32(item, 0);
  const h2 = murmurHash3_32(item, 1580931817); // arbitrary second seed
  return ((h1 + Math.imul(i, h2)) >>> 0) % size;
}

function setBit(bits: Uint8Array, pos: number): void {
  const byteIdx = pos >>> 3;
  const bitIdx = pos & 7;
  bits[byteIdx] = (bits[byteIdx]! | (1 << bitIdx)) & 0xff;
}

function testBit(bits: Uint8Array, pos: number): boolean {
  const byteIdx = pos >>> 3;
  const bitIdx = pos & 7;
  return (bits[byteIdx]! & (1 << bitIdx)) !== 0;
}

// ---------------------------------------------------------------------------
// Add
// ---------------------------------------------------------------------------

/**
 * Add an item to the Bloom filter. Mutates state in-place.
 *
 * Note: There is no way to determine if the item was already present
 * (due to the probabilistic nature of Bloom filters). The count is
 * always incremented as an approximation of items added.
 */
export function bloomAdd(state: BloomFilterState, item: string): void {
  for (let i = 0; i < state.numHashes; i++) {
    const pos = hashPosition(item, i, state.size);
    setBit(state.bits, pos);
  }
  // Mutate count (readonly in type but we own the object)
  (state as { count: number }).count++;
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

/**
 * Test whether an item might be in the set.
 *
 * - Returns `true` if the item is *probably* in the set (may be a false positive).
 * - Returns `false` if the item is *definitely not* in the set.
 */
export function bloomTest(state: BloomFilterState, item: string): boolean {
  for (let i = 0; i < state.numHashes; i++) {
    const pos = hashPosition(item, i, state.size);
    if (!testBit(state.bits, pos)) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// False positive rate estimation
// ---------------------------------------------------------------------------

/**
 * Estimate the current false positive rate based on the number of items
 * inserted and the filter parameters.
 *
 * Formula: (1 - e^(-k * n / m))^k
 *
 * where k = numHashes, n = count (items inserted), m = size (bits).
 */
export function bloomFalsePositiveRate(state: BloomFilterState): number {
  const k = state.numHashes;
  const n = state.count;
  const m = state.size;
  if (m === 0) return 1;
  return Math.pow(1 - Math.exp((-k * n) / m), k);
}

// ---------------------------------------------------------------------------
// Merge
// ---------------------------------------------------------------------------

/**
 * Merge two Bloom filters with the same configuration (same size and numHashes)
 * by OR-ing their bit arrays. The resulting filter contains elements from both.
 *
 * @throws If the filters have different sizes or hash counts.
 */
export function bloomMerge(a: BloomFilterState, b: BloomFilterState): BloomFilterState {
  if (a.size !== b.size || a.numHashes !== b.numHashes) {
    throw new Error('Cannot merge Bloom filters with different configurations');
  }

  const merged = new Uint8Array(a.bits.length);
  for (let i = 0; i < merged.length; i++) {
    merged[i] = (a.bits[i]! | b.bits[i]!) & 0xff;
  }

  return {
    bits: merged,
    numHashes: a.numHashes,
    size: a.size,
    count: a.count + b.count,
  };
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

/** Clear all bits and reset the count to zero. */
export function bloomReset(state: BloomFilterState): void {
  state.bits.fill(0);
  (state as { count: number }).count = 0;
}
