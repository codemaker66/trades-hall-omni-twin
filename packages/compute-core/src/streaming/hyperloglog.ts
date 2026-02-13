// ---------------------------------------------------------------------------
// HPC-6: Streaming Algorithms — HyperLogLog
// ---------------------------------------------------------------------------
// Probabilistic cardinality (distinct count) estimator. Uses ~1.6KB of memory
// at default precision (p=14) for ~0.8% standard error.
// Pure TypeScript, zero external dependencies.
// ---------------------------------------------------------------------------

import type { HyperLogLogState } from '../types.js';
import { murmurHash3_32 } from '../types.js';

// ---------------------------------------------------------------------------
// Create
// ---------------------------------------------------------------------------

/**
 * Create a HyperLogLog estimator.
 *
 * @param precision  Number of bits used for register addressing (default 14).
 *                   Valid range: 4-18. Higher precision = more registers =
 *                   lower error but more memory.
 *                   - p=14: 16384 registers, ~0.8% error, 16KB memory.
 *                   - p=10: 1024 registers, ~3.2% error, 1KB memory.
 *                   - p=4:  16 registers, ~26% error, 16 bytes.
 */
export function createHyperLogLog(precision?: number): HyperLogLogState {
  const p = Math.max(4, Math.min(18, precision ?? 14));
  const m = 1 << p;

  return {
    registers: new Uint8Array(m),
    precision: p,
    numRegisters: m,
  };
}

// ---------------------------------------------------------------------------
// Add
// ---------------------------------------------------------------------------

/**
 * Add an item to the HyperLogLog estimator. Mutates state in-place.
 *
 * Algorithm:
 * 1. Hash the item to a 32-bit value.
 * 2. Use the first `p` bits as the register index.
 * 3. Count leading zeros of the remaining (32-p) bits, plus 1 = rank.
 * 4. Update the register to the maximum of its current value and the rank.
 */
export function hllAdd(state: HyperLogLogState, item: string): void {
  const hash = murmurHash3_32(item, 0x5f61726d); // arbitrary seed

  const p = state.precision;
  // First p bits determine the register index
  const registerIdx = hash >>> (32 - p);
  // Remaining (32-p) bits — count leading zeros + 1
  const remaining = (hash << p) | 0;
  const rank = clz32(remaining, 32 - p) + 1;

  if (rank > state.registers[registerIdx]!) {
    state.registers[registerIdx] = rank;
  }
}

/**
 * Count leading zeros in the first `maxBits` bits of a 32-bit integer.
 */
function clz32(value: number, maxBits: number): number {
  if (maxBits <= 0) return 0;
  // Use Math.clz32 on the value, but only count up to maxBits
  if (value === 0) return maxBits;

  // Shift value so that the relevant bits are the top bits of a 32-bit int
  // value is already left-shifted by `p` bits, so the top (32-p) bits are what we want.
  // But since we pass maxBits = 32 - p, we need to count leading zeros in
  // the top `maxBits` bits.
  const shifted = value >>> 0; // ensure unsigned
  const totalClz = Math.clz32(shifted);
  return Math.min(totalClz, maxBits);
}

// ---------------------------------------------------------------------------
// Count (cardinality estimation)
// ---------------------------------------------------------------------------

/**
 * Estimate the number of distinct items added to the HyperLogLog.
 *
 * Uses the standard HLL algorithm with bias correction:
 * 1. Compute raw estimate = alpha_m * m^2 / sum(2^(-register[i]))
 * 2. Small range correction: if estimate <= 5/2 * m and there are empty
 *    registers, use linear counting.
 * 3. Large range correction: if estimate > 2^32 / 30, apply correction
 *    for hash collisions.
 */
export function hllCount(state: HyperLogLogState): number {
  const m = state.numRegisters;

  // Alpha correction constant
  const alpha = alphaM(m);

  // Harmonic mean: sum of 2^(-register[i])
  let harmonicSum = 0;
  let emptyRegisters = 0;

  for (let i = 0; i < m; i++) {
    const val = state.registers[i]!;
    harmonicSum += Math.pow(2, -val);
    if (val === 0) emptyRegisters++;
  }

  // Raw estimate
  let estimate = alpha * m * m / harmonicSum;

  // Small range correction (linear counting)
  if (estimate <= 2.5 * m && emptyRegisters > 0) {
    estimate = m * Math.log(m / emptyRegisters);
  }

  // Large range correction (hash collision compensation)
  const POW_2_32 = 4294967296; // 2^32
  if (estimate > POW_2_32 / 30) {
    estimate = -POW_2_32 * Math.log(1 - estimate / POW_2_32);
  }

  return Math.round(estimate);
}

/**
 * Compute the alpha_m bias correction constant.
 *
 * For m >= 128: alpha = 0.7213 / (1 + 1.079 / m)
 * For smaller m, use tabulated values.
 */
function alphaM(m: number): number {
  if (m >= 128) return 0.7213 / (1 + 1.079 / m);

  switch (m) {
    case 16:
      return 0.673;
    case 32:
      return 0.697;
    case 64:
      return 0.709;
    default:
      // Fallback for very small m (shouldn't happen with p >= 4)
      return 0.7213 / (1 + 1.079 / m);
  }
}

// ---------------------------------------------------------------------------
// Merge
// ---------------------------------------------------------------------------

/**
 * Merge two HyperLogLog estimators by taking the element-wise maximum
 * of their register arrays. Both must have the same precision.
 *
 * @throws If the estimators have different precisions.
 */
export function hllMerge(a: HyperLogLogState, b: HyperLogLogState): HyperLogLogState {
  if (a.precision !== b.precision) {
    throw new Error('Cannot merge HyperLogLog states with different precisions');
  }

  const merged = new Uint8Array(a.numRegisters);
  for (let i = 0; i < a.numRegisters; i++) {
    merged[i] = Math.max(a.registers[i]!, b.registers[i]!);
  }

  return {
    registers: merged,
    precision: a.precision,
    numRegisters: a.numRegisters,
  };
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

/** Clear all registers to zero. */
export function hllReset(state: HyperLogLogState): void {
  state.registers.fill(0);
}

// ---------------------------------------------------------------------------
// Error estimation
// ---------------------------------------------------------------------------

/**
 * Compute the standard error of the cardinality estimate.
 *
 * Standard error = 1.04 / sqrt(m)
 *
 * For p=14 (m=16384): ~0.81%
 * For p=10 (m=1024):  ~3.25%
 */
export function hllError(state: HyperLogLogState): number {
  return 1.04 / Math.sqrt(state.numRegisters);
}
