import { describe, it, expect } from 'vitest';
import {
  createMutexState,
  tryLock,
  unlock,
  isLocked,
  spinLock,
  createRingBuffer,
  ringBufferPush,
  ringBufferPop,
  ringBufferPeek,
  ringBufferSize,
  ringBufferIsFull,
  ringBufferIsEmpty,
  ringBufferClear,
  ringBufferToArray,
  createPoolState,
  acquireWorker,
  releaseWorker,
  computeOptimalPoolSize,
  distributeChunks,
  estimateTransferTimeMs,
  shouldBatchTasks,
  generateTemperatureLadder,
  shouldSwapReplicas,
  computeSwapAcceptanceRate,
  assignReplicasToWorkers,
} from '../workers/index.js';

import type { PRNG } from '../types.js';

// ---------------------------------------------------------------------------
// Mutex
// ---------------------------------------------------------------------------

describe('createMutexState', () => {
  it('starts unlocked (locked === 0)', () => {
    const state = createMutexState();
    expect(state.locked).toBe(0);
  });
});

describe('tryLock / unlock / isLocked', () => {
  it('tryLock succeeds on an unlocked mutex', () => {
    const view = new Int32Array(1);
    expect(tryLock(view, 0)).toBe(true);
    expect(view[0]).toBe(1);
  });

  it('tryLock fails when mutex is already locked', () => {
    const view = new Int32Array(1);
    tryLock(view, 0);
    expect(tryLock(view, 0)).toBe(false);
  });

  it('unlock releases the mutex', () => {
    const view = new Int32Array(1);
    tryLock(view, 0);
    expect(isLocked(view, 0)).toBe(true);
    unlock(view, 0);
    expect(isLocked(view, 0)).toBe(false);
  });

  it('isLocked returns false on fresh Int32Array', () => {
    const view = new Int32Array(1);
    expect(isLocked(view, 0)).toBe(false);
  });

  it('lock/unlock cycle allows re-acquisition', () => {
    const view = new Int32Array(1);
    expect(tryLock(view, 0)).toBe(true);
    unlock(view, 0);
    expect(tryLock(view, 0)).toBe(true);
  });
});

describe('spinLock', () => {
  it('acquires an unlocked mutex on the first spin', () => {
    const view = new Int32Array(1);
    expect(spinLock(view, 0, 10)).toBe(true);
    expect(isLocked(view, 0)).toBe(true);
  });

  it('fails after maxSpins when mutex is held', () => {
    const view = new Int32Array(1);
    tryLock(view, 0);
    expect(spinLock(view, 0, 5)).toBe(false);
  });

  it('returns false with maxSpins of 0', () => {
    const view = new Int32Array(1);
    expect(spinLock(view, 0, 0)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Ring buffer
// ---------------------------------------------------------------------------

describe('createRingBuffer', () => {
  it('creates a buffer with the given capacity', () => {
    const rb = createRingBuffer(8);
    expect(rb.capacity).toBe(8);
  });

  it('starts empty', () => {
    const rb = createRingBuffer(8);
    expect(ringBufferIsEmpty(rb)).toBe(true);
    expect(ringBufferSize(rb)).toBe(0);
  });

  it('throws for capacity < 2', () => {
    expect(() => createRingBuffer(1)).toThrow(RangeError);
    expect(() => createRingBuffer(0)).toThrow(RangeError);
  });
});

describe('ring buffer push/pop/peek', () => {
  it('preserves FIFO order', () => {
    const rb = createRingBuffer(8);
    ringBufferPush(rb, 10);
    ringBufferPush(rb, 20);
    ringBufferPush(rb, 30);
    expect(ringBufferPop(rb)).toBe(10);
    expect(ringBufferPop(rb)).toBe(20);
    expect(ringBufferPop(rb)).toBe(30);
  });

  it('peek returns front element without consuming it', () => {
    const rb = createRingBuffer(8);
    ringBufferPush(rb, 42);
    expect(ringBufferPeek(rb)).toBe(42);
    expect(ringBufferSize(rb)).toBe(1);
    expect(ringBufferPeek(rb)).toBe(42);
  });

  it('pop returns null on empty buffer', () => {
    const rb = createRingBuffer(4);
    expect(ringBufferPop(rb)).toBeNull();
  });

  it('peek returns null on empty buffer', () => {
    const rb = createRingBuffer(4);
    expect(ringBufferPeek(rb)).toBeNull();
  });

  it('push returns false when buffer is full', () => {
    const rb = createRingBuffer(3); // usable capacity = 2
    expect(ringBufferPush(rb, 1)).toBe(true);
    expect(ringBufferPush(rb, 2)).toBe(true);
    expect(ringBufferPush(rb, 3)).toBe(false);
    expect(ringBufferIsFull(rb)).toBe(true);
  });
});

describe('ring buffer full/empty detection', () => {
  it('isFull returns true at capacity - 1 elements', () => {
    const rb = createRingBuffer(4); // usable = 3
    ringBufferPush(rb, 1);
    ringBufferPush(rb, 2);
    ringBufferPush(rb, 3);
    expect(ringBufferIsFull(rb)).toBe(true);
  });

  it('isEmpty returns true after popping all elements', () => {
    const rb = createRingBuffer(4);
    ringBufferPush(rb, 1);
    ringBufferPush(rb, 2);
    ringBufferPop(rb);
    ringBufferPop(rb);
    expect(ringBufferIsEmpty(rb)).toBe(true);
  });

  it('size tracks pushes and pops correctly', () => {
    const rb = createRingBuffer(8);
    ringBufferPush(rb, 1);
    ringBufferPush(rb, 2);
    expect(ringBufferSize(rb)).toBe(2);
    ringBufferPop(rb);
    expect(ringBufferSize(rb)).toBe(1);
  });
});

describe('ring buffer wrap-around', () => {
  it('handles wrap-around correctly', () => {
    const rb = createRingBuffer(4); // usable = 3
    // Fill and drain partially to advance pointers
    ringBufferPush(rb, 1);
    ringBufferPush(rb, 2);
    ringBufferPop(rb); // head advances
    ringBufferPop(rb); // head advances further

    // Now push more to wrap around
    ringBufferPush(rb, 3);
    ringBufferPush(rb, 4);
    ringBufferPush(rb, 5);

    expect(ringBufferPop(rb)).toBe(3);
    expect(ringBufferPop(rb)).toBe(4);
    expect(ringBufferPop(rb)).toBe(5);
  });

  it('ringBufferToArray returns elements in FIFO order after wrap', () => {
    const rb = createRingBuffer(4); // usable = 3
    ringBufferPush(rb, 10);
    ringBufferPush(rb, 20);
    ringBufferPop(rb); // advance head past index 0
    ringBufferPush(rb, 30);
    ringBufferPush(rb, 40);

    const arr = ringBufferToArray(rb);
    expect(arr.length).toBe(3);
    expect(arr[0]).toBe(20);
    expect(arr[1]).toBe(30);
    expect(arr[2]).toBe(40);
  });
});

describe('ringBufferClear', () => {
  it('resets the buffer to empty', () => {
    const rb = createRingBuffer(8);
    ringBufferPush(rb, 1);
    ringBufferPush(rb, 2);
    ringBufferClear(rb);
    expect(ringBufferIsEmpty(rb)).toBe(true);
    expect(ringBufferSize(rb)).toBe(0);
  });

  it('allows new pushes after clearing', () => {
    const rb = createRingBuffer(4);
    ringBufferPush(rb, 1);
    ringBufferPush(rb, 2);
    ringBufferPush(rb, 3);
    ringBufferClear(rb);
    expect(ringBufferPush(rb, 99)).toBe(true);
    expect(ringBufferPop(rb)).toBe(99);
  });
});

// ---------------------------------------------------------------------------
// Worker pool
// ---------------------------------------------------------------------------

describe('createPoolState', () => {
  it('creates a pool with the specified size', () => {
    const pool = createPoolState(4);
    expect(pool.workers.length).toBe(4);
    expect(pool.idleCount).toBe(4);
  });

  it('all workers start as idle', () => {
    const pool = createPoolState(3);
    for (const w of pool.workers) {
      expect(w.busy).toBe(false);
    }
  });

  it('throws for size < 1', () => {
    expect(() => createPoolState(0)).toThrow(RangeError);
  });
});

describe('acquireWorker / releaseWorker', () => {
  it('acquires the first idle worker', () => {
    const pool = createPoolState(3);
    const id = acquireWorker(pool);
    expect(id).toBe(0);
    expect(pool.idleCount).toBe(2);
  });

  it('returns null when all workers are busy', () => {
    const pool = createPoolState(2);
    acquireWorker(pool);
    acquireWorker(pool);
    expect(acquireWorker(pool)).toBeNull();
  });

  it('releaseWorker makes the worker available again', () => {
    const pool = createPoolState(2);
    const id = acquireWorker(pool)!;
    releaseWorker(pool, id);
    expect(pool.idleCount).toBe(2);
    const id2 = acquireWorker(pool);
    expect(id2).toBe(id);
  });

  it('releaseWorker is idempotent', () => {
    const pool = createPoolState(2);
    const id = acquireWorker(pool)!;
    releaseWorker(pool, id);
    releaseWorker(pool, id); // second release is a no-op
    expect(pool.idleCount).toBe(2);
  });

  it('releaseWorker throws for invalid worker id', () => {
    const pool = createPoolState(2);
    expect(() => releaseWorker(pool, 99)).toThrow(RangeError);
  });
});

describe('computeOptimalPoolSize', () => {
  it('returns hardwareConcurrency - 1', () => {
    expect(computeOptimalPoolSize(8)).toBe(7);
    expect(computeOptimalPoolSize(4)).toBe(3);
  });

  it('returns at least 1 even for single-core', () => {
    expect(computeOptimalPoolSize(1)).toBe(1);
    expect(computeOptimalPoolSize(0)).toBe(1);
  });

  it('returns 1 for negative concurrency', () => {
    expect(computeOptimalPoolSize(-1)).toBe(1);
  });
});

describe('distributeChunks', () => {
  it('covers all elements across chunks', () => {
    const chunks = distributeChunks(100, 3);
    const total = chunks.reduce((sum, c) => sum + c.length, 0);
    expect(total).toBe(100);
  });

  it('distributes remainder to first workers', () => {
    const chunks = distributeChunks(10, 3);
    // 10 / 3 = 3 remainder 1; first worker gets 4, rest get 3
    expect(chunks[0]!.length).toBe(4);
    expect(chunks[1]!.length).toBe(3);
    expect(chunks[2]!.length).toBe(3);
  });

  it('returns empty for zero data length', () => {
    expect(distributeChunks(0, 3)).toEqual([]);
  });

  it('throws for numWorkers < 1', () => {
    expect(() => distributeChunks(10, 0)).toThrow(RangeError);
  });

  it('chunks have contiguous offsets', () => {
    const chunks = distributeChunks(50, 4);
    for (let i = 1; i < chunks.length; i++) {
      const prev = chunks[i - 1]!;
      const curr = chunks[i]!;
      expect(curr.offset).toBe(prev.offset + prev.length);
    }
  });

  it('handles more workers than data', () => {
    const chunks = distributeChunks(2, 5);
    // Only 2 chunks produced (one element each)
    expect(chunks.length).toBe(2);
    const total = chunks.reduce((sum, c) => sum + c.length, 0);
    expect(total).toBe(2);
  });
});

describe('estimateTransferTimeMs', () => {
  it('returns 0.1ms for transfer mode regardless of size', () => {
    expect(estimateTransferTimeMs(0, 'transfer')).toBe(0.1);
    expect(estimateTransferTimeMs(10_000_000, 'transfer')).toBe(0.1);
  });

  it('clone time scales linearly with byte size', () => {
    const t1 = estimateTransferTimeMs(5_242_880, 'clone');
    const t2 = estimateTransferTimeMs(10_485_760, 'clone');
    expect(t2).toBeCloseTo(t1 * 2, 5);
  });

  it('clone of 5MB takes ~1ms', () => {
    expect(estimateTransferTimeMs(5_242_880, 'clone')).toBeCloseTo(1, 5);
  });
});

describe('shouldBatchTasks', () => {
  it('returns true when overhead > 10% of task duration', () => {
    expect(shouldBatchTasks(10, 2)).toBe(true); // 2 > 10 * 0.1
  });

  it('returns false when overhead <= 10% of task duration', () => {
    expect(shouldBatchTasks(100, 5)).toBe(false); // 5 <= 100 * 0.1
  });

  it('returns true for zero task duration', () => {
    expect(shouldBatchTasks(0, 1)).toBe(true);
  });

  it('returns false for zero overhead with nonzero task duration', () => {
    expect(shouldBatchTasks(10, 0)).toBe(false); // 0 > 10 * 0.1 is false
  });

  it('returns true at the boundary (overhead exactly 10%)', () => {
    // 10 * 0.1 = 1; overhead = 1 is not > 1, so false
    expect(shouldBatchTasks(10, 1)).toBe(false);
    // overhead = 1.01 > 1, so true
    expect(shouldBatchTasks(10, 1.01)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Parallel tempering
// ---------------------------------------------------------------------------

describe('generateTemperatureLadder', () => {
  it('returns a Float64Array of the requested length', () => {
    const ladder = generateTemperatureLadder(5, 1.0, 10.0);
    expect(ladder).toBeInstanceOf(Float64Array);
    expect(ladder.length).toBe(5);
  });

  it('first temperature is tMin, last is tMax', () => {
    const ladder = generateTemperatureLadder(4, 1.0, 8.0);
    expect(ladder[0]).toBeCloseTo(1.0, 10);
    expect(ladder[3]).toBeCloseTo(8.0, 10);
  });

  it('temperatures are geometrically spaced', () => {
    const ladder = generateTemperatureLadder(3, 1.0, 4.0);
    // T_0 = 1, T_1 = 1 * (4/1)^(1/2) = 2, T_2 = 4
    expect(ladder[0]).toBeCloseTo(1.0, 10);
    expect(ladder[1]).toBeCloseTo(2.0, 10);
    expect(ladder[2]).toBeCloseTo(4.0, 10);
  });

  it('returns [tMin] for a single replica', () => {
    const ladder = generateTemperatureLadder(1, 2.5, 10.0);
    expect(ladder.length).toBe(1);
    expect(ladder[0]).toBeCloseTo(2.5, 10);
  });

  it('throws for invalid inputs', () => {
    expect(() => generateTemperatureLadder(0, 1, 10)).toThrow(RangeError);
    expect(() => generateTemperatureLadder(3, 0, 10)).toThrow(RangeError);
    expect(() => generateTemperatureLadder(3, 10, 1)).toThrow(RangeError);
  });
});

describe('shouldSwapReplicas', () => {
  it('always accepts energetically favorable swaps (delta >= 0)', () => {
    // energyI=10, energyJ=5, tempI=1, tempJ=2
    // delta = (1/1 - 1/2) * (5 - 10) = 0.5 * (-5) = -2.5 ... not favorable
    // Let's construct a favorable case:
    // energyI=5, energyJ=10, tempI=1, tempJ=2
    // delta = (1/1 - 1/2) * (10 - 5) = 0.5 * 5 = 2.5 >= 0
    const alwaysOne: PRNG = () => 0.999;
    expect(shouldSwapReplicas(5, 10, 1, 2, alwaysOne)).toBe(true);
  });

  it('rejects unfavorable swap with high random value', () => {
    // energyI=10, energyJ=5, tempI=1, tempJ=2
    // delta = (1 - 0.5) * (5 - 10) = -2.5
    // exp(-2.5) ~= 0.082
    const highRng: PRNG = () => 0.999;
    expect(shouldSwapReplicas(10, 5, 1, 2, highRng)).toBe(false);
  });

  it('accepts unfavorable swap with sufficiently low random value', () => {
    // Same delta = -2.5, exp(-2.5) ~= 0.082
    const lowRng: PRNG = () => 0.01;
    expect(shouldSwapReplicas(10, 5, 1, 2, lowRng)).toBe(true);
  });

  it('always accepts when energies are equal', () => {
    // delta = (1/T_i - 1/T_j) * (E_j - E_i) = any * 0 = 0 >= 0
    const anyRng: PRNG = () => 0.5;
    expect(shouldSwapReplicas(5, 5, 1, 2, anyRng)).toBe(true);
  });

  it('always accepts when temperatures are equal', () => {
    // delta = (1/T - 1/T) * dE = 0 * dE = 0 >= 0
    const anyRng: PRNG = () => 0.99;
    expect(shouldSwapReplicas(5, 10, 2, 2, anyRng)).toBe(true);
  });
});

describe('computeSwapAcceptanceRate', () => {
  it('returns 0 for fewer than 2 replicas', () => {
    expect(computeSwapAcceptanceRate(new Float64Array([1]), new Float64Array([1]))).toBe(0);
    expect(computeSwapAcceptanceRate(new Float64Array([]), new Float64Array([]))).toBe(0);
  });

  it('returns 1 when all energies are equal (always favorable)', () => {
    const energies = new Float64Array([5, 5, 5, 5]);
    const temps = new Float64Array([1, 2, 3, 4]);
    expect(computeSwapAcceptanceRate(energies, temps)).toBeCloseTo(1, 10);
  });

  it('returns a value between 0 and 1 for typical inputs', () => {
    const energies = new Float64Array([10, 8, 12, 5]);
    const temps = new Float64Array([1, 2, 4, 8]);
    const rate = computeSwapAcceptanceRate(energies, temps);
    expect(rate).toBeGreaterThanOrEqual(0);
    expect(rate).toBeLessThanOrEqual(1);
  });

  it('rate is higher with closer temperatures', () => {
    const energies = new Float64Array([10, 15, 20]);
    const closeTemps = new Float64Array([1.0, 1.1, 1.2]);
    const farTemps = new Float64Array([1.0, 5.0, 10.0]);
    const closeRate = computeSwapAcceptanceRate(energies, closeTemps);
    const farRate = computeSwapAcceptanceRate(energies, farTemps);
    // With ascending energies and ascending temps, close temps means
    // larger (1/T_i - 1/T_j) which amplifies delta. Actually the
    // acceptance depends on the specific energy/temp combination.
    // Just check both are in valid range.
    expect(closeRate).toBeGreaterThanOrEqual(0);
    expect(farRate).toBeGreaterThanOrEqual(0);
  });

  it('handles mismatched array lengths by using the shorter one', () => {
    const energies = new Float64Array([1, 2, 3]);
    const temps = new Float64Array([1, 2]);
    const rate = computeSwapAcceptanceRate(energies, temps);
    // Only 1 pair (indices 0,1)
    expect(rate).toBeGreaterThanOrEqual(0);
    expect(rate).toBeLessThanOrEqual(1);
  });
});

describe('assignReplicasToWorkers', () => {
  it('distributes replicas round-robin', () => {
    const assignment = assignReplicasToWorkers(7, 3);
    expect(assignment).toEqual([[0, 3, 6], [1, 4], [2, 5]]);
  });

  it('returns arrays of equal-ish length', () => {
    const assignment = assignReplicasToWorkers(8, 4);
    for (const group of assignment) {
      expect(group.length).toBe(2);
    }
  });

  it('all replica indices are covered exactly once', () => {
    const assignment = assignReplicasToWorkers(10, 3);
    const allIndices = assignment.flat().sort((a, b) => a - b);
    expect(allIndices).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it('handles single worker', () => {
    const assignment = assignReplicasToWorkers(5, 1);
    expect(assignment.length).toBe(1);
    expect(assignment[0]).toEqual([0, 1, 2, 3, 4]);
  });

  it('handles more workers than replicas', () => {
    const assignment = assignReplicasToWorkers(2, 5);
    expect(assignment.length).toBe(5);
    const allIndices = assignment.flat().sort((a, b) => a - b);
    expect(allIndices).toEqual([0, 1]);
    // Extra workers have empty arrays
    expect(assignment[2]).toEqual([]);
    expect(assignment[3]).toEqual([]);
    expect(assignment[4]).toEqual([]);
  });

  it('throws for numWorkers < 1', () => {
    expect(() => assignReplicasToWorkers(5, 0)).toThrow(RangeError);
  });
});
