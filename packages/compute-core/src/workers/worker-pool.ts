// ---------------------------------------------------------------------------
// HPC-3: Workers — Worker Pool Scheduling Logic
// ---------------------------------------------------------------------------
// Pure scheduling and dispatch logic for a Web Worker pool.
// No actual Worker API dependency — just state management and heuristics
// for acquiring/releasing workers, distributing work, and batching decisions.
// ---------------------------------------------------------------------------

/** State of the worker pool. */
export type PoolState = {
  workers: Array<{ id: number; busy: boolean }>;
  idleCount: number;
};

/**
 * Create a pool state with `size` workers, all initially idle.
 */
export function createPoolState(size: number): PoolState {
  if (size < 1) {
    throw new RangeError('Pool size must be at least 1');
  }
  const workers: Array<{ id: number; busy: boolean }> = [];
  for (let i = 0; i < size; i++) {
    workers.push({ id: i, busy: false });
  }
  return { workers, idleCount: size };
}

/**
 * Acquire an idle worker from the pool.
 * Returns the worker index, or null if all workers are busy.
 */
export function acquireWorker(pool: PoolState): number | null {
  for (let i = 0; i < pool.workers.length; i++) {
    const worker = pool.workers[i]!;
    if (!worker.busy) {
      worker.busy = true;
      pool.idleCount--;
      return worker.id;
    }
  }
  return null;
}

/**
 * Release a worker back to the idle pool.
 */
export function releaseWorker(pool: PoolState, workerId: number): void {
  const worker = pool.workers[workerId];
  if (worker === undefined) {
    throw new RangeError(`Worker ${workerId} does not exist in pool`);
  }
  if (!worker.busy) {
    return; // already idle — idempotent
  }
  worker.busy = false;
  pool.idleCount++;
}

/**
 * Compute the optimal pool size based on hardware concurrency.
 * Reserves one thread for the main thread: `max(1, hardwareConcurrency - 1)`.
 */
export function computeOptimalPoolSize(hardwareConcurrency: number): number {
  return Math.max(1, hardwareConcurrency - 1);
}

/**
 * Distribute a contiguous data range across `numWorkers` as evenly as possible.
 * Returns an array of `{ offset, length }` chunks. The last chunk absorbs
 * any remainder from integer division.
 */
export function distributeChunks(
  dataLength: number,
  numWorkers: number,
): Array<{ offset: number; length: number }> {
  if (numWorkers < 1) {
    throw new RangeError('numWorkers must be at least 1');
  }
  if (dataLength <= 0) {
    return [];
  }

  const chunks: Array<{ offset: number; length: number }> = [];
  const baseSize = Math.floor(dataLength / numWorkers);
  const remainder = dataLength % numWorkers;
  let currentOffset = 0;

  for (let i = 0; i < numWorkers; i++) {
    // Distribute remainder one extra element to the first `remainder` workers
    const chunkLength = baseSize + (i < remainder ? 1 : 0);
    if (chunkLength === 0) break; // no more data to distribute
    chunks.push({ offset: currentOffset, length: chunkLength });
    currentOffset += chunkLength;
  }

  return chunks;
}

/**
 * Estimate the time to move `byteSize` bytes to/from a worker.
 *
 * - `transfer` mode (Transferable): near-zero-copy, fixed ~0.1ms overhead.
 * - `clone` mode (structured clone): proportional to data size.
 *   Approximation: ~1ms per 5 MB (5_242_880 bytes).
 */
export function estimateTransferTimeMs(
  byteSize: number,
  mode: 'transfer' | 'clone',
): number {
  if (mode === 'transfer') {
    return 0.1;
  }
  // clone: linear cost ~1ms per 5MB
  return byteSize / 5_242_880;
}

/**
 * Decide whether small tasks should be batched before posting to a worker.
 * Returns true when the postMessage overhead exceeds 10% of the task duration,
 * meaning the overhead is significant relative to the actual work.
 */
export function shouldBatchTasks(
  taskDurationMs: number,
  postMessageOverheadMs: number,
): boolean {
  if (taskDurationMs <= 0) return true;
  return postMessageOverheadMs > taskDurationMs * 0.1;
}
