// ---------------------------------------------------------------------------
// HPC-3: Workers — Barrel Export
// ---------------------------------------------------------------------------

export {
  createMutexState,
  tryLock,
  unlock,
  isLocked,
  spinLock,
} from './atomics-mutex.js';

export type { RingBufState } from './ringbuf.js';
export {
  createRingBuffer,
  ringBufferPush,
  ringBufferPop,
  ringBufferPeek,
  ringBufferSize,
  ringBufferIsFull,
  ringBufferIsEmpty,
  ringBufferClear,
  ringBufferToArray,
} from './ringbuf.js';

export type { PoolState } from './worker-pool.js';
export {
  createPoolState,
  acquireWorker,
  releaseWorker,
  computeOptimalPoolSize,
  distributeChunks,
  estimateTransferTimeMs,
  shouldBatchTasks,
} from './worker-pool.js';

export {
  generateTemperatureLadder,
  shouldSwapReplicas,
  computeSwapAcceptanceRate,
  optimizeTemperatures,
  assignReplicasToWorkers,
} from './parallel-tempering.js';
