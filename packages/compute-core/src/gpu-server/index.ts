// ---------------------------------------------------------------------------
// HPC-4: GPU Server Compute — Barrel Export
// ---------------------------------------------------------------------------

export {
  createServerTask,
  estimateGPUMemoryMB,
  isTaskTooLargeForBrowser,
  taskPriorityScore,
  estimateServerCost,
} from './task-types.js';

export {
  createResult,
  isSuccessful,
  totalLatencyMs,
  computeThroughput,
  summarizeResults,
} from './result-types.js';

export {
  selectProvider,
  estimateQueueWaitMs,
  shouldRetry,
  computeBackoffMs,
  batchTasks,
} from './client.js';
