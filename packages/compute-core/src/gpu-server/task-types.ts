// ---------------------------------------------------------------------------
// HPC-4: GPU Server Compute — Task Types
// ---------------------------------------------------------------------------
// Server task creation, sizing, and priority logic for offloading compute
// to remote GPU instances (Modal, RunPod, Lambda, etc.).
// ---------------------------------------------------------------------------

import type { ServerComputeTask, GPUCloudConfig } from '../types.js';

/** Auto-incrementing task ID counter. */
let nextTaskId = 0;

/**
 * Create a new server compute task with a unique ID.
 * Priority defaults to `'medium'`.
 */
export function createServerTask(
  type: ServerComputeTask['type'],
  dataSize: number,
  estimatedTimeMs: number,
): ServerComputeTask {
  return {
    id: `task-${nextTaskId++}`,
    type,
    dataSize,
    estimatedTimeMs,
    priority: 'medium',
  };
}

/**
 * Estimate the GPU memory (in MB) required for a given task.
 *
 * Heuristics per task type:
 * - `sinkhorn`:             ~8 bytes per element squared (distance matrix)
 * - `layout_optimization`:  ~16 bytes per element (position + gradient)
 * - `monte_carlo`:          ~8 bytes per sample
 * - `diffusion`:            ~32 bytes per element (U-Net activations)
 * - `faiss_search`:         ~4 bytes per vector dimension + index overhead
 *
 * All results include a 64 MB base overhead for the GPU runtime.
 */
export function estimateGPUMemoryMB(task: ServerComputeTask): number {
  const BASE_OVERHEAD_MB = 64;
  const bytesPerMB = 1_048_576;

  let dataBytes: number;
  switch (task.type) {
    case 'sinkhorn':
      // Distance matrix is n x n of float64
      dataBytes = task.dataSize * task.dataSize * 8;
      break;
    case 'layout_optimization':
      dataBytes = task.dataSize * 16;
      break;
    case 'monte_carlo':
      dataBytes = task.dataSize * 8;
      break;
    case 'diffusion':
      dataBytes = task.dataSize * 32;
      break;
    case 'faiss_search':
      dataBytes = task.dataSize * 4 + task.dataSize * 2; // vectors + index
      break;
  }

  return BASE_OVERHEAD_MB + dataBytes / bytesPerMB;
}

/**
 * Determine whether a task is too large to run in the browser.
 * Criteria: estimated GPU memory > 500 MB or estimated time > 5000 ms.
 */
export function isTaskTooLargeForBrowser(task: ServerComputeTask): boolean {
  return estimateGPUMemoryMB(task) > 500 || task.estimatedTimeMs > 5000;
}

/**
 * Compute a numeric priority score for queue ordering.
 *
 * - `critical`: 100
 * - `high`:     75
 * - `medium`:   50
 * - `low`:      25
 */
export function taskPriorityScore(task: ServerComputeTask): number {
  switch (task.priority) {
    case 'critical': return 100;
    case 'high':     return 75;
    case 'medium':   return 50;
    case 'low':      return 25;
  }
}

/**
 * Estimate the dollar cost of running a task on a given GPU cloud provider.
 *
 * Cost = (estimatedTimeMs / 3_600_000) * costPerHour
 *
 * This is a simple time-proportional model; real providers may have
 * minimum billing increments.
 */
export function estimateServerCost(
  task: ServerComputeTask,
  config: GPUCloudConfig,
): number {
  const hours = task.estimatedTimeMs / 3_600_000;
  return hours * config.costPerHour;
}
