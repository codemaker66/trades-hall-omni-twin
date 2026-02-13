// ---------------------------------------------------------------------------
// HPC-4: GPU Server Compute — Client Logic
// ---------------------------------------------------------------------------
// Client-side logic for interacting with remote GPU compute servers:
// provider selection, queue estimation, retry/backoff, and task batching.
// ---------------------------------------------------------------------------

import type { GPUCloudConfig, ServerComputeTask, ServerComputeResult } from '../types.js';
import { estimateGPUMemoryMB } from './task-types.js';

/**
 * Select the cheapest GPU cloud provider that meets the task's requirements.
 *
 * A provider "meets requirements" if:
 * - Its estimated memory capacity (based on GPU type) can handle the task.
 *
 * GPU memory heuristics (total VRAM):
 * - `t4`:   16 GB
 * - `a10g`: 24 GB
 * - `a100`: 80 GB
 * - `h100`: 80 GB
 *
 * Among qualifying providers, the one with the lowest `costPerHour` wins.
 * If no provider qualifies, falls back to the first config.
 */
export function selectProvider(
  configs: GPUCloudConfig[],
  task: ServerComputeTask,
): GPUCloudConfig {
  if (configs.length === 0) {
    throw new Error('No GPU cloud configs provided');
  }

  const requiredMB = estimateGPUMemoryMB(task);

  const gpuMemoryMB: Record<GPUCloudConfig['gpuType'], number> = {
    t4: 16_384,
    a10g: 24_576,
    a100: 81_920,
    h100: 81_920,
  };

  // Filter to providers whose GPU can fit the task
  const qualifying = configs.filter((c) => {
    const availableMB = gpuMemoryMB[c.gpuType];
    return availableMB >= requiredMB;
  });

  if (qualifying.length === 0) {
    // Fallback to first config if nothing qualifies
    return configs[0]!;
  }

  // Pick cheapest
  let best = qualifying[0]!;
  for (let i = 1; i < qualifying.length; i++) {
    const c = qualifying[i]!;
    if (c.costPerHour < best.costPerHour) {
      best = c;
    }
  }

  return best;
}

/**
 * Estimate the queue wait time in milliseconds.
 *
 * Simple model: tasks ahead in queue / concurrency * average task duration.
 *
 *   waitMs = (queueDepth / concurrency) * avgTaskMs
 */
export function estimateQueueWaitMs(
  queueDepth: number,
  avgTaskMs: number,
  concurrency: number,
): number {
  if (concurrency <= 0) return Infinity;
  if (queueDepth <= 0) return 0;
  return (queueDepth / concurrency) * avgTaskMs;
}

/**
 * Determine whether a failed result should be retried.
 *
 * Retry when:
 * - The result status is 'failed' or 'timeout' (not 'completed')
 * - The current attempt number is less than `maxRetries`
 */
export function shouldRetry(
  result: ServerComputeResult,
  attempt: number,
  maxRetries: number,
): boolean {
  if (result.status === 'completed') return false;
  return attempt < maxRetries;
}

/**
 * Compute exponential backoff with jitter for retry delays.
 *
 *   delay = min(maxMs, baseMs * 2^attempt) * (0.5 + 0.5 * random)
 *
 * The jitter factor uses a deterministic formula based on the attempt
 * number to keep this function pure (no Math.random dependency).
 * Jitter range: [0.5, 1.0) of the exponential delay.
 */
export function computeBackoffMs(
  attempt: number,
  baseMs: number,
  maxMs: number,
): number {
  const exponential = Math.min(maxMs, baseMs * Math.pow(2, attempt));
  // Deterministic jitter based on attempt: golden-ratio-based hash
  const jitter = 0.5 + 0.5 * ((attempt * 0.6180339887) % 1);
  return exponential * jitter;
}

/**
 * Group tasks into batches of at most `maxBatchSize`.
 *
 * Returns an array of arrays, where each inner array has at most
 * `maxBatchSize` tasks. Tasks preserve their original order.
 */
export function batchTasks(
  tasks: ServerComputeTask[],
  maxBatchSize: number,
): ServerComputeTask[][] {
  if (maxBatchSize < 1) {
    throw new RangeError('maxBatchSize must be at least 1');
  }

  const batches: ServerComputeTask[][] = [];
  for (let i = 0; i < tasks.length; i += maxBatchSize) {
    batches.push(tasks.slice(i, i + maxBatchSize));
  }
  return batches;
}
