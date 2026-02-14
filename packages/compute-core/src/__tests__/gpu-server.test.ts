import { describe, it, expect } from 'vitest';

import {
  createServerTask,
  estimateGPUMemoryMB,
  isTaskTooLargeForBrowser,
  taskPriorityScore,
  estimateServerCost,
  createResult,
  isSuccessful,
  totalLatencyMs,
  computeThroughput,
  summarizeResults,
  selectProvider,
  estimateQueueWaitMs,
  shouldRetry,
  computeBackoffMs,
  batchTasks,
} from '../gpu-server/index.js';

import type {
  GPUCloudConfig,
  ServerComputeTask,
  ServerComputeResult,
} from '../types.js';

// ---------------------------------------------------------------------------
// Task creation
// ---------------------------------------------------------------------------

describe('createServerTask', () => {
  it('creates a task with the correct type, dataSize, and estimatedTimeMs', () => {
    const task = createServerTask('sinkhorn', 1024, 3000);
    expect(task.type).toBe('sinkhorn');
    expect(task.dataSize).toBe(1024);
    expect(task.estimatedTimeMs).toBe(3000);
  });

  it('defaults priority to medium', () => {
    const task = createServerTask('monte_carlo', 500, 1000);
    expect(task.priority).toBe('medium');
  });

  it('assigns unique IDs to successive tasks', () => {
    const t1 = createServerTask('diffusion', 100, 500);
    const t2 = createServerTask('diffusion', 100, 500);
    expect(t1.id).not.toBe(t2.id);
  });
});

// ---------------------------------------------------------------------------
// GPU memory estimation
// ---------------------------------------------------------------------------

describe('estimateGPUMemoryMB', () => {
  it('includes a base overhead of 64 MB', () => {
    // layout_optimization: dataSize * 16 bytes
    const task = createServerTask('layout_optimization', 0, 100);
    const mem = estimateGPUMemoryMB(task);
    expect(mem).toBe(64);
  });

  it('computes sinkhorn memory as n^2 * 8 bytes + 64 MB overhead', () => {
    const task = createServerTask('sinkhorn', 1024, 100);
    const expected = 64 + (1024 * 1024 * 8) / 1_048_576;
    expect(estimateGPUMemoryMB(task)).toBe(expected);
  });

  it('computes diffusion memory as n * 32 bytes + 64 MB overhead', () => {
    const task = createServerTask('diffusion', 1_048_576, 100);
    const expected = 64 + (1_048_576 * 32) / 1_048_576;
    expect(estimateGPUMemoryMB(task)).toBe(expected);
  });
});

// ---------------------------------------------------------------------------
// Browser size limit
// ---------------------------------------------------------------------------

describe('isTaskTooLargeForBrowser', () => {
  it('returns true when GPU memory exceeds 500 MB', () => {
    // sinkhorn with n=8192 => 8192^2 * 8 = 512 MB data + 64 overhead
    const task = createServerTask('sinkhorn', 8192, 100);
    expect(isTaskTooLargeForBrowser(task)).toBe(true);
  });

  it('returns true when estimatedTimeMs exceeds 5000', () => {
    const task = createServerTask('monte_carlo', 1, 6000);
    expect(isTaskTooLargeForBrowser(task)).toBe(true);
  });

  it('returns false for a small, fast task', () => {
    const task = createServerTask('monte_carlo', 100, 100);
    expect(isTaskTooLargeForBrowser(task)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Priority scoring
// ---------------------------------------------------------------------------

describe('taskPriorityScore', () => {
  it('returns 50 for medium priority (default)', () => {
    const task = createServerTask('monte_carlo', 100, 100);
    expect(taskPriorityScore(task)).toBe(50);
  });

  it('returns 100 for critical priority', () => {
    const task: ServerComputeTask = {
      id: 'test-critical',
      type: 'sinkhorn',
      dataSize: 10,
      estimatedTimeMs: 10,
      priority: 'critical',
    };
    expect(taskPriorityScore(task)).toBe(100);
  });

  it('returns 75 for high and 25 for low', () => {
    const high: ServerComputeTask = {
      id: 'h',
      type: 'diffusion',
      dataSize: 1,
      estimatedTimeMs: 1,
      priority: 'high',
    };
    const low: ServerComputeTask = {
      id: 'l',
      type: 'diffusion',
      dataSize: 1,
      estimatedTimeMs: 1,
      priority: 'low',
    };
    expect(taskPriorityScore(high)).toBe(75);
    expect(taskPriorityScore(low)).toBe(25);
  });
});

// ---------------------------------------------------------------------------
// Cost estimation
// ---------------------------------------------------------------------------

describe('estimateServerCost', () => {
  it('returns a positive cost for a non-zero task', () => {
    const task = createServerTask('monte_carlo', 100, 3600_000);
    const config: GPUCloudConfig = {
      provider: 'modal',
      gpuType: 'a100',
      costPerHour: 2.5,
      maxConcurrency: 4,
    };
    const cost = estimateServerCost(task, config);
    expect(cost).toBeGreaterThan(0);
  });

  it('computes cost proportional to time and costPerHour', () => {
    const task = createServerTask('diffusion', 100, 1_800_000); // 0.5 hours
    const config: GPUCloudConfig = {
      provider: 'runpod',
      gpuType: 'h100',
      costPerHour: 4.0,
      maxConcurrency: 2,
    };
    expect(estimateServerCost(task, config)).toBeCloseTo(2.0, 5);
  });
});

// ---------------------------------------------------------------------------
// Result creation and inspection
// ---------------------------------------------------------------------------

describe('createResult', () => {
  it('creates a result with status completed', () => {
    const data = new Float64Array([1, 2, 3]);
    const result = createResult('task-0', data, 120, 30);
    expect(result.taskId).toBe('task-0');
    expect(result.status).toBe('completed');
    expect(result.data).toBe(data);
    expect(result.gpuTimeMs).toBe(120);
    expect(result.transferTimeMs).toBe(30);
  });
});

describe('isSuccessful', () => {
  it('returns true for completed status', () => {
    const result = createResult('t', new Float64Array(1), 10, 5);
    expect(isSuccessful(result)).toBe(true);
  });

  it('returns false for failed status', () => {
    const failed: ServerComputeResult = {
      taskId: 't',
      status: 'failed',
      data: new Float64Array(0),
      gpuTimeMs: 0,
      transferTimeMs: 0,
    };
    expect(isSuccessful(failed)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Latency and throughput
// ---------------------------------------------------------------------------

describe('totalLatencyMs', () => {
  it('sums gpuTimeMs and transferTimeMs', () => {
    const result = createResult('t', new Float64Array(1), 200, 50);
    expect(totalLatencyMs(result)).toBe(250);
  });
});

describe('computeThroughput', () => {
  it('returns positive throughput for a valid result', () => {
    const result = createResult('t', new Float64Array(1000), 500, 500);
    // 1000 elements / 1000 ms * 1000 = 1000 elements/sec
    expect(computeThroughput(result)).toBe(1000);
  });

  it('returns 0 when total latency is zero', () => {
    const result = createResult('t', new Float64Array(10), 0, 0);
    expect(computeThroughput(result)).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Summarize results
// ---------------------------------------------------------------------------

describe('summarizeResults', () => {
  it('returns zeros for an empty array', () => {
    const summary = summarizeResults([]);
    expect(summary.successRate).toBe(0);
    expect(summary.avgLatencyMs).toBe(0);
    expect(summary.totalElements).toBe(0);
  });

  it('computes correct aggregates for multiple results', () => {
    const r1 = createResult('a', new Float64Array(100), 100, 50);
    const r2 = createResult('b', new Float64Array(200), 200, 100);
    const failed: ServerComputeResult = {
      taskId: 'c',
      status: 'failed',
      data: new Float64Array(50),
      gpuTimeMs: 10,
      transferTimeMs: 5,
    };
    const summary = summarizeResults([r1, r2, failed]);
    expect(summary.successRate).toBeCloseTo(2 / 3, 5);
    expect(summary.totalElements).toBe(350);
    // Avg latency: (150 + 300 + 15) / 3 = 155
    expect(summary.avgLatencyMs).toBeCloseTo(155, 5);
  });
});

// ---------------------------------------------------------------------------
// Provider selection
// ---------------------------------------------------------------------------

describe('selectProvider', () => {
  const configs: GPUCloudConfig[] = [
    { provider: 'runpod', gpuType: 't4', costPerHour: 0.5, maxConcurrency: 4 },
    { provider: 'modal', gpuType: 'a100', costPerHour: 3.0, maxConcurrency: 2 },
    { provider: 'lambda', gpuType: 'a10g', costPerHour: 1.2, maxConcurrency: 8 },
  ];

  it('selects the cheapest qualifying provider', () => {
    // Small task: all providers qualify, t4 at $0.50 is cheapest
    const task = createServerTask('monte_carlo', 10, 100);
    const selected = selectProvider(configs, task);
    expect(selected.gpuType).toBe('t4');
  });

  it('throws when no configs are provided', () => {
    const task = createServerTask('sinkhorn', 10, 100);
    expect(() => selectProvider([], task)).toThrow('No GPU cloud configs provided');
  });

  it('falls back to first config when no provider qualifies', () => {
    // sinkhorn with huge n: requires enormous memory
    const task = createServerTask('sinkhorn', 1_000_000, 100);
    const selected = selectProvider(configs, task);
    expect(selected).toBe(configs[0]!);
  });
});

// ---------------------------------------------------------------------------
// Queue estimation
// ---------------------------------------------------------------------------

describe('estimateQueueWaitMs', () => {
  it('returns 0 for an empty queue', () => {
    expect(estimateQueueWaitMs(0, 1000, 4)).toBe(0);
  });

  it('returns Infinity for zero concurrency', () => {
    expect(estimateQueueWaitMs(5, 1000, 0)).toBe(Infinity);
  });

  it('computes (depth / concurrency) * avgTaskMs', () => {
    expect(estimateQueueWaitMs(10, 500, 2)).toBe(2500);
  });
});

// ---------------------------------------------------------------------------
// Retry logic
// ---------------------------------------------------------------------------

describe('shouldRetry', () => {
  it('returns false for a completed result', () => {
    const result = createResult('t', new Float64Array(1), 10, 5);
    expect(shouldRetry(result, 0, 3)).toBe(false);
  });

  it('returns true for a failed result when attempts remain', () => {
    const failed: ServerComputeResult = {
      taskId: 't',
      status: 'failed',
      data: new Float64Array(0),
      gpuTimeMs: 0,
      transferTimeMs: 0,
    };
    expect(shouldRetry(failed, 1, 3)).toBe(true);
  });

  it('returns false when attempt equals maxRetries', () => {
    const timeout: ServerComputeResult = {
      taskId: 't',
      status: 'timeout',
      data: new Float64Array(0),
      gpuTimeMs: 0,
      transferTimeMs: 0,
    };
    expect(shouldRetry(timeout, 3, 3)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Exponential backoff
// ---------------------------------------------------------------------------

describe('computeBackoffMs', () => {
  it('increases with higher attempt numbers', () => {
    const b0 = computeBackoffMs(0, 1000, 60000);
    const b1 = computeBackoffMs(1, 1000, 60000);
    const b2 = computeBackoffMs(2, 1000, 60000);
    expect(b1).toBeGreaterThan(b0);
    expect(b2).toBeGreaterThan(b1);
  });

  it('is capped by maxMs', () => {
    const b10 = computeBackoffMs(10, 1000, 5000);
    // Base * 2^10 = 1024000, capped at 5000, then jitter applied (max factor 1.0)
    expect(b10).toBeLessThanOrEqual(5000);
  });

  it('is deterministic for the same inputs', () => {
    const a = computeBackoffMs(3, 500, 10000);
    const b = computeBackoffMs(3, 500, 10000);
    expect(a).toBe(b);
  });
});

// ---------------------------------------------------------------------------
// Task batching
// ---------------------------------------------------------------------------

describe('batchTasks', () => {
  it('groups tasks into batches of maxBatchSize', () => {
    const tasks = Array.from({ length: 7 }, (_, i) =>
      createServerTask('monte_carlo', i, 100),
    );
    const batches = batchTasks(tasks, 3);
    expect(batches.length).toBe(3);
    expect(batches[0]!.length).toBe(3);
    expect(batches[1]!.length).toBe(3);
    expect(batches[2]!.length).toBe(1);
  });

  it('returns an empty array for empty input', () => {
    expect(batchTasks([], 5)).toEqual([]);
  });

  it('throws RangeError when maxBatchSize < 1', () => {
    expect(() => batchTasks([], 0)).toThrow(RangeError);
  });

  it('preserves original task order', () => {
    const tasks = Array.from({ length: 4 }, (_, i) =>
      createServerTask('diffusion', i * 10, 100),
    );
    const batches = batchTasks(tasks, 2);
    expect(batches[0]![0]!.dataSize).toBe(0);
    expect(batches[0]![1]!.dataSize).toBe(10);
    expect(batches[1]![0]!.dataSize).toBe(20);
    expect(batches[1]![1]!.dataSize).toBe(30);
  });
});
