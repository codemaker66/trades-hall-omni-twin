// ---------------------------------------------------------------------------
// HPC-4: GPU Server Compute — Result Processing
// ---------------------------------------------------------------------------
// Functions for creating, inspecting, and summarizing GPU compute results.
// ---------------------------------------------------------------------------

import type { ServerComputeResult } from '../types.js';

/**
 * Create a successful result record.
 */
export function createResult(
  taskId: string,
  data: Float64Array,
  gpuTimeMs: number,
  transferTimeMs: number,
): ServerComputeResult {
  return {
    taskId,
    status: 'completed',
    data,
    gpuTimeMs,
    transferTimeMs,
  };
}

/**
 * Returns true if the result completed successfully (status === 'completed').
 */
export function isSuccessful(result: ServerComputeResult): boolean {
  return result.status === 'completed';
}

/**
 * Total end-to-end latency: GPU compute time + network transfer time.
 */
export function totalLatencyMs(result: ServerComputeResult): number {
  return result.gpuTimeMs + result.transferTimeMs;
}

/**
 * Compute throughput in elements per second.
 * Returns 0 if total latency is zero to avoid division by zero.
 */
export function computeThroughput(result: ServerComputeResult): number {
  const latency = totalLatencyMs(result);
  if (latency <= 0) return 0;
  return (result.data.length / latency) * 1000;
}

/**
 * Aggregate summary over multiple results.
 *
 * Returns:
 * - `successRate`:    fraction of results with status 'completed'
 * - `avgLatencyMs`:   mean total latency across all results
 * - `totalElements`:  total number of output elements across all results
 */
export function summarizeResults(
  results: ServerComputeResult[],
): { successRate: number; avgLatencyMs: number; totalElements: number } {
  if (results.length === 0) {
    return { successRate: 0, avgLatencyMs: 0, totalElements: 0 };
  }

  let successCount = 0;
  let totalLatency = 0;
  let totalElements = 0;

  for (let i = 0; i < results.length; i++) {
    const r = results[i]!;
    if (r.status === 'completed') {
      successCount++;
    }
    totalLatency += totalLatencyMs(r);
    totalElements += r.data.length;
  }

  return {
    successRate: successCount / results.length,
    avgLatencyMs: totalLatency / results.length,
    totalElements,
  };
}
