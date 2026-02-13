// ---------------------------------------------------------------------------
// @omni-twin/compute-core — HPC-1: Compute Dispatch Manager
// ---------------------------------------------------------------------------
// Dispatch orchestration, profiling aggregation, and utilization analysis
// for WebGPU compute pipelines. Operates on descriptors and timing data —
// does not call the WebGPU API directly.
// ---------------------------------------------------------------------------

import type {
  ComputeDispatch,
  GPUCapabilities,
  TimestampResult,
} from '../types.js';

// ---------------------------------------------------------------------------
// Dispatch planning
// ---------------------------------------------------------------------------

/**
 * Reorders a list of compute dispatches for optimal GPU execution.
 *
 * Strategy:
 * 1. Group dispatches by pipeline name (reduces pipeline state switches).
 * 2. Within each pipeline group, sort by total buffer size descending
 *    (launch memory-heavy dispatches first to overlap with smaller ones).
 * 3. Filters out dispatches that exceed device capabilities.
 *
 * The returned array is a new sorted copy; the input is not mutated.
 */
export function planDispatches(
  tasks: ComputeDispatch[],
  capabilities: GPUCapabilities,
): ComputeDispatch[] {
  // Filter to only dispatches within device limits
  const valid = tasks.filter((task) => {
    const [wx, wy, wz] = task.workgroups;
    const maxDim = capabilities.maxWorkgroupsPerDimension;

    if (
      (wx !== undefined && wx > maxDim) ||
      (wy !== undefined && wy > maxDim) ||
      (wz !== undefined && wz > maxDim)
    ) {
      return false;
    }

    for (const buf of task.buffers) {
      if (buf.size > capabilities.maxStorageBufferSize) {
        return false;
      }
    }

    return true;
  });

  // Sort: primary key = pipeline name (grouping), secondary = total buffer size desc
  const sorted = valid.slice().sort((a, b) => {
    const nameCmp = a.pipeline.localeCompare(b.pipeline);
    if (nameCmp !== 0) return nameCmp;

    const sizeA = totalBufferSize(a);
    const sizeB = totalBufferSize(b);
    return sizeB - sizeA; // descending
  });

  return sorted;
}

// ---------------------------------------------------------------------------
// Time estimation
// ---------------------------------------------------------------------------

/**
 * Estimates the GPU compute time in milliseconds for a single dispatch.
 *
 * The model is intentionally simple and conservative:
 * - Base overhead per dispatch: 0.01 ms (command buffer submission).
 * - Per-workgroup cost: scales with the total workgroup count.
 * - Memory cost: scales with total buffer bytes (cache/bandwidth limited).
 *
 * Actual performance varies wildly across GPUs; this provides a rough
 * order-of-magnitude estimate for scheduling decisions.
 */
export function estimateComputeTimeMs(
  dispatch: ComputeDispatch,
  capabilities: GPUCapabilities,
): number {
  const baseOverheadMs = 0.01;

  const [wx, wy, wz] = dispatch.workgroups;
  const totalWorkgroups = (wx ?? 1) * (wy ?? 1) * (wz ?? 1);

  // Assume ~100 GFLOPS effective throughput for a midrange GPU,
  // with each workgroup doing ~256 * 10 FLOPs = 2560 FLOPs.
  // That gives ~0.0000256 ms per workgroup as a rough baseline.
  const workgroupCostMs = totalWorkgroups * 0.0000256;

  // Memory bandwidth estimate: assume 200 GB/s effective bandwidth.
  // Convert bytes to GB, then to ms.
  const totalBytes = totalBufferSize(dispatch);
  const memCostMs = (totalBytes / (200 * 1024 * 1024 * 1024)) * 1000;

  // Scale down if capabilities indicate a smaller GPU (heuristic:
  // smaller max buffer size suggests a lower-end device).
  const deviceScale =
    capabilities.maxStorageBufferSize < 256 * 1024 * 1024 ? 2.0 : 1.0;

  return (baseOverheadMs + workgroupCostMs + memCostMs) * deviceScale;
}

// ---------------------------------------------------------------------------
// Timestamp aggregation
// ---------------------------------------------------------------------------

/**
 * Aggregates GPU timestamp query results into a total and per-pipeline
 * breakdown.
 *
 * @returns `totalNs` — sum of all durations, and `breakdown` — a map from
 * pipeline name to cumulative nanoseconds spent in that pipeline.
 */
export function aggregateTimestamps(
  results: TimestampResult[],
): { totalNs: number; breakdown: Map<string, number> } {
  let totalNs = 0;
  const breakdown = new Map<string, number>();

  for (const r of results) {
    totalNs += r.durationNs;

    const prev = breakdown.get(r.pipelineName) ?? 0;
    breakdown.set(r.pipelineName, prev + r.durationNs);
  }

  return { totalNs, breakdown };
}

// ---------------------------------------------------------------------------
// Utilization
// ---------------------------------------------------------------------------

/**
 * Computes the GPU utilization ratio: how much of the frame budget is
 * consumed by GPU compute work.
 *
 * @param computeTimeNs - Total GPU compute time in nanoseconds.
 * @param frameBudgetMs - Available frame budget in milliseconds
 *                        (e.g., 16.67 for 60 FPS, 8.33 for 120 FPS).
 * @returns A value in [0, Infinity). Values > 1.0 indicate the compute
 *          work exceeds the frame budget.
 */
export function computeGPUUtilization(
  computeTimeNs: number,
  frameBudgetMs: number,
): number {
  if (frameBudgetMs <= 0) {
    return 0;
  }
  const computeTimeMs = computeTimeNs / 1_000_000;
  return computeTimeMs / frameBudgetMs;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function totalBufferSize(dispatch: ComputeDispatch): number {
  let total = 0;
  for (const buf of dispatch.buffers) {
    total += buf.size;
  }
  return total;
}
