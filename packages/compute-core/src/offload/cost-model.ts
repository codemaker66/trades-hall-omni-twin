// ---------------------------------------------------------------------------
// HPC-9: Offload — Compute cost estimation
// ---------------------------------------------------------------------------
// Estimates the monetary cost and latency of running a compute task on a
// given target (browser or server), factoring in network transfer, GPU time,
// and per-millisecond pricing models.
// ---------------------------------------------------------------------------

import type { ComputeTask, CostModel, OffloadDecision } from '../types.js';

// ---------------------------------------------------------------------------
// estimateServerCostUSD
// ---------------------------------------------------------------------------

/**
 * Estimate the dollar cost for executing a task on a remote GPU server.
 *
 * Cost = GPU time cost + network transfer cost (approximated from data size
 * divided by the model's transfer rate, multiplied by the edge cost rate).
 */
export function estimateServerCostUSD(
  task: ComputeTask,
  model: CostModel,
): number {
  const gpuCost = task.estimatedTimeMs * model.serverGPUCostPerMs;
  const transferTimeMs =
    model.transferBytesPerMs > 0 ? task.dataSize / model.transferBytesPerMs : 0;
  const transferCost = transferTimeMs * model.edgeCostPerMs;
  return gpuCost + transferCost;
}

// ---------------------------------------------------------------------------
// estimateBrowserCostMs
// ---------------------------------------------------------------------------

/**
 * Estimate the latency (in ms) for executing a task entirely in the browser.
 *
 * When a GPU is available, assume 5x speedup over CPU execution for compute-
 * heavy tasks. Without GPU, return the task's own estimate directly.
 */
export function estimateBrowserCostMs(
  task: ComputeTask,
  gpuAvailable: boolean,
): number {
  if (gpuAvailable && task.requiresGPU) {
    // GPU execution is roughly 5x faster for parallelisable work
    return task.estimatedTimeMs / 5;
  }
  return task.estimatedTimeMs;
}

// ---------------------------------------------------------------------------
// estimateNetworkLatencyMs
// ---------------------------------------------------------------------------

/**
 * Estimate the round-trip network latency including data transfer.
 *
 * Total = base RTT + upload + download (symmetric transfer assumed).
 *
 * @param dataSizeMB    - payload size in megabytes
 * @param bandwidthMbps - available bandwidth in megabits per second
 * @param baseLatencyMs - base round-trip time (ping) in milliseconds
 */
export function estimateNetworkLatencyMs(
  dataSizeMB: number,
  bandwidthMbps: number,
  baseLatencyMs: number,
): number {
  if (bandwidthMbps <= 0) return Infinity;
  // Convert MB to megabits: 1 MB = 8 Mb
  const transferTimeMsOneWay = (dataSizeMB * 8) / bandwidthMbps * 1_000;
  // Round trip: upload request + download result
  return baseLatencyMs + transferTimeMsOneWay * 2;
}

// ---------------------------------------------------------------------------
// breakEvenDataSize
// ---------------------------------------------------------------------------

/**
 * Compute the data size (in bytes) at which server execution becomes faster
 * than browser execution, accounting for network overhead.
 *
 * Below this threshold the browser is faster; above it the server wins.
 *
 * Derivation:
 *   browser_time(dataSize) ~ dataSize / localRate
 *   server_time(dataSize)  ~ networkLatency + dataSize / transferRate + gpuTime
 *
 * We simplify by modelling browser throughput as 1 byte / ms and solving
 *   dataSize = networkLatencyMs * transferBytesPerMs / (transferBytesPerMs - 1)
 * when transferBytesPerMs > 1.
 */
export function breakEvenDataSize(model: CostModel): number {
  // If transfer is slower than local processing, server is never faster
  if (model.transferBytesPerMs <= 1) return Infinity;

  return (
    (model.networkLatencyMs * model.transferBytesPerMs) /
    (model.transferBytesPerMs - 1)
  );
}

// ---------------------------------------------------------------------------
// totalCost
// ---------------------------------------------------------------------------

/**
 * Compute the combined latency and dollar cost for executing a task on the
 * specified target.
 */
export function totalCost(
  task: ComputeTask,
  target: OffloadDecision['target'],
  model: CostModel,
): { latencyMs: number; costUSD: number } {
  switch (target) {
    case 'server-gpu':
    case 'edge': {
      const transferMs =
        model.transferBytesPerMs > 0
          ? task.dataSize / model.transferBytesPerMs
          : 0;
      const latencyMs =
        model.networkLatencyMs + transferMs + task.estimatedTimeMs;
      const costUSD = estimateServerCostUSD(task, model);
      return { latencyMs, costUSD };
    }
    case 'browser-gpu': {
      const latencyMs = estimateBrowserCostMs(task, true);
      return { latencyMs, costUSD: 0 };
    }
    case 'browser-wasm':
    case 'browser-js': {
      const latencyMs = estimateBrowserCostMs(task, false);
      return { latencyMs, costUSD: 0 };
    }
  }
}

// ---------------------------------------------------------------------------
// optimalTarget
// ---------------------------------------------------------------------------

/** All candidate targets to evaluate. */
const ALL_TARGETS: readonly OffloadDecision['target'][] = [
  'browser-js',
  'browser-wasm',
  'browser-gpu',
  'server-gpu',
  'edge',
] as const;

/**
 * Find the target that minimises total cost (latency + dollar cost weighted).
 *
 * Dollar cost is converted to an equivalent latency penalty at a rate of
 * 1 USD = 10 000 ms penalty. Browser targets that require GPU but have none
 * available are filtered out.
 */
export function optimalTarget(
  task: ComputeTask,
  model: CostModel,
  gpuAvailable: boolean,
): OffloadDecision {
  const COST_TO_LATENCY_FACTOR = 10_000; // 1 USD ≈ 10 s penalty

  let bestTarget: OffloadDecision['target'] = 'browser-wasm';
  let bestScore = Infinity;
  let bestLatency = Infinity;
  let bestCost = 0;

  for (let i = 0; i < ALL_TARGETS.length; i++) {
    const t = ALL_TARGETS[i]!;

    // Skip browser-gpu if no GPU
    if (t === 'browser-gpu' && !gpuAvailable) continue;

    const { latencyMs, costUSD } = totalCost(task, t, model);
    const score = latencyMs + costUSD * COST_TO_LATENCY_FACTOR;

    if (score < bestScore) {
      bestScore = score;
      bestTarget = t;
      bestLatency = latencyMs;
      bestCost = costUSD;
    }
  }

  return {
    target: bestTarget,
    reason: `Optimal target by cost model (score=${bestScore.toFixed(2)})`,
    estimatedLatencyMs: bestLatency,
    estimatedCost: bestCost,
  };
}
