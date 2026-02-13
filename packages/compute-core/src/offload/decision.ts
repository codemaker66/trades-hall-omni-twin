// ---------------------------------------------------------------------------
// HPC-9: Offload — Browser <-> Server offload decision logic
// ---------------------------------------------------------------------------
// Determines whether a compute task should run in-browser (JS / WASM / WebGPU)
// or be offloaded to a remote GPU server, based on data size, estimated time,
// GPU availability, memory requirements, and task type.
// ---------------------------------------------------------------------------

import type { ComputeTask, OffloadDecision } from '../types.js';

// ---------------------------------------------------------------------------
// Thresholds (constants)
// ---------------------------------------------------------------------------

const SERVER_DATA_SIZE_THRESHOLD = 500 * 1024 * 1024; // 500 MB
const SERVER_TIME_THRESHOLD = 5_000; // 5 seconds
const SERVER_MEMORY_THRESHOLD = 2_048; // 2 GB
const BROWSER_JS_DATA_SIZE_CEILING = 1 * 1024 * 1024; // 1 MB
const BROWSER_JS_TIME_CEILING = 16; // 1 frame at 60 fps
const BROWSER_GPU_DATA_SIZE_FLOOR = 1 * 1024 * 1024; // 1 MB

// ---------------------------------------------------------------------------
// shouldOffload — main decision function
// ---------------------------------------------------------------------------

/**
 * Determine where a compute task should execute.
 *
 * Decision waterfall:
 * 1. Large data (>500 MB)         -> server-gpu
 * 2. Long running (>5 s)          -> server-gpu
 * 3. Needs GPU but unavailable    -> server-gpu
 * 4. High memory (>2 GB)          -> server-gpu
 * 5. Diffusion layout tasks       -> server-gpu
 * 6. Tiny (<1 MB, <16 ms)         -> browser-js
 * 7. GPU available & >1 MB        -> browser-gpu
 * 8. Fallback                     -> browser-wasm
 */
export function shouldOffload(
  task: ComputeTask,
  gpuAvailable: boolean,
): OffloadDecision {
  // Rule 1: Large data
  if (task.dataSize > SERVER_DATA_SIZE_THRESHOLD) {
    return {
      target: 'server-gpu',
      reason: 'Data size exceeds 500 MB browser limit',
      estimatedLatencyMs: task.estimatedTimeMs,
      estimatedCost: 0,
    };
  }

  // Rule 2: Long running
  if (task.estimatedTimeMs > SERVER_TIME_THRESHOLD) {
    return {
      target: 'server-gpu',
      reason: 'Estimated time exceeds 5 s threshold',
      estimatedLatencyMs: task.estimatedTimeMs,
      estimatedCost: 0,
    };
  }

  // Rule 3: Requires GPU but none available
  if (task.requiresGPU && !gpuAvailable) {
    return {
      target: 'server-gpu',
      reason: 'Task requires GPU but none available in browser',
      estimatedLatencyMs: task.estimatedTimeMs,
      estimatedCost: 0,
    };
  }

  // Rule 4: High memory
  if (task.memoryMB > SERVER_MEMORY_THRESHOLD) {
    return {
      target: 'server-gpu',
      reason: 'Memory requirement exceeds 2 GB browser limit',
      estimatedLatencyMs: task.estimatedTimeMs,
      estimatedCost: 0,
    };
  }

  // Rule 5: Diffusion layout is always server-side
  if (task.type === 'diffusion_layout') {
    return {
      target: 'server-gpu',
      reason: 'Diffusion layout tasks always run on server GPU',
      estimatedLatencyMs: task.estimatedTimeMs,
      estimatedCost: 0,
    };
  }

  // Rule 6: Tiny tasks stay in the main thread
  if (
    task.dataSize < BROWSER_JS_DATA_SIZE_CEILING &&
    task.estimatedTimeMs < BROWSER_JS_TIME_CEILING
  ) {
    return {
      target: 'browser-js',
      reason: 'Task is small enough for main-thread JS execution',
      estimatedLatencyMs: task.estimatedTimeMs,
      estimatedCost: 0,
    };
  }

  // Rule 7: GPU available and non-trivial data
  if (gpuAvailable && task.dataSize > BROWSER_GPU_DATA_SIZE_FLOOR) {
    return {
      target: 'browser-gpu',
      reason: 'GPU available for medium-size compute task',
      estimatedLatencyMs: task.estimatedTimeMs,
      estimatedCost: 0,
    };
  }

  // Rule 8: Fallback to WASM
  return {
    target: 'browser-wasm',
    reason: 'Falling back to browser WASM execution',
    estimatedLatencyMs: task.estimatedTimeMs,
    estimatedCost: 0,
  };
}

// ---------------------------------------------------------------------------
// classifyTask — classify by type string
// ---------------------------------------------------------------------------

/** Latency-sensitive task types that must finish within a single frame. */
const LATENCY_SENSITIVE_TYPES = new Set([
  'ui_update',
  'cursor_move',
  'hover',
  'selection',
  'highlight',
  'drag',
  'input',
]);

/** Batch task types suited for background processing. */
const BATCH_TYPES = new Set([
  'export',
  'report',
  'precompute',
  'training',
  'diffusion_layout',
  'batch_transform',
  'full_reindex',
]);

/**
 * Classify a task type string into one of three scheduling categories.
 */
export function classifyTask(
  type: string,
): 'latency-sensitive' | 'throughput' | 'batch' {
  if (LATENCY_SENSITIVE_TYPES.has(type)) return 'latency-sensitive';
  if (BATCH_TYPES.has(type)) return 'batch';
  return 'throughput';
}

// ---------------------------------------------------------------------------
// recommendTier — recommend execution tier
// ---------------------------------------------------------------------------

/**
 * Recommend the optimal execution tier considering cross-origin isolation
 * (required for SharedArrayBuffer / WASM threads).
 *
 * Tier priority:
 *   server > webgpu > wasm-worker (requires crossOriginIsolated) > wasm-main > js-main
 */
export function recommendTier(
  task: ComputeTask,
  gpuAvailable: boolean,
  crossOriginIsolated: boolean,
): 'webgpu' | 'wasm-worker' | 'wasm-main' | 'js-main' | 'server' {
  const decision = shouldOffload(task, gpuAvailable);

  // Server offload takes priority
  if (decision.target === 'server-gpu') return 'server';

  // Browser GPU → WebGPU tier
  if (decision.target === 'browser-gpu') return 'webgpu';

  // Browser JS → JS main thread
  if (decision.target === 'browser-js') return 'js-main';

  // browser-wasm: prefer threaded workers when cross-origin-isolated
  if (crossOriginIsolated) return 'wasm-worker';

  return 'wasm-main';
}
