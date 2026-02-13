// ---------------------------------------------------------------------------
// @omni-twin/compute-core — HPC-1: WebGPU Compute Infrastructure (barrel)
// ---------------------------------------------------------------------------

export {
  REDUCE_WGSL,
  MATMUL_WGSL,
  PREFIX_SUM_WGSL,
  SINKHORN_ROW_WGSL,
  ISING_METROPOLIS_WGSL,
} from './shaders.js';

export {
  defaultCapabilities,
  estimateWorkgroups,
  computeOptimalWorkgroupSize,
  validateDispatch,
} from './device.js';

export {
  createPipelineRegistry,
  registerPipeline,
  getPipelineDescriptor,
  listPipelines,
  estimateCompilationTimeMs,
} from './pipeline-cache.js';
export type { PipelineRegistry } from './pipeline-cache.js';

export {
  planDispatches,
  estimateComputeTimeMs,
  aggregateTimestamps,
  computeGPUUtilization,
} from './compute-manager.js';
