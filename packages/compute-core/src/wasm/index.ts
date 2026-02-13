// ---------------------------------------------------------------------------
// @omni-twin/compute-core — HPC-2: WASM Pipeline (barrel)
// ---------------------------------------------------------------------------

export {
  detectWASMCapabilities,
  estimateModuleSize,
  buildFlags,
  optimalMemoryPages,
  wasmPageSize,
} from './config.js';

export {
  computeAlignment,
  bytesPerElement,
  regionOverlaps,
  allocateRegion,
  validateRegion,
  splitRegion,
} from './zero-copy.js';

export {
  estimateLoadTimeMs,
  prioritizeModules,
  computeCompressionRatio,
  shouldPreload,
} from './module-loader.js';
