// ---------------------------------------------------------------------------
// @omni-twin/compute-core — HPC-11: Deployment Architecture (barrel)
// ---------------------------------------------------------------------------

export {
  createEdgeConfig,
  validateEdgeDeployment,
  fitsEdge,
  selectEdgeProvider,
} from './edge-config.js';

export {
  createLoadingStrategy,
  buildLoadingPlan,
  estimateInitialLoadMs,
  estimateTotalLoadMs,
  shouldStreamCompile,
  compressionSavings,
} from './wasm-loader.js';

export {
  computeCacheKey,
  estimateCacheSize,
  shouldInvalidateCache,
  cachePriority,
  evictionCandidates,
  estimateCacheHitRate,
} from './cache-strategy.js';
