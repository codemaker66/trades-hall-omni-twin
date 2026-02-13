// ---------------------------------------------------------------------------
// @omni-twin/compute-core — HPC-9: Browser <-> Server Offload (barrel)
// ---------------------------------------------------------------------------

export {
  shouldOffload,
  classifyTask,
  recommendTier,
} from './decision.js';

export {
  estimateServerCostUSD,
  estimateBrowserCostMs,
  estimateNetworkLatencyMs,
  breakEvenDataSize,
  totalCost,
  optimalTarget,
} from './cost-model.js';

export {
  splitWorkload,
  parallelExecutionTimeMs,
  isWorthParallelizing,
  progressiveRefinement,
  fallbackChain,
} from './hybrid.js';
