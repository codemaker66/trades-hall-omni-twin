// ---------------------------------------------------------------------------
// CV-7: Real-time Rendering — barrel export
// ---------------------------------------------------------------------------

export { computeScreenSpaceError, selectLOD, buildLODChain } from './lod.js';
export {
  buildInstanceBuffer,
  frustumCull,
  extractFrustumPlanes,
} from './instancing.js';
export {
  computeCascadeSplits,
  computeShadowMatrix,
  contactShadowRaymarch,
} from './shadow.js';
export {
  createPerfBudget,
  estimateVRAM,
  checkBudget,
  adaptiveQuality,
} from './perf-budget.js';
