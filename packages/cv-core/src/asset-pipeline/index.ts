// ---------------------------------------------------------------------------
// CV-12: Asset Pipeline — barrel export
// ---------------------------------------------------------------------------

export {
  validateTriangleCount,
  validateUVBounds,
  validateScale,
  validateManifold,
} from './validation.js';

export {
  createAssetMetadata,
  computePhysicalDimensions,
  computeMeshComplexity,
} from './metadata.js';

export {
  createQualityGate,
  scoreMeshQuality,
  checkAllGates,
} from './quality-gates.js';

export {
  planLODChain,
  estimateLODSizes,
  selectLODForBandwidth,
} from './progressive.js';
