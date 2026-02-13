// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-4: Spatial Layout Understanding
// Barrel re-exports for the spatial layout module.
// ---------------------------------------------------------------------------

export { buildLayoutGraph, layoutItemsToFeatures } from './layout-gnn.js';
export { layoutQualityGNN, globalMeanPool, globalSumPool } from './layout-quality.js';
export { graphMatchingScore } from './layout-gmn.js';
export { sceneGraphForward } from './scene-graph.js';
