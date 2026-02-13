// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — Integration (GNN-11)
// Barrel re-exports for OT, TDA, sheaf, and demand-prediction integration.
// ---------------------------------------------------------------------------

// Optimal Transport + GNN
export { wassersteinReadout, fgwDistance } from './ot-gnn.js';

// Topological GNN Layer (TOGL)
export { computePersistenceDiagram, persistenceImage, toglLayer } from './togl.js';

// Sheaf Neural Networks
export { buildRestrictionMaps, sheafLaplacian, neuralSheafDiffusion } from './sheaf.js';

// Demand Prediction + Stochastic Pricing
export { demandPredictorGNN, stochasticPricingOptimizer } from './demand-gnn.js';
