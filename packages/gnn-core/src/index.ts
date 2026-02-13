// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — Graph Neural Networks for Venue Intelligence
// ---------------------------------------------------------------------------
// Barrel re-export for all 12 GNN sub-domains (GNN-1 through GNN-12).
// ---------------------------------------------------------------------------

// Infrastructure: types, graph primitives, tensor ops
export * from './types.js';
export * from './graph.js';
export * from './tensor.js';

// GNN-1: MPNN Foundation (GCN, GraphSAGE, GAT/GATv2, GIN, over-smoothing)
export * from './mpnn/index.js';

// GNN-2: Heterogeneous Graphs (R-GCN, HAN, HGT, Simple-HGN, venue builder)
export * from './heterogeneous/index.js';

// GNN-3: Recommendation (LightGCN, PinSage, SR-GNN, KGAT, cold-start)
export * from './recommendation/index.js';

// GNN-4: Spatial Layout Understanding (layout GNN, quality, GMN, scene graph)
export * from './spatial/index.js';

// GNN-5: Graph Generation (GraphRNN, GRAN, DiGress, force-directed, surrogate)
export * from './generation/index.js';

// GNN-6: Temporal Networks (TGN, TGAT, time encodings)
export * from './temporal/index.js';

// GNN-7: Graph Transformers (GPS, positional encoding, Exphormer)
export * from './transformers/index.js';

// GNN-8: Combinatorial Optimization (attention model, assignment, MIP-GNN)
export * from './optimization/index.js';

// GNN-9: Scalable Inference (neighbor sampling, ClusterGCN, GLNN, IVF search)
export * from './inference/index.js';

// GNN-10: Explainability (GNNExplainer, PGExplainer, counterfactual, templates)
export * from './explainability/index.js';

// GNN-11: Integration (OT-GNN, TOGL, sheaf diffusion, demand prediction)
export * from './integration/index.js';

// GNN-12: Production Architecture (graph store, cache, event processor, serving)
export * from './production/index.js';
