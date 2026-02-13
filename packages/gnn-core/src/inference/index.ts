// ---------------------------------------------------------------------------
// GNN-9 Scalable Inference — Barrel Re-exports
// ---------------------------------------------------------------------------

export { neighborSample, createMiniBatchIterator } from './neighbor-loader.js';
export { spectralBipartition, recursivePartition, getClusterSubgraph } from './cluster-gcn.js';
export { distillGNNToMLP, mlpInference } from './glnn-distill.js';
export { buildIVFIndex, searchIVF, exactKNN } from './embedding-search.js';
