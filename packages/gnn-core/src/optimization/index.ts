// ---------------------------------------------------------------------------
// GNN-8: Combinatorial Optimization — Barrel Re-exports
// ---------------------------------------------------------------------------

export {
  attentionModelEncode,
  attentionModelDecode,
  greedyDecode,
} from './attention-model.js';

export {
  sinkhornAssignment,
  hungarianAlgorithm,
  bipartiteGNNAssignment,
} from './event-room-assignment.js';

export {
  encodeMIP,
  mipGNNPredict,
} from './mip-gnn.js';

export type {
  AttentionModelConfig,
  AttentionModelWeights,
  SinkhornConfig,
  AssignmentResult,
  MIPVariable,
  MIPConstraint,
  MIPEncoding,
} from '../types.js';
