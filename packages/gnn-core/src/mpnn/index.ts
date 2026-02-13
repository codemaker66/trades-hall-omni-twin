// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — MPNN Foundation (GNN-1)
// Barrel re-exports for all message-passing neural network layers.
// ---------------------------------------------------------------------------

// GCN (Kipf & Welling 2017)
export { gcnLayer, gcnForward } from './gcn.js';

// GraphSAGE (Hamilton et al. 2017)
export { sageMeanLayer, sageMaxLayer, sageForward } from './sage.js';

// GAT / GATv2 (Velickovic et al. 2018, Brody et al. 2022)
export { gatLayer, gatv2Layer, gatForward, getAttentionWeights } from './gat.js';

// GIN (Xu et al. 2019)
export { ginLayer, ginGraphReadout, ginForward } from './gin.js';

// Over-smoothing mitigations (residual, JK-Net, DropEdge)
export { residualConnection, jkNetCombine, dropEdge } from './over-smoothing.js';
