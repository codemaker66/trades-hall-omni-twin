// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-4: Spatial Layout Understanding
// layout-quality.ts — GNN-based layout quality scoring
//
// Multi-layer GAT with residual connections, global pooling, and a linear
// scoring head. Produces a single [0,1] quality score for an entire
// furniture layout graph plus per-node attention weights.
// ---------------------------------------------------------------------------

import type { Graph, GATConfig, GATWeights, SurrogateEnergyModel, LayoutQualityResult, PRNG } from '../types.js';
import { createPRNG } from '../types.js';
import { relu, sigmoid, matVecMul, add, dot } from '../tensor.js';
import { gatLayer, gatv2Layer, getAttentionWeights } from '../mpnn/gat.js';

// ---------------------------------------------------------------------------
// 1. layoutQualityGNN — Multi-layer GAT quality scorer
// ---------------------------------------------------------------------------

/**
 * Score a furniture layout graph for spatial quality using a multi-layer GAT.
 *
 * Algorithm:
 * 1. For each GAT layer in the model:
 *    - h' = relu(GAT(h, graph))
 *    - If h' and h have the same dimension, apply residual: h' = h' + h
 *    - h = h'
 * 2. Global mean pooling: graph_emb = mean(h_i for all nodes i).
 * 3. Linear scoring head: score = sigmoid(W_head * graph_emb + bias).
 * 4. Optionally extract attention weights from the last GAT layer.
 *
 * @param graph - Layout graph (CSR with nodeFeatures).
 * @param model - SurrogateEnergyModel containing GAT weights and scoring head.
 * @returns LayoutQualityResult with score in [0, 1] and optional nodeAttentionWeights.
 */
export function layoutQualityGNN(
  graph: Graph,
  model: SurrogateEnergyModel,
): LayoutQualityResult {
  const { gatWeights, config } = model;
  const n = graph.numNodes;

  if (n === 0) {
    return { score: 0.5 };
  }

  // Use a deterministic PRNG with a fixed seed (no dropout at inference)
  const rng = createPRNG(42);

  // Create inference config with dropout disabled
  const inferenceConfig: GATConfig = {
    ...config,
    dropout: 0,
  };

  let H = graph.nodeFeatures;
  let currentDim = graph.featureDim;

  // Multi-layer GAT with residual connections
  for (let l = 0; l < gatWeights.length; l++) {
    const weights = gatWeights[l]!;

    // Determine output dimension for this layer
    const layerConfig: GATConfig = {
      ...inferenceConfig,
      inDim: currentDim,
    };
    const outDim = layerConfig.concat
      ? layerConfig.outDim * layerConfig.heads
      : layerConfig.outDim;

    // Forward pass through GAT layer
    let hPrime: Float64Array;
    if (layerConfig.v2) {
      hPrime = gatv2Layer(graph, H, weights, layerConfig, rng);
    } else {
      hPrime = gatLayer(graph, H, weights, layerConfig, rng);
    }

    // Apply ReLU activation
    hPrime = relu(hPrime);

    // Residual connection (only if dimensions match)
    if (currentDim === outDim) {
      hPrime = add(hPrime, H);
    }

    H = hPrime;
    currentDim = outDim;
  }

  // Global mean pooling: average all node embeddings
  const graphEmb = globalMeanPool(H, n, currentDim);

  // Linear scoring head: score = sigmoid(W_head * graph_emb + bias)
  // First apply pooling projection if provided: proj = poolingW * graph_emb
  let projected: Float64Array;
  if (model.poolingW.length > 0) {
    const poolingOutDim = model.poolingW.length / currentDim;
    projected = matVecMul(model.poolingW, graphEmb, poolingOutDim, currentDim);
  } else {
    projected = graphEmb;
  }

  // Final score: sigmoid(W_head * projected + headBias)
  const headDim = projected.length;
  const logit = dot(model.headW.slice(0, headDim), projected) + model.headBias[0]!;

  // Numerically stable sigmoid
  let score: number;
  if (logit >= 0) {
    score = 1.0 / (1.0 + Math.exp(-logit));
  } else {
    const ez = Math.exp(logit);
    score = ez / (1.0 + ez);
  }

  // Extract attention weights from the last GAT layer for interpretability
  let nodeAttentionWeights: Float64Array | undefined;
  if (gatWeights.length > 0) {
    const lastWeights = gatWeights[gatWeights.length - 1]!;
    // Recompute features up to the second-to-last layer to get correct input
    if (gatWeights.length === 1) {
      const lastConfig: GATConfig = {
        ...inferenceConfig,
        inDim: graph.featureDim,
      };
      const rawAttn = getAttentionWeights(graph, graph.nodeFeatures, lastWeights, lastConfig);
      // Average attention over heads per edge, then aggregate per node
      nodeAttentionWeights = aggregateNodeAttention(rawAttn, graph, lastConfig.heads);
    }
    // For multi-layer, we use the final H before pooling as a proxy for importance
    // (computing exact per-node attention from intermediate layers is expensive)
    if (!nodeAttentionWeights) {
      nodeAttentionWeights = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let norm = 0;
        const off = i * currentDim;
        for (let d = 0; d < currentDim; d++) {
          norm += H[off + d]! * H[off + d]!;
        }
        nodeAttentionWeights[i] = Math.sqrt(norm);
      }
      // Normalize to [0, 1]
      let maxNorm = 0;
      for (let i = 0; i < n; i++) {
        if (nodeAttentionWeights[i]! > maxNorm) maxNorm = nodeAttentionWeights[i]!;
      }
      if (maxNorm > 0) {
        for (let i = 0; i < n; i++) {
          nodeAttentionWeights[i] = nodeAttentionWeights[i]! / maxNorm;
        }
      }
    }
  }

  return { score, nodeAttentionWeights };
}

// ---------------------------------------------------------------------------
// 2. globalMeanPool — Average all node features
// ---------------------------------------------------------------------------

/**
 * Global mean pooling: average all node feature vectors into a single vector.
 *
 * Given a feature matrix X of shape (numNodes x featureDim), computes:
 *   out[d] = (1/numNodes) * sum_i X[i * featureDim + d]
 *
 * @param X          - Node feature matrix, row-major (numNodes x featureDim).
 * @param numNodes   - Number of nodes.
 * @param featureDim - Dimension of each node's feature vector.
 * @returns Float64Array of length featureDim.
 */
export function globalMeanPool(
  X: Float64Array,
  numNodes: number,
  featureDim: number,
): Float64Array {
  const out = new Float64Array(featureDim);
  if (numNodes === 0) return out;

  for (let i = 0; i < numNodes; i++) {
    const offset = i * featureDim;
    for (let d = 0; d < featureDim; d++) {
      out[d] = out[d]! + X[offset + d]!;
    }
  }

  const invN = 1.0 / numNodes;
  for (let d = 0; d < featureDim; d++) {
    out[d] = out[d]! * invN;
  }

  return out;
}

// ---------------------------------------------------------------------------
// 3. globalSumPool — Sum all node features
// ---------------------------------------------------------------------------

/**
 * Global sum pooling: sum all node feature vectors into a single vector.
 *
 * Given a feature matrix X of shape (numNodes x featureDim), computes:
 *   out[d] = sum_i X[i * featureDim + d]
 *
 * @param X          - Node feature matrix, row-major (numNodes x featureDim).
 * @param numNodes   - Number of nodes.
 * @param featureDim - Dimension of each node's feature vector.
 * @returns Float64Array of length featureDim.
 */
export function globalSumPool(
  X: Float64Array,
  numNodes: number,
  featureDim: number,
): Float64Array {
  const out = new Float64Array(featureDim);

  for (let i = 0; i < numNodes; i++) {
    const offset = i * featureDim;
    for (let d = 0; d < featureDim; d++) {
      out[d] = out[d]! + X[offset + d]!;
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// Internal helper — Aggregate per-edge attention to per-node importance
// ---------------------------------------------------------------------------

/**
 * Aggregate edge-level attention weights (numEdges x heads) into per-node
 * importance scores by averaging incoming attention across all edges and heads.
 */
function aggregateNodeAttention(
  edgeAttn: Float64Array,
  graph: Graph,
  heads: number,
): Float64Array {
  const n = graph.numNodes;
  const nodeAttn = new Float64Array(n);
  const nodeCounts = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;

    for (let e = start; e < end; e++) {
      // Average attention across heads for this edge
      let avgAttn = 0;
      for (let h = 0; h < heads; h++) {
        avgAttn += edgeAttn[e * heads + h]!;
      }
      avgAttn /= heads;

      // Accumulate on source node (how much attention node i pays)
      nodeAttn[i] = nodeAttn[i]! + avgAttn;
      nodeCounts[i] = nodeCounts[i]! + 1;
    }
  }

  // Normalize by count
  for (let i = 0; i < n; i++) {
    if (nodeCounts[i]! > 0) {
      nodeAttn[i] = nodeAttn[i]! / nodeCounts[i]!;
    }
  }

  return nodeAttn;
}
