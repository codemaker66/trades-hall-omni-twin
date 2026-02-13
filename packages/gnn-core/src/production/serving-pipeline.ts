// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-12 Production Architecture: Serving Pipeline
// End-to-end serving orchestrator for GNN inference.
// ---------------------------------------------------------------------------

import type { MLPWeights, IVFIndex } from '../types.js';
import { dot, l2Norm } from '../tensor.js';
import type { GraphStore } from './graph-store.js';
import type { EmbeddingCache } from './cache-manager.js';

// ---- Types ----

/** Configuration for the serving pipeline. */
export interface ServingPipelineConfig {
  /** Inference mode: distilled MLP, approximate nearest neighbor, or realtime GNN. */
  readonly mode: 'mlp' | 'ann' | 'realtime';
  /** Reference to the shared graph store. */
  readonly graphStore: GraphStore;
  /** Reference to the shared embedding cache. */
  readonly cache: EmbeddingCache;
  /** Dimension of embedding vectors. */
  readonly embeddingDim: number;
}

/** A request to the serving pipeline. */
export interface ServingRequest {
  /** The query node whose embedding/prediction is needed. */
  readonly queryNodeId: string;
  /** The task to perform. */
  readonly taskType: 'recommend' | 'predict' | 'explain';
  /** Number of results to return for recommendation tasks. */
  readonly topK?: number;
}

/** The response from the serving pipeline. */
export interface ServingResponse {
  /** Raw result vector (scores, logits, or importance weights). */
  readonly results: Float64Array;
  /** Node IDs corresponding to the results. */
  readonly nodeIds: string[];
  /** Time taken for inference in milliseconds. */
  readonly latencyMs: number;
  /** Whether the result was served from cache. */
  readonly cached: boolean;
}

/** Health check result. */
export interface HealthCheckResult {
  readonly status: 'ok' | 'degraded';
  readonly cacheStats: { hits: number; misses: number; evictions: number; size: number };
  readonly graphStats: { nodeCount: number; edgeCount: number };
}

// ---------------------------------------------------------------------------
// ServingPipeline — Orchestrates inference across modes
// ---------------------------------------------------------------------------

/**
 * End-to-end serving orchestrator that supports three inference modes:
 *
 * - **mlp**: Uses a distilled MLP for fast single-node inference.
 * - **ann**: Uses an IVF index for approximate nearest neighbor search.
 * - **realtime**: Computes embeddings on-the-fly from the graph (mean-pool
 *   neighbors as a lightweight approximation).
 *
 * All modes check the embedding cache first. The pipeline is designed to
 * demonstrate the production architecture pattern rather than serve as a
 * fully optimized inference engine.
 */
export class ServingPipeline {
  private readonly graphStore: GraphStore;
  private readonly cache: EmbeddingCache;
  private readonly embeddingDim: number;
  private readonly mode: 'mlp' | 'ann' | 'realtime';

  private mlpWeights: MLPWeights | null = null;
  private ivfIndex: IVFIndex | null = null;

  constructor(config: ServingPipelineConfig) {
    this.mode = config.mode;
    this.graphStore = config.graphStore;
    this.cache = config.cache;
    this.embeddingDim = config.embeddingDim;
  }

  // ---- Configuration ----

  /** Set the MLP weights for distilled inference mode. */
  setMLPWeights(weights: MLPWeights): void {
    this.mlpWeights = weights;
  }

  /** Set the IVF index for approximate nearest neighbor mode. */
  setIVFIndex(index: IVFIndex): void {
    this.ivfIndex = index;
  }

  // ---- Serving ----

  /**
   * Serve an inference request.
   *
   * Flow:
   * 1. Check cache for a precomputed embedding.
   * 2. If not cached, compute the embedding according to the current mode.
   * 3. Depending on `taskType`, produce recommendation scores, a prediction
   *    vector, or feature importance weights.
   * 4. Return the result with latency and cache-hit metadata.
   */
  serve(request: ServingRequest): ServingResponse {
    const startTime = performance.now();
    const topK = request.topK ?? 10;

    // Step 1: Check cache
    let embedding = this.cache.get(request.queryNodeId);
    let cached = false;

    if (embedding) {
      cached = true;
    } else {
      // Step 2: Compute embedding based on mode
      embedding = this.computeEmbedding(request.queryNodeId);
      if (embedding) {
        this.cache.set(request.queryNodeId, embedding);
      }
    }

    // If the node does not exist, return an empty response
    if (!embedding) {
      return {
        results: new Float64Array(0),
        nodeIds: [],
        latencyMs: performance.now() - startTime,
        cached: false,
      };
    }

    // Step 3: Task-specific processing
    let results: Float64Array;
    let nodeIds: string[];

    switch (request.taskType) {
      case 'recommend': {
        const rec = this.recommend(embedding, request.queryNodeId, topK);
        results = rec.scores;
        nodeIds = rec.nodeIds;
        break;
      }
      case 'predict': {
        results = this.predict(embedding);
        nodeIds = [request.queryNodeId];
        break;
      }
      case 'explain': {
        const exp = this.explain(request.queryNodeId, embedding);
        results = exp.importance;
        nodeIds = exp.nodeIds;
        break;
      }
      default: {
        results = embedding;
        nodeIds = [request.queryNodeId];
      }
    }

    const latencyMs = performance.now() - startTime;

    return { results, nodeIds, latencyMs, cached };
  }

  // ---- Cache warming ----

  /**
   * Pre-compute and cache embeddings for a set of node IDs.
   * Useful for warming the cache before peak traffic.
   */
  warmCache(nodeIds: string[]): void {
    for (const nodeId of nodeIds) {
      if (!this.cache.get(nodeId)) {
        const embedding = this.computeEmbedding(nodeId);
        if (embedding) {
          this.cache.set(nodeId, embedding);
        }
      }
    }
  }

  // ---- Health check ----

  /** Return the health status of the serving pipeline. */
  healthCheck(): HealthCheckResult {
    const cacheStats = this.cache.stats();
    const graphStats = {
      nodeCount: this.graphStore.nodeCount(),
      edgeCount: this.graphStore.edgeCount(),
    };

    // Pipeline is degraded if the cache has a high miss rate or graph is empty
    const totalRequests = cacheStats.hits + cacheStats.misses;
    const hitRate = totalRequests > 0 ? cacheStats.hits / totalRequests : 1;
    const status: 'ok' | 'degraded' =
      graphStats.nodeCount === 0 || hitRate < 0.1 ? 'degraded' : 'ok';

    return { status, cacheStats, graphStats };
  }

  // ---- Private: Embedding computation ----

  /**
   * Compute an embedding for the given node according to the current mode.
   */
  private computeEmbedding(nodeId: string): Float64Array | undefined {
    const node = this.graphStore.getNode(nodeId);
    if (!node) return undefined;

    switch (this.mode) {
      case 'mlp':
        return this.computeMLPEmbedding(node.features);
      case 'ann':
        // For ANN mode, the embedding is the node feature itself
        // (embeddings should have been pre-computed and stored as features)
        return this.padOrTruncate(node.features);
      case 'realtime':
        return this.computeRealtimeEmbedding(nodeId, node.features);
      default:
        return this.padOrTruncate(node.features);
    }
  }

  /**
   * MLP-based embedding: forward pass through the distilled MLP layers.
   * Each layer: output = relu(W * input + bias), except the last which is linear.
   */
  private computeMLPEmbedding(features: Float64Array): Float64Array {
    if (!this.mlpWeights || this.mlpWeights.layers.length === 0) {
      return this.padOrTruncate(features);
    }

    let current = features;

    for (let l = 0; l < this.mlpWeights.layers.length; l++) {
      const layer = this.mlpWeights.layers[l]!;
      const { W, bias, inDim, outDim } = layer;

      const output = new Float64Array(outDim);

      for (let o = 0; o < outDim; o++) {
        let val = bias[o]!;
        const inputLen = Math.min(inDim, current.length);
        for (let i = 0; i < inputLen; i++) {
          val += W[o * inDim + i]! * current[i]!;
        }
        // Apply ReLU for all layers except the last
        if (l < this.mlpWeights.layers.length - 1) {
          output[o] = val > 0 ? val : 0;
        } else {
          output[o] = val;
        }
      }

      current = output;
    }

    return current;
  }

  /**
   * Realtime embedding: lightweight 1-hop mean-pool aggregation.
   *
   * Computes the mean of the node's own features and its neighbors' features.
   * This is a simplified stand-in for a full GNN forward pass.
   */
  private computeRealtimeEmbedding(
    nodeId: string,
    features: Float64Array,
  ): Float64Array {
    const neighbors = this.graphStore.getNeighborIds(nodeId);

    if (neighbors.length === 0) {
      return this.padOrTruncate(features);
    }

    const dim = this.embeddingDim;
    const agg = new Float64Array(dim);

    // Add self features
    const selfFeat = this.padOrTruncate(features);
    for (let d = 0; d < dim; d++) {
      agg[d] = selfFeat[d]!;
    }

    // Add neighbor features
    let count = 1;
    for (const neighborId of neighbors) {
      const neighbor = this.graphStore.getNode(neighborId);
      if (neighbor) {
        const nFeat = this.padOrTruncate(neighbor.features);
        for (let d = 0; d < dim; d++) {
          agg[d] = agg[d]! + nFeat[d]!;
        }
        count++;
      }
    }

    // Mean pool
    for (let d = 0; d < dim; d++) {
      agg[d] = agg[d]! / count;
    }

    return agg;
  }

  // ---- Private: Task-specific processing ----

  /**
   * Recommend: find the top-K most similar nodes by cosine similarity
   * to the query embedding.
   */
  private recommend(
    queryEmbedding: Float64Array,
    queryNodeId: string,
    topK: number,
  ): { scores: Float64Array; nodeIds: string[] } {
    if (this.mode === 'ann' && this.ivfIndex) {
      return this.annRecommend(queryEmbedding, queryNodeId, topK);
    }

    // Brute-force cosine similarity against all nodes
    const scored: { nodeId: string; score: number }[] = [];
    const queryNorm = l2Norm(queryEmbedding);

    if (queryNorm === 0) {
      return { scores: new Float64Array(0), nodeIds: [] };
    }

    // Get all nodes from the graph store via the CSR conversion
    const { nodeIds: allNodeIds } = this.graphStore.toGraph();

    for (const candidateId of allNodeIds) {
      if (candidateId === queryNodeId) continue;

      // Check cache first, then compute
      let candidateEmb = this.cache.get(candidateId);
      if (!candidateEmb) {
        candidateEmb = this.computeEmbedding(candidateId) ?? undefined;
        if (candidateEmb) {
          this.cache.set(candidateId, candidateEmb);
        }
      }
      if (!candidateEmb) continue;

      const candidateNorm = l2Norm(candidateEmb);
      if (candidateNorm === 0) continue;

      const similarity = dot(queryEmbedding, candidateEmb) / (queryNorm * candidateNorm);
      scored.push({ nodeId: candidateId, score: similarity });
    }

    // Sort descending by score
    scored.sort((a, b) => b.score - a.score);
    const k = Math.min(topK, scored.length);

    const scores = new Float64Array(k);
    const nodeIds: string[] = [];
    for (let i = 0; i < k; i++) {
      const entry = scored[i]!;
      scores[i] = entry.score;
      nodeIds.push(entry.nodeId);
    }

    return { scores, nodeIds };
  }

  /**
   * ANN-based recommendation using a pre-built IVF index.
   * Falls back to brute force if the index is not suitable.
   */
  private annRecommend(
    queryEmbedding: Float64Array,
    queryNodeId: string,
    topK: number,
  ): { scores: Float64Array; nodeIds: string[] } {
    const index = this.ivfIndex!;
    const dim = index.dim;

    // Find nearest centroid
    let bestCluster = 0;
    let bestDist = Infinity;
    for (let c = 0; c < index.nClusters; c++) {
      let dist = 0;
      for (let d = 0; d < dim; d++) {
        const diff = queryEmbedding[d]! - index.centroids[c * dim + d]!;
        dist += diff * diff;
      }
      if (dist < bestDist) {
        bestDist = dist;
        bestCluster = c;
      }
    }

    // Search within the best cluster
    const scored: { idx: number; score: number }[] = [];
    const queryNorm = l2Norm(queryEmbedding);

    for (let i = 0; i < index.numEmbeddings; i++) {
      if (index.assignments[i] !== bestCluster) continue;

      const embSlice = index.embeddings.subarray(i * dim, (i + 1) * dim);
      const embNorm = l2Norm(embSlice);
      if (embNorm === 0 || queryNorm === 0) continue;

      const similarity = dot(queryEmbedding, embSlice) / (queryNorm * embNorm);
      scored.push({ idx: i, score: similarity });
    }

    scored.sort((a, b) => b.score - a.score);
    const k = Math.min(topK, scored.length);

    const scores = new Float64Array(k);
    const nodeIds: string[] = [];

    // Map index back to node IDs via graph ordering
    const { nodeIds: allNodeIds } = this.graphStore.toGraph();
    for (let i = 0; i < k; i++) {
      const entry = scored[i]!;
      scores[i] = entry.score;
      const nid = allNodeIds[entry.idx];
      nodeIds.push(nid ?? `node_${entry.idx}`);
    }

    return { scores, nodeIds };
  }

  /**
   * Predict: run the embedding through the MLP (if available) to produce
   * a prediction logit vector. Otherwise just return the embedding.
   */
  private predict(embedding: Float64Array): Float64Array {
    if (this.mode === 'mlp' && this.mlpWeights) {
      // The embedding is already the MLP output; return it as the prediction
      return embedding;
    }
    // For other modes, the embedding itself serves as the prediction vector
    return embedding;
  }

  /**
   * Explain: compute per-neighbor feature importance as a simple
   * attribution heuristic (absolute dot-product contribution).
   */
  private explain(
    nodeId: string,
    embedding: Float64Array,
  ): { importance: Float64Array; nodeIds: string[] } {
    const neighbors = this.graphStore.getNeighborIds(nodeId);

    if (neighbors.length === 0) {
      return {
        importance: new Float64Array([1.0]),
        nodeIds: [nodeId],
      };
    }

    // Include self + neighbors
    const allIds = [nodeId, ...neighbors];
    const importance = new Float64Array(allIds.length);

    const embNorm = l2Norm(embedding);
    if (embNorm === 0) {
      return { importance, nodeIds: allIds };
    }

    for (let i = 0; i < allIds.length; i++) {
      const nid = allIds[i]!;
      const node = this.graphStore.getNode(nid);
      if (!node) continue;

      const feat = this.padOrTruncate(node.features);
      const featNorm = l2Norm(feat);
      if (featNorm === 0) continue;

      // Absolute cosine similarity as importance proxy
      importance[i] = Math.abs(dot(embedding, feat) / (embNorm * featNorm));
    }

    // Normalize importance so it sums to 1
    let total = 0;
    for (let i = 0; i < importance.length; i++) {
      total += importance[i]!;
    }
    if (total > 0) {
      for (let i = 0; i < importance.length; i++) {
        importance[i] = importance[i]! / total;
      }
    }

    return { importance, nodeIds: allIds };
  }

  // ---- Private: Utilities ----

  /**
   * Pad a feature vector to `embeddingDim` (zero-pad) or truncate if longer.
   */
  private padOrTruncate(features: Float64Array): Float64Array {
    if (features.length === this.embeddingDim) {
      return features;
    }

    const result = new Float64Array(this.embeddingDim);
    const len = Math.min(features.length, this.embeddingDim);
    for (let i = 0; i < len; i++) {
      result[i] = features[i]!;
    }
    return result;
  }
}
