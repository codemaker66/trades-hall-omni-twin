// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — Graph Infrastructure (CSR Sparse Format)
// Graph construction, manipulation, and spectral operations for GNNs.
// ---------------------------------------------------------------------------

import type { PRNG, Graph, GraphBatch } from './types.js';

// ---------------------------------------------------------------------------
// 1. buildCSR — Edge list to Compressed Sparse Row graph
// ---------------------------------------------------------------------------

/**
 * Build a CSR (Compressed Sparse Row) graph from an edge list.
 *
 * Algorithm:
 * 1. Sort edges by source node.
 * 2. Count outgoing edges per source to build rowPtr.
 * 3. Fill colIdx and optional edgeWeights from the sorted edge list.
 *
 * @param edges - Array of [src, dst] pairs (0-indexed node IDs).
 * @param numNodes - Total number of nodes in the graph.
 * @param weights - Optional edge weights (must match edges.length).
 * @returns A Graph in CSR format with empty nodeFeatures (featureDim = 0).
 */
export function buildCSR(
  edges: [number, number][],
  numNodes: number,
  weights?: number[],
): Graph {
  const numEdges = edges.length;

  // Create indexed copy for sorting so we can track original weight indices
  const indexed = edges.map((e, i) => ({ src: e[0]!, dst: e[1]!, idx: i }));
  indexed.sort((a, b) => a.src - b.src || a.dst - b.dst);

  const rowPtr = new Uint32Array(numNodes + 1);
  const colIdx = new Uint32Array(numEdges);
  const edgeWeights = weights ? new Float64Array(numEdges) : undefined;

  // Fill colIdx (and weights) from sorted edges
  for (let i = 0; i < indexed.length; i++) {
    const entry = indexed[i]!;
    colIdx[i] = entry.dst;
    if (edgeWeights && weights) {
      edgeWeights[i] = weights[entry.idx]!;
    }
  }

  // Build rowPtr by counting edges per source
  // First pass: count edges for each source node
  const counts = new Uint32Array(numNodes);
  for (let i = 0; i < indexed.length; i++) {
    counts[indexed[i]!.src]!++;
  }

  // Prefix sum to get rowPtr
  rowPtr[0] = 0;
  for (let i = 0; i < numNodes; i++) {
    rowPtr[i + 1] = rowPtr[i]! + counts[i]!;
  }

  return {
    numNodes,
    numEdges,
    rowPtr,
    colIdx,
    edgeWeights,
    nodeFeatures: new Float64Array(0),
    featureDim: 0,
  };
}

// ---------------------------------------------------------------------------
// 2. buildFromAdjacency — Dense adjacency matrix to CSR
// ---------------------------------------------------------------------------

/**
 * Build a CSR graph from a dense adjacency matrix.
 *
 * Algorithm:
 * 1. Scan the n x n row-major adjacency matrix for non-zero entries.
 * 2. Collect edges and weights, then delegate to buildCSR.
 *
 * @param adj - Dense adjacency matrix as Float64Array of length n*n, row-major.
 * @param n - Number of nodes (matrix is n x n).
 * @returns A Graph in CSR format.
 */
export function buildFromAdjacency(adj: Float64Array, n: number): Graph {
  const edges: [number, number][] = [];
  const weights: number[] = [];

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const val = adj[i * n + j]!;
      if (val !== 0) {
        edges.push([i, j]);
        weights.push(val);
      }
    }
  }

  return buildCSR(edges, n, weights);
}

// ---------------------------------------------------------------------------
// 3. buildKNNGraph — k-Nearest Neighbor graph from positions
// ---------------------------------------------------------------------------

/**
 * Build an undirected k-nearest neighbor graph from positional data.
 *
 * Algorithm:
 * 1. For each node, compute Euclidean distance to all other nodes.
 * 2. Select the k nearest neighbors (brute force O(n^2 * dim)).
 * 3. Add both directions (i->j and j->i) to make the graph undirected.
 * 4. Deduplicate and build CSR.
 *
 * @param positions - Flat Float64Array of shape (numNodes x dim), row-major.
 * @param k - Number of nearest neighbors per node.
 * @param dim - Dimensionality of each position vector.
 * @returns An undirected CSR Graph with edge weights = Euclidean distances.
 */
export function buildKNNGraph(
  positions: Float64Array,
  k: number,
  dim: number,
): Graph {
  const numNodes = positions.length / dim;
  const edgeSet = new Set<string>();
  const edgeList: [number, number][] = [];
  const weights: number[] = [];

  for (let i = 0; i < numNodes; i++) {
    // Compute distances from node i to all others
    const dists: { node: number; dist: number }[] = [];
    for (let j = 0; j < numNodes; j++) {
      if (i === j) continue;
      let sumSq = 0;
      for (let d = 0; d < dim; d++) {
        const diff = positions[i * dim + d]! - positions[j * dim + d]!;
        sumSq += diff * diff;
      }
      dists.push({ node: j, dist: Math.sqrt(sumSq) });
    }

    // Sort by distance and take top k
    dists.sort((a, b) => a.dist - b.dist);
    const neighborCount = Math.min(k, dists.length);

    for (let ni = 0; ni < neighborCount; ni++) {
      const neighbor = dists[ni]!;
      const j = neighbor.node;
      const d = neighbor.dist;

      // Add both directions for undirected graph, dedup by canonical key
      const keyFwd = `${i},${j}`;
      const keyBwd = `${j},${i}`;

      if (!edgeSet.has(keyFwd)) {
        edgeSet.add(keyFwd);
        edgeList.push([i, j]);
        weights.push(d);
      }
      if (!edgeSet.has(keyBwd)) {
        edgeSet.add(keyBwd);
        edgeList.push([j, i]);
        weights.push(d);
      }
    }
  }

  return buildCSR(edgeList, numNodes, weights);
}

// ---------------------------------------------------------------------------
// 4. getNeighbors — Retrieve neighbors from CSR
// ---------------------------------------------------------------------------

/**
 * Get the neighbors of a given node from the CSR graph representation.
 *
 * Algorithm:
 * Uses rowPtr to locate the range [rowPtr[node], rowPtr[node+1]) in colIdx
 * and optionally edgeWeights.
 *
 * @param graph - A CSR Graph.
 * @param node - The node index to query.
 * @returns Object with `indices` (Uint32Array of neighbor node IDs) and
 *          optional `weights` (Float64Array of corresponding edge weights).
 */
export function getNeighbors(
  graph: Graph,
  node: number,
): { indices: Uint32Array; weights?: Float64Array } {
  const start = graph.rowPtr[node]!;
  const end = graph.rowPtr[node + 1]!;

  const indices = graph.colIdx.slice(start, end);
  const weights = graph.edgeWeights
    ? graph.edgeWeights.slice(start, end)
    : undefined;

  return { indices, weights };
}

// ---------------------------------------------------------------------------
// 5. addSelfLoops — Add self-loop edges for every node
// ---------------------------------------------------------------------------

/**
 * Return a new graph with a self-loop (i, i) added for every node.
 *
 * Algorithm:
 * 1. Collect all existing edges.
 * 2. For each node, check if a self-loop already exists.
 * 3. Add missing self-loops with weight 1.0.
 * 4. Rebuild CSR from the augmented edge list.
 *
 * @param graph - Input CSR Graph.
 * @returns A new Graph with self-loops added (weight 1.0 for new loops).
 */
export function addSelfLoops(graph: Graph): Graph {
  // Collect existing edges and weights
  const edges: [number, number][] = [];
  const weights: number[] = [];
  const hasSelfLoop = new Uint8Array(graph.numNodes);

  for (let i = 0; i < graph.numNodes; i++) {
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;
    for (let e = start; e < end; e++) {
      const dst = graph.colIdx[e]!;
      edges.push([i, dst]);
      weights.push(graph.edgeWeights ? graph.edgeWeights[e]! : 1.0);
      if (dst === i) {
        hasSelfLoop[i] = 1;
      }
    }
  }

  // Add missing self-loops
  for (let i = 0; i < graph.numNodes; i++) {
    if (!hasSelfLoop[i]) {
      edges.push([i, i]);
      weights.push(1.0);
    }
  }

  const result = buildCSR(edges, graph.numNodes, weights);

  return {
    ...result,
    nodeFeatures: graph.nodeFeatures,
    featureDim: graph.featureDim,
  };
}

// ---------------------------------------------------------------------------
// 6. normalizeAdjacency — Symmetric or row normalization
// ---------------------------------------------------------------------------

/**
 * Normalize adjacency matrix edge weights for GCN-style message passing.
 *
 * Algorithms:
 * - **symmetric** (D^{-1/2} A D^{-1/2}): For edge (i,j), weight becomes
 *   A[i,j] / sqrt(deg(i) * deg(j)). Used by Kipf & Welling (2017) GCN.
 * - **row** (D^{-1} A): For edge (i,j), weight becomes A[i,j] / deg(i).
 *   Equivalent to row-stochastic transition matrix.
 *
 * @param graph - Input CSR Graph.
 * @param type - 'symmetric' for D^{-1/2}AD^{-1/2}, 'row' for D^{-1}A.
 * @returns A new Graph with updated edgeWeights.
 */
export function normalizeAdjacency(
  graph: Graph,
  type: 'symmetric' | 'row',
): Graph {
  const n = graph.numNodes;

  // Compute degree for each node
  const deg = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;
    for (let e = start; e < end; e++) {
      deg[i] = deg[i]! + (graph.edgeWeights ? graph.edgeWeights[e]! : 1.0);
    }
  }

  // Compute normalized weights
  const newWeights = new Float64Array(graph.numEdges);

  for (let i = 0; i < n; i++) {
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;
    for (let e = start; e < end; e++) {
      const j = graph.colIdx[e]!;
      const w = graph.edgeWeights ? graph.edgeWeights[e]! : 1.0;

      if (type === 'symmetric') {
        // D^{-1/2} A D^{-1/2}: w / sqrt(deg_i * deg_j)
        const denom = Math.sqrt(deg[i]! * deg[j]!);
        newWeights[e] = denom > 0 ? w / denom : 0;
      } else {
        // D^{-1} A: w / deg_i
        newWeights[e] = deg[i]! > 0 ? w / deg[i]! : 0;
      }
    }
  }

  return {
    numNodes: graph.numNodes,
    numEdges: graph.numEdges,
    rowPtr: graph.rowPtr,
    colIdx: graph.colIdx,
    edgeWeights: newWeights,
    nodeFeatures: graph.nodeFeatures,
    featureDim: graph.featureDim,
  };
}

// ---------------------------------------------------------------------------
// 7. subgraph — Extract induced subgraph
// ---------------------------------------------------------------------------

/**
 * Extract the induced subgraph containing only the specified nodes.
 *
 * Algorithm:
 * 1. Create a mapping from old node indices to new contiguous indices.
 * 2. Iterate over all edges; keep only those where both endpoints are in nodeSet.
 * 3. Remap node indices and rebuild CSR.
 * 4. Copy corresponding node features.
 *
 * @param graph - Input CSR Graph.
 * @param nodeSet - Set of node indices to include.
 * @returns A new Graph with remapped node indices (sorted order).
 */
export function subgraph(graph: Graph, nodeSet: Set<number>): Graph {
  // Build sorted node list and index mapping
  const sortedNodes = Array.from(nodeSet).sort((a, b) => a - b);
  const nodeMap = new Map<number, number>();
  for (let i = 0; i < sortedNodes.length; i++) {
    nodeMap.set(sortedNodes[i]!, i);
  }

  const newNumNodes = sortedNodes.length;

  // Collect edges within the subgraph
  const edges: [number, number][] = [];
  const weights: number[] = [];

  for (const oldSrc of sortedNodes) {
    const start = graph.rowPtr[oldSrc]!;
    const end = graph.rowPtr[oldSrc + 1]!;
    const newSrc = nodeMap.get(oldSrc)!;

    for (let e = start; e < end; e++) {
      const oldDst = graph.colIdx[e]!;
      const newDst = nodeMap.get(oldDst);
      if (newDst !== undefined) {
        edges.push([newSrc, newDst]);
        weights.push(graph.edgeWeights ? graph.edgeWeights[e]! : 1.0);
      }
    }
  }

  const result = buildCSR(edges, newNumNodes, weights.length > 0 ? weights : undefined);

  // Copy node features for the subgraph
  let nodeFeatures: Float64Array;
  let featureDim: number;

  if (graph.featureDim > 0) {
    featureDim = graph.featureDim;
    nodeFeatures = new Float64Array(newNumNodes * featureDim);
    for (let i = 0; i < sortedNodes.length; i++) {
      const oldIdx = sortedNodes[i]!;
      const srcOffset = oldIdx * featureDim;
      const dstOffset = i * featureDim;
      for (let f = 0; f < featureDim; f++) {
        nodeFeatures[dstOffset + f] = graph.nodeFeatures[srcOffset + f]!;
      }
    }
  } else {
    featureDim = 0;
    nodeFeatures = new Float64Array(0);
  }

  return {
    ...result,
    nodeFeatures,
    featureDim,
  };
}

// ---------------------------------------------------------------------------
// 8. randomWalk — Single random walk on the graph
// ---------------------------------------------------------------------------

/**
 * Perform a single random walk on the graph.
 *
 * Algorithm:
 * Starting from `start`, at each step uniformly sample one neighbor
 * and move to it. If a node has no neighbors, the walk terminates early.
 * The walk array includes the starting node as the first element.
 *
 * @param graph - Input CSR Graph.
 * @param start - Starting node index.
 * @param length - Number of steps (output array length = length + 1 including start).
 * @param rng - Deterministic PRNG function returning values in [0, 1).
 * @returns Uint32Array of visited node indices (length + 1 or shorter if stuck).
 */
export function randomWalk(
  graph: Graph,
  start: number,
  length: number,
  rng: PRNG,
): Uint32Array {
  const walk = new Uint32Array(length + 1);
  walk[0] = start;
  let current = start;

  for (let step = 1; step <= length; step++) {
    const rowStart = graph.rowPtr[current]!;
    const rowEnd = graph.rowPtr[current + 1]!;
    const deg = rowEnd - rowStart;

    if (deg === 0) {
      // Dead end: return truncated walk
      return walk.slice(0, step);
    }

    const neighborIdx = rowStart + Math.floor(rng() * deg);
    current = graph.colIdx[neighborIdx]!;
    walk[step] = current;
  }

  return walk;
}

// ---------------------------------------------------------------------------
// 9. sampleNeighbors — GraphSAGE-style neighbor sampling
// ---------------------------------------------------------------------------

/**
 * Sample up to `fanout` neighbors for each node in the input set.
 *
 * Algorithm (GraphSAGE-style):
 * For each node, if degree <= fanout, return all neighbors.
 * Otherwise, sample `fanout` neighbors with replacement using the PRNG.
 *
 * @param graph - Input CSR Graph.
 * @param nodes - Array of node indices to sample neighbors for.
 * @param fanout - Maximum number of neighbors to sample per node.
 * @param rng - Deterministic PRNG function.
 * @returns Map from each node to its sampled neighbor Uint32Array.
 */
export function sampleNeighbors(
  graph: Graph,
  nodes: Uint32Array,
  fanout: number,
  rng: PRNG,
): Map<number, Uint32Array> {
  const result = new Map<number, Uint32Array>();

  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i]!;
    const rowStart = graph.rowPtr[node]!;
    const rowEnd = graph.rowPtr[node + 1]!;
    const deg = rowEnd - rowStart;

    if (deg === 0) {
      result.set(node, new Uint32Array(0));
      continue;
    }

    if (deg <= fanout) {
      // Return all neighbors
      result.set(node, graph.colIdx.slice(rowStart, rowEnd));
    } else {
      // Sample with replacement
      const sampled = new Uint32Array(fanout);
      for (let s = 0; s < fanout; s++) {
        const idx = rowStart + Math.floor(rng() * deg);
        sampled[s] = graph.colIdx[idx]!;
      }
      result.set(node, sampled);
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// 10. spectralPartition — Spectral graph partitioning via Fiedler vector
// ---------------------------------------------------------------------------

/**
 * Spectral graph partitioning using the Fiedler vector.
 *
 * Algorithm:
 * 1. Compute the graph Laplacian L = D - A.
 * 2. Approximate the 2nd smallest eigenvector (Fiedler vector) of L using
 *    power iteration with deflation against the trivial eigenvector (all-ones).
 * 3. For k=2: partition nodes by the sign of the Fiedler vector.
 *    For k>2: recursively split or use k-means on the Fiedler vector values.
 *
 * This implementation uses inverse power iteration (shift-invert) approximation
 * via repeated Laplacian multiplication + deflation for simplicity.
 *
 * @param graph - Input CSR Graph.
 * @param k - Number of partitions (clusters).
 * @param rng - Deterministic PRNG function for initialization.
 * @returns Uint32Array mapping each node to a cluster ID in [0, k).
 */
export function spectralPartition(
  graph: Graph,
  k: number,
  rng: PRNG,
): Uint32Array {
  const n = graph.numNodes;
  const assignment = new Uint32Array(n);

  if (n === 0 || k <= 1) return assignment;

  // Compute dense Laplacian
  const L = graphLaplacian(graph);

  // Find Fiedler vector using power iteration on (maxLambda*I - L)
  // which finds the largest eigenvector of (maxLambda*I - L), corresponding
  // to the smallest non-trivial eigenvector of L.

  // Estimate max eigenvalue (Gershgorin bound: max diagonal entry + max row sum offset)
  let maxLambda = 0;
  for (let i = 0; i < n; i++) {
    let rowSum = 0;
    for (let j = 0; j < n; j++) {
      rowSum += Math.abs(L[i * n + j]!);
    }
    if (rowSum > maxLambda) maxLambda = rowSum;
  }

  // M = maxLambda * I - L (shifted so smallest eigenvalues of L become largest)
  const M = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      M[i * n + j] = -L[i * n + j]!;
    }
    M[i * n + i] = M[i * n + i]! + maxLambda;
  }

  // Power iteration to find top eigenvector of M (= bottom eigvec of L)
  // but we must deflate against the trivial eigenvector (constant vector)
  let v = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    v[i] = rng() - 0.5;
  }

  // Orthogonalize against constant vector
  const oneOverSqrtN = 1.0 / Math.sqrt(n);

  const deflate = (vec: Float64Array): void => {
    let dot = 0;
    for (let i = 0; i < n; i++) {
      dot += vec[i]! * oneOverSqrtN;
    }
    for (let i = 0; i < n; i++) {
      vec[i] = vec[i]! - dot * oneOverSqrtN;
    }
  };

  const normalize = (vec: Float64Array): number => {
    let norm = 0;
    for (let i = 0; i < n; i++) {
      norm += vec[i]! * vec[i]!;
    }
    norm = Math.sqrt(norm);
    if (norm > 1e-15) {
      for (let i = 0; i < n; i++) {
        vec[i] = vec[i]! / norm;
      }
    }
    return norm;
  };

  deflate(v);
  normalize(v);

  const maxIter = 300;
  const tmp = new Float64Array(n);

  for (let iter = 0; iter < maxIter; iter++) {
    // tmp = M * v
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < n; j++) {
        sum += M[i * n + j]! * v[j]!;
      }
      tmp[i] = sum;
    }

    // Deflate against constant eigenvector
    deflate(tmp);

    // Normalize
    normalize(tmp);

    // Copy tmp -> v
    v.set(tmp);
  }

  // The Fiedler vector is now in v
  // For k=2, partition by sign
  if (k === 2) {
    for (let i = 0; i < n; i++) {
      assignment[i] = v[i]! >= 0 ? 0 : 1;
    }
    return assignment;
  }

  // For k>2, use simple 1D k-means on the Fiedler vector values
  // Initialize centroids uniformly from the range of v
  let vMin = Infinity;
  let vMax = -Infinity;
  for (let i = 0; i < n; i++) {
    if (v[i]! < vMin) vMin = v[i]!;
    if (v[i]! > vMax) vMax = v[i]!;
  }

  const centroids = new Float64Array(k);
  for (let c = 0; c < k; c++) {
    centroids[c] = vMin + ((c + 0.5) / k) * (vMax - vMin);
  }

  // K-means iterations
  const kmeansIter = 50;
  for (let iter = 0; iter < kmeansIter; iter++) {
    // Assign each node to nearest centroid
    for (let i = 0; i < n; i++) {
      let bestDist = Infinity;
      let bestC = 0;
      for (let c = 0; c < k; c++) {
        const d = Math.abs(v[i]! - centroids[c]!);
        if (d < bestDist) {
          bestDist = d;
          bestC = c;
        }
      }
      assignment[i] = bestC;
    }

    // Update centroids
    const sums = new Float64Array(k);
    const counts = new Uint32Array(k);
    for (let i = 0; i < n; i++) {
      const c = assignment[i]!;
      sums[c] = sums[c]! + v[i]!;
      counts[c] = counts[c]! + 1;
    }
    for (let c = 0; c < k; c++) {
      if (counts[c]! > 0) {
        centroids[c] = sums[c]! / counts[c]!;
      }
    }
  }

  return assignment;
}

// ---------------------------------------------------------------------------
// 11. graphLaplacian — Dense Laplacian matrix L = D - A
// ---------------------------------------------------------------------------

/**
 * Compute the graph Laplacian as a dense matrix L = D - A.
 *
 * Algorithm:
 * 1. Initialize n x n zero matrix.
 * 2. For each edge (i, j) with weight w, set L[i][j] = -w and add w to L[i][i].
 * 3. D is the diagonal degree matrix; A is the adjacency matrix.
 *
 * @param graph - Input CSR Graph.
 * @returns Float64Array of length n*n, row-major, representing L = D - A.
 */
export function graphLaplacian(graph: Graph): Float64Array {
  const n = graph.numNodes;
  const L = new Float64Array(n * n);

  for (let i = 0; i < n; i++) {
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;

    for (let e = start; e < end; e++) {
      const j = graph.colIdx[e]!;
      const w = graph.edgeWeights ? graph.edgeWeights[e]! : 1.0;

      // Off-diagonal: L[i][j] = -A[i][j]
      L[i * n + j] = L[i * n + j]! - w;

      // Diagonal: L[i][i] += degree contribution
      L[i * n + i] = L[i * n + i]! + w;
    }
  }

  return L;
}

// ---------------------------------------------------------------------------
// 12. batchGraphs — Combine multiple graphs into a single GraphBatch
// ---------------------------------------------------------------------------

/**
 * Batch multiple graphs into a single combined GraphBatch for graph-level tasks.
 *
 * Algorithm:
 * 1. Compute node and edge offsets for each graph.
 * 2. Merge rowPtr arrays with offsets applied to edge indices.
 * 3. Merge colIdx arrays with offsets applied to node indices.
 * 4. Concatenate node features and edge weights.
 * 5. Build batchIndex mapping each node to its source graph.
 *
 * @param graphs - Array of Graph objects to batch together.
 * @returns A GraphBatch containing the merged graph and bookkeeping arrays.
 */
export function batchGraphs(graphs: Graph[]): GraphBatch {
  const numGraphs = graphs.length;

  // Compute offsets
  const nodeOffsets = new Uint32Array(numGraphs + 1);
  const edgeOffsets = new Uint32Array(numGraphs + 1);

  for (let g = 0; g < numGraphs; g++) {
    nodeOffsets[g + 1] = nodeOffsets[g]! + graphs[g]!.numNodes;
    edgeOffsets[g + 1] = edgeOffsets[g]! + graphs[g]!.numEdges;
  }

  const totalNodes = nodeOffsets[numGraphs]!;
  const totalEdges = edgeOffsets[numGraphs]!;

  // Determine unified feature dimension (use first graph's featureDim; 0 if empty)
  const featureDim = numGraphs > 0 ? graphs[0]!.featureDim : 0;

  // Allocate merged arrays
  const rowPtr = new Uint32Array(totalNodes + 1);
  const colIdx = new Uint32Array(totalEdges);
  const batchIndex = new Uint32Array(totalNodes);
  const nodeFeatures = new Float64Array(totalNodes * featureDim);

  // Determine if any graph has edge weights
  const hasWeights = graphs.some((g) => g.edgeWeights !== undefined);
  const edgeWeights = hasWeights ? new Float64Array(totalEdges) : undefined;

  for (let g = 0; g < numGraphs; g++) {
    const gr = graphs[g]!;
    const nOff = nodeOffsets[g]!;
    const eOff = edgeOffsets[g]!;

    // Merge rowPtr
    for (let i = 0; i < gr.numNodes; i++) {
      rowPtr[nOff + i] = gr.rowPtr[i]! + eOff;
    }
    // The last entry for this graph's nodes
    if (g === numGraphs - 1) {
      rowPtr[totalNodes] = totalEdges;
    }

    // Merge colIdx with node offset
    for (let e = 0; e < gr.numEdges; e++) {
      colIdx[eOff + e] = gr.colIdx[e]! + nOff;
    }

    // Merge edge weights
    if (edgeWeights) {
      for (let e = 0; e < gr.numEdges; e++) {
        edgeWeights[eOff + e] = gr.edgeWeights ? gr.edgeWeights[e]! : 1.0;
      }
    }

    // Merge node features
    if (featureDim > 0) {
      for (let i = 0; i < gr.numNodes; i++) {
        for (let f = 0; f < featureDim; f++) {
          nodeFeatures[(nOff + i) * featureDim + f] =
            gr.nodeFeatures[i * gr.featureDim + f]!;
        }
      }
    }

    // Build batch index
    for (let i = 0; i < gr.numNodes; i++) {
      batchIndex[nOff + i] = g;
    }
  }

  // Fix rowPtr boundary between graphs: ensure continuity
  // The rowPtr for the start of each graph (except first) should also be set correctly.
  // The last rowPtr entry should be totalEdges.
  rowPtr[totalNodes] = totalEdges;

  const mergedGraph: Graph = {
    numNodes: totalNodes,
    numEdges: totalEdges,
    rowPtr,
    colIdx,
    edgeWeights,
    nodeFeatures,
    featureDim,
  };

  return {
    graph: mergedGraph,
    batchIndex,
    numGraphs,
    nodeOffsets: nodeOffsets.slice(0, numGraphs),
  };
}

// ---------------------------------------------------------------------------
// 13. degree — Node degree from CSR
// ---------------------------------------------------------------------------

/**
 * Get the degree of a node from the CSR representation.
 *
 * Algorithm:
 * Degree = rowPtr[node + 1] - rowPtr[node].
 *
 * @param graph - Input CSR Graph.
 * @param node - The node index.
 * @returns The degree (number of outgoing edges) of the node.
 */
export function degree(graph: Graph, node: number): number {
  return graph.rowPtr[node + 1]! - graph.rowPtr[node]!;
}

// ---------------------------------------------------------------------------
// 14. getEdgeIndex — Extract COO-style edge arrays from CSR
// ---------------------------------------------------------------------------

/**
 * Convert CSR format to COO-style edge index arrays [srcArray, dstArray].
 *
 * Algorithm:
 * For each node i, iterate over its edges [rowPtr[i], rowPtr[i+1]).
 * Each edge contributes i to srcArray and colIdx[e] to dstArray.
 *
 * @param graph - Input CSR Graph.
 * @returns A tuple [srcArray, dstArray] where each is Uint32Array of length numEdges.
 */
export function getEdgeIndex(
  graph: Graph,
): [Uint32Array, Uint32Array] {
  const srcArray = new Uint32Array(graph.numEdges);
  const dstArray = new Uint32Array(graph.numEdges);

  for (let i = 0; i < graph.numNodes; i++) {
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;
    for (let e = start; e < end; e++) {
      srcArray[e] = i;
      dstArray[e] = graph.colIdx[e]!;
    }
  }

  return [srcArray, dstArray];
}
