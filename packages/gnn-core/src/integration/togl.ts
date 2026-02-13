// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — Topological GNN Layer (GNN-11)
// TOGL: Topological Graph Layer — learns filtration functions, computes
// persistence diagrams (H_0 via Union-Find), converts to persistence images,
// and concatenates topological features to node embeddings.
// ---------------------------------------------------------------------------

import type { Graph, PersistenceDiagram } from '../types.js';
import { dot } from '../tensor.js';

// ---------------------------------------------------------------------------
// 1. computePersistenceDiagram — H_0 persistence via Union-Find
// ---------------------------------------------------------------------------

/**
 * Compute the H_0 (connected components) persistence diagram from node
 * filtration values using a Union-Find data structure.
 *
 * Algorithm:
 * 1. Extract all edges from the graph, assigning each edge a filtration
 *    value = max(filtration[src], filtration[dst]).
 * 2. Sort edges by ascending filtration value.
 * 3. Initialize each node as its own connected component (born at its
 *    filtration value). Each component is "born" at the minimum filtration
 *    of its nodes.
 * 4. Process edges in order: when merging two components, the younger one
 *    (higher birth time) "dies" at the current edge's filtration value.
 * 5. The oldest surviving component lives forever (death = +Infinity, or
 *    we cap it at max filtration).
 *
 * @param graph - CSR graph.
 * @param filtration - Node filtration values (length = numNodes).
 * @returns PersistenceDiagram with births, deaths arrays and dim = 0.
 */
export function computePersistenceDiagram(
  graph: Graph,
  filtration: Float64Array,
): PersistenceDiagram {
  const n = graph.numNodes;

  // --- Union-Find ---
  const parent = new Int32Array(n);
  const rank = new Int32Array(n);
  const birthTime = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    parent[i] = i;
    rank[i] = 0;
    birthTime[i] = filtration[i]!;
  }

  function find(x: number): number {
    let root = x;
    while (parent[root]! !== root) {
      root = parent[root]!;
    }
    // Path compression
    let cur = x;
    while (cur !== root) {
      const next = parent[cur]!;
      parent[cur] = root;
      cur = next;
    }
    return root;
  }

  // --- Extract and sort edges ---
  interface FilteredEdge {
    src: number;
    dst: number;
    filtrationValue: number;
  }

  const edges: FilteredEdge[] = [];
  for (let i = 0; i < n; i++) {
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;
    for (let e = start; e < end; e++) {
      const j = graph.colIdx[e]!;
      // Only add each undirected edge once (i < j)
      if (i < j) {
        edges.push({
          src: i,
          dst: j,
          filtrationValue: Math.max(filtration[i]!, filtration[j]!),
        });
      }
    }
  }

  edges.sort((a, b) => a.filtrationValue - b.filtrationValue);

  // --- Process edges and record persistence pairs ---
  const birthsList: number[] = [];
  const deathsList: number[] = [];

  for (let eIdx = 0; eIdx < edges.length; eIdx++) {
    const edge = edges[eIdx]!;
    const rootA = find(edge.src);
    const rootB = find(edge.dst);

    if (rootA === rootB) continue; // Already connected

    // Determine which component is older (lower birth) and which is younger
    const birthA = birthTime[rootA]!;
    const birthB = birthTime[rootB]!;

    let older: number;
    let younger: number;

    if (birthA <= birthB) {
      older = rootA;
      younger = rootB;
    } else {
      older = rootB;
      younger = rootA;
    }

    // The younger component dies at this edge's filtration value
    const deathValue = edge.filtrationValue;
    const birthValue = birthTime[younger]!;

    // Only record if there is non-zero persistence
    if (deathValue > birthValue) {
      birthsList.push(birthValue);
      deathsList.push(deathValue);
    }

    // Union by rank — older survives
    if (rank[older]! < rank[younger]!) {
      parent[older] = younger;
      birthTime[younger] = Math.min(birthTime[younger]!, birthTime[older]!);
    } else if (rank[older]! > rank[younger]!) {
      parent[younger] = older;
      birthTime[older] = Math.min(birthTime[older]!, birthTime[younger]!);
    } else {
      parent[younger] = older;
      birthTime[older] = Math.min(birthTime[older]!, birthTime[younger]!);
      rank[older] = rank[older]! + 1;
    }
  }

  return {
    births: new Float64Array(birthsList),
    deaths: new Float64Array(deathsList),
    dim: 0,
  };
}

// ---------------------------------------------------------------------------
// 2. persistenceImage — Fixed-size vectorisation of persistence diagram
// ---------------------------------------------------------------------------

/**
 * Convert a persistence diagram to a persistence image — a fixed-size
 * vector representation suitable for machine learning.
 *
 * Algorithm:
 * 1. Transform each (birth, death) pair to (birth, persistence) where
 *    persistence = death - birth.
 * 2. Construct a grid of resolution x resolution over the range of
 *    birth and persistence values.
 * 3. Place a Gaussian kernel at each transformed point, weighted by
 *    its persistence (longer-lived features are more important).
 * 4. Sum contributions on the grid and return the flattened image.
 *
 * @param diagram - Input persistence diagram.
 * @param resolution - Grid resolution (output is resolution x resolution).
 * @param sigma - Standard deviation of the Gaussian kernel.
 * @returns Flattened persistence image of length resolution^2.
 */
export function persistenceImage(
  diagram: PersistenceDiagram,
  resolution: number,
  sigma: number,
): Float64Array {
  const numPoints = diagram.births.length;
  const image = new Float64Array(resolution * resolution);

  if (numPoints === 0) return image;

  // Transform to (birth, persistence) coordinates
  const bCoords = new Float64Array(numPoints);
  const pCoords = new Float64Array(numPoints);

  let bMin = Infinity;
  let bMax = -Infinity;
  let pMin = Infinity;
  let pMax = -Infinity;

  for (let i = 0; i < numPoints; i++) {
    const b = diagram.births[i]!;
    const p = diagram.deaths[i]! - b;
    bCoords[i] = b;
    pCoords[i] = p;

    if (b < bMin) bMin = b;
    if (b > bMax) bMax = b;
    if (p < pMin) pMin = p;
    if (p > pMax) pMax = p;
  }

  // Handle degenerate ranges
  const bRange = bMax - bMin > 1e-12 ? bMax - bMin : 1.0;
  const pRange = pMax - pMin > 1e-12 ? pMax - pMin : 1.0;

  // Extend ranges slightly for padding
  const bStart = bMin - 0.05 * bRange;
  const bEnd = bMax + 0.05 * bRange;
  const pStart = pMin - 0.05 * pRange;
  const pEnd = pMax + 0.05 * pRange;

  const bStep = (bEnd - bStart) / resolution;
  const pStep = (pEnd - pStart) / resolution;

  const invTwoSigmaSq = 1.0 / (2.0 * sigma * sigma);

  // Accumulate Gaussian contributions weighted by persistence
  for (let ptIdx = 0; ptIdx < numPoints; ptIdx++) {
    const b = bCoords[ptIdx]!;
    const p = pCoords[ptIdx]!;
    const weight = p; // Weight by persistence

    for (let gi = 0; gi < resolution; gi++) {
      const gridB = bStart + (gi + 0.5) * bStep;
      const dbSq = (gridB - b) * (gridB - b);

      for (let gj = 0; gj < resolution; gj++) {
        const gridP = pStart + (gj + 0.5) * pStep;
        const dpSq = (gridP - p) * (gridP - p);

        const gaussVal = Math.exp(-(dbSq + dpSq) * invTwoSigmaSq);
        image[gi * resolution + gj] = image[gi * resolution + gj]! + weight * gaussVal;
      }
    }
  }

  return image;
}

// ---------------------------------------------------------------------------
// 3. toglLayer — Topological Graph Layer (TOGL)
// ---------------------------------------------------------------------------

/**
 * TOGL layer: learn a filtration from node features, compute persistent
 * homology, convert to a persistence image, and concatenate the topological
 * features to each node's feature vector.
 *
 * Algorithm:
 * 1. Compute learned filtration: f(v) = filtrationWeights^T * x_v for each
 *    node v (a linear projection of node features to a scalar).
 * 2. Compute the H_0 persistence diagram using Union-Find on the graph
 *    with the learned filtration.
 * 3. Convert the persistence diagram to a persistence image of size
 *    resolution x resolution.
 * 4. Concatenate the flattened persistence image to each node's feature
 *    vector, producing augmented features of dimension
 *    (featureDim + resolution^2) per node.
 *
 * @param graph - CSR graph.
 * @param X - Node features, flat row-major (numNodes x featureDim).
 * @param numNodes - Number of nodes.
 * @param featureDim - Original feature dimension per node.
 * @param filtrationWeights - Weight vector of length featureDim for linear filtration.
 * @param resolution - Persistence image grid resolution.
 * @param sigma - Gaussian kernel bandwidth for persistence image.
 * @returns Augmented features, flat row-major (numNodes x (featureDim + resolution^2)).
 */
export function toglLayer(
  graph: Graph,
  X: Float64Array,
  numNodes: number,
  featureDim: number,
  filtrationWeights: Float64Array,
  resolution: number,
  sigma: number,
): Float64Array {
  // Step 1: Compute learned filtration f(v) = w^T * x_v
  const filtration = new Float64Array(numNodes);
  for (let v = 0; v < numNodes; v++) {
    const nodeFeats = X.subarray(v * featureDim, (v + 1) * featureDim);
    filtration[v] = dot(filtrationWeights, nodeFeats);
  }

  // Step 2: Compute persistence diagram
  const diagram = computePersistenceDiagram(graph, filtration);

  // Step 3: Convert to persistence image
  const piImage = persistenceImage(diagram, resolution, sigma);
  const imageDim = resolution * resolution;

  // Step 4: Concatenate image features to each node's features
  const augDim = featureDim + imageDim;
  const augmented = new Float64Array(numNodes * augDim);

  for (let v = 0; v < numNodes; v++) {
    // Copy original features
    const srcOffset = v * featureDim;
    const dstOffset = v * augDim;
    for (let f = 0; f < featureDim; f++) {
      augmented[dstOffset + f] = X[srcOffset + f]!;
    }
    // Append persistence image (shared across all nodes — graph-level topology)
    for (let f = 0; f < imageDim; f++) {
      augmented[dstOffset + featureDim + f] = piImage[f]!;
    }
  }

  return augmented;
}
