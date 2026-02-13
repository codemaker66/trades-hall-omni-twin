// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-4: Spatial Layout Understanding
// layout-gnn.ts — Furniture layout as graph
//
// Converts venue furniture layouts into graph representations suitable for
// GNN-based spatial reasoning. Each furniture item becomes a node with
// type-one-hot + geometric features. Edges encode spatial adjacency and
// optionally functional relationships (e.g., chair -> table).
// ---------------------------------------------------------------------------

import type { Graph, LayoutItem, LayoutGraphConfig } from '../types.js';
import { buildCSR } from '../graph.js';

// ---------------------------------------------------------------------------
// 1. buildLayoutGraph — Layout items to CSR Graph with spatial edges
// ---------------------------------------------------------------------------

/**
 * Build a CSR graph from venue furniture layout items.
 *
 * Algorithm:
 * 1. For each LayoutItem, create a node feature vector:
 *    [type_onehot(numTypes), width, depth, height, x, y, sin(rotation), cos(rotation)]
 *    So featureDim = numTypes + 7.
 * 2. Add spatial adjacency edges: for each pair (i, j) where
 *    Euclidean distance(item_i, item_j) < config.distanceThreshold,
 *    add edges (i, j) and (j, i) (undirected).
 * 3. Edge features: [dx, dy, euclidean_distance, delta_rotation, 1.0]
 *    where dx = x_j - x_i, dy = y_j - y_i, delta_rotation = rot_j - rot_i.
 * 4. Optionally add functional edges (chair -> table) if config.functionalEdges.
 *
 * @param items  - Array of LayoutItem furniture pieces.
 * @param config - Graph construction configuration.
 * @returns A Graph in CSR format with nodeFeatures, edgeFeatures, edgeFeatureDim=5.
 */
export function buildLayoutGraph(
  items: LayoutItem[],
  config: LayoutGraphConfig,
): Graph {
  const numItems = items.length;
  if (numItems === 0) {
    return {
      numNodes: 0,
      numEdges: 0,
      rowPtr: new Uint32Array(1),
      colIdx: new Uint32Array(0),
      nodeFeatures: new Float64Array(0),
      featureDim: 0,
      edgeFeatures: new Float64Array(0),
      edgeFeatureDim: 5,
    };
  }

  // Determine featureDim from first item's numTypes
  const numTypes = items[0]!.numTypes;
  const featureDim = numTypes + 7;

  // Build node features
  const nodeFeatures = layoutItemsToFeatures(items);

  // Collect edges with features
  const edges: [number, number][] = [];
  const edgeFeaturesList: number[][] = [];

  const threshold = config.distanceThreshold;
  const thresholdSq = threshold * threshold;

  // Spatial adjacency edges (undirected: add both directions)
  for (let i = 0; i < numItems; i++) {
    const itemI = items[i]!;
    for (let j = i + 1; j < numItems; j++) {
      const itemJ = items[j]!;
      const dx = itemJ.x - itemI.x;
      const dy = itemJ.y - itemI.y;
      const distSq = dx * dx + dy * dy;

      if (distSq < thresholdSq) {
        const dist = Math.sqrt(distSq);
        const deltaRot = itemJ.rotation - itemI.rotation;

        // Edge i -> j
        edges.push([i, j]);
        edgeFeaturesList.push([dx, dy, dist, deltaRot, 1.0]);

        // Edge j -> i (reverse direction)
        edges.push([j, i]);
        edgeFeaturesList.push([-dx, -dy, dist, -deltaRot, 1.0]);
      }
    }
  }

  // Functional edges: chair (type=0) -> nearest table (type=1) within threshold
  if (config.functionalEdges) {
    const CHAIR_TYPE = 0;
    const TABLE_TYPE = 1;

    for (let i = 0; i < numItems; i++) {
      const itemI = items[i]!;
      if (itemI.type !== CHAIR_TYPE) continue;

      let nearestTable = -1;
      let nearestDistSq = Infinity;

      for (let j = 0; j < numItems; j++) {
        if (i === j) continue;
        const itemJ = items[j]!;
        if (itemJ.type !== TABLE_TYPE) continue;

        const dx = itemJ.x - itemI.x;
        const dy = itemJ.y - itemI.y;
        const distSq = dx * dx + dy * dy;

        if (distSq < thresholdSq && distSq < nearestDistSq) {
          nearestDistSq = distSq;
          nearestTable = j;
        }
      }

      if (nearestTable >= 0) {
        const itemJ = items[nearestTable]!;
        const dx = itemJ.x - itemI.x;
        const dy = itemJ.y - itemI.y;
        const dist = Math.sqrt(nearestDistSq);
        const deltaRot = itemJ.rotation - itemI.rotation;

        // Check if this edge already exists (from spatial adjacency)
        const edgeKey = `${i},${nearestTable}`;
        let alreadyExists = false;
        for (let e = 0; e < edges.length; e++) {
          if (edges[e]![0] === i && edges[e]![1] === nearestTable) {
            alreadyExists = true;
            break;
          }
        }

        if (!alreadyExists) {
          // Functional edge chair -> table
          edges.push([i, nearestTable]);
          edgeFeaturesList.push([dx, dy, dist, deltaRot, 1.0]);

          // Reverse edge table -> chair
          edges.push([nearestTable, i]);
          edgeFeaturesList.push([-dx, -dy, dist, -deltaRot, 1.0]);
        }
      }
    }
  }

  // Build CSR graph
  const csrGraph = buildCSR(edges, numItems);

  // Build edge features in CSR order
  // buildCSR sorts edges by (src, dst), so we need to map original edge
  // features to the sorted order.
  const numEdges = edges.length;
  const edgeFeatureDim = 5;

  // Create a map from (src, dst) -> edge feature to handle the sort
  // Since buildCSR sorts edges, we need to reconstruct edge features in sorted order.
  // We'll build a lookup by iterating the CSR output.
  const edgeFeatMap = new Map<string, number[]>();
  for (let e = 0; e < numEdges; e++) {
    const key = `${edges[e]![0]},${edges[e]![1]}`;
    // If duplicate edges exist (shouldn't, but safety), keep first
    if (!edgeFeatMap.has(key)) {
      edgeFeatMap.set(key, edgeFeaturesList[e]!);
    }
  }

  const edgeFeatures = new Float64Array(csrGraph.numEdges * edgeFeatureDim);
  for (let i = 0; i < numItems; i++) {
    const start = csrGraph.rowPtr[i]!;
    const end = csrGraph.rowPtr[i + 1]!;
    for (let e = start; e < end; e++) {
      const j = csrGraph.colIdx[e]!;
      const key = `${i},${j}`;
      const feat = edgeFeatMap.get(key);
      if (feat) {
        for (let d = 0; d < edgeFeatureDim; d++) {
          edgeFeatures[e * edgeFeatureDim + d] = feat[d]!;
        }
      }
    }
  }

  return {
    numNodes: csrGraph.numNodes,
    numEdges: csrGraph.numEdges,
    rowPtr: csrGraph.rowPtr,
    colIdx: csrGraph.colIdx,
    edgeWeights: csrGraph.edgeWeights,
    nodeFeatures,
    featureDim,
    edgeFeatures,
    edgeFeatureDim,
  };
}

// ---------------------------------------------------------------------------
// 2. layoutItemsToFeatures — Convert LayoutItems to feature matrix
// ---------------------------------------------------------------------------

/**
 * Convert an array of LayoutItems to a flat feature matrix.
 *
 * For each item, the feature vector is:
 *   [type_onehot(numTypes), width, depth, height, x, y, sin(rotation), cos(rotation)]
 *
 * Feature dimension = numTypes + 7.
 *
 * @param items - Array of LayoutItem furniture pieces.
 * @returns Float64Array of shape (numItems x featureDim), row-major.
 */
export function layoutItemsToFeatures(items: LayoutItem[]): Float64Array {
  if (items.length === 0) return new Float64Array(0);

  const numTypes = items[0]!.numTypes;
  const featureDim = numTypes + 7;
  const numItems = items.length;
  const features = new Float64Array(numItems * featureDim);

  for (let i = 0; i < numItems; i++) {
    const item = items[i]!;
    const offset = i * featureDim;

    // One-hot encoding of type
    if (item.type >= 0 && item.type < numTypes) {
      features[offset + item.type] = 1.0;
    }

    // Geometric features
    features[offset + numTypes] = item.width;
    features[offset + numTypes + 1] = item.depth;
    features[offset + numTypes + 2] = item.height;
    features[offset + numTypes + 3] = item.x;
    features[offset + numTypes + 4] = item.y;
    features[offset + numTypes + 5] = Math.sin(item.rotation);
    features[offset + numTypes + 6] = Math.cos(item.rotation);
  }

  return features;
}
