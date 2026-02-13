// ---------------------------------------------------------------------------
// GNN-9 Scalable Inference — Cluster-GCN (METIS-style Graph Partitioning)
// ---------------------------------------------------------------------------
//
// Implements graph partitioning for Cluster-GCN training, where the graph is
// split into disjoint clusters and each training step operates on the induced
// subgraph of one (or a few) clusters. This avoids neighborhood explosion
// by restricting message passing to within-cluster edges.
//
// References:
//   Chiang et al., "Cluster-GCN: An Efficient Algorithm for Training Deep
//   and Large Graph Convolutional Networks" (KDD 2019)
// ---------------------------------------------------------------------------

import type { Graph, PRNG, ClusterPartition } from '../types.js';
import { graphLaplacian, subgraph, buildCSR } from '../graph.js';

// ---------------------------------------------------------------------------
// 1. spectralBipartition — Binary partition using Fiedler vector
// ---------------------------------------------------------------------------

/**
 * Binary partition of a graph using the Fiedler vector (2nd smallest
 * eigenvector of the graph Laplacian).
 *
 * Algorithm:
 * 1. Compute the dense graph Laplacian L = D - A.
 * 2. Use power iteration on the shifted matrix M = lambda_max * I - L
 *    with deflation against the trivial all-ones eigenvector to find the
 *    eigenvector corresponding to the 2nd smallest eigenvalue (Fiedler vector).
 * 3. Assign node i to partition 0 if fiedler[i] >= 0, else partition 1.
 *
 * @param graph - Input CSR Graph.
 * @param rng - Deterministic PRNG function for initial vector randomization.
 * @returns Uint32Array of length numNodes with values 0 or 1.
 */
export function spectralBipartition(
  graph: Graph,
  rng: PRNG,
): Uint32Array {
  const n = graph.numNodes;
  const assignment = new Uint32Array(n);

  if (n <= 1) return assignment;

  // Compute dense Laplacian L = D - A
  const L = graphLaplacian(graph);

  // Estimate max eigenvalue via Gershgorin bound
  let maxLambda = 0;
  for (let i = 0; i < n; i++) {
    let rowSum = 0;
    for (let j = 0; j < n; j++) {
      rowSum += Math.abs(L[i * n + j]!);
    }
    if (rowSum > maxLambda) maxLambda = rowSum;
  }

  // Avoid degenerate case (single connected component with no edges)
  if (maxLambda < 1e-12) {
    // No structure to partition; assign first half to 0, rest to 1
    for (let i = 0; i < n; i++) {
      assignment[i] = i < Math.ceil(n / 2) ? 0 : 1;
    }
    return assignment;
  }

  // Shifted matrix M = maxLambda * I - L
  // Top eigenvector of M = bottom non-trivial eigenvector of L (Fiedler)
  const M = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      M[i * n + j] = -L[i * n + j]!;
    }
    M[i * n + i] = M[i * n + i]! + maxLambda;
  }

  // Random initial vector
  const v = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    v[i] = rng() - 0.5;
  }

  const oneOverSqrtN = 1.0 / Math.sqrt(n);

  // Deflate: remove component along the trivial eigenvector (constant vector)
  function deflate(vec: Float64Array): void {
    let d = 0;
    for (let i = 0; i < n; i++) {
      d += vec[i]! * oneOverSqrtN;
    }
    for (let i = 0; i < n; i++) {
      vec[i] = vec[i]! - d * oneOverSqrtN;
    }
  }

  // Normalize to unit length
  function normalizeVec(vec: Float64Array): void {
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
  }

  deflate(v);
  normalizeVec(v);

  const tmp = new Float64Array(n);
  const maxIter = 300;

  // Power iteration
  for (let iter = 0; iter < maxIter; iter++) {
    // tmp = M * v
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) {
        s += M[i * n + j]! * v[j]!;
      }
      tmp[i] = s;
    }
    deflate(tmp);
    normalizeVec(tmp);
    v.set(tmp);
  }

  // Partition by sign of the Fiedler vector
  for (let i = 0; i < n; i++) {
    assignment[i] = v[i]! >= 0 ? 0 : 1;
  }

  return assignment;
}

// ---------------------------------------------------------------------------
// 2. recursivePartition — Recursive bisection to get k clusters
// ---------------------------------------------------------------------------

/**
 * Recursively bisect the graph to produce approximately k clusters.
 *
 * Algorithm:
 * 1. Start with all nodes in a single partition.
 * 2. Pick the largest partition and split it using spectralBipartition.
 * 3. Repeat until we have k partitions (or all partitions are singletons).
 * 4. Compute cluster sizes and return the partition assignment.
 *
 * For non-power-of-2 k, the algorithm still produces k clusters by always
 * splitting the largest cluster next.
 *
 * @param graph - Input CSR Graph.
 * @param k - Desired number of clusters.
 * @param rng - Deterministic PRNG function.
 * @returns ClusterPartition with assignment, numClusters, and clusterSizes.
 */
export function recursivePartition(
  graph: Graph,
  k: number,
  rng: PRNG,
): ClusterPartition {
  const n = graph.numNodes;
  const assignment = new Uint32Array(n);

  if (k <= 1 || n <= 1) {
    return {
      assignment,
      numClusters: n === 0 ? 0 : 1,
      clusterSizes: n === 0 ? new Uint32Array(0) : new Uint32Array([n]),
    };
  }

  // Track partitions as arrays of original node indices
  // Each partition gets a cluster ID
  let partitions: number[][] = [[]];
  for (let i = 0; i < n; i++) {
    partitions[0]!.push(i);
  }

  let numClusters = 1;

  while (numClusters < k) {
    // Find the largest partition to split
    let largestIdx = 0;
    let largestSize = partitions[0]!.length;
    for (let p = 1; p < partitions.length; p++) {
      if (partitions[p]!.length > largestSize) {
        largestSize = partitions[p]!.length;
        largestIdx = p;
      }
    }

    // If the largest partition has only 1 node, we cannot split further
    if (largestSize <= 1) break;

    const partitionNodes = partitions[largestIdx]!;

    // Build induced subgraph for this partition
    const nodeSet = new Set(partitionNodes);
    const sub = subgraph(graph, nodeSet);

    // Perform spectral bipartition on the subgraph
    const bipartition = spectralBipartition(sub, rng);

    // Split the partition into two groups
    // The subgraph nodes are in sorted order of the original IDs
    const sortedNodes = Array.from(nodeSet).sort((a, b) => a - b);
    const group0: number[] = [];
    const group1: number[] = [];

    for (let i = 0; i < sortedNodes.length; i++) {
      if (bipartition[i]! === 0) {
        group0.push(sortedNodes[i]!);
      } else {
        group1.push(sortedNodes[i]!);
      }
    }

    // Handle degenerate case: if one group is empty, force a split
    if (group0.length === 0 || group1.length === 0) {
      const half = Math.ceil(sortedNodes.length / 2);
      group0.length = 0;
      group1.length = 0;
      for (let i = 0; i < sortedNodes.length; i++) {
        if (i < half) {
          group0.push(sortedNodes[i]!);
        } else {
          group1.push(sortedNodes[i]!);
        }
      }
    }

    // Replace the largest partition with the two new groups
    partitions[largestIdx] = group0;
    partitions.push(group1);
    numClusters++;
  }

  // Build assignment array and cluster sizes
  const clusterSizes = new Uint32Array(numClusters);
  for (let c = 0; c < partitions.length; c++) {
    const nodes = partitions[c]!;
    clusterSizes[c] = nodes.length;
    for (const node of nodes) {
      assignment[node] = c;
    }
  }

  return {
    assignment,
    numClusters,
    clusterSizes,
  };
}

// ---------------------------------------------------------------------------
// 3. getClusterSubgraph — Extract subgraph for a single cluster
// ---------------------------------------------------------------------------

/**
 * Extract the induced subgraph containing only nodes in the specified cluster.
 *
 * Algorithm:
 * 1. Identify all nodes assigned to `clusterId` from the partition.
 * 2. Build a node set and extract the induced subgraph using graph.subgraph().
 * 3. The returned graph has contiguous node indices [0, clusterSize).
 *
 * @param graph - Input CSR Graph.
 * @param partition - ClusterPartition from recursivePartition.
 * @param clusterId - The cluster ID to extract.
 * @returns A new Graph containing only the nodes and edges of the cluster.
 */
export function getClusterSubgraph(
  graph: Graph,
  partition: ClusterPartition,
  clusterId: number,
): Graph {
  const nodeSet = new Set<number>();

  for (let i = 0; i < partition.assignment.length; i++) {
    if (partition.assignment[i]! === clusterId) {
      nodeSet.add(i);
    }
  }

  return subgraph(graph, nodeSet);
}
