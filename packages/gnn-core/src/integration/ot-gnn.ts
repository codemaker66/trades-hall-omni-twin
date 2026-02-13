// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — Optimal Transport + GNN Integration (GNN-11)
// Wasserstein readout and Fused Gromov-Wasserstein distance for graph-level
// comparisons blending feature similarity with structural similarity.
// ---------------------------------------------------------------------------

import type { Graph, WassersteinReadoutConfig, WassersteinReadoutResult } from '../types.js';
import { getEdgeIndex } from '../graph.js';

// ---------------------------------------------------------------------------
// 1. wassersteinReadout — W_2 distances from node embeddings to prototypes
// ---------------------------------------------------------------------------

/**
 * Compute Wasserstein-2 distances between the empirical distribution of node
 * embeddings and each prototype distribution via Sinkhorn iterations.
 *
 * Algorithm (Sinkhorn–Knopp):
 * 1. Build a cost matrix C between each node embedding and each prototype
 *    using squared Euclidean distance.
 * 2. Form the Gibbs kernel K = exp(-C / epsilon).
 * 3. Iteratively update dual variables (u, v) to approximate the optimal
 *    transport coupling.
 * 4. The approximate Wasserstein distance is the Frobenius inner product
 *    <coupling, C>.
 *
 * @param nodeEmbeddings - Flat row-major (numNodes x embDim) node embeddings.
 * @param numNodes - Number of graph nodes.
 * @param embDim - Embedding dimensionality.
 * @param prototypes - Flat row-major (numPrototypes x embDim) prototype vectors.
 * @param numPrototypes - Number of prototypes.
 * @param config - Sinkhorn iteration parameters.
 * @returns WassersteinReadoutResult with per-prototype distances and transport plans.
 */
export function wassersteinReadout(
  nodeEmbeddings: Float64Array,
  numNodes: number,
  embDim: number,
  prototypes: Float64Array,
  numPrototypes: number,
  config: WassersteinReadoutConfig,
): WassersteinReadoutResult {
  const { sinkhornIterations, epsilon } = config;

  const distances = new Float64Array(numPrototypes);
  const transportPlans: Float64Array[] = [];

  // Uniform distribution over nodes
  const mu = new Float64Array(numNodes);
  const muVal = 1.0 / numNodes;
  for (let i = 0; i < numNodes; i++) {
    mu[i] = muVal;
  }

  // For each prototype, compute the OT distance from node empirical dist
  // to a single Dirac at the prototype (extended to numNodes copies).
  // We treat each prototype as a point mass and compute distance from the
  // uniform distribution over node embeddings to the prototype.
  // More usefully: treat prototypes as a second set of support points with
  // uniform weight. We compute OT(nodes-uniform, prototype-as-single-point)
  // for a graph-level feature vector.

  for (let p = 0; p < numPrototypes; p++) {
    // Cost matrix C: numNodes x 1 (distance from each node embedding to prototype p)
    // For a richer readout, we compute the Sinkhorn distance between the
    // uniform distribution over all node embeddings and a uniform distribution
    // over a single prototype (essentially the mean squared distance), but
    // the Sinkhorn formulation generalises to multi-prototype sets.

    // Cost matrix: numNodes x numPrototypes would give a single graph feature vector.
    // We compute per-prototype: C_ij = ||node_i - prototype_j||^2 for a single j=p.
    // Since the target has a single point, the transport plan is trivially mu.
    // Instead, compute the full numNodes x numPrototypes cost matrix once.

    // For simplicity with the Sinkhorn framework, compute the OT distance between
    // the node distribution (uniform over numNodes) and a uniform distribution
    // over numPrototypes as a single readout, then extract per-prototype distances.
    // But the spec asks for per-prototype distances, so we compute
    // W_2(empirical node dist, single prototype p) for each p.

    const costVec = new Float64Array(numNodes);
    for (let i = 0; i < numNodes; i++) {
      let sqDist = 0;
      for (let d = 0; d < embDim; d++) {
        const diff = nodeEmbeddings[i * embDim + d]! - prototypes[p * embDim + d]!;
        sqDist += diff * diff;
      }
      costVec[i] = sqDist;
    }

    // For a single target point, the Wasserstein distance is simply
    // sum_i mu_i * cost_i. But we still use the Sinkhorn framework
    // for consistency when extended to multi-point prototype sets.
    // Here: OT between mu (numNodes uniform) and nu = Dirac(prototype_p).
    // Coupling is T = mu (column vector) — trivial.
    // W_2 = sum_i mu_i * ||x_i - prototype_p||^2

    let dist = 0;
    for (let i = 0; i < numNodes; i++) {
      dist += mu[i]! * costVec[i]!;
    }
    distances[p] = Math.sqrt(Math.max(dist, 0));

    // Store transport plan (trivially mu for single-point target)
    const plan = new Float64Array(numNodes);
    plan.set(mu);
    transportPlans.push(plan);
  }

  // Now do a full Sinkhorn computation between nodes and prototypes as a
  // richer alternative — this gives a coupling matrix.
  // Overwrite distances with the Sinkhorn-based OT distances.

  // Full cost matrix: numNodes x numPrototypes
  const C = new Float64Array(numNodes * numPrototypes);
  for (let i = 0; i < numNodes; i++) {
    for (let j = 0; j < numPrototypes; j++) {
      let sqDist = 0;
      for (let d = 0; d < embDim; d++) {
        const diff = nodeEmbeddings[i * embDim + d]! - prototypes[j * embDim + d]!;
        sqDist += diff * diff;
      }
      C[i * numPrototypes + j] = sqDist;
    }
  }

  // Gibbs kernel K = exp(-C / epsilon)
  const K = new Float64Array(numNodes * numPrototypes);
  for (let idx = 0; idx < K.length; idx++) {
    K[idx] = Math.exp(-C[idx]! / epsilon);
  }

  // Source (nodes) and target (prototypes) marginals — uniform
  const nu = new Float64Array(numPrototypes);
  const nuVal = 1.0 / numPrototypes;
  for (let j = 0; j < numPrototypes; j++) {
    nu[j] = nuVal;
  }

  // Sinkhorn iterations
  const u = new Float64Array(numNodes);
  u.fill(1.0);
  const v = new Float64Array(numPrototypes);
  v.fill(1.0);

  for (let iter = 0; iter < sinkhornIterations; iter++) {
    // Update u: u = mu / (K * v)
    for (let i = 0; i < numNodes; i++) {
      let kv = 0;
      for (let j = 0; j < numPrototypes; j++) {
        kv += K[i * numPrototypes + j]! * v[j]!;
      }
      u[i] = kv > 1e-30 ? mu[i]! / kv : mu[i]!;
    }

    // Update v: v = nu / (K^T * u)
    for (let j = 0; j < numPrototypes; j++) {
      let ktu = 0;
      for (let i = 0; i < numNodes; i++) {
        ktu += K[i * numPrototypes + j]! * u[i]!;
      }
      v[j] = ktu > 1e-30 ? nu[j]! / ktu : nu[j]!;
    }
  }

  // Compute transport plan T = diag(u) * K * diag(v)
  const T = new Float64Array(numNodes * numPrototypes);
  for (let i = 0; i < numNodes; i++) {
    for (let j = 0; j < numPrototypes; j++) {
      T[i * numPrototypes + j] = u[i]! * K[i * numPrototypes + j]! * v[j]!;
    }
  }

  // Compute per-prototype Wasserstein distance: sum over i of T[i,j] * C[i,j]
  // then take sqrt for W_2
  for (let j = 0; j < numPrototypes; j++) {
    let d = 0;
    for (let i = 0; i < numNodes; i++) {
      d += T[i * numPrototypes + j]! * C[i * numPrototypes + j]!;
    }
    distances[j] = Math.sqrt(Math.max(d, 0));

    // Store per-prototype transport column
    const plan = new Float64Array(numNodes);
    for (let i = 0; i < numNodes; i++) {
      plan[i] = T[i * numPrototypes + j]!;
    }
    transportPlans[j] = plan;
  }

  return { distances, transportPlans };
}

// ---------------------------------------------------------------------------
// 2. fgwDistance — Fused Gromov-Wasserstein distance
// ---------------------------------------------------------------------------

/**
 * Compute the Fused Gromov-Wasserstein distance between two graphs.
 *
 * FGW blends:
 *   - Wasserstein (feature-based): sum T_ij * d(x_i, y_j)
 *   - Gromov-Wasserstein (structure-based): sum T_ij * T_kl * |d_X(i,k) - d_Y(j,l)|^2
 *
 * The objective is:
 *   min_T alpha * sum T_ij * c(x_i, y_j)
 *          + (1-alpha) * sum T_ij T_kl |D1(i,k) - D2(j,l)|^2
 *
 * where c is squared Euclidean distance on features, D1 and D2 are shortest-path
 * distance matrices (here approximated by adjacency-based hop distances).
 *
 * Uses Sinkhorn-like iterations on the coupling matrix.
 *
 * @param graph1 - First graph (CSR).
 * @param features1 - Node features for graph1, flat row-major (n1 x d).
 * @param graph2 - Second graph (CSR).
 * @param features2 - Node features for graph2, flat row-major (n2 x d).
 * @param alpha - Blending coefficient in [0,1]. alpha=1 => pure Wasserstein, alpha=0 => pure GW.
 * @param numIter - Number of Sinkhorn-like outer iterations.
 * @returns Scalar FGW distance.
 */
export function fgwDistance(
  graph1: Graph,
  features1: Float64Array,
  graph2: Graph,
  features2: Float64Array,
  alpha: number,
  numIter: number,
): number {
  const n1 = graph1.numNodes;
  const n2 = graph2.numNodes;
  const d = features1.length / n1;

  // Compute intra-graph distance matrices (shortest path via BFS)
  const D1 = computeShortestPathMatrix(graph1);
  const D2 = computeShortestPathMatrix(graph2);

  // Feature cost matrix: c(x_i, y_j) = ||x_i - y_j||^2
  const M = new Float64Array(n1 * n2);
  for (let i = 0; i < n1; i++) {
    for (let j = 0; j < n2; j++) {
      let sqDist = 0;
      for (let k = 0; k < d; k++) {
        const diff = features1[i * d + k]! - features2[j * d + k]!;
        sqDist += diff * diff;
      }
      M[i * n2 + j] = sqDist;
    }
  }

  // Uniform marginals
  const mu = 1.0 / n1;
  const nu = 1.0 / n2;

  // Initialize coupling T = mu * nu (outer product of uniform distributions)
  const T = new Float64Array(n1 * n2);
  for (let i = 0; i < n1; i++) {
    for (let j = 0; j < n2; j++) {
      T[i * n2 + j] = mu * nu;
    }
  }

  const epsilon = 0.01; // Sinkhorn regularization

  for (let iter = 0; iter < numIter; iter++) {
    // Compute the GW cost tensor contracted with T:
    // G_ij = sum_{k,l} T_kl * |D1(i,k) - D2(j,l)|^2
    const G = new Float64Array(n1 * n2);
    for (let i = 0; i < n1; i++) {
      for (let j = 0; j < n2; j++) {
        let gSum = 0;
        for (let k = 0; k < n1; k++) {
          for (let l = 0; l < n2; l++) {
            const dDiff = D1[i * n1 + k]! - D2[j * n2 + l]!;
            gSum += T[k * n2 + l]! * dDiff * dDiff;
          }
        }
        G[i * n2 + j] = gSum;
      }
    }

    // Combined cost: alpha * M + (1 - alpha) * G
    const cost = new Float64Array(n1 * n2);
    for (let idx = 0; idx < n1 * n2; idx++) {
      cost[idx] = alpha * M[idx]! + (1 - alpha) * G[idx]!;
    }

    // Sinkhorn step on the combined cost
    // K = exp(-cost / epsilon)
    const K = new Float64Array(n1 * n2);
    for (let idx = 0; idx < n1 * n2; idx++) {
      K[idx] = Math.exp(-cost[idx]! / epsilon);
    }

    const u = new Float64Array(n1);
    u.fill(1.0);
    const v = new Float64Array(n2);
    v.fill(1.0);

    const sinkhornIter = 50;
    for (let s = 0; s < sinkhornIter; s++) {
      // u = mu / (K * v)
      for (let i = 0; i < n1; i++) {
        let kv = 0;
        for (let j = 0; j < n2; j++) {
          kv += K[i * n2 + j]! * v[j]!;
        }
        u[i] = kv > 1e-30 ? mu / kv : mu;
      }
      // v = nu / (K^T * u)
      for (let j = 0; j < n2; j++) {
        let ktu = 0;
        for (let i = 0; i < n1; i++) {
          ktu += K[i * n2 + j]! * u[i]!;
        }
        v[j] = ktu > 1e-30 ? nu / ktu : nu;
      }
    }

    // Update coupling T = diag(u) * K * diag(v)
    for (let i = 0; i < n1; i++) {
      for (let j = 0; j < n2; j++) {
        T[i * n2 + j] = u[i]! * K[i * n2 + j]! * v[j]!;
      }
    }
  }

  // Compute final FGW distance
  // Wasserstein part
  let wDist = 0;
  for (let i = 0; i < n1; i++) {
    for (let j = 0; j < n2; j++) {
      wDist += T[i * n2 + j]! * M[i * n2 + j]!;
    }
  }

  // GW part
  let gwDist = 0;
  for (let i = 0; i < n1; i++) {
    for (let j = 0; j < n2; j++) {
      for (let k = 0; k < n1; k++) {
        for (let l = 0; l < n2; l++) {
          const dDiff = D1[i * n1 + k]! - D2[j * n2 + l]!;
          gwDist += T[i * n2 + j]! * T[k * n2 + l]! * dDiff * dDiff;
        }
      }
    }
  }

  return alpha * wDist + (1 - alpha) * gwDist;
}

// ---------------------------------------------------------------------------
// Helper: BFS shortest-path distance matrix
// ---------------------------------------------------------------------------

/**
 * Compute all-pairs shortest-path distances using BFS (unweighted).
 * Returns n x n flat row-major Float64Array.
 */
function computeShortestPathMatrix(graph: Graph): Float64Array {
  const n = graph.numNodes;
  const dist = new Float64Array(n * n);
  dist.fill(Infinity);

  for (let src = 0; src < n; src++) {
    // BFS from src
    dist[src * n + src] = 0;
    const queue: number[] = [src];
    let head = 0;

    while (head < queue.length) {
      const cur = queue[head]!;
      head++;
      const curDist = dist[src * n + cur]!;
      const start = graph.rowPtr[cur]!;
      const end = graph.rowPtr[cur + 1]!;

      for (let e = start; e < end; e++) {
        const neighbor = graph.colIdx[e]!;
        if (dist[src * n + neighbor]! === Infinity) {
          dist[src * n + neighbor] = curDist + 1;
          queue.push(neighbor);
        }
      }
    }

    // Replace remaining Infinity with a large finite value (disconnected nodes)
    for (let j = 0; j < n; j++) {
      if (dist[src * n + j]! === Infinity) {
        dist[src * n + j] = n; // diameter upper bound
      }
    }
  }

  return dist;
}
