// ---------------------------------------------------------------------------
// GNN-9 Scalable Inference — Embedding Search (FAISS-style ANN)
// ---------------------------------------------------------------------------
//
// Implements approximate nearest neighbor (ANN) search using an Inverted
// File Index (IVF) structure. Embeddings are partitioned into Voronoi cells
// via k-means clustering. At query time, only a few cells (nProbes) are
// searched, trading a small accuracy loss for significant speedup.
//
// Also provides exact brute-force k-NN for comparison and small datasets.
//
// References:
//   Johnson et al., "Billion-scale similarity search with GPUs"
//   (IEEE TDBG 2021) — FAISS
// ---------------------------------------------------------------------------

import type { PRNG, IVFIndex, ANNSearchResult } from '../types.js';
import { dot, l2Norm } from '../tensor.js';

// ---------------------------------------------------------------------------
// 1. buildIVFIndex — Build IVF index via k-means clustering
// ---------------------------------------------------------------------------

/**
 * Build an Inverted File Index (IVF) for approximate nearest neighbor search.
 *
 * Algorithm (Lloyd's k-means):
 * 1. Initialize nClusters centroids using k-means++ seeding.
 * 2. Iterate until convergence (or max iterations):
 *    a. Assign each embedding to its nearest centroid (L2 distance).
 *    b. Recompute centroids as the mean of assigned embeddings.
 * 3. Store the final centroids and per-embedding cluster assignments.
 *
 * @param embeddings - All embeddings, row-major (numEmbeddings * dim).
 * @param numEmbeddings - Number of embedding vectors.
 * @param dim - Dimensionality of each embedding.
 * @param nClusters - Number of Voronoi cells (clusters).
 * @param rng - Deterministic PRNG function.
 * @returns IVFIndex with centroids, assignments, and metadata.
 */
export function buildIVFIndex(
  embeddings: Float64Array,
  numEmbeddings: number,
  dim: number,
  nClusters: number,
  rng: PRNG,
): IVFIndex {
  // Clamp nClusters to not exceed the number of embeddings
  const k = Math.min(nClusters, numEmbeddings);

  // --- k-means++ initialization ---
  const centroids = new Float64Array(k * dim);

  // Pick first centroid uniformly at random
  const firstIdx = Math.floor(rng() * numEmbeddings);
  for (let d = 0; d < dim; d++) {
    centroids[d] = embeddings[firstIdx * dim + d]!;
  }

  // Distance from each point to its nearest chosen centroid
  const minDist = new Float64Array(numEmbeddings);
  minDist.fill(Infinity);

  for (let c = 1; c < k; c++) {
    // Update min distances to the newly added centroid (c-1)
    const prevCentroidOff = (c - 1) * dim;
    let totalDist = 0;

    for (let i = 0; i < numEmbeddings; i++) {
      let dist = 0;
      const embOff = i * dim;
      for (let d = 0; d < dim; d++) {
        const diff = embeddings[embOff + d]! - centroids[prevCentroidOff + d]!;
        dist += diff * diff;
      }
      if (dist < minDist[i]!) {
        minDist[i] = dist;
      }
      totalDist += minDist[i]!;
    }

    // Sample next centroid proportional to distance squared
    let r = rng() * totalDist;
    let chosenIdx = 0;
    for (let i = 0; i < numEmbeddings; i++) {
      r -= minDist[i]!;
      if (r <= 0) {
        chosenIdx = i;
        break;
      }
    }

    // Copy the chosen embedding as the next centroid
    const chosenOff = chosenIdx * dim;
    const centroidOff = c * dim;
    for (let d = 0; d < dim; d++) {
      centroids[centroidOff + d] = embeddings[chosenOff + d]!;
    }
  }

  // --- Lloyd's k-means iterations ---
  const assignments = new Uint32Array(numEmbeddings);
  const maxIter = 50;

  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false;

    // Assignment step: assign each embedding to nearest centroid
    for (let i = 0; i < numEmbeddings; i++) {
      const embOff = i * dim;
      let bestCluster = 0;
      let bestDist = Infinity;

      for (let c = 0; c < k; c++) {
        const centOff = c * dim;
        let dist = 0;
        for (let d = 0; d < dim; d++) {
          const diff = embeddings[embOff + d]! - centroids[centOff + d]!;
          dist += diff * diff;
        }
        if (dist < bestDist) {
          bestDist = dist;
          bestCluster = c;
        }
      }

      if (assignments[i] !== bestCluster) {
        assignments[i] = bestCluster;
        changed = true;
      }
    }

    if (!changed) break;

    // Update step: recompute centroids
    const sums = new Float64Array(k * dim);
    const counts = new Uint32Array(k);

    for (let i = 0; i < numEmbeddings; i++) {
      const cluster = assignments[i]!;
      counts[cluster]!++;
      const embOff = i * dim;
      const sumOff = cluster * dim;
      for (let d = 0; d < dim; d++) {
        sums[sumOff + d] = sums[sumOff + d]! + embeddings[embOff + d]!;
      }
    }

    for (let c = 0; c < k; c++) {
      const cnt = counts[c]!;
      if (cnt > 0) {
        const off = c * dim;
        for (let d = 0; d < dim; d++) {
          centroids[off + d] = sums[off + d]! / cnt;
        }
      }
      // If a cluster is empty, leave its centroid as is (from previous iteration)
    }
  }

  return {
    centroids,
    assignments,
    dim,
    nClusters: k,
    embeddings,
    numEmbeddings,
  };
}

// ---------------------------------------------------------------------------
// 2. searchIVF — Approximate nearest neighbor search using IVF index
// ---------------------------------------------------------------------------

/**
 * Search for the k nearest embeddings to a query using the IVF index.
 *
 * Algorithm:
 * 1. Compute L2 distance from the query to all centroids.
 * 2. Select the nProbes nearest centroids.
 * 3. Within those clusters, compute exact L2 distance to each embedding.
 * 4. Return the top-k nearest embeddings across all probed clusters.
 *
 * @param index - Pre-built IVF index.
 * @param query - Query embedding (length = dim).
 * @param k - Number of nearest neighbors to return.
 * @param nProbes - Number of Voronoi cells to search.
 * @returns ANNSearchResult with indices and distances, sorted by distance.
 */
export function searchIVF(
  index: IVFIndex,
  query: Float64Array,
  k: number,
  nProbes: number,
): ANNSearchResult {
  const { centroids, assignments, dim, nClusters, embeddings, numEmbeddings } = index;
  const probes = Math.min(nProbes, nClusters);

  // Step 1: Find the nProbes nearest centroids
  const centroidDists: { cluster: number; dist: number }[] = [];
  for (let c = 0; c < nClusters; c++) {
    let dist = 0;
    const centOff = c * dim;
    for (let d = 0; d < dim; d++) {
      const diff = query[d]! - centroids[centOff + d]!;
      dist += diff * diff;
    }
    centroidDists.push({ cluster: c, dist });
  }
  centroidDists.sort((a, b) => a.dist - b.dist);

  // Collect the probed cluster IDs
  const probedClusters = new Set<number>();
  for (let p = 0; p < probes; p++) {
    probedClusters.add(centroidDists[p]!.cluster);
  }

  // Step 2: Search within probed clusters for nearest embeddings
  // Use a max-heap (sorted array) to track top-k candidates
  const candidates: { idx: number; dist: number }[] = [];

  for (let i = 0; i < numEmbeddings; i++) {
    if (!probedClusters.has(assignments[i]!)) continue;

    let dist = 0;
    const embOff = i * dim;
    for (let d = 0; d < dim; d++) {
      const diff = query[d]! - embeddings[embOff + d]!;
      dist += diff * diff;
    }

    if (candidates.length < k) {
      candidates.push({ idx: i, dist });
      // Keep sorted descending by distance (worst at front for easy removal)
      candidates.sort((a, b) => b.dist - a.dist);
    } else if (dist < candidates[0]!.dist) {
      candidates[0] = { idx: i, dist };
      candidates.sort((a, b) => b.dist - a.dist);
    }
  }

  // Sort ascending by distance for output
  candidates.sort((a, b) => a.dist - b.dist);

  const resultK = candidates.length;
  const indices = new Uint32Array(resultK);
  const distances = new Float64Array(resultK);

  for (let i = 0; i < resultK; i++) {
    indices[i] = candidates[i]!.idx;
    distances[i] = Math.sqrt(candidates[i]!.dist); // Return L2 distance, not squared
  }

  return { indices, distances };
}

// ---------------------------------------------------------------------------
// 3. exactKNN — Brute-force exact k-nearest neighbor search
// ---------------------------------------------------------------------------

/**
 * Brute-force exact k-nearest neighbor search for comparison or small datasets.
 *
 * Algorithm:
 * 1. For every embedding, compute L2 distance to the query.
 * 2. Maintain a sorted list of the k smallest distances seen so far.
 * 3. Return the k nearest indices and their distances.
 *
 * Time complexity: O(numEmbeddings * dim + numEmbeddings * log k)
 *
 * @param embeddings - All embeddings, row-major (numEmbeddings * dim).
 * @param numEmbeddings - Number of embedding vectors.
 * @param dim - Dimensionality of each embedding.
 * @param query - Query embedding (length = dim).
 * @param k - Number of nearest neighbors to return.
 * @returns ANNSearchResult with indices and distances, sorted by distance.
 */
export function exactKNN(
  embeddings: Float64Array,
  numEmbeddings: number,
  dim: number,
  query: Float64Array,
  k: number,
): ANNSearchResult {
  const resultK = Math.min(k, numEmbeddings);

  // Maintain a max-heap of size k (sorted descending by dist for easy eviction)
  const heap: { idx: number; dist: number }[] = [];

  for (let i = 0; i < numEmbeddings; i++) {
    let distSq = 0;
    const embOff = i * dim;
    for (let d = 0; d < dim; d++) {
      const diff = query[d]! - embeddings[embOff + d]!;
      distSq += diff * diff;
    }

    if (heap.length < resultK) {
      heap.push({ idx: i, dist: distSq });
      // Keep sorted descending so heap[0] is the worst (largest distance)
      heap.sort((a, b) => b.dist - a.dist);
    } else if (distSq < heap[0]!.dist) {
      heap[0] = { idx: i, dist: distSq };
      heap.sort((a, b) => b.dist - a.dist);
    }
  }

  // Sort ascending by distance for output
  heap.sort((a, b) => a.dist - b.dist);

  const indices = new Uint32Array(heap.length);
  const distances = new Float64Array(heap.length);

  for (let i = 0; i < heap.length; i++) {
    indices[i] = heap[i]!.idx;
    distances[i] = Math.sqrt(heap[i]!.dist); // Return L2 distance, not squared
  }

  return { indices, distances };
}
