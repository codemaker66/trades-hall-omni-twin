// ---------------------------------------------------------------------------
// SP-5: Spectral Clustering of Venues by Demand Frequency Profile
// ---------------------------------------------------------------------------
// 1. Compute normalized PSD for each venue
// 2. Build similarity: W_ij = exp(-||PSD_i - PSD_j||²/2σ²)
// 3. Graph Laplacian eigenvectors
// 4. k-means on eigenvectors
// Clusters: "weekend-dominant", "holiday-driven", "corporate-weekday", etc.

import type { SpectralCluster, WelchResult } from '../types.js';
import { welchPSD } from '../fourier/welch.js';

/**
 * Compute normalized PSD for a set of venue time series.
 * Normalization: PSD / sum(PSD) → probability distribution over frequencies.
 */
export function computeNormalizedPSDs(
  signals: Float64Array[],
  fs: number = 1,
  nperseg: number = 128,
): WelchResult[] {
  return signals.map(sig => {
    const result = welchPSD(sig, fs, Math.min(nperseg, sig.length));
    // Normalize PSD to sum to 1
    let total = 0;
    for (let i = 0; i < result.psd.length; i++) total += result.psd[i]!;
    if (total > 0) {
      for (let i = 0; i < result.psd.length; i++) result.psd[i] = result.psd[i]! / total;
    }
    return result;
  });
}

/**
 * Build Gaussian similarity matrix from PSD profiles.
 * W_ij = exp(-||PSD_i - PSD_j||²/2σ²)
 */
function buildSimilarityMatrix(psds: Float64Array[], sigma: number): Float64Array {
  const n = psds.length;
  const W = new Float64Array(n * n);

  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      if (i === j) {
        W[i * n + j] = 1;
        continue;
      }
      let dist2 = 0;
      const len = Math.min(psds[i]!.length, psds[j]!.length);
      for (let k = 0; k < len; k++) {
        const diff = psds[i]![k]! - psds[j]![k]!;
        dist2 += diff * diff;
      }
      const sim = Math.exp(-dist2 / (2 * sigma * sigma));
      W[i * n + j] = sim;
      W[j * n + i] = sim;
    }
  }

  return W;
}

/**
 * Simple k-means clustering on low-dimensional vectors.
 */
function kMeans(
  vectors: Float64Array[],
  k: number,
  maxIter: number = 100,
): number[] {
  const n = vectors.length;
  const d = vectors[0]?.length ?? 0;
  if (n === 0 || d === 0) return [];

  // Initialize centroids via k-means++ (simplified: evenly spaced)
  const assignments = new Array<number>(n).fill(0);
  const centroids: Float64Array[] = [];
  const step = Math.max(1, Math.floor(n / k));
  for (let i = 0; i < k; i++) {
    centroids.push(new Float64Array(vectors[Math.min(i * step, n - 1)]!));
  }

  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false;

    // Assign to nearest centroid
    for (let i = 0; i < n; i++) {
      let bestCluster = 0;
      let bestDist = Infinity;
      for (let c = 0; c < k; c++) {
        let dist = 0;
        for (let j = 0; j < d; j++) {
          const diff = vectors[i]![j]! - centroids[c]![j]!;
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

    // Recompute centroids
    for (let c = 0; c < k; c++) {
      const newCentroid = new Float64Array(d);
      let count = 0;
      for (let i = 0; i < n; i++) {
        if (assignments[i] === c) {
          for (let j = 0; j < d; j++) newCentroid[j] = newCentroid[j]! + vectors[i]![j]!;
          count++;
        }
      }
      if (count > 0) {
        for (let j = 0; j < d; j++) newCentroid[j] = newCentroid[j]! / count;
        centroids[c] = newCentroid;
      }
    }
  }

  return assignments;
}

/**
 * Power iteration to find top-k eigenvectors of a symmetric matrix.
 * Simplified for the graph Laplacian use case.
 */
function topEigenvectors(L: Float64Array, n: number, k: number, maxIter: number = 200): Float64Array[] {
  const eigvecs: Float64Array[] = [];

  for (let ek = 0; ek < k; ek++) {
    // Random initial vector
    let v = new Float64Array(n);
    for (let i = 0; i < n; i++) v[i] = Math.random() - 0.5;

    for (let iter = 0; iter < maxIter; iter++) {
      // Matrix-vector multiply
      const Av = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < n; j++) {
          sum += L[i * n + j]! * v[j]!;
        }
        Av[i] = sum;
      }

      // Deflate against previous eigenvectors (Gram-Schmidt)
      for (const prev of eigvecs) {
        let dot = 0;
        for (let i = 0; i < n; i++) dot += Av[i]! * prev[i]!;
        for (let i = 0; i < n; i++) Av[i] = Av[i]! - dot * prev[i]!;
      }

      // Normalize
      let norm = 0;
      for (let i = 0; i < n; i++) norm += Av[i]! * Av[i]!;
      norm = Math.sqrt(norm);
      if (norm < 1e-14) break;
      for (let i = 0; i < n; i++) Av[i] = Av[i]! / norm;

      v = Av;
    }

    eigvecs.push(v);
  }

  return eigvecs;
}

/**
 * Spectral clustering of venues by their demand frequency profiles.
 */
export function spectralClusterVenues(
  signals: Float64Array[],
  nClusters: number = 4,
  fs: number = 1,
  sigma: number = 0.1,
): SpectralCluster[] {
  const n = signals.length;
  if (n <= nClusters) {
    // Degenerate case: each venue is its own cluster
    return signals.map((_, i) => ({
      clusterId: i,
      label: `cluster-${i}`,
      memberIndices: [i],
      centroidPSD: new Float64Array(0),
    }));
  }

  // Step 1: Compute normalized PSDs
  const psdResults = computeNormalizedPSDs(signals, fs);
  const psds = psdResults.map(r => r.psd);

  // Step 2: Build similarity matrix
  const W = buildSimilarityMatrix(psds, sigma);

  // Step 3: Compute normalized graph Laplacian
  // D = diag(sum of rows), L = D^(-1/2) · W · D^(-1/2)
  const D = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) sum += W[i * n + j]!;
    D[i] = sum;
  }

  const Lnorm = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const dI = D[i]! > 0 ? 1 / Math.sqrt(D[i]!) : 0;
      const dJ = D[j]! > 0 ? 1 / Math.sqrt(D[j]!) : 0;
      Lnorm[i * n + j] = dI * W[i * n + j]! * dJ;
    }
  }

  // Step 4: Top-k eigenvectors
  const eigvecs = topEigenvectors(Lnorm, n, nClusters);

  // Build embedding vectors (each venue → k-dimensional vector)
  const embeddings: Float64Array[] = [];
  for (let i = 0; i < n; i++) {
    const vec = new Float64Array(nClusters);
    for (let k = 0; k < nClusters; k++) {
      vec[k] = eigvecs[k]?.[i] ?? 0;
    }
    // Row-normalize
    let norm = 0;
    for (let k = 0; k < nClusters; k++) norm += vec[k]! * vec[k]!;
    norm = Math.sqrt(norm);
    if (norm > 1e-14) {
      for (let k = 0; k < nClusters; k++) vec[k] = vec[k]! / norm;
    }
    embeddings.push(vec);
  }

  // Step 5: k-means on embeddings
  const assignments = kMeans(embeddings, nClusters);

  // Build cluster result
  const clusterLabels = [
    'weekend-dominant', 'corporate-weekday', 'holiday-driven', 'seasonal-mixed',
    'cluster-4', 'cluster-5', 'cluster-6', 'cluster-7',
  ];

  const clusters: SpectralCluster[] = [];
  for (let c = 0; c < nClusters; c++) {
    const members = assignments
      .map((a, i) => (a === c ? i : -1))
      .filter(i => i >= 0);

    // Compute centroid PSD
    const psdLen = psds[0]?.length ?? 0;
    const centroid = new Float64Array(psdLen);
    if (members.length > 0) {
      for (const m of members) {
        const psd = psds[m]!;
        for (let k = 0; k < psdLen; k++) centroid[k] = centroid[k]! + psd[k]!;
      }
      for (let k = 0; k < psdLen; k++) centroid[k] = centroid[k]! / members.length;
    }

    clusters.push({
      clusterId: c,
      label: clusterLabels[c] ?? `cluster-${c}`,
      memberIndices: members,
      centroidPSD: centroid,
    });
  }

  return clusters;
}
