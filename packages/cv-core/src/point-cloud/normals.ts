// ---------------------------------------------------------------------------
// CV-8: Normal Estimation — PCA-based per-point normal estimation and
// consistent normal orientation via propagation.
// ---------------------------------------------------------------------------

import type { PointCloud } from '../types.js';

// ---------------------------------------------------------------------------
// estimateNormals
// ---------------------------------------------------------------------------

/**
 * Estimate per-point normals using PCA on the k-nearest-neighbour
 * neighbourhood of each point.
 *
 * For each point:
 * 1. Find the k nearest neighbours (brute-force).
 * 2. Compute the 3x3 covariance matrix of the neighbourhood.
 * 3. The normal is the eigenvector corresponding to the smallest eigenvalue,
 *    computed via the analytic eigendecomposition of a symmetric 3x3 matrix.
 *
 * The returned array is packed as [nx0, ny0, nz0, nx1, ny1, nz1, ...] with
 * length `cloud.count * 3`.  Normals are unit-length but their sign
 * (orientation) is arbitrary; use {@link orientNormals} to make them
 * consistent.
 *
 * @param cloud      Input point cloud.
 * @param kNeighbors Number of nearest neighbours for local PCA.
 * @returns          Float64Array of packed per-point normals.
 */
export function estimateNormals(
  cloud: PointCloud,
  kNeighbors: number,
): Float64Array {
  const n = cloud.count;
  const pos = cloud.positions;
  const k = Math.min(kNeighbors, n - 1);
  const normals = new Float64Array(n * 3);

  if (n < 3 || k < 2) {
    // Not enough points for meaningful normal estimation.
    // Return zero normals.
    return normals;
  }

  // Temporary arrays reused per point.
  const dists = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const pi = i * 3;
    const px = pos[pi]!;
    const py = pos[pi + 1]!;
    const pz = pos[pi + 2]!;

    // --- Step 1: Find k nearest neighbours (brute-force) ---
    for (let j = 0; j < n; j++) {
      if (j === i) {
        dists[j] = Infinity;
        continue;
      }
      const qj = j * 3;
      const dx = px - pos[qj]!;
      const dy = py - pos[qj + 1]!;
      const dz = pz - pos[qj + 2]!;
      dists[j] = dx * dx + dy * dy + dz * dz;
    }

    // Get indices of the k smallest distances.
    const knnIndices = kSmallestIndices(dists, k, n);

    // --- Step 2: Compute covariance of the neighbourhood ---
    // Compute centroid of neighbours.
    let cx = 0, cy = 0, cz = 0;
    for (let ki = 0; ki < k; ki++) {
      const ni = knnIndices[ki]! * 3;
      cx += pos[ni]!;
      cy += pos[ni + 1]!;
      cz += pos[ni + 2]!;
    }
    cx /= k;
    cy /= k;
    cz /= k;

    // Build 3x3 covariance (symmetric).
    let c00 = 0, c01 = 0, c02 = 0;
    let c11 = 0, c12 = 0, c22 = 0;

    for (let ki = 0; ki < k; ki++) {
      const ni = knnIndices[ki]! * 3;
      const dx = pos[ni]! - cx;
      const dy = pos[ni + 1]! - cy;
      const dz = pos[ni + 2]! - cz;
      c00 += dx * dx;
      c01 += dx * dy;
      c02 += dx * dz;
      c11 += dy * dy;
      c12 += dy * dz;
      c22 += dz * dz;
    }

    // --- Step 3: Smallest eigenvector of the symmetric 3x3 covariance ---
    const normal = smallestEigenvector3x3(c00, c01, c02, c11, c12, c22);

    normals[pi] = normal[0]!;
    normals[pi + 1] = normal[1]!;
    normals[pi + 2] = normal[2]!;
  }

  return normals;
}

// ---------------------------------------------------------------------------
// orientNormals
// ---------------------------------------------------------------------------

/**
 * Orient normals consistently across the point cloud using a propagation
 * approach.
 *
 * Starting from a seed point (defaults to the point with the largest z,
 * assuming the scanner is above the scene), propagate orientation to
 * neighbours: if a neighbour's normal has a negative dot product with the
 * current point's normal, flip it.
 *
 * Uses a BFS traversal over a k-NN adjacency graph (k is derived from the
 * normals array length / cloud count, capped at a reasonable maximum).
 *
 * @param cloud     Input point cloud.
 * @param normals   Packed per-point normals (length = cloud.count * 3).
 * @param seedIndex Optional starting point index.  Defaults to the point
 *                  with the maximum z coordinate.
 * @returns         A new Float64Array with consistently oriented normals.
 */
export function orientNormals(
  cloud: PointCloud,
  normals: Float64Array,
  seedIndex?: number,
): Float64Array {
  const n = cloud.count;
  const pos = cloud.positions;
  const out = Float64Array.from(normals);

  if (n < 2) return out;

  // Determine the seed.  Default: point with max z.
  let seed = seedIndex ?? 0;
  if (seedIndex === undefined) {
    let maxZ = -Infinity;
    for (let i = 0; i < n; i++) {
      const z = pos[i * 3 + 2]!;
      if (z > maxZ) {
        maxZ = z;
        seed = i;
      }
    }
  }

  // Ensure the seed normal points "up" (positive z).
  // If the seed's normal z-component is negative, flip it.
  if (out[seed * 3 + 2]! < 0) {
    out[seed * 3] = -out[seed * 3]!;
    out[seed * 3 + 1] = -out[seed * 3 + 1]!;
    out[seed * 3 + 2] = -out[seed * 3 + 2]!;
  }

  // BFS propagation.
  const visited = new Uint8Array(n);
  const queue: number[] = [seed];
  visited[seed] = 1;

  // Number of neighbours to consider per point for propagation.
  const kProp = Math.min(10, n - 1);

  const dists = new Float64Array(n);

  while (queue.length > 0) {
    const cur = queue.shift()!;
    const ci = cur * 3;
    const cnx = out[ci]!;
    const cny = out[ci + 1]!;
    const cnz = out[ci + 2]!;

    const cpx = pos[ci]!;
    const cpy = pos[ci + 1]!;
    const cpz = pos[ci + 2]!;

    // Find k nearest unvisited neighbours.
    for (let j = 0; j < n; j++) {
      if (j === cur) {
        dists[j] = Infinity;
        continue;
      }
      const qj = j * 3;
      const dx = cpx - pos[qj]!;
      const dy = cpy - pos[qj + 1]!;
      const dz = cpz - pos[qj + 2]!;
      dists[j] = dx * dx + dy * dy + dz * dz;
    }

    const knn = kSmallestIndices(dists, kProp, n);

    for (let ki = 0; ki < knn.length; ki++) {
      const ni = knn[ki]!;
      if (visited[ni]) continue;
      visited[ni] = 1;

      // If the neighbour's normal points away from the current normal, flip it.
      const nii = ni * 3;
      const dot =
        cnx * out[nii]! + cny * out[nii + 1]! + cnz * out[nii + 2]!;

      if (dot < 0) {
        out[nii] = -out[nii]!;
        out[nii + 1] = -out[nii + 1]!;
        out[nii + 2] = -out[nii + 2]!;
      }

      queue.push(ni);
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Return the indices of the k smallest values in an array.
 * Uses a simple partial-sort approach.
 */
function kSmallestIndices(
  arr: Float64Array,
  k: number,
  n: number,
): Uint32Array {
  // Build index-value pairs and partial sort.
  const pairs: Array<{ idx: number; val: number }> = [];
  for (let i = 0; i < n; i++) {
    pairs.push({ idx: i, val: arr[i]! });
  }
  pairs.sort((a, b) => a.val - b.val);

  const result = new Uint32Array(k);
  for (let i = 0; i < k; i++) {
    result[i] = pairs[i]!.idx;
  }
  return result;
}

/**
 * Compute the eigenvector corresponding to the smallest eigenvalue of a
 * real symmetric 3x3 matrix.
 *
 * Uses the analytic formula for eigenvalues of a 3x3 symmetric matrix
 * (Cardano's method) then extracts the corresponding eigenvector via
 * the cofactor method.
 *
 * Matrix:  [a, b, c]
 *          [b, d, e]
 *          [c, e, f]
 *
 * @returns [nx, ny, nz] unit eigenvector.
 */
function smallestEigenvector3x3(
  a: number,
  b: number,
  c: number,
  d: number,
  e: number,
  f: number,
): Float64Array {
  // Eigenvalues via Cardano's method for the characteristic polynomial
  // det(M - lambda I) = 0.
  //
  // p1 = b^2 + c^2 + e^2
  const p1 = b * b + c * c + e * e;

  if (p1 < 1e-30) {
    // Matrix is diagonal.  Eigenvalues are a, d, f.
    const eigs = [a, d, f];
    let minIdx = 0;
    if (eigs[1]! < eigs[0]!) minIdx = 1;
    if (eigs[2]! < eigs[minIdx]!) minIdx = 2;

    const result = new Float64Array(3);
    result[minIdx] = 1;
    return result;
  }

  const q = (a + d + f) / 3; // trace / 3
  const p2 =
    (a - q) * (a - q) +
    (d - q) * (d - q) +
    (f - q) * (f - q) +
    2 * p1;
  const p = Math.sqrt(p2 / 6);

  // B = (1/p) * (M - q*I)
  const inv_p = 1 / p;
  const b00 = (a - q) * inv_p;
  const b01 = b * inv_p;
  const b02 = c * inv_p;
  const b11 = (d - q) * inv_p;
  const b12 = e * inv_p;
  const b22 = (f - q) * inv_p;

  // det(B) = b00*(b11*b22 - b12^2) - b01*(b01*b22 - b12*b02) + b02*(b01*b12 - b11*b02)
  const detB =
    b00 * (b11 * b22 - b12 * b12) -
    b01 * (b01 * b22 - b12 * b02) +
    b02 * (b01 * b12 - b11 * b02);

  // r = det(B) / 2, clamped to [-1, 1].
  let r = detB / 2;
  r = Math.max(-1, Math.min(1, r));

  const phi = Math.acos(r) / 3;

  // Eigenvalues (sorted: eig0 >= eig1 >= eig2).
  const eig0 = q + 2 * p * Math.cos(phi);
  const eig2 = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3);
  // const eig1 = 3 * q - eig0 - eig2; // not needed

  // We want the eigenvector for eig2 (the smallest eigenvalue).
  // Compute (M - eig2 * I) and find its null space via cross product of two rows.
  const m00 = a - eig2;
  const m01 = b;
  const m02 = c;
  const m10 = b;
  const m11 = d - eig2;
  const m12 = e;
  const m20 = c;
  const m21 = e;
  const m22 = f - eig2;

  // Cross product of row 0 and row 1.
  let nx = m01 * m12 - m02 * m11;
  let ny = m02 * m10 - m00 * m12;
  let nz = m00 * m11 - m01 * m10;
  let len = Math.sqrt(nx * nx + ny * ny + nz * nz);

  if (len < 1e-15) {
    // Try cross product of row 0 and row 2.
    nx = m01 * m22 - m02 * m21;
    ny = m02 * m20 - m00 * m22;
    nz = m00 * m21 - m01 * m20;
    len = Math.sqrt(nx * nx + ny * ny + nz * nz);
  }

  if (len < 1e-15) {
    // Try cross product of row 1 and row 2.
    nx = m11 * m22 - m12 * m21;
    ny = m12 * m20 - m10 * m22;
    nz = m10 * m21 - m11 * m20;
    len = Math.sqrt(nx * nx + ny * ny + nz * nz);
  }

  if (len < 1e-15) {
    // Fallback: return z-axis.
    const result = new Float64Array(3);
    result[2] = 1;
    return result;
  }

  const result = new Float64Array(3);
  result[0] = nx / len;
  result[1] = ny / len;
  result[2] = nz / len;
  return result;
}
