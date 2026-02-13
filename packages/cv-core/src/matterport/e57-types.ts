// ---------------------------------------------------------------------------
// CV-1: E57 Point Cloud Header Parsing & Coordinate Transforms
// ---------------------------------------------------------------------------

import type {
  Vector3,
  E57Header,
  CoordinateTransform,
  Matrix4x4,
  Matrix3x3,
} from '../types.js';
import {
  vec3Sub,
  vec3Scale,
  vec3Add,
  vec3Dot,
  vec3Cross,
  vec3Normalize,
  vec3Length,
  mat4Identity,
  mat3Transpose,
  mat3Multiply,
} from '../types.js';

/**
 * Parse an E57 header from a Float64Array.
 *
 * The layout convention is:
 *   [0]  scanCount
 *   [1]  totalPointCount
 *   [2]  boundsMin.x
 *   [3]  boundsMin.y
 *   [4]  boundsMin.z
 *   [5]  boundsMax.x
 *   [6]  boundsMax.y
 *   [7]  boundsMax.z
 *
 * String fields (guid, coordinateSystem, creationDate) are given
 * placeholder values because they cannot be encoded in a Float64Array.
 *
 * @param data - Float64Array of at least 8 elements.
 * @returns Parsed {@link E57Header}.
 */
export function parseE57Header(data: Float64Array): E57Header {
  if (data.length < 8) {
    throw new Error(`E57 header data too short: expected >= 8 elements, got ${data.length}`);
  }

  return {
    guid: '00000000-0000-0000-0000-000000000000',
    scanCount: data[0]!,
    totalPointCount: data[1]!,
    coordinateSystem: 'unknown',
    creationDate: new Date().toISOString(),
    boundsMin: { x: data[2]!, y: data[3]!, z: data[4]! },
    boundsMax: { x: data[5]!, y: data[6]!, z: data[7]! },
  };
}

/**
 * Apply a rigid coordinate transform (rotation + translation) to a 3D point.
 *
 * The transform's 4x4 matrix (column-major) is used: p' = M * [p, 1].
 * The scale factor is applied uniformly before the matrix transform.
 *
 * @param point     - Input 3D point.
 * @param transform - The rigid coordinate transform.
 * @returns The transformed point.
 */
export function transformPoint(
  point: Vector3,
  transform: CoordinateTransform,
): Vector3 {
  const s = transform.scale;
  const m = transform.matrix;

  // Scale then multiply by the upper-left 3x4 of the 4x4 matrix
  const px = point.x * s;
  const py = point.y * s;
  const pz = point.z * s;

  return {
    x: m[0]! * px + m[4]! * py + m[8]!  * pz + m[12]!,
    y: m[1]! * px + m[5]! * py + m[9]!  * pz + m[13]!,
    z: m[2]! * px + m[6]! * py + m[10]! * pz + m[14]!,
  };
}

/**
 * Compute a rigid coordinate transform that maps `src` points to `dst`
 * points using a simplified Kabsch algorithm (Procrustes alignment).
 *
 * Steps:
 *   1. Compute centroids of src and dst.
 *   2. Centre both point sets.
 *   3. Compute the 3x3 cross-covariance matrix H = sum(src_i * dst_i^T).
 *   4. Compute rotation via an iterative polar decomposition of H
 *      (since we cannot use a full SVD without a large linear-algebra
 *      library, we approximate R = H * (H^T H)^{-1/2} via repeated
 *      normalisation — this converges rapidly for near-orthogonal H).
 *   5. Translation = centroid_dst - R * centroid_src.
 *
 * The resulting scale is set to 1.0 (uniform-scale estimation is not
 * performed in this simplified variant).
 *
 * @param src - Source point set (minimum 3 non-collinear points).
 * @param dst - Destination point set (same length as src).
 * @returns A {@link CoordinateTransform} that maps src -> dst.
 */
export function computeTransformFromPoints(
  src: Vector3[],
  dst: Vector3[],
): CoordinateTransform {
  if (src.length !== dst.length) {
    throw new Error('Source and destination point sets must have the same length');
  }
  if (src.length < 3) {
    throw new Error('At least 3 point correspondences are required');
  }

  const n = src.length;

  // --- 1. Compute centroids ---
  let srcCentroid: Vector3 = { x: 0, y: 0, z: 0 };
  let dstCentroid: Vector3 = { x: 0, y: 0, z: 0 };
  for (let i = 0; i < n; i++) {
    srcCentroid = vec3Add(srcCentroid, src[i]!);
    dstCentroid = vec3Add(dstCentroid, dst[i]!);
  }
  srcCentroid = vec3Scale(srcCentroid, 1 / n);
  dstCentroid = vec3Scale(dstCentroid, 1 / n);

  // --- 2. Centre the point sets ---
  const srcCentered: Vector3[] = [];
  const dstCentered: Vector3[] = [];
  for (let i = 0; i < n; i++) {
    srcCentered.push(vec3Sub(src[i]!, srcCentroid));
    dstCentered.push(vec3Sub(dst[i]!, dstCentroid));
  }

  // --- 3. Compute cross-covariance matrix H (3x3, column-major) ---
  // H = sum_i (dst_i * src_i^T) — note the order: we want R such that
  // dst ≈ R * src, so H_jk = sum_i dst_i[j] * src_i[k].
  const H = new Float64Array(9); // column-major 3x3
  for (let i = 0; i < n; i++) {
    const s = srcCentered[i]!;
    const d = dstCentered[i]!;
    // Column 0: H[0..2] += d * s.x
    H[0] = H[0]! + d.x * s.x;
    H[1] = H[1]! + d.y * s.x;
    H[2] = H[2]! + d.z * s.x;
    // Column 1: H[3..5] += d * s.y
    H[3] = H[3]! + d.x * s.y;
    H[4] = H[4]! + d.y * s.y;
    H[5] = H[5]! + d.z * s.y;
    // Column 2: H[6..8] += d * s.z
    H[6] = H[6]! + d.x * s.z;
    H[7] = H[7]! + d.y * s.z;
    H[8] = H[8]! + d.z * s.z;
  }

  // --- 4. Polar decomposition via iterative normalisation ---
  // Start with R = H, then repeatedly apply R = 0.5 * (R + (R^{-T}))
  // until convergence (this extracts the nearest rotation matrix).
  const R = new Float64Array(9);
  for (let i = 0; i < 9; i++) R[i] = H[i]!;

  for (let iter = 0; iter < 100; iter++) {
    // Compute R^T
    const Rt = mat3Transpose(R);

    // Invert R^T  (= (R^{-1})^T = R^{-T})
    // For 3x3 we use the explicit inverse via cofactors/determinant
    const RtInv = mat3Inverse3x3(Rt);
    if (RtInv === null) {
      // Singular — can't improve further
      break;
    }

    // R_next = 0.5 * (R + RtInv)
    let maxDelta = 0;
    for (let j = 0; j < 9; j++) {
      const newVal = 0.5 * (R[j]! + RtInv[j]!);
      const delta = Math.abs(newVal - R[j]!);
      if (delta > maxDelta) maxDelta = delta;
      R[j] = newVal;
    }

    if (maxDelta < 1e-12) break;
  }

  // Ensure determinant is +1 (proper rotation, not reflection)
  const det = mat3Det(R);
  if (det < 0) {
    // Negate the column with smallest singular value — approximate by
    // negating the third column.
    R[6] = -R[6]!;
    R[7] = -R[7]!;
    R[8] = -R[8]!;
  }

  // --- 5. Compute translation ---
  // t = dstCentroid - R * srcCentroid
  const rSrc: Vector3 = {
    x: R[0]! * srcCentroid.x + R[3]! * srcCentroid.y + R[6]! * srcCentroid.z,
    y: R[1]! * srcCentroid.x + R[4]! * srcCentroid.y + R[7]! * srcCentroid.z,
    z: R[2]! * srcCentroid.x + R[5]! * srcCentroid.y + R[8]! * srcCentroid.z,
  };
  const t = vec3Sub(dstCentroid, rSrc);

  // --- 6. Build 4x4 column-major transform matrix ---
  const matrix = mat4Identity();
  // Rotation (upper-left 3x3)
  matrix[0]  = R[0]!;
  matrix[1]  = R[1]!;
  matrix[2]  = R[2]!;
  matrix[4]  = R[3]!;
  matrix[5]  = R[4]!;
  matrix[6]  = R[5]!;
  matrix[8]  = R[6]!;
  matrix[9]  = R[7]!;
  matrix[10] = R[8]!;
  // Translation (column 3)
  matrix[12] = t.x;
  matrix[13] = t.y;
  matrix[14] = t.z;

  return {
    matrix,
    sourceSystem: 'source',
    targetSystem: 'target',
    scale: 1.0,
  };
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Determinant of a 3x3 column-major matrix. */
function mat3Det(m: Float64Array): number {
  const a = m[0]!, b = m[3]!, c = m[6]!;
  const d = m[1]!, e = m[4]!, f = m[7]!;
  const g = m[2]!, h = m[5]!, i = m[8]!;
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

/** Inverse of a 3x3 column-major matrix, or null if singular. */
function mat3Inverse3x3(m: Float64Array): Float64Array | null {
  const det = mat3Det(m);
  if (Math.abs(det) < 1e-15) return null;

  const invDet = 1 / det;
  const a = m[0]!, b = m[3]!, c = m[6]!;
  const d = m[1]!, e = m[4]!, f = m[7]!;
  const g = m[2]!, h = m[5]!, i = m[8]!;

  const out = new Float64Array(9);
  // Cofactor matrix transposed, scaled by 1/det
  // Column 0
  out[0] = (e * i - f * h) * invDet;
  out[1] = (f * g - d * i) * invDet;
  out[2] = (d * h - e * g) * invDet;
  // Column 1
  out[3] = (c * h - b * i) * invDet;
  out[4] = (a * i - c * g) * invDet;
  out[5] = (b * g - a * h) * invDet;
  // Column 2
  out[6] = (b * f - c * e) * invDet;
  out[7] = (c * d - a * f) * invDet;
  out[8] = (a * e - b * d) * invDet;

  return out;
}
