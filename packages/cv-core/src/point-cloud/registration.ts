// ---------------------------------------------------------------------------
// CV-8: Point Cloud Registration — Iterative Closest Point (ICP),
// RANSAC-based global registration, and brute-force nearest-neighbour search.
// ---------------------------------------------------------------------------

import type { ICPResult, PointCloud, PRNG, Vector3 } from '../types.js';
import { mat4Identity, mat4Multiply } from '../types.js';

// ---------------------------------------------------------------------------
// findClosestPoint
// ---------------------------------------------------------------------------

/**
 * Find the closest point in `cloud` to the query point using brute-force
 * linear search.
 *
 * @param point Query point.
 * @param cloud Target point cloud.
 * @returns     Index and Euclidean distance of the closest point.
 */
export function findClosestPoint(
  point: Vector3,
  cloud: PointCloud,
): { index: number; distance: number } {
  let bestIdx = 0;
  let bestDist = Infinity;

  for (let i = 0; i < cloud.count; i++) {
    const pi = i * 3;
    const dx = point.x - cloud.positions[pi]!;
    const dy = point.y - cloud.positions[pi + 1]!;
    const dz = point.z - cloud.positions[pi + 2]!;
    const d = dx * dx + dy * dy + dz * dz;
    if (d < bestDist) {
      bestDist = d;
      bestIdx = i;
    }
  }

  return { index: bestIdx, distance: Math.sqrt(bestDist) };
}

// ---------------------------------------------------------------------------
// icpPointToPoint
// ---------------------------------------------------------------------------

/**
 * Iterative Closest Point (point-to-point variant).
 *
 * At each iteration:
 * 1. For each source point, find the closest target point.
 * 2. Compute the rigid transform (rotation + translation) that minimises the
 *    sum of squared distances between correspondences using SVD via the
 *    Kabsch algorithm (simplified cross-covariance approach).
 * 3. Apply the incremental transform to the source.
 * 4. Repeat until convergence or max iterations.
 *
 * @param source    Source point cloud (will be transformed to align with target).
 * @param target    Target (reference) point cloud.
 * @param maxIter   Maximum number of iterations.
 * @param tolerance Convergence tolerance on mean squared error change.
 * @returns         {@link ICPResult} with the accumulated rigid transform.
 */
export function icpPointToPoint(
  source: PointCloud,
  target: PointCloud,
  maxIter: number,
  tolerance: number,
): ICPResult {
  const n = source.count;
  if (n === 0 || target.count === 0) {
    return {
      transform: mat4Identity(),
      mse: Infinity,
      iterations: 0,
      converged: false,
      inlierCount: 0,
      fitness: 0,
    };
  }

  // Working copy of source positions that we iteratively transform.
  const src = Float64Array.from(source.positions);

  let accumulatedTransform = mat4Identity();
  let prevMSE = Infinity;
  let iter = 0;
  let mse = Infinity;

  for (iter = 0; iter < maxIter; iter++) {
    // --- Step 1: find correspondences ---
    const corrIdx = new Uint32Array(n);
    let mseSum = 0;

    for (let i = 0; i < n; i++) {
      const si = i * 3;
      const pt: Vector3 = {
        x: src[si]!,
        y: src[si + 1]!,
        z: src[si + 2]!,
      };
      const closest = findClosestPoint(pt, target);
      corrIdx[i] = closest.index;
      mseSum += closest.distance * closest.distance;
    }

    mse = mseSum / n;

    // Check convergence.
    if (Math.abs(prevMSE - mse) < tolerance) {
      iter++;
      break;
    }
    prevMSE = mse;

    // --- Step 2: compute centroids ---
    let scx = 0,
      scy = 0,
      scz = 0;
    let tcx = 0,
      tcy = 0,
      tcz = 0;

    for (let i = 0; i < n; i++) {
      const si = i * 3;
      scx += src[si]!;
      scy += src[si + 1]!;
      scz += src[si + 2]!;

      const ti = corrIdx[i]! * 3;
      tcx += target.positions[ti]!;
      tcy += target.positions[ti + 1]!;
      tcz += target.positions[ti + 2]!;
    }

    const invN = 1 / n;
    scx *= invN;
    scy *= invN;
    scz *= invN;
    tcx *= invN;
    tcy *= invN;
    tcz *= invN;

    // --- Step 3: compute cross-covariance H = sum (src_i - centS)(tgt_i - centT)^T ---
    // H is a 3x3 matrix stored row-major for the SVD below.
    let h00 = 0, h01 = 0, h02 = 0;
    let h10 = 0, h11 = 0, h12 = 0;
    let h20 = 0, h21 = 0, h22 = 0;

    for (let i = 0; i < n; i++) {
      const si = i * 3;
      const sx = src[si]! - scx;
      const sy = src[si + 1]! - scy;
      const sz = src[si + 2]! - scz;

      const ti = corrIdx[i]! * 3;
      const tx = target.positions[ti]! - tcx;
      const ty = target.positions[ti + 1]! - tcy;
      const tz = target.positions[ti + 2]! - tcz;

      h00 += sx * tx; h01 += sx * ty; h02 += sx * tz;
      h10 += sy * tx; h11 += sy * ty; h12 += sy * tz;
      h20 += sz * tx; h21 += sz * ty; h22 += sz * tz;
    }

    // --- Step 4: approximate rotation via polar decomposition ---
    // R = V * U^T from SVD of H.  For a lightweight implementation we use
    // the iterative normalisation approach for 3x3 rotation extraction.
    const R = extractRotation3x3(
      h00, h01, h02,
      h10, h11, h12,
      h20, h21, h22,
    );

    // Translation: t = centT - R * centS
    const tx = tcx - (R[0]! * scx + R[1]! * scy + R[2]! * scz);
    const ty = tcy - (R[3]! * scx + R[4]! * scy + R[5]! * scz);
    const tz = tcz - (R[6]! * scx + R[7]! * scy + R[8]! * scz);

    // --- Step 5: build 4x4 incremental transform (column-major) ---
    const inc = new Float64Array(16);
    inc[0] = R[0]!;  inc[1] = R[3]!;  inc[2] = R[6]!;  inc[3] = 0;
    inc[4] = R[1]!;  inc[5] = R[4]!;  inc[6] = R[7]!;  inc[7] = 0;
    inc[8] = R[2]!;  inc[9] = R[5]!;  inc[10] = R[8]!; inc[11] = 0;
    inc[12] = tx;     inc[13] = ty;     inc[14] = tz;    inc[15] = 1;

    // Apply incremental transform to the working source positions.
    for (let i = 0; i < n; i++) {
      const si = i * 3;
      const px = src[si]!;
      const py = src[si + 1]!;
      const pz = src[si + 2]!;
      src[si] = R[0]! * px + R[1]! * py + R[2]! * pz + tx;
      src[si + 1] = R[3]! * px + R[4]! * py + R[5]! * pz + ty;
      src[si + 2] = R[6]! * px + R[7]! * py + R[8]! * pz + tz;
    }

    // Accumulate transform.
    accumulatedTransform = mat4Multiply(inc, accumulatedTransform);
  }

  return {
    transform: accumulatedTransform,
    mse,
    iterations: iter,
    converged: Math.abs(prevMSE - mse) < tolerance,
    inlierCount: n,
    fitness: 1,
  };
}

// ---------------------------------------------------------------------------
// ransacGlobalRegistration
// ---------------------------------------------------------------------------

/**
 * RANSAC-based global registration for an initial coarse alignment between
 * two point clouds.
 *
 * At each iteration:
 * 1. Randomly sample 3 source points.
 * 2. Find their closest counterparts in the target cloud.
 * 3. Compute the rigid transform from those 3 correspondences.
 * 4. Count inliers (source points that, once transformed, lie within
 *    `threshold` of their closest target point).
 * 5. Keep the transform with the most inliers.
 *
 * @param source    Source point cloud.
 * @param target    Target point cloud.
 * @param threshold Inlier distance threshold (metres).
 * @param maxIter   Maximum RANSAC iterations.
 * @param rng       Seedable PRNG.
 * @returns         {@link ICPResult} with the best-found transform.
 */
export function ransacGlobalRegistration(
  source: PointCloud,
  target: PointCloud,
  threshold: number,
  maxIter: number,
  rng: PRNG,
): ICPResult {
  const n = source.count;
  if (n < 3 || target.count < 3) {
    return {
      transform: mat4Identity(),
      mse: Infinity,
      iterations: 0,
      converged: false,
      inlierCount: 0,
      fitness: 0,
    };
  }

  let bestTransform = mat4Identity();
  let bestInliers = 0;
  let bestMSE = Infinity;

  const thresholdSq = threshold * threshold;

  for (let iter = 0; iter < maxIter; iter++) {
    // --- Sample 3 distinct source indices ---
    const sampleIndices = sampleDistinct(3, n, rng);

    // --- Find closest target points for each sample ---
    const srcPts: Vector3[] = [];
    const tgtPts: Vector3[] = [];
    for (let k = 0; k < 3; k++) {
      const si = sampleIndices[k]! * 3;
      const sp: Vector3 = {
        x: source.positions[si]!,
        y: source.positions[si + 1]!,
        z: source.positions[si + 2]!,
      };
      srcPts.push(sp);

      const closest = findClosestPoint(sp, target);
      const ti = closest.index * 3;
      tgtPts.push({
        x: target.positions[ti]!,
        y: target.positions[ti + 1]!,
        z: target.positions[ti + 2]!,
      });
    }

    // --- Compute rigid transform from 3 correspondences (Kabsch) ---
    const transform = rigidFrom3(srcPts, tgtPts);

    // --- Count inliers ---
    let inlierCount = 0;
    let mseSum = 0;

    for (let i = 0; i < n; i++) {
      const si = i * 3;
      const px = source.positions[si]!;
      const py = source.positions[si + 1]!;
      const pz = source.positions[si + 2]!;

      // Apply transform (column-major).
      const qx =
        transform[0]! * px + transform[4]! * py + transform[8]! * pz + transform[12]!;
      const qy =
        transform[1]! * px + transform[5]! * py + transform[9]! * pz + transform[13]!;
      const qz =
        transform[2]! * px + transform[6]! * py + transform[10]! * pz + transform[14]!;

      const closest = findClosestPoint({ x: qx, y: qy, z: qz }, target);
      const dSq = closest.distance * closest.distance;

      if (dSq <= thresholdSq) {
        inlierCount++;
        mseSum += dSq;
      }
    }

    if (inlierCount > bestInliers) {
      bestInliers = inlierCount;
      bestTransform = transform;
      bestMSE = inlierCount > 0 ? mseSum / inlierCount : Infinity;
    }
  }

  return {
    transform: bestTransform,
    mse: bestMSE,
    iterations: maxIter,
    converged: bestInliers > 0,
    inlierCount: bestInliers,
    fitness: n > 0 ? bestInliers / n : 0,
  };
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Sample `k` distinct integers from [0, n) using a PRNG.
 */
function sampleDistinct(k: number, n: number, rng: PRNG): number[] {
  const result: number[] = [];
  const used = new Set<number>();

  while (result.length < k) {
    const idx = Math.floor(rng() * n);
    if (!used.has(idx)) {
      used.add(idx);
      result.push(idx);
    }
  }

  return result;
}

/**
 * Compute a rigid 4x4 transform (column-major) from exactly 3
 * correspondences using the Kabsch algorithm.
 */
function rigidFrom3(src: Vector3[], tgt: Vector3[]): Float64Array {
  // Centroids.
  let scx = 0, scy = 0, scz = 0;
  let tcx = 0, tcy = 0, tcz = 0;
  for (let i = 0; i < 3; i++) {
    scx += src[i]!.x; scy += src[i]!.y; scz += src[i]!.z;
    tcx += tgt[i]!.x; tcy += tgt[i]!.y; tcz += tgt[i]!.z;
  }
  scx /= 3; scy /= 3; scz /= 3;
  tcx /= 3; tcy /= 3; tcz /= 3;

  // Cross-covariance H (row-major 3x3).
  let h00 = 0, h01 = 0, h02 = 0;
  let h10 = 0, h11 = 0, h12 = 0;
  let h20 = 0, h21 = 0, h22 = 0;
  for (let i = 0; i < 3; i++) {
    const sx = src[i]!.x - scx;
    const sy = src[i]!.y - scy;
    const sz = src[i]!.z - scz;
    const tx = tgt[i]!.x - tcx;
    const ty = tgt[i]!.y - tcy;
    const tz = tgt[i]!.z - tcz;
    h00 += sx * tx; h01 += sx * ty; h02 += sx * tz;
    h10 += sy * tx; h11 += sy * ty; h12 += sy * tz;
    h20 += sz * tx; h21 += sz * ty; h22 += sz * tz;
  }

  const R = extractRotation3x3(h00, h01, h02, h10, h11, h12, h20, h21, h22);

  const tx = tcx - (R[0]! * scx + R[1]! * scy + R[2]! * scz);
  const ty = tcy - (R[3]! * scx + R[4]! * scy + R[5]! * scz);
  const tz = tcz - (R[6]! * scx + R[7]! * scy + R[8]! * scz);

  // Column-major 4x4.
  const m = new Float64Array(16);
  m[0] = R[0]!;  m[1] = R[3]!;  m[2] = R[6]!;  m[3] = 0;
  m[4] = R[1]!;  m[5] = R[4]!;  m[6] = R[7]!;  m[7] = 0;
  m[8] = R[2]!;  m[9] = R[5]!;  m[10] = R[8]!; m[11] = 0;
  m[12] = tx;     m[13] = ty;     m[14] = tz;    m[15] = 1;
  return m;
}

/**
 * Extract a 3x3 rotation matrix from a cross-covariance matrix H using
 * iterative normalisation (a lightweight substitute for SVD).
 *
 * Returns 9 elements in row-major order: [r00, r01, r02, r10, r11, r12, r20, r21, r22].
 *
 * The approach:
 * 1. Start with H^T * H to get a symmetric positive-definite matrix.
 * 2. Compute its approximate inverse square root via Newton iteration.
 * 3. R = H * (H^T H)^{-1/2}.
 *
 * For small inputs (3 correspondences) this is sufficiently accurate.
 */
function extractRotation3x3(
  h00: number, h01: number, h02: number,
  h10: number, h11: number, h12: number,
  h20: number, h21: number, h22: number,
): Float64Array {
  // Transpose of H.
  const t00 = h00, t01 = h10, t02 = h20;
  const t10 = h01, t11 = h11, t12 = h21;
  const t20 = h02, t21 = h12, t22 = h22;

  // S = H^T * H (symmetric 3x3).
  const s00 = t00 * h00 + t01 * h10 + t02 * h20;
  const s01 = t00 * h01 + t01 * h11 + t02 * h21;
  const s02 = t00 * h02 + t01 * h12 + t02 * h22;
  const s11 = t10 * h01 + t11 * h11 + t12 * h21;
  const s12 = t10 * h02 + t11 * h12 + t12 * h22;
  const s22 = t20 * h02 + t21 * h12 + t22 * h22;

  // Newton iteration for (H^T H)^{-1/2} starting from identity.
  // Y_k+1 = 0.5 * Y_k * (3I - S * Y_k^2)
  // We iterate a fixed number of times for stability.
  let y00 = 1, y01 = 0, y02 = 0;
  let y10 = 0, y11 = 1, y12 = 0;
  let y20 = 0, y21 = 0, y22 = 1;

  // Scale initial guess by 1/sqrt(trace(S)) for better conditioning.
  const traceS = s00 + s11 + s22;
  if (traceS > 1e-15) {
    const invSqrtTrace = 1 / Math.sqrt(traceS / 3);
    y00 = invSqrtTrace;
    y11 = invSqrtTrace;
    y22 = invSqrtTrace;
  }

  for (let it = 0; it < 20; it++) {
    // Compute Y^2.
    const yy00 = y00 * y00 + y01 * y10 + y02 * y20;
    const yy01 = y00 * y01 + y01 * y11 + y02 * y21;
    const yy02 = y00 * y02 + y01 * y12 + y02 * y22;
    const yy10 = y10 * y00 + y11 * y10 + y12 * y20;
    const yy11 = y10 * y01 + y11 * y11 + y12 * y21;
    const yy12 = y10 * y02 + y11 * y12 + y12 * y22;
    const yy20 = y20 * y00 + y21 * y10 + y22 * y20;
    const yy21 = y20 * y01 + y21 * y11 + y22 * y21;
    const yy22 = y20 * y02 + y21 * y12 + y22 * y22;

    // Compute S * Y^2.
    const sy00 = s00 * yy00 + s01 * yy10 + s02 * yy20;
    const sy01 = s00 * yy01 + s01 * yy11 + s02 * yy21;
    const sy02 = s00 * yy02 + s01 * yy12 + s02 * yy22;
    const sy10 = s01 * yy00 + s11 * yy10 + s12 * yy20;
    const sy11 = s01 * yy01 + s11 * yy11 + s12 * yy21;
    const sy12 = s01 * yy02 + s11 * yy12 + s12 * yy22;
    const sy20 = s02 * yy00 + s12 * yy10 + s22 * yy20;
    const sy21 = s02 * yy01 + s12 * yy11 + s22 * yy21;
    const sy22 = s02 * yy02 + s12 * yy12 + s22 * yy22;

    // Z = 3I - S*Y^2.
    const z00 = 3 - sy00, z01 = -sy01,    z02 = -sy02;
    const z10 = -sy10,    z11 = 3 - sy11, z12 = -sy12;
    const z20 = -sy20,    z21 = -sy21,    z22 = 3 - sy22;

    // Y_new = 0.5 * Y * Z.
    const n00 = 0.5 * (y00 * z00 + y01 * z10 + y02 * z20);
    const n01 = 0.5 * (y00 * z01 + y01 * z11 + y02 * z21);
    const n02 = 0.5 * (y00 * z02 + y01 * z12 + y02 * z22);
    const n10 = 0.5 * (y10 * z00 + y11 * z10 + y12 * z20);
    const n11 = 0.5 * (y10 * z01 + y11 * z11 + y12 * z21);
    const n12 = 0.5 * (y10 * z02 + y11 * z12 + y12 * z22);
    const n20 = 0.5 * (y20 * z00 + y21 * z10 + y22 * z20);
    const n21 = 0.5 * (y20 * z01 + y21 * z11 + y22 * z21);
    const n22 = 0.5 * (y20 * z02 + y21 * z12 + y22 * z22);

    y00 = n00; y01 = n01; y02 = n02;
    y10 = n10; y11 = n11; y12 = n12;
    y20 = n20; y21 = n21; y22 = n22;
  }

  // R = H * Y  (where Y approximates (H^T H)^{-1/2}).
  const r = new Float64Array(9);
  r[0] = h00 * y00 + h01 * y10 + h02 * y20;
  r[1] = h00 * y01 + h01 * y11 + h02 * y21;
  r[2] = h00 * y02 + h01 * y12 + h02 * y22;
  r[3] = h10 * y00 + h11 * y10 + h12 * y20;
  r[4] = h10 * y01 + h11 * y11 + h12 * y21;
  r[5] = h10 * y02 + h11 * y12 + h12 * y22;
  r[6] = h20 * y00 + h21 * y10 + h22 * y20;
  r[7] = h20 * y01 + h21 * y11 + h22 * y21;
  r[8] = h20 * y02 + h21 * y12 + h22 * y22;

  return r;
}
