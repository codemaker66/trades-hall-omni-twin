// ---------------------------------------------------------------------------
// CV-4: Structure from Motion — homography estimation, fundamental matrix
// via RANSAC, triangulation, and the normalised 8-point algorithm.
// ---------------------------------------------------------------------------

import type {
  FundamentalMatrixResult,
  Match,
  Matrix3x3,
  PRNG,
  Vec2,
  Vector3,
} from '../types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Solve a homogeneous linear system A x = 0 for x (null-space vector).
 *
 * We compute the right-singular-vector associated with the smallest singular
 * value of A using the normal equations: A^T A v = sigma^2 v.  We perform a
 * simple power-iteration / inverse-iteration approach on A^T A to extract the
 * eigenvector with the smallest eigenvalue.
 *
 * For the small matrices in SfM (8x9, 2Nx9, etc.) this is numerically
 * adequate and avoids pulling in a full SVD library.
 */
function solveNullSpace(A: Float64Array, rows: number, cols: number): Float64Array {
  // Compute A^T A (cols x cols)
  const ATA = new Float64Array(cols * cols);
  for (let i = 0; i < cols; i++) {
    for (let j = i; j < cols; j++) {
      let sum = 0;
      for (let k = 0; k < rows; k++) {
        sum += A[k * cols + i]! * A[k * cols + j]!;
      }
      ATA[i * cols + j] = sum;
      ATA[j * cols + i] = sum;
    }
  }

  // Inverse iteration: repeatedly solve (ATA - sigma*I) v = v_old to
  // converge to the eigenvector with smallest eigenvalue.
  // We use a small shift to avoid singularity.
  const shift = 1e-10;

  // Start with a uniform vector
  let v = new Float64Array(cols);
  for (let i = 0; i < cols; i++) v[i] = 1 / Math.sqrt(cols);

  // Build shifted matrix: M = ATA + shift*I
  const M = new Float64Array(cols * cols);
  for (let i = 0; i < cols * cols; i++) M[i] = ATA[i]!;
  for (let i = 0; i < cols; i++) M[i * cols + i] = M[i * cols + i]! + shift;

  // LU decomposition of M (in-place with partial pivoting)
  const LU = new Float64Array(cols * cols);
  for (let i = 0; i < cols * cols; i++) LU[i] = M[i]!;
  const piv = new Int32Array(cols);
  for (let i = 0; i < cols; i++) piv[i] = i;

  for (let k = 0; k < cols; k++) {
    // Find pivot
    let maxVal = Math.abs(LU[piv[k]! * cols + k]!);
    let maxIdx = k;
    for (let i = k + 1; i < cols; i++) {
      const val = Math.abs(LU[piv[i]! * cols + k]!);
      if (val > maxVal) {
        maxVal = val;
        maxIdx = i;
      }
    }
    if (maxIdx !== k) {
      const tmp = piv[k]!;
      piv[k] = piv[maxIdx]!;
      piv[maxIdx] = tmp;
    }

    const pivotRow = piv[k]!;
    const pivotVal = LU[pivotRow * cols + k]!;
    if (Math.abs(pivotVal) < 1e-30) continue;

    for (let i = k + 1; i < cols; i++) {
      const row = piv[i]!;
      const factor = LU[row * cols + k]! / pivotVal;
      LU[row * cols + k] = factor; // store L factor in-place
      for (let j = k + 1; j < cols; j++) {
        LU[row * cols + j] = LU[row * cols + j]! - factor * LU[pivotRow * cols + j]!;
      }
    }
  }

  // Inverse iteration (30 iterations is more than enough for 9x9)
  for (let iter = 0; iter < 30; iter++) {
    // Forward substitution: Ly = Pv
    const y = new Float64Array(cols);
    for (let i = 0; i < cols; i++) {
      let sum = v[piv[i]!]!;
      const row = piv[i]!;
      for (let j = 0; j < i; j++) {
        sum -= LU[row * cols + j]! * y[j]!;
      }
      y[i] = sum;
    }

    // Back substitution: Ux = y
    const x = new Float64Array(cols);
    for (let i = cols - 1; i >= 0; i--) {
      const row = piv[i]!;
      let sum = y[i]!;
      for (let j = i + 1; j < cols; j++) {
        sum -= LU[row * cols + j]! * x[j]!;
      }
      const diag = LU[row * cols + i]!;
      x[i] = Math.abs(diag) > 1e-30 ? sum / diag : 0;
    }

    // Normalise
    let norm = 0;
    for (let i = 0; i < cols; i++) norm += x[i]! * x[i]!;
    norm = Math.sqrt(norm);
    if (norm < 1e-30) break;
    for (let i = 0; i < cols; i++) x[i] = x[i]! / norm;

    v = x;
  }

  return v;
}

/** 3x3 matrix-vector multiply for homogeneous 2D points: y = M * [vx, vy, 1]^T. */
function mat3MulVec2H(M: Float64Array, vx: number, vy: number): { x: number; y: number; w: number } {
  // column-major 3x3
  const x = M[0]! * vx + M[3]! * vy + M[6]!;
  const y = M[1]! * vx + M[4]! * vy + M[7]!;
  const w = M[2]! * vx + M[5]! * vy + M[8]!;
  return { x, y, w };
}

// ---------------------------------------------------------------------------
// computeHomography
// ---------------------------------------------------------------------------

/**
 * Compute a 3x3 homography mapping `src` points to `dst` points via the
 * Direct Linear Transform (DLT).
 *
 * Requires at least 4 point correspondences.  The homography H satisfies
 * `dst ~ H * src` (in homogeneous coordinates).
 *
 * @param src Source 2D points (at least 4).
 * @param dst Destination 2D points (same length as src).
 * @returns   3x3 homography matrix as a Float64Array (column-major).
 */
export function computeHomography(src: Vec2[], dst: Vec2[]): Float64Array {
  const n = src.length;
  if (n < 4) throw new Error('computeHomography requires at least 4 correspondences');
  if (n !== dst.length) throw new Error('src and dst must have the same length');

  // Normalise points for numerical stability.
  // Compute centroid and mean distance, then scale so mean distance = sqrt(2).
  const normSrc = normalisePoints(src);
  const normDst = normalisePoints(dst);

  // Build the 2N x 9 system
  const rows = 2 * n;
  const cols = 9;
  const A = new Float64Array(rows * cols);

  for (let i = 0; i < n; i++) {
    const sx = normSrc.pts[i]!.x;
    const sy = normSrc.pts[i]!.y;
    const dx = normDst.pts[i]!.x;
    const dy = normDst.pts[i]!.y;

    const r1 = 2 * i;
    const r2 = r1 + 1;

    // Row 1: [0,0,0, -sx,-sy,-1, dy*sx, dy*sy, dy]
    A[r1 * cols + 3] = -sx;
    A[r1 * cols + 4] = -sy;
    A[r1 * cols + 5] = -1;
    A[r1 * cols + 6] = dy * sx;
    A[r1 * cols + 7] = dy * sy;
    A[r1 * cols + 8] = dy;

    // Row 2: [sx,sy,1, 0,0,0, -dx*sx, -dx*sy, -dx]
    A[r2 * cols + 0] = sx;
    A[r2 * cols + 1] = sy;
    A[r2 * cols + 2] = 1;
    A[r2 * cols + 6] = -dx * sx;
    A[r2 * cols + 7] = -dx * sy;
    A[r2 * cols + 8] = -dx;
  }

  // Solve for the null-space of A
  const h = solveNullSpace(A, rows, cols);

  // Reshape into 3x3 column-major H (null-space vector is row-major H entries)
  const Hnorm = new Float64Array(9);
  // h is laid out as [h00, h01, h02, h10, h11, h12, h20, h21, h22] row-major
  // Convert to column-major:
  Hnorm[0] = h[0]!; // (0,0)
  Hnorm[1] = h[3]!; // (1,0)
  Hnorm[2] = h[6]!; // (2,0)
  Hnorm[3] = h[1]!; // (0,1)
  Hnorm[4] = h[4]!; // (1,1)
  Hnorm[5] = h[7]!; // (2,1)
  Hnorm[6] = h[2]!; // (0,2)
  Hnorm[7] = h[5]!; // (1,2)
  Hnorm[8] = h[8]!; // (2,2)

  // De-normalise: H = T_dst^{-1} * H_norm * T_src
  return denormaliseHomography(Hnorm, normSrc.T, normDst.T);
}

// --- point normalisation helpers ---

interface NormalisedPoints {
  pts: Vec2[];
  /** 3x3 normalisation matrix (column-major). */
  T: Float64Array;
}

function normalisePoints(pts: Vec2[]): NormalisedPoints {
  const n = pts.length;
  let cx = 0;
  let cy = 0;
  for (let i = 0; i < n; i++) {
    cx += pts[i]!.x;
    cy += pts[i]!.y;
  }
  cx /= n;
  cy /= n;

  let meanDist = 0;
  for (let i = 0; i < n; i++) {
    const dx = pts[i]!.x - cx;
    const dy = pts[i]!.y - cy;
    meanDist += Math.sqrt(dx * dx + dy * dy);
  }
  meanDist /= n;

  const s = meanDist > 1e-15 ? Math.SQRT2 / meanDist : 1;

  const normalised: Vec2[] = new Array(n);
  for (let i = 0; i < n; i++) {
    normalised[i] = {
      x: (pts[i]!.x - cx) * s,
      y: (pts[i]!.y - cy) * s,
    };
  }

  // T (column-major 3x3):
  //  s  0  -s*cx
  //  0  s  -s*cy
  //  0  0   1
  const T = new Float64Array(9);
  T[0] = s;     // (0,0)
  T[4] = s;     // (1,1)
  T[8] = 1;     // (2,2)
  T[6] = -s * cx; // (0,2)
  T[7] = -s * cy; // (1,2)

  return { pts: normalised, T };
}

/** Compute T_dst_inv * Hnorm * T_src (all column-major 3x3). */
function denormaliseHomography(
  Hnorm: Float64Array,
  Tsrc: Float64Array,
  Tdst: Float64Array,
): Float64Array {
  // Invert Tdst (it's a similarity: scale + translation)
  const s = Tdst[0]!; // scale
  const tx = Tdst[6]!;
  const ty = Tdst[7]!;
  const invS = s !== 0 ? 1 / s : 0;

  const TdstInv = new Float64Array(9);
  TdstInv[0] = invS;
  TdstInv[4] = invS;
  TdstInv[8] = 1;
  TdstInv[6] = -tx * invS;
  TdstInv[7] = -ty * invS;

  // Multiply: TdstInv * Hnorm
  const tmp = mat3Mul(TdstInv, Hnorm);
  // Multiply: tmp * Tsrc
  return mat3Mul(tmp, Tsrc);
}

/** Multiply two column-major 3x3 matrices. */
function mat3Mul(A: Float64Array, B: Float64Array): Float64Array {
  const C = new Float64Array(9);
  for (let col = 0; col < 3; col++) {
    for (let row = 0; row < 3; row++) {
      let sum = 0;
      for (let k = 0; k < 3; k++) {
        sum += A[k * 3 + row]! * B[col * 3 + k]!;
      }
      C[col * 3 + row] = sum;
    }
  }
  return C;
}

// ---------------------------------------------------------------------------
// eightPointAlgorithm
// ---------------------------------------------------------------------------

/**
 * Compute the fundamental matrix from 8 or more point correspondences
 * using the normalised 8-point algorithm (Hartley).
 *
 * The returned 3x3 matrix F satisfies `x2^T F x1 = 0` for corresponding
 * points x1, x2 in homogeneous coordinates.
 *
 * @param pts1 Points in image 1 (at least 8).
 * @param pts2 Corresponding points in image 2.
 * @returns    3x3 fundamental matrix (column-major Float64Array).
 */
export function eightPointAlgorithm(pts1: Vec2[], pts2: Vec2[]): Float64Array {
  const n = pts1.length;
  if (n < 8) throw new Error('eightPointAlgorithm requires at least 8 correspondences');

  // Normalise
  const norm1 = normalisePoints(pts1);
  const norm2 = normalisePoints(pts2);

  // Build the Nx9 constraint matrix
  const cols = 9;
  const A = new Float64Array(n * cols);

  for (let i = 0; i < n; i++) {
    const x1 = norm1.pts[i]!.x;
    const y1 = norm1.pts[i]!.y;
    const x2 = norm2.pts[i]!.x;
    const y2 = norm2.pts[i]!.y;

    const row = i * cols;
    A[row + 0] = x2 * x1;
    A[row + 1] = x2 * y1;
    A[row + 2] = x2;
    A[row + 3] = y2 * x1;
    A[row + 4] = y2 * y1;
    A[row + 5] = y2;
    A[row + 6] = x1;
    A[row + 7] = y1;
    A[row + 8] = 1;
  }

  // Solve null space
  const f = solveNullSpace(A, n, cols);

  // f is row-major [f00,f01,f02,f10,f11,f12,f20,f21,f22]
  // Convert to column-major 3x3
  const Fnorm = new Float64Array(9);
  Fnorm[0] = f[0]!;
  Fnorm[1] = f[3]!;
  Fnorm[2] = f[6]!;
  Fnorm[3] = f[1]!;
  Fnorm[4] = f[4]!;
  Fnorm[5] = f[7]!;
  Fnorm[6] = f[2]!;
  Fnorm[7] = f[5]!;
  Fnorm[8] = f[8]!;

  // Enforce rank-2 constraint by zeroing the smallest singular value.
  // We use a simplified SVD for 3x3: compute eigendecomposition of F^T F.
  enforceRank2(Fnorm);

  // De-normalise: F = T2^T * Fnorm * T1
  const T2t = mat3Transpose3(norm2.T);
  const tmp = mat3Mul(T2t, Fnorm);
  return mat3Mul(tmp, norm1.T);
}

/** Transpose a column-major 3x3 matrix. */
function mat3Transpose3(M: Float64Array): Float64Array {
  const out = new Float64Array(9);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[c * 3 + r] = M[r * 3 + c]!;
    }
  }
  return out;
}

/**
 * Enforce rank-2 on a 3x3 matrix by performing an approximate SVD via
 * Jacobi rotations and zeroing the smallest singular value.
 *
 * Modifies the input matrix in place (column-major 3x3).
 */
function enforceRank2(F: Float64Array): void {
  // Compute F^T F (symmetric 3x3)
  const Ft = mat3Transpose3(F);
  const FtF = mat3Mul(Ft, F);

  // Power iteration to find the eigenvector with the largest eigenvalue
  // of F^T F.  We then deflate and repeat to find all three eigenvectors.
  const eigenvecs = powerIterationEigen3(FtF);

  // Compute singular values: sigma_i = sqrt(v_i^T FtF v_i)
  const sigmas: number[] = [];
  for (let i = 0; i < 3; i++) {
    const vi = eigenvecs[i]!;
    // Compute FtF * vi
    let dot = 0;
    for (let r = 0; r < 3; r++) {
      let sum = 0;
      for (let c = 0; c < 3; c++) {
        sum += FtF[c * 3 + r]! * vi[c]!;
      }
      dot += sum * vi[r]!;
    }
    sigmas.push(Math.sqrt(Math.max(0, dot)));
  }

  // Find the index of the smallest singular value
  let minIdx = 0;
  let minVal = sigmas[0]!;
  for (let i = 1; i < 3; i++) {
    if (sigmas[i]! < minVal) {
      minVal = sigmas[i]!;
      minIdx = i;
    }
  }

  // Zero out the smallest singular value component.
  // F_new = F - sigma_min * u_min * v_min^T
  // where u_min = F * v_min / sigma_min.
  const vMin = eigenvecs[minIdx]!;
  if (minVal < 1e-15) return; // Already rank-deficient

  // Compute u = F * v / sigma
  const u = new Float64Array(3);
  for (let r = 0; r < 3; r++) {
    let sum = 0;
    for (let c = 0; c < 3; c++) {
      sum += F[c * 3 + r]! * vMin[c]!;
    }
    u[r] = sum / minVal;
  }

  // Subtract sigma * u * v^T from F (column-major)
  for (let c = 0; c < 3; c++) {
    for (let r = 0; r < 3; r++) {
      F[c * 3 + r] = F[c * 3 + r]! - minVal * u[r]! * vMin[c]!;
    }
  }
}

/**
 * Extract 3 eigenvectors of a symmetric 3x3 matrix via deflated power
 * iteration. Returns array of 3 eigenvectors (each Float64Array of length 3)
 * ordered by descending eigenvalue.
 */
function powerIterationEigen3(M: Float64Array): Float64Array[] {
  const result: Float64Array[] = [];
  const A = new Float64Array(9);
  for (let i = 0; i < 9; i++) A[i] = M[i]!;

  for (let ev = 0; ev < 3; ev++) {
    // Start with a unit vector
    const v = new Float64Array(3);
    v[ev % 3] = 1;

    for (let iter = 0; iter < 100; iter++) {
      // w = A * v
      const w = new Float64Array(3);
      for (let r = 0; r < 3; r++) {
        let sum = 0;
        for (let c = 0; c < 3; c++) {
          sum += A[c * 3 + r]! * v[c]!;
        }
        w[r] = sum;
      }

      let norm = 0;
      for (let i = 0; i < 3; i++) norm += w[i]! * w[i]!;
      norm = Math.sqrt(norm);
      if (norm < 1e-30) {
        // Degenerate: eigenvalue is 0
        v[0] = 0;
        v[1] = 0;
        v[2] = 0;
        v[ev % 3] = 1;
        break;
      }
      for (let i = 0; i < 3; i++) v[i] = w[i]! / norm;
    }

    result.push(new Float64Array(v));

    // Compute eigenvalue: lambda = v^T A v
    let lambda = 0;
    for (let r = 0; r < 3; r++) {
      let sum = 0;
      for (let c = 0; c < 3; c++) {
        sum += A[c * 3 + r]! * v[c]!;
      }
      lambda += sum * v[r]!;
    }

    // Deflate: A = A - lambda * v * v^T
    for (let c = 0; c < 3; c++) {
      for (let r = 0; r < 3; r++) {
        A[c * 3 + r] = A[c * 3 + r]! - lambda * v[r]! * v[c]!;
      }
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// ransacFundamental
// ---------------------------------------------------------------------------

/**
 * RANSAC estimation of the fundamental matrix using the 8-point algorithm
 * as the minimal solver.
 *
 * @param matches   Feature matches.
 * @param points1   2D points in image 1 (indexed by Match.queryIdx).
 * @param points2   2D points in image 2 (indexed by Match.trainIdx).
 * @param threshold Inlier threshold in pixels (Sampson distance).
 * @param maxIter   Maximum RANSAC iterations.
 * @param rng       Seedable PRNG.
 * @returns         The best fundamental matrix with inlier mask.
 */
export function ransacFundamental(
  matches: Match[],
  points1: Vec2[],
  points2: Vec2[],
  threshold: number,
  maxIter: number,
  rng: PRNG,
): FundamentalMatrixResult {
  const n = matches.length;
  if (n < 8) throw new Error('ransacFundamental requires at least 8 matches');

  let bestF: Float64Array = new Float64Array(9);
  let bestInlierMask = new Uint8Array(n);
  let bestInlierCount = 0;
  let bestError = Infinity;

  const thresholdSq = threshold * threshold;
  const sampleSize = 8;

  for (let iter = 0; iter < maxIter; iter++) {
    // Random sample of 8 matches (Fisher-Yates partial shuffle)
    const indices = sampleIndices(n, sampleSize, rng);

    const samplePts1: Vec2[] = new Array(sampleSize);
    const samplePts2: Vec2[] = new Array(sampleSize);
    for (let i = 0; i < sampleSize; i++) {
      const m = matches[indices[i]!]!;
      samplePts1[i] = points1[m.queryIdx]!;
      samplePts2[i] = points2[m.trainIdx]!;
    }

    // Compute F from the 8-point sample
    let F: Float64Array;
    try {
      F = eightPointAlgorithm(samplePts1, samplePts2);
    } catch {
      continue; // Degenerate configuration
    }

    // Count inliers using the Sampson distance
    const inlierMask = new Uint8Array(n);
    let inlierCount = 0;
    let totalError = 0;

    for (let i = 0; i < n; i++) {
      const m = matches[i]!;
      const p1 = points1[m.queryIdx]!;
      const p2 = points2[m.trainIdx]!;
      const err = sampsonDistance(F, p1, p2);

      if (err < thresholdSq) {
        inlierMask[i] = 1;
        inlierCount++;
        totalError += err;
      }
    }

    if (inlierCount > bestInlierCount || (inlierCount === bestInlierCount && totalError < bestError)) {
      bestF = F;
      bestInlierMask = inlierMask;
      bestInlierCount = inlierCount;
      bestError = totalError;
    }
  }

  // Refine F using all inliers
  if (bestInlierCount >= 8) {
    const inlierPts1: Vec2[] = [];
    const inlierPts2: Vec2[] = [];
    for (let i = 0; i < n; i++) {
      if (bestInlierMask[i]) {
        const m = matches[i]!;
        inlierPts1.push(points1[m.queryIdx]!);
        inlierPts2.push(points2[m.trainIdx]!);
      }
    }
    try {
      bestF = eightPointAlgorithm(inlierPts1, inlierPts2);
    } catch {
      // Keep the previous best F
    }
  }

  // Compute mean error over inliers
  let meanError = 0;
  if (bestInlierCount > 0) {
    let totalErr = 0;
    for (let i = 0; i < n; i++) {
      if (bestInlierMask[i]) {
        const m = matches[i]!;
        totalErr += Math.sqrt(
          sampsonDistance(bestF, points1[m.queryIdx]!, points2[m.trainIdx]!),
        );
      }
    }
    meanError = totalErr / bestInlierCount;
  }

  return {
    F: bestF as Matrix3x3,
    inlierMask: bestInlierMask,
    inlierCount: bestInlierCount,
    meanError,
  };
}

/** Sampson distance (squared) for the fundamental matrix constraint. */
function sampsonDistance(F: Float64Array, p1: Vec2, p2: Vec2): number {
  // F * x1
  const Fx1 = mat3MulVec2H(F, p1.x, p1.y);
  // F^T * x2
  const Ft = mat3Transpose3(F);
  const Ftx2 = mat3MulVec2H(Ft, p2.x, p2.y);

  // x2^T F x1
  const x2tFx1 = p2.x * Fx1.x + p2.y * Fx1.y + Fx1.w;

  const denom =
    Fx1.x * Fx1.x + Fx1.y * Fx1.y + Ftx2.x * Ftx2.x + Ftx2.y * Ftx2.y;

  if (denom < 1e-30) return Infinity;
  return (x2tFx1 * x2tFx1) / denom;
}

/** Sample `k` unique indices from [0, n) using a PRNG. */
function sampleIndices(n: number, k: number, rng: PRNG): number[] {
  const pool: number[] = new Array(n);
  for (let i = 0; i < n; i++) pool[i] = i;

  const result: number[] = new Array(k);
  for (let i = 0; i < k; i++) {
    const j = i + Math.floor(rng() * (n - i));
    result[i] = pool[j]!;
    pool[j] = pool[i]!;
  }
  return result;
}

// ---------------------------------------------------------------------------
// triangulatePoint
// ---------------------------------------------------------------------------

/**
 * Linear triangulation of a 3D point from two 2D observations and their
 * corresponding 3x4 camera projection matrices, using the DLT approach.
 *
 * Each projection matrix P is a 3x4 matrix stored as a 12-element
 * Float64Array in **row-major** order:
 *   [p00, p01, p02, p03,   p10, p11, p12, p13,   p20, p21, p22, p23]
 *
 * The linear system is formed from the cross-product formulation:
 *   x * (p3^T X) - (p1^T X) = 0
 *   y * (p3^T X) - (p2^T X) = 0
 * for each image, yielding a 4x4 system A.  The solution is the null-space
 * vector of A (smallest singular value).
 *
 * @param P1 3x4 projection matrix of camera 1 (row-major Float64Array, 12 elements).
 * @param P2 3x4 projection matrix of camera 2 (row-major Float64Array, 12 elements).
 * @param x1 Observed 2D point in image 1.
 * @param x2 Observed 2D point in image 2.
 * @returns  Triangulated 3D point.
 */
export function triangulatePoint(
  P1: Float64Array,
  P2: Float64Array,
  x1: Vec2,
  x2: Vec2,
): Vector3 {
  // Build the 4x4 matrix A (row-major)
  const A = new Float64Array(16);

  // Row 0: x1 * P1[2,:] - P1[0,:]
  A[0]  = x1.x * P1[8]!  - P1[0]!;
  A[1]  = x1.x * P1[9]!  - P1[1]!;
  A[2]  = x1.x * P1[10]! - P1[2]!;
  A[3]  = x1.x * P1[11]! - P1[3]!;

  // Row 1: y1 * P1[2,:] - P1[1,:]
  A[4]  = x1.y * P1[8]!  - P1[4]!;
  A[5]  = x1.y * P1[9]!  - P1[5]!;
  A[6]  = x1.y * P1[10]! - P1[6]!;
  A[7]  = x1.y * P1[11]! - P1[7]!;

  // Row 2: x2 * P2[2,:] - P2[0,:]
  A[8]  = x2.x * P2[8]!  - P2[0]!;
  A[9]  = x2.x * P2[9]!  - P2[1]!;
  A[10] = x2.x * P2[10]! - P2[2]!;
  A[11] = x2.x * P2[11]! - P2[3]!;

  // Row 3: y2 * P2[2,:] - P2[1,:]
  A[12] = x2.y * P2[8]!  - P2[4]!;
  A[13] = x2.y * P2[9]!  - P2[5]!;
  A[14] = x2.y * P2[10]! - P2[6]!;
  A[15] = x2.y * P2[11]! - P2[7]!;

  // Solve for null-space of A (4x4)
  const v = solveNullSpace(A, 4, 4);

  // Convert from homogeneous: X = [v0, v1, v2] / v3
  const w = v[3]!;
  if (Math.abs(w) < 1e-15) {
    // Point at infinity — return the direction
    return { x: v[0]!, y: v[1]!, z: v[2]! };
  }

  return {
    x: v[0]! / w,
    y: v[1]! / w,
    z: v[2]! / w,
  };
}
