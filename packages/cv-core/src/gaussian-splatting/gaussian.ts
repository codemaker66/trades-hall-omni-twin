// ---------------------------------------------------------------------------
// CV-2: Gaussian Splatting — Core Gaussian Primitives
// ---------------------------------------------------------------------------

import type {
  Vector3,
  Quaternion,
  Vec2,
  Gaussian3D,
  Matrix3x3,
  Matrix4x4,
} from '../types.js';
import { quatToMat3, mat3Transpose, mat3Multiply } from '../types.js';

/**
 * Create a fully-populated {@link Gaussian3D} from compact parameters.
 *
 * The covariance matrix is computed from scales and rotation via
 * {@link covarianceFromScaleRotation}.
 *
 * @param center   - World-space centre position.
 * @param scales   - Per-axis standard deviations (sx, sy, sz).
 * @param rotation - Orientation quaternion.
 * @param color    - RGB colour, each channel in [0,1].
 * @param opacity  - Opacity in [0,1].
 * @returns A fully initialised Gaussian3D.
 */
export function createGaussian3D(
  center: Vector3,
  scales: Vector3,
  rotation: Quaternion,
  color: Vector3,
  opacity: number,
): Gaussian3D {
  return {
    center,
    covariance: covarianceFromScaleRotation(scales, rotation),
    color,
    opacity,
    sh_coeffs: new Float64Array(0),
  };
}

/**
 * Compute a 3x3 symmetric covariance matrix from anisotropic scales and
 * a rotation quaternion.
 *
 *   Sigma = R * S * S^T * R^T
 *
 * where S = diag(sx, sy, sz) and R is the 3x3 rotation matrix derived
 * from the quaternion.
 *
 * @param scales   - Per-axis standard deviations.
 * @param rotation - Orientation quaternion.
 * @returns 3x3 column-major covariance matrix (Float64Array of 9 elements).
 */
export function covarianceFromScaleRotation(
  scales: Vector3,
  rotation: Quaternion,
): Float64Array {
  const R = quatToMat3(rotation);

  // M = R * S  (S is diagonal, so just scale the columns of R)
  const M = new Float64Array(9);
  // Column 0: R[:,0] * sx
  M[0] = R[0]! * scales.x;
  M[1] = R[1]! * scales.x;
  M[2] = R[2]! * scales.x;
  // Column 1: R[:,1] * sy
  M[3] = R[3]! * scales.y;
  M[4] = R[4]! * scales.y;
  M[5] = R[5]! * scales.y;
  // Column 2: R[:,2] * sz
  M[6] = R[6]! * scales.z;
  M[7] = R[7]! * scales.z;
  M[8] = R[8]! * scales.z;

  // Sigma = M * M^T
  const Mt = mat3Transpose(M);
  return mat3Multiply(M, Mt);
}

/**
 * Project a 3D Gaussian to 2D screen space via EWA (Elliptical Weighted
 * Average) splatting.
 *
 * Steps:
 *   1. Transform the Gaussian centre to camera space using the view matrix.
 *   2. Compute the Jacobian of the perspective projection at that depth.
 *   3. Project the 3D covariance to 2D: Sigma_2D = J * W * Sigma_3D * W^T * J^T
 *      where W is the upper-left 3x3 of the view matrix.
 *   4. Project the centre to normalised device coordinates via the projection matrix,
 *      then to screen coordinates.
 *
 * @param gaussian   - The 3D Gaussian to project.
 * @param viewMatrix - 4x4 column-major world-to-camera matrix.
 * @param projMatrix - 4x4 column-major projection matrix.
 * @param width      - Viewport width in pixels.
 * @param height     - Viewport height in pixels.
 * @returns Projected 2D centre, 2x2 covariance (column-major, 4 elements), and depth.
 */
export function projectGaussian2D(
  gaussian: Gaussian3D,
  viewMatrix: Float64Array,
  projMatrix: Float64Array,
  width: number,
  height: number,
): { center: Vec2; cov2D: Float64Array; depth: number } {
  const c = gaussian.center;

  // --- Transform centre to camera space ---
  const cx =
    viewMatrix[0]! * c.x +
    viewMatrix[4]! * c.y +
    viewMatrix[8]! * c.z +
    viewMatrix[12]!;
  const cy =
    viewMatrix[1]! * c.x +
    viewMatrix[5]! * c.y +
    viewMatrix[9]! * c.z +
    viewMatrix[13]!;
  const cz =
    viewMatrix[2]! * c.x +
    viewMatrix[6]! * c.y +
    viewMatrix[10]! * c.z +
    viewMatrix[14]!;

  const depth = cz;

  // Guard against divide-by-zero for points at or behind camera
  const invZ = cz !== 0 ? 1.0 / cz : 0;
  const invZ2 = invZ * invZ;

  // --- Jacobian of perspective projection ---
  // For a pinhole camera, the projection Jacobian at camera-space point (cx, cy, cz) is:
  //   J = | fx/cz    0     -fx*cx/cz^2 |
  //       |   0    fy/cz   -fy*cy/cz^2 |
  //
  // We extract focal lengths from the projection matrix:
  //   fx = projMatrix[0]  (element (0,0))
  //   fy = projMatrix[5]  (element (1,1))
  const fx = projMatrix[0]!;
  const fy = projMatrix[5]!;

  const J00 = fx * invZ;
  const J02 = -fx * cx * invZ2;
  const J11 = fy * invZ;
  const J12 = -fy * cy * invZ2;

  // --- Extract upper-left 3x3 of the view matrix (W) ---
  const W00 = viewMatrix[0]!, W01 = viewMatrix[4]!, W02 = viewMatrix[8]!;
  const W10 = viewMatrix[1]!, W11 = viewMatrix[5]!, W12 = viewMatrix[9]!;
  const W20 = viewMatrix[2]!, W21 = viewMatrix[6]!, W22 = viewMatrix[10]!;

  // --- T = J * W (2x3 matrix) ---
  const T00 = J00 * W00 + J02 * W20;
  const T01 = J00 * W01 + J02 * W21;
  const T02 = J00 * W02 + J02 * W22;
  const T10 = J11 * W10 + J12 * W20;
  const T11 = J11 * W11 + J12 * W21;
  const T12 = J11 * W12 + J12 * W22;

  // --- Sigma_3D (column-major 3x3) ---
  const S = gaussian.covariance;
  const S00 = S[0]!, S10 = S[1]!, S20 = S[2]!;
  const S01 = S[3]!, S11 = S[4]!, S21 = S[5]!;
  const S02 = S[6]!, S12 = S[7]!, S22 = S[8]!;

  // --- Compute M = T * Sigma_3D  (2x3) ---
  const M00 = T00 * S00 + T01 * S10 + T02 * S20;
  const M01 = T00 * S01 + T01 * S11 + T02 * S21;
  const M02 = T00 * S02 + T01 * S12 + T02 * S22;
  const M10 = T10 * S00 + T11 * S10 + T12 * S20;
  const M11 = T10 * S01 + T11 * S11 + T12 * S21;
  const M12 = T10 * S02 + T11 * S12 + T12 * S22;

  // --- cov2D = M * T^T  (2x2, symmetric) ---
  const cov2D = new Float64Array(4); // column-major 2x2
  cov2D[0] = M00 * T00 + M01 * T01 + M02 * T02; // (0,0)
  cov2D[1] = M10 * T00 + M11 * T01 + M12 * T02; // (1,0)
  cov2D[2] = cov2D[1]!;                          // (0,1) = (1,0) by symmetry
  cov2D[3] = M10 * T10 + M11 * T11 + M12 * T12; // (1,1)

  // --- Project centre to screen coordinates ---
  // Apply projection matrix to camera-space point
  const clipX =
    projMatrix[0]! * cx +
    projMatrix[4]! * cy +
    projMatrix[8]! * cz +
    projMatrix[12]!;
  const clipY =
    projMatrix[1]! * cx +
    projMatrix[5]! * cy +
    projMatrix[9]! * cz +
    projMatrix[13]!;
  const clipW =
    projMatrix[3]! * cx +
    projMatrix[7]! * cy +
    projMatrix[11]! * cz +
    projMatrix[15]!;

  const ndcX = clipW !== 0 ? clipX / clipW : 0;
  const ndcY = clipW !== 0 ? clipY / clipW : 0;

  // NDC [-1, 1] -> screen [0, width/height]
  const screenX = (ndcX * 0.5 + 0.5) * width;
  const screenY = (ndcY * 0.5 + 0.5) * height;

  return {
    center: { x: screenX, y: screenY },
    cov2D,
    depth,
  };
}

/**
 * Evaluate a 2D Gaussian at a given point.
 *
 *   G(p) = exp(-0.5 * (p - mu)^T * Sigma^{-1} * (p - mu))
 *
 * @param point      - 2D query point.
 * @param center     - 2D Gaussian mean.
 * @param covInverse - Inverse of the 2x2 covariance matrix (column-major, 4 elements).
 * @returns The unnormalised Gaussian weight in [0, 1].
 */
export function evaluateGaussian2D(
  point: Vec2,
  center: Vec2,
  covInverse: Float64Array,
): number {
  const dx = point.x - center.x;
  const dy = point.y - center.y;

  // Mahalanobis distance squared: d^T * Sigma^{-1} * d
  // covInverse is 2x2 column-major: [a, c, b, d] -> row-major [[a,b],[c,d]]
  const a = covInverse[0]!;
  const c = covInverse[1]!;
  const b = covInverse[2]!;
  const d = covInverse[3]!;

  const mahal = dx * (a * dx + b * dy) + dy * (c * dx + d * dy);

  return Math.exp(-0.5 * mahal);
}
