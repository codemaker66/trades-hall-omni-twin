// ---------------------------------------------------------------------------
// CV-11: XR — 6-DoF Pose Utilities
// Construction, interpolation, matrix conversion, composition, and inversion
// of 6 degrees-of-freedom poses.
// ---------------------------------------------------------------------------

import type { Pose6DoF, Vector3, Quaternion, Matrix4x4 } from '../types.js';
import {
  vec3Add,
  vec3Scale,
  vec3Sub,
  quatMultiply,
  quatNormalize,
  quatSlerp,
  quatToMat3,
  mat4Identity,
} from '../types.js';

// ---------------------------------------------------------------------------
// createPose6DoF
// ---------------------------------------------------------------------------

/**
 * Create a 6-DoF pose from a position and an orientation quaternion.
 *
 * The quaternion is normalised on construction to ensure a valid rotation.
 *
 * @param position    World-space position.
 * @param orientation Orientation as a quaternion (will be normalised).
 * @returns           A `Pose6DoF` value.
 */
export function createPose6DoF(
  position: Vector3,
  orientation: Quaternion,
): Pose6DoF {
  return {
    position: { x: position.x, y: position.y, z: position.z },
    orientation: quatNormalize(orientation),
  };
}

// ---------------------------------------------------------------------------
// interpolatePose
// ---------------------------------------------------------------------------

/**
 * Linearly interpolate position and spherically interpolate orientation
 * between two 6-DoF poses.
 *
 * @param a Source pose (t = 0).
 * @param b Target pose (t = 1).
 * @param t Interpolation parameter in [0, 1].
 * @returns Interpolated pose.
 */
export function interpolatePose(
  a: Pose6DoF,
  b: Pose6DoF,
  t: number,
): Pose6DoF {
  // Clamp t to [0, 1].
  const tc = Math.max(0, Math.min(1, t));

  // Lerp position.
  const pos: Vector3 = vec3Add(
    vec3Scale(a.position, 1 - tc),
    vec3Scale(b.position, tc),
  );

  // Slerp orientation.
  const ori = quatSlerp(a.orientation, b.orientation, tc);

  return { position: pos, orientation: ori };
}

// ---------------------------------------------------------------------------
// poseToMatrix
// ---------------------------------------------------------------------------

/**
 * Convert a 6-DoF pose to a 4x4 homogeneous transformation matrix in
 * column-major order.
 *
 * The upper-left 3x3 block is the rotation derived from the quaternion and
 * the fourth column holds the translation.
 *
 * @param pose The pose to convert.
 * @returns    A 16-element `Float64Array` in column-major layout.
 */
export function poseToMatrix(pose: Pose6DoF): Float64Array {
  const m = mat4Identity();
  const r = quatToMat3(quatNormalize(pose.orientation));

  // Column-major 4x4: element at (row, col) = index col*4 + row.
  // Copy the 3x3 rotation (column-major) into the upper-left block.
  // Column 0
  m[0] = r[0]!;
  m[1] = r[1]!;
  m[2] = r[2]!;
  // Column 1
  m[4] = r[3]!;
  m[5] = r[4]!;
  m[6] = r[5]!;
  // Column 2
  m[8] = r[6]!;
  m[9] = r[7]!;
  m[10] = r[8]!;

  // Translation in column 3.
  m[12] = pose.position.x;
  m[13] = pose.position.y;
  m[14] = pose.position.z;

  return m;
}

// ---------------------------------------------------------------------------
// matrixToPose
// ---------------------------------------------------------------------------

/**
 * Extract a `Pose6DoF` from a 4x4 homogeneous transformation matrix.
 *
 * The translation is taken directly from the fourth column. The rotation is
 * recovered by converting the upper-left 3x3 sub-matrix to a quaternion
 * using Shepperd's method (numerically stable for all rotations).
 *
 * @param mat A 16-element column-major `Float64Array`.
 * @returns   The corresponding `Pose6DoF`.
 */
export function matrixToPose(mat: Float64Array): Pose6DoF {
  // Translation: column 3.
  const position: Vector3 = {
    x: mat[12]!,
    y: mat[13]!,
    z: mat[14]!,
  };

  // Extract rotation matrix elements (column-major).
  const m00 = mat[0]!;
  const m10 = mat[1]!;
  const m20 = mat[2]!;
  const m01 = mat[4]!;
  const m11 = mat[5]!;
  const m21 = mat[6]!;
  const m02 = mat[8]!;
  const m12 = mat[9]!;
  const m22 = mat[10]!;

  // Shepperd's method — pick the largest diagonal element to avoid division
  // by near-zero.
  const trace = m00 + m11 + m22;
  let q: Quaternion;

  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1.0);
    q = {
      w: 0.25 / s,
      x: (m21 - m12) * s,
      y: (m02 - m20) * s,
      z: (m10 - m01) * s,
    };
  } else if (m00 > m11 && m00 > m22) {
    const s = 2.0 * Math.sqrt(1.0 + m00 - m11 - m22);
    q = {
      w: (m21 - m12) / s,
      x: 0.25 * s,
      y: (m01 + m10) / s,
      z: (m02 + m20) / s,
    };
  } else if (m11 > m22) {
    const s = 2.0 * Math.sqrt(1.0 + m11 - m00 - m22);
    q = {
      w: (m02 - m20) / s,
      x: (m01 + m10) / s,
      y: 0.25 * s,
      z: (m12 + m21) / s,
    };
  } else {
    const s = 2.0 * Math.sqrt(1.0 + m22 - m00 - m11);
    q = {
      w: (m10 - m01) / s,
      x: (m02 + m20) / s,
      y: (m12 + m21) / s,
      z: 0.25 * s,
    };
  }

  return { position, orientation: quatNormalize(q) };
}

// ---------------------------------------------------------------------------
// composePoses
// ---------------------------------------------------------------------------

/**
 * Compose (chain) two 6-DoF transforms: result = parent * child.
 *
 * The child's position is rotated by the parent's orientation and then offset
 * by the parent's position. The orientations are composed by quaternion
 * multiplication.
 *
 * @param parent The parent (outer) transform.
 * @param child  The child (inner) transform, expressed in the parent's frame.
 * @returns      The combined transform in world space.
 */
export function composePoses(parent: Pose6DoF, child: Pose6DoF): Pose6DoF {
  // Rotate child position by parent orientation.
  const rotatedChildPos = rotateVectorByQuat(
    parent.orientation,
    child.position,
  );

  const position = vec3Add(parent.position, rotatedChildPos);
  const orientation = quatNormalize(
    quatMultiply(parent.orientation, child.orientation),
  );

  return { position, orientation };
}

// ---------------------------------------------------------------------------
// inversePose
// ---------------------------------------------------------------------------

/**
 * Compute the inverse of a 6-DoF pose.
 *
 * If `pose` maps from frame A to frame B, then `inversePose(pose)` maps from
 * frame B back to frame A.
 *
 * @param pose The pose to invert.
 * @returns    The inverse pose.
 */
export function inversePose(pose: Pose6DoF): Pose6DoF {
  // Inverse quaternion: conjugate of a unit quaternion.
  const invOri: Quaternion = quatNormalize({
    x: -pose.orientation.x,
    y: -pose.orientation.y,
    z: -pose.orientation.z,
    w: pose.orientation.w,
  });

  // Inverse position: -R^{-1} * t  =  rotate(-t) by invOri.
  const negPos: Vector3 = {
    x: -pose.position.x,
    y: -pose.position.y,
    z: -pose.position.z,
  };
  const invPos = rotateVectorByQuat(invOri, negPos);

  return { position: invPos, orientation: invOri };
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Rotate a vector by a unit quaternion: v' = q * v * q^{-1}.
 *
 * Optimised formulation that avoids full quaternion multiplications:
 *   v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
 */
function rotateVectorByQuat(q: Quaternion, v: Vector3): Vector3 {
  const qx = q.x;
  const qy = q.y;
  const qz = q.z;
  const qw = q.w;

  // t = 2 * cross(q.xyz, v)
  const tx = 2 * (qy * v.z - qz * v.y);
  const ty = 2 * (qz * v.x - qx * v.z);
  const tz = 2 * (qx * v.y - qy * v.x);

  return {
    x: v.x + qw * tx + (qy * tz - qz * ty),
    y: v.y + qw * ty + (qz * tx - qx * tz),
    z: v.z + qw * tz + (qx * ty - qy * tx),
  };
}
