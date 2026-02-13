// ---------------------------------------------------------------------------
// CV-10: Dimension Estimation — real-world measurements from bounding
// boxes, depth, and 3D points.
// ---------------------------------------------------------------------------

import type { BBox2D, BBox3D, CameraIntrinsics, Vector3 } from '../types.js';

// ---------------------------------------------------------------------------
// estimateRealDimensions
// ---------------------------------------------------------------------------

/**
 * Estimate real-world width and height of an object given its 2D bounding
 * box, depth (distance from camera), and camera intrinsics.
 *
 * Uses the pinhole camera model to back-project the bounding box edges:
 *
 *   realWidth  = bbox.width  * depth / fx
 *   realHeight = bbox.height * depth / fy
 *
 * @param bbox       2D bounding box of the object in the image.
 * @param depth      Estimated depth of the object in metres.
 * @param intrinsics Camera intrinsics (pinhole model).
 * @returns Object with real-world `width` and `height` in metres.
 */
export function estimateRealDimensions(
  bbox: BBox2D,
  depth: number,
  intrinsics: CameraIntrinsics,
): { width: number; height: number } {
  return {
    width: (bbox.width * depth) / intrinsics.fx,
    height: (bbox.height * depth) / intrinsics.fy,
  };
}

// ---------------------------------------------------------------------------
// measureDistance
// ---------------------------------------------------------------------------

/**
 * Compute the Euclidean distance between two 3D points.
 *
 * @param p1 First point.
 * @param p2 Second point.
 * @returns Distance in the same units as the input coordinates.
 */
export function measureDistance(p1: Vector3, p2: Vector3): number {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const dz = p2.z - p1.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

// ---------------------------------------------------------------------------
// estimateVolume
// ---------------------------------------------------------------------------

/**
 * Compute the volume of a 3D axis-aligned bounding box.
 *
 * The volume is calculated as `2 * halfExtents.x * 2 * halfExtents.y *
 * 2 * halfExtents.z` — i.e., the full extents multiplied together.
 *
 * @param bbox3D A 3D bounding box with centre and half-extents.
 * @returns Volume in cubic units.
 */
export function estimateVolume(bbox3D: BBox3D): number {
  return (
    2 * bbox3D.halfExtents.x *
    2 * bbox3D.halfExtents.y *
    2 * bbox3D.halfExtents.z
  );
}
