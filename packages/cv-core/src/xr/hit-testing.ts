// ---------------------------------------------------------------------------
// CV-11: XR — Hit Testing
// Ray-plane intersection, surface snapping, and floor hit tests for AR/XR.
// ---------------------------------------------------------------------------

import type { Vector3, Quaternion, HitTestResult, Pose6DoF } from '../types.js';
import { vec3Dot, vec3Sub, vec3Scale, vec3Add } from '../types.js';

// ---------------------------------------------------------------------------
// rayPlaneIntersection
// ---------------------------------------------------------------------------

/**
 * Compute the intersection of a ray with an infinite plane.
 *
 * The plane is defined in Hessian normal form: dot(planeNormal, p) + planeD = 0.
 * Returns `null` if the ray is parallel to the plane (or nearly so) or if the
 * intersection lies behind the ray origin (t < 0).
 *
 * @param rayOrigin   Origin of the ray in world space.
 * @param rayDir      Direction of the ray (need not be unit-length but must not be zero).
 * @param planeNormal Unit normal of the plane.
 * @param planeD      Signed distance from the origin to the plane (negative side of normal).
 * @returns           The intersection point, or `null` if no valid intersection exists.
 */
export function rayPlaneIntersection(
  rayOrigin: Vector3,
  rayDir: Vector3,
  planeNormal: Vector3,
  planeD: number,
): Vector3 | null {
  const denom = vec3Dot(planeNormal, rayDir);

  // Ray is parallel to (or nearly parallel to) the plane.
  if (Math.abs(denom) < 1e-12) return null;

  // Solve for parametric t:
  //   dot(planeNormal, rayOrigin + t * rayDir) + planeD = 0
  //   t = -(dot(planeNormal, rayOrigin) + planeD) / dot(planeNormal, rayDir)
  const t = -(vec3Dot(planeNormal, rayOrigin) + planeD) / denom;

  // Intersection behind the ray origin.
  if (t < 0) return null;

  return vec3Add(rayOrigin, vec3Scale(rayDir, t));
}

// ---------------------------------------------------------------------------
// snapToSurface
// ---------------------------------------------------------------------------

/**
 * Project a point onto a plane defined by a surface normal and a point on the
 * surface.
 *
 * This is useful for "snapping" AR content to a detected surface — the
 * returned point lies on the plane closest to the input point.
 *
 * @param point         The point to snap.
 * @param surfaceNormal Unit normal of the surface plane.
 * @param surfacePoint  Any point that lies on the surface plane.
 * @returns             The projected point on the surface.
 */
export function snapToSurface(
  point: Vector3,
  surfaceNormal: Vector3,
  surfacePoint: Vector3,
): Vector3 {
  // Signed distance from the point to the plane.
  const diff = vec3Sub(point, surfacePoint);
  const signedDist = vec3Dot(diff, surfaceNormal);

  // Move the point along the negative normal direction by the signed distance.
  return vec3Sub(point, vec3Scale(surfaceNormal, signedDist));
}

// ---------------------------------------------------------------------------
// hitTestFloor
// ---------------------------------------------------------------------------

/**
 * Perform an AR-style hit test against a horizontal floor plane at a given Y
 * height.
 *
 * The floor plane normal is (0, 1, 0) and the plane equation is y = floorY.
 * Returns a full `HitTestResult` including a `Pose6DoF` oriented to the floor
 * surface, or `null` if the ray does not intersect the floor.
 *
 * @param rayOrigin Origin of the ray (e.g. camera position).
 * @param rayDir    Direction of the ray (e.g. screen-space pick direction).
 * @param floorY    The Y coordinate of the floor plane.
 * @returns         A hit test result, or `null` if no intersection.
 */
export function hitTestFloor(
  rayOrigin: Vector3,
  rayDir: Vector3,
  floorY: number,
): HitTestResult | null {
  const planeNormal: Vector3 = { x: 0, y: 1, z: 0 };
  // Plane equation: dot((0,1,0), p) - floorY = 0  =>  planeD = -floorY
  const planeD = -floorY;

  const hitPoint = rayPlaneIntersection(rayOrigin, rayDir, planeNormal, planeD);
  if (hitPoint === null) return null;

  const diff = vec3Sub(hitPoint, rayOrigin);
  const distance = Math.sqrt(
    diff.x * diff.x + diff.y * diff.y + diff.z * diff.z,
  );

  // Build a pose with the floor normal as "up" — identity orientation for a
  // horizontal surface (the floor normal aligns with world Y).
  const pose: Pose6DoF = {
    position: hitPoint,
    orientation: { x: 0, y: 0, z: 0, w: 1 } as Quaternion,
  };

  return {
    pose,
    distance,
    type: 'plane',
    confidence: 1.0,
    planeType: 'floor',
  };
}
