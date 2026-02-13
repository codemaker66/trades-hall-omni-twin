// ---------------------------------------------------------------------------
// CV-3: Raycasting — ray-primitive intersection routines for compositing
// and picking in the Gaussian-splat / mesh hybrid renderer.
// ---------------------------------------------------------------------------

import type { RaycastHit, Vector3 } from '../types.js';
import { vec3Cross, vec3Dot, vec3Sub, vec3Add, vec3Scale } from '../types.js';

// ---------------------------------------------------------------------------
// rayEllipsoidIntersection
// ---------------------------------------------------------------------------

/**
 * Ray vs axis-aligned ellipsoid intersection.
 *
 * The ellipsoid is defined by a centre point and three radii along the world
 * axes.  We transform the problem into unit-sphere space (divide by radii),
 * solve the standard ray-sphere quadratic, and return the nearest positive
 * parameter `t` or `null` on miss.
 *
 * @param rayOrigin Ray origin.
 * @param rayDir    Ray direction (need not be normalised).
 * @param center    Ellipsoid centre.
 * @param radii     Semi-axis lengths along x, y, z.
 * @returns         Parameter t of the nearest intersection, or null.
 */
export function rayEllipsoidIntersection(
  rayOrigin: Vector3,
  rayDir: Vector3,
  center: Vector3,
  radii: Vector3,
): number | null {
  // Guard against degenerate radii
  if (radii.x === 0 || radii.y === 0 || radii.z === 0) return null;

  // Transform ray into unit-sphere space: divide origin offset and direction
  // by the radii so the ellipsoid becomes a unit sphere at the origin.
  const ox = (rayOrigin.x - center.x) / radii.x;
  const oy = (rayOrigin.y - center.y) / radii.y;
  const oz = (rayOrigin.z - center.z) / radii.z;

  const dx = rayDir.x / radii.x;
  const dy = rayDir.y / radii.y;
  const dz = rayDir.z / radii.z;

  // Solve quadratic: |o + t*d|^2 = 1
  const a = dx * dx + dy * dy + dz * dz;
  const b = 2 * (ox * dx + oy * dy + oz * dz);
  const c = ox * ox + oy * oy + oz * oz - 1;

  const discriminant = b * b - 4 * a * c;
  if (discriminant < 0) return null;

  const sqrtDisc = Math.sqrt(discriminant);
  const t0 = (-b - sqrtDisc) / (2 * a);
  const t1 = (-b + sqrtDisc) / (2 * a);

  // Return the smallest positive t
  if (t0 > 0) return t0;
  if (t1 > 0) return t1;
  return null;
}

// ---------------------------------------------------------------------------
// rayTriangleIntersection
// ---------------------------------------------------------------------------

const EPSILON = 1e-12;

/**
 * Moller-Trumbore ray-triangle intersection.
 *
 * Returns a {@link RaycastHit} on intersection (with distance, hit point,
 * face normal, and primitiveId = 0) or `null` on miss.
 *
 * @param rayOrigin Ray origin.
 * @param rayDir    Ray direction (need not be normalised).
 * @param v0        Triangle vertex 0.
 * @param v1        Triangle vertex 1.
 * @param v2        Triangle vertex 2.
 */
export function rayTriangleIntersection(
  rayOrigin: Vector3,
  rayDir: Vector3,
  v0: Vector3,
  v1: Vector3,
  v2: Vector3,
): RaycastHit | null {
  const edge1 = vec3Sub(v1, v0);
  const edge2 = vec3Sub(v2, v0);

  const h = vec3Cross(rayDir, edge2);
  const det = vec3Dot(edge1, h);

  // Ray parallel to the triangle plane
  if (det > -EPSILON && det < EPSILON) return null;

  const invDet = 1 / det;
  const s = vec3Sub(rayOrigin, v0);
  const u = vec3Dot(s, h) * invDet;
  if (u < 0 || u > 1) return null;

  const q = vec3Cross(s, edge1);
  const v = vec3Dot(rayDir, q) * invDet;
  if (v < 0 || u + v > 1) return null;

  const t = vec3Dot(edge2, q) * invDet;
  if (t < EPSILON) return null;

  // Compute the hit point
  const point = vec3Add(rayOrigin, vec3Scale(rayDir, t));

  // Face normal (not normalised to unit length; caller can normalise)
  const normal = vec3Cross(edge1, edge2);
  const nLen = Math.sqrt(vec3Dot(normal, normal));
  const normalised: Vector3 =
    nLen > EPSILON
      ? { x: normal.x / nLen, y: normal.y / nLen, z: normal.z / nLen }
      : { x: 0, y: 0, z: 0 };

  return {
    distance: t,
    point,
    normal: normalised,
    primitiveId: 0,
  };
}

// ---------------------------------------------------------------------------
// rayAABBIntersection
// ---------------------------------------------------------------------------

/**
 * Slab-method ray vs axis-aligned bounding box (AABB) intersection.
 *
 * Returns the parameter `t` of the nearest intersection (entry point) or
 * `null` if the ray misses the box.  Handles rays originating inside the box
 * (returns 0).
 *
 * @param rayOrigin Ray origin.
 * @param rayDir    Ray direction (need not be normalised but must not be zero).
 * @param boxMin    AABB minimum corner.
 * @param boxMax    AABB maximum corner.
 */
export function rayAABBIntersection(
  rayOrigin: Vector3,
  rayDir: Vector3,
  boxMin: Vector3,
  boxMax: Vector3,
): number | null {
  let tMin = -Infinity;
  let tMax = Infinity;

  // --- X slab ---
  if (Math.abs(rayDir.x) < EPSILON) {
    // Ray is parallel to the X slab
    if (rayOrigin.x < boxMin.x || rayOrigin.x > boxMax.x) return null;
  } else {
    const invD = 1 / rayDir.x;
    let t1 = (boxMin.x - rayOrigin.x) * invD;
    let t2 = (boxMax.x - rayOrigin.x) * invD;
    if (t1 > t2) {
      const tmp = t1;
      t1 = t2;
      t2 = tmp;
    }
    tMin = Math.max(tMin, t1);
    tMax = Math.min(tMax, t2);
    if (tMin > tMax) return null;
  }

  // --- Y slab ---
  if (Math.abs(rayDir.y) < EPSILON) {
    if (rayOrigin.y < boxMin.y || rayOrigin.y > boxMax.y) return null;
  } else {
    const invD = 1 / rayDir.y;
    let t1 = (boxMin.y - rayOrigin.y) * invD;
    let t2 = (boxMax.y - rayOrigin.y) * invD;
    if (t1 > t2) {
      const tmp = t1;
      t1 = t2;
      t2 = tmp;
    }
    tMin = Math.max(tMin, t1);
    tMax = Math.min(tMax, t2);
    if (tMin > tMax) return null;
  }

  // --- Z slab ---
  if (Math.abs(rayDir.z) < EPSILON) {
    if (rayOrigin.z < boxMin.z || rayOrigin.z > boxMax.z) return null;
  } else {
    const invD = 1 / rayDir.z;
    let t1 = (boxMin.z - rayOrigin.z) * invD;
    let t2 = (boxMax.z - rayOrigin.z) * invD;
    if (t1 > t2) {
      const tmp = t1;
      t1 = t2;
      t2 = tmp;
    }
    tMin = Math.max(tMin, t1);
    tMax = Math.min(tMax, t2);
    if (tMin > tMax) return null;
  }

  // If the closest intersection is behind the ray, the ray starts inside
  // the box (or misses in the negative half-space).
  if (tMax < 0) return null;

  // If tMin < 0 the ray originates inside the box; return 0 (entry at origin).
  return tMin >= 0 ? tMin : 0;
}
