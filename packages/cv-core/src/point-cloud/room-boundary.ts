// ---------------------------------------------------------------------------
// CV-8: Room Boundary Extraction — RANSAC plane detection, Douglas-Peucker
// polyline simplification, and floor-boundary extraction from point clouds.
// ---------------------------------------------------------------------------

import type { PointCloud, PRNG, RANSACPlaneResult, Vec2 } from '../types.js';
import { vec3Dot, vec3Normalize, vec3Cross, vec3Sub } from '../types.js';

// ---------------------------------------------------------------------------
// ransacPlaneDetection
// ---------------------------------------------------------------------------

/**
 * Fit a plane to a 3D point cloud using RANSAC.
 *
 * Each iteration samples 3 random non-collinear points, computes the plane
 * through them, and counts the number of inlier points (those within
 * `threshold` of the plane).  The plane with the most inliers is returned.
 *
 * Plane equation: dot(normal, point) - distance = 0, where `normal` is a
 * unit vector and `distance = dot(normal, pointOnPlane)`.
 *
 * @param cloud     Input point cloud.
 * @param threshold Maximum point-to-plane distance to count as an inlier (metres).
 * @param maxIter   Maximum RANSAC iterations.
 * @param rng       Seedable PRNG.
 * @returns         {@link RANSACPlaneResult} with the best-fit plane.
 */
export function ransacPlaneDetection(
  cloud: PointCloud,
  threshold: number,
  maxIter: number,
  rng: PRNG,
): RANSACPlaneResult {
  const n = cloud.count;
  const pos = cloud.positions;

  let bestNormal = { x: 0, y: 1, z: 0 };
  let bestDist = 0;
  let bestInlierCount = 0;
  let bestInlierIndices = new Uint32Array(0);

  if (n < 3) {
    return {
      normal: bestNormal,
      distance: 0,
      inlierIndices: bestInlierIndices,
      inlierCount: 0,
      inlierRatio: 0,
    };
  }

  for (let iter = 0; iter < maxIter; iter++) {
    // Sample 3 distinct random indices.
    const i0 = Math.floor(rng() * n);
    let i1 = Math.floor(rng() * n);
    while (i1 === i0) i1 = Math.floor(rng() * n);
    let i2 = Math.floor(rng() * n);
    while (i2 === i0 || i2 === i1) i2 = Math.floor(rng() * n);

    const p0 = { x: pos[i0 * 3]!, y: pos[i0 * 3 + 1]!, z: pos[i0 * 3 + 2]! };
    const p1 = { x: pos[i1 * 3]!, y: pos[i1 * 3 + 1]!, z: pos[i1 * 3 + 2]! };
    const p2 = { x: pos[i2 * 3]!, y: pos[i2 * 3 + 1]!, z: pos[i2 * 3 + 2]! };

    // Plane normal from cross product of two edge vectors.
    const normal = vec3Normalize(vec3Cross(vec3Sub(p1, p0), vec3Sub(p2, p0)));

    // Skip degenerate (collinear) samples.
    const len = Math.sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    if (len < 0.5) continue;

    const dist = vec3Dot(normal, p0);

    // Count inliers.
    const inliers: number[] = [];
    for (let j = 0; j < n; j++) {
      const pj = j * 3;
      const d = Math.abs(
        normal.x * pos[pj]! +
        normal.y * pos[pj + 1]! +
        normal.z * pos[pj + 2]! -
        dist,
      );
      if (d <= threshold) {
        inliers.push(j);
      }
    }

    if (inliers.length > bestInlierCount) {
      bestInlierCount = inliers.length;
      bestNormal = normal;
      bestDist = dist;
      bestInlierIndices = new Uint32Array(inliers);
    }
  }

  return {
    normal: bestNormal,
    distance: bestDist,
    inlierIndices: bestInlierIndices,
    inlierCount: bestInlierCount,
    inlierRatio: n > 0 ? bestInlierCount / n : 0,
  };
}

// ---------------------------------------------------------------------------
// douglasPeucker
// ---------------------------------------------------------------------------

/**
 * Simplify a 2D polyline using the Ramer-Douglas-Peucker algorithm.
 *
 * Recursively finds the point with the greatest perpendicular distance from
 * the line segment between the first and last points.  If that distance
 * exceeds `epsilon`, the polyline is split and simplified recursively;
 * otherwise the intermediate points are discarded.
 *
 * @param points  Ordered 2D polyline vertices.
 * @param epsilon Distance tolerance (metres).
 * @returns       Simplified polyline.
 */
export function douglasPeucker(points: Vec2[], epsilon: number): Vec2[] {
  if (points.length <= 2) return points.slice();

  // Find the point with the maximum distance from the line between first and last.
  const first = points[0]!;
  const last = points[points.length - 1]!;

  let maxDist = 0;
  let maxIdx = 0;

  for (let i = 1; i < points.length - 1; i++) {
    const d = perpendicularDistance(points[i]!, first, last);
    if (d > maxDist) {
      maxDist = d;
      maxIdx = i;
    }
  }

  if (maxDist > epsilon) {
    // Recurse on both halves.
    const left = douglasPeucker(points.slice(0, maxIdx + 1), epsilon);
    const right = douglasPeucker(points.slice(maxIdx), epsilon);

    // Merge (drop the duplicate point at the junction).
    return left.slice(0, -1).concat(right);
  }

  // All intermediate points are within tolerance; keep only endpoints.
  return [first, last];
}

/**
 * Perpendicular distance from a point to a line segment defined by two endpoints.
 */
function perpendicularDistance(p: Vec2, a: Vec2, b: Vec2): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lenSq = dx * dx + dy * dy;

  if (lenSq < 1e-15) {
    // Degenerate segment — return distance to a.
    const ex = p.x - a.x;
    const ey = p.y - a.y;
    return Math.sqrt(ex * ex + ey * ey);
  }

  // Area of the parallelogram / base length = perpendicular distance.
  const area = Math.abs(dx * (a.y - p.y) - (a.x - p.x) * dy);
  return area / Math.sqrt(lenSq);
}

// ---------------------------------------------------------------------------
// extractFloorBoundary
// ---------------------------------------------------------------------------

/**
 * Extract a 2D floor boundary from a 3D point cloud.
 *
 * Steps:
 * 1. Select points near the floor plane (|z - floorHeight| <= tolerance).
 * 2. Project those points to 2D (x, y).
 * 3. Compute a convex hull of the projected points.
 * 4. Return the hull as an ordered polygon.
 *
 * @param cloud       Input 3D point cloud.
 * @param floorHeight Z-coordinate of the floor plane (metres).
 * @param tolerance   Thickness of the floor slab to include points (metres).
 * @returns           Ordered 2D polygon vertices of the floor boundary.
 */
export function extractFloorBoundary(
  cloud: PointCloud,
  floorHeight: number,
  tolerance: number,
): Vec2[] {
  const n = cloud.count;
  const pos = cloud.positions;

  // Step 1 + 2: filter and project.
  const pts2d: Vec2[] = [];
  for (let i = 0; i < n; i++) {
    const z = pos[i * 3 + 2]!;
    if (Math.abs(z - floorHeight) <= tolerance) {
      pts2d.push({ x: pos[i * 3]!, y: pos[i * 3 + 1]! });
    }
  }

  if (pts2d.length < 3) return pts2d;

  // Step 3: convex hull (Andrew's monotone chain).
  return convexHull2D(pts2d);
}

// ---------------------------------------------------------------------------
// convexHull2D  (Andrew's monotone chain)
// ---------------------------------------------------------------------------

/**
 * Compute the 2D convex hull of a set of points using Andrew's monotone
 * chain algorithm.  Returns the hull vertices in counter-clockwise order.
 */
function convexHull2D(points: Vec2[]): Vec2[] {
  const sorted = points.slice().sort((a, b) =>
    a.x !== b.x ? a.x - b.x : a.y - b.y,
  );

  const n = sorted.length;
  if (n <= 1) return sorted;

  // Build lower hull.
  const lower: Vec2[] = [];
  for (let i = 0; i < n; i++) {
    while (lower.length >= 2 && cross2D(lower[lower.length - 2]!, lower[lower.length - 1]!, sorted[i]!) <= 0) {
      lower.pop();
    }
    lower.push(sorted[i]!);
  }

  // Build upper hull.
  const upper: Vec2[] = [];
  for (let i = n - 1; i >= 0; i--) {
    while (upper.length >= 2 && cross2D(upper[upper.length - 2]!, upper[upper.length - 1]!, sorted[i]!) <= 0) {
      upper.pop();
    }
    upper.push(sorted[i]!);
  }

  // Remove the last point of each half because it's repeated.
  lower.pop();
  upper.pop();

  return lower.concat(upper);
}

/** 2D cross product of vectors OA and OB. Positive if counter-clockwise. */
function cross2D(o: Vec2, a: Vec2, b: Vec2): number {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}
