// ---------------------------------------------------------------------------
// CV-10: Bounding Box Utilities — creation, area, containment, oriented
// bbox fitting, and merging.
// ---------------------------------------------------------------------------

import type { BBox2D, BBox3D, OrientedBBox3D, Vec2, Vector3 } from '../types.js';

// ---------------------------------------------------------------------------
// createBBox2D
// ---------------------------------------------------------------------------

/**
 * Create a 2D axis-aligned bounding box.
 *
 * @param x      Top-left x coordinate (pixels).
 * @param y      Top-left y coordinate (pixels).
 * @param width  Box width (pixels).
 * @param height Box height (pixels).
 * @returns A new {@link BBox2D} with confidence 1 and empty label.
 */
export function createBBox2D(
  x: number,
  y: number,
  width: number,
  height: number,
): BBox2D {
  return { x, y, width, height, confidence: 1, label: '' };
}

// ---------------------------------------------------------------------------
// createBBox3D
// ---------------------------------------------------------------------------

/**
 * Create a 3D axis-aligned bounding box from min/max corners.
 *
 * The centre is computed as the midpoint and the half-extents as half the
 * difference along each axis.
 *
 * @param min Minimum corner of the AABB.
 * @param max Maximum corner of the AABB.
 * @returns A new {@link BBox3D} with confidence 1 and empty label.
 */
export function createBBox3D(min: Vector3, max: Vector3): BBox3D {
  return {
    center: {
      x: (min.x + max.x) / 2,
      y: (min.y + max.y) / 2,
      z: (min.z + max.z) / 2,
    },
    halfExtents: {
      x: (max.x - min.x) / 2,
      y: (max.y - min.y) / 2,
      z: (max.z - min.z) / 2,
    },
    confidence: 1,
    label: '',
  };
}

// ---------------------------------------------------------------------------
// bboxArea
// ---------------------------------------------------------------------------

/**
 * Compute the area of a 2D bounding box.
 *
 * @param bbox A 2D bounding box.
 * @returns Area in square pixels.
 */
export function bboxArea(bbox: BBox2D): number {
  return bbox.width * bbox.height;
}

// ---------------------------------------------------------------------------
// bboxContains
// ---------------------------------------------------------------------------

/**
 * Test whether a 2D bounding box contains a point.
 *
 * The test is inclusive on the min edges and exclusive on the max edges:
 * `x <= px < x + width` and `y <= py < y + height`.
 *
 * @param bbox  A 2D bounding box.
 * @param point A 2D point.
 * @returns `true` if the point lies within the box.
 */
export function bboxContains(bbox: BBox2D, point: Vec2): boolean {
  return (
    point.x >= bbox.x &&
    point.x < bbox.x + bbox.width &&
    point.y >= bbox.y &&
    point.y < bbox.y + bbox.height
  );
}

// ---------------------------------------------------------------------------
// fitOrientedBBox
// ---------------------------------------------------------------------------

/**
 * Fit a minimum oriented bounding box to a set of 2D points using a
 * PCA-based approach.
 *
 * 1. Compute the centroid and covariance matrix of the points.
 * 2. Find the principal axes via the 2x2 eigenvector decomposition.
 * 3. Project all points onto the principal axes to determine the extents.
 * 4. Build an {@link OrientedBBox3D} with z-extents set to 0.
 *
 * The orientation quaternion encodes a rotation about the Z axis matching
 * the principal direction.
 *
 * @param points Array of 2D points (at least 1).
 * @returns An oriented bounding box in the XY plane (z = 0).
 */
export function fitOrientedBBox(points: Vec2[]): OrientedBBox3D {
  const n = points.length;

  if (n === 0) {
    return {
      center: { x: 0, y: 0, z: 0 },
      halfExtents: { x: 0, y: 0, z: 0 },
      orientation: { x: 0, y: 0, z: 0, w: 1 },
      confidence: 0,
      label: '',
    };
  }

  // 1. Centroid
  let cx = 0;
  let cy = 0;
  for (let i = 0; i < n; i++) {
    const p = points[i]!;
    cx += p.x;
    cy += p.y;
  }
  cx /= n;
  cy /= n;

  // 2. Covariance matrix (2x2 symmetric)
  let cxx = 0;
  let cxy = 0;
  let cyy = 0;
  for (let i = 0; i < n; i++) {
    const p = points[i]!;
    const dx = p.x - cx;
    const dy = p.y - cy;
    cxx += dx * dx;
    cxy += dx * dy;
    cyy += dy * dy;
  }
  cxx /= n;
  cxy /= n;
  cyy /= n;

  // 3. Eigendecomposition of 2x2 symmetric matrix
  //    eigenvalues of [[a, b],[b, c]]:
  //    lambda = 0.5 * (a + c) +/- 0.5 * sqrt((a - c)^2 + 4b^2)
  const trace = cxx + cyy;
  const det = cxx * cyy - cxy * cxy;
  const disc = Math.sqrt(Math.max(0, trace * trace / 4 - det));
  // We only need the principal eigenvector (largest eigenvalue)
  const _lambda1 = trace / 2 + disc;

  // Eigenvector for the largest eigenvalue
  let evx: number;
  let evy: number;

  if (Math.abs(cxy) > 1e-15) {
    evx = _lambda1 - cyy;
    evy = cxy;
  } else if (cxx >= cyy) {
    evx = 1;
    evy = 0;
  } else {
    evx = 0;
    evy = 1;
  }

  // Normalize
  const len = Math.sqrt(evx * evx + evy * evy);
  if (len > 1e-15) {
    evx /= len;
    evy /= len;
  }

  // Secondary axis (perpendicular)
  const svx = -evy;
  const svy = evx;

  // 4. Project points onto axes to find extents
  let minP = Infinity;
  let maxP = -Infinity;
  let minS = Infinity;
  let maxS = -Infinity;

  for (let i = 0; i < n; i++) {
    const p = points[i]!;
    const dx = p.x - cx;
    const dy = p.y - cy;
    const projP = dx * evx + dy * evy;
    const projS = dx * svx + dy * svy;
    if (projP < minP) minP = projP;
    if (projP > maxP) maxP = projP;
    if (projS < minS) minS = projS;
    if (projS > maxS) maxS = projS;
  }

  const halfW = (maxP - minP) / 2;
  const halfH = (maxS - minS) / 2;
  const midP = (minP + maxP) / 2;
  const midS = (minS + maxS) / 2;

  // Oriented centre in world space
  const centerX = cx + midP * evx + midS * svx;
  const centerY = cy + midP * evy + midS * svy;

  // Build quaternion for rotation about Z axis by angle theta
  const theta = Math.atan2(evy, evx);
  const halfTheta = theta / 2;

  return {
    center: { x: centerX, y: centerY, z: 0 },
    halfExtents: { x: halfW, y: halfH, z: 0 },
    orientation: {
      x: 0,
      y: 0,
      z: Math.sin(halfTheta),
      w: Math.cos(halfTheta),
    },
    confidence: 1,
    label: '',
  };
}

// ---------------------------------------------------------------------------
// mergeBBoxes
// ---------------------------------------------------------------------------

/**
 * Compute the smallest axis-aligned bounding box that encloses all
 * given bounding boxes.
 *
 * If the input array is empty the returned box has zero dimensions at
 * the origin.
 *
 * @param boxes Array of 2D bounding boxes.
 * @returns The enclosing bounding box.
 */
export function mergeBBoxes(boxes: BBox2D[]): BBox2D {
  if (boxes.length === 0) {
    return { x: 0, y: 0, width: 0, height: 0, confidence: 0, label: '' };
  }

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (let i = 0; i < boxes.length; i++) {
    const b = boxes[i]!;
    if (b.x < minX) minX = b.x;
    if (b.y < minY) minY = b.y;
    const bx2 = b.x + b.width;
    const by2 = b.y + b.height;
    if (bx2 > maxX) maxX = bx2;
    if (by2 > maxY) maxY = by2;
  }

  // Average confidence across merged boxes
  let totalConf = 0;
  for (let i = 0; i < boxes.length; i++) {
    totalConf += boxes[i]!.confidence;
  }

  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY,
    confidence: totalConf / boxes.length,
    label: '',
  };
}
