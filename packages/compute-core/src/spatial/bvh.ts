// ---------------------------------------------------------------------------
// HPC-5: Spatial Indexing — Bounding Volume Hierarchy
// ---------------------------------------------------------------------------
// Top-down median split BVH for fast collision detection and raycasting.
// Pure TypeScript, zero external dependencies.
// ---------------------------------------------------------------------------

import type { BoundingBox2D, BVHNode, SpatialItem } from '../types.js';

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

/**
 * Build a BVH from spatial items using top-down median split on the longest
 * axis of the enclosing bounding box at each level.
 *
 * Leaf nodes store a single item index; internal nodes store the union bbox
 * of their children.
 *
 * @param items Array of spatial items. Item indices correspond to positions
 *              in this array and are used in query results.
 * @returns Root of the BVH, or null if items is empty.
 */
export function buildBVH(items: SpatialItem[]): BVHNode | null {
  if (items.length === 0) return null;

  const indices = new Array<number>(items.length);
  for (let i = 0; i < items.length; i++) indices[i] = i;

  return buildRecursive(items, indices, 0, indices.length);
}

function buildRecursive(
  items: SpatialItem[],
  indices: number[],
  lo: number,
  hi: number,
): BVHNode | null {
  const count = hi - lo;
  if (count <= 0) return null;

  // Single item: leaf node
  if (count === 1) {
    const idx = indices[lo]!;
    return {
      bbox: items[idx]!.bbox,
      left: null,
      right: null,
      itemIndex: idx,
    };
  }

  // Compute enclosing bbox for all items in [lo, hi)
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (let i = lo; i < hi; i++) {
    const bb = items[indices[i]!]!.bbox;
    if (bb.minX < minX) minX = bb.minX;
    if (bb.minY < minY) minY = bb.minY;
    if (bb.maxX > maxX) maxX = bb.maxX;
    if (bb.maxY > maxY) maxY = bb.maxY;
  }

  // Choose split axis: longest extent
  const extentX = maxX - minX;
  const extentY = maxY - minY;
  const splitOnX = extentX >= extentY;

  // Sort by centroid on chosen axis and split at median
  if (splitOnX) {
    sortIndicesByCenter(items, indices, lo, hi, true);
  } else {
    sortIndicesByCenter(items, indices, lo, hi, false);
  }

  const mid = lo + (count >> 1);
  const left = buildRecursive(items, indices, lo, mid);
  const right = buildRecursive(items, indices, mid, hi);

  // Union bbox
  const bbox: BoundingBox2D = {
    minX,
    minY,
    maxX,
    maxY,
  };

  return {
    bbox,
    left,
    right,
    itemIndex: -1,
  };
}

/**
 * In-place sort of `indices[lo..hi)` by item centroid on the chosen axis.
 */
function sortIndicesByCenter(
  items: SpatialItem[],
  indices: number[],
  lo: number,
  hi: number,
  useX: boolean,
): void {
  // Simple insertion sort for small ranges, otherwise use quicksort-like sort
  const sub = indices.slice(lo, hi);
  if (useX) {
    sub.sort((a, b) => {
      const ca = items[a]!.bbox.minX + items[a]!.bbox.maxX;
      const cb = items[b]!.bbox.minX + items[b]!.bbox.maxX;
      return ca - cb;
    });
  } else {
    sub.sort((a, b) => {
      const ca = items[a]!.bbox.minY + items[a]!.bbox.maxY;
      const cb = items[b]!.bbox.minY + items[b]!.bbox.maxY;
      return ca - cb;
    });
  }
  for (let i = 0; i < sub.length; i++) {
    indices[lo + i] = sub[i]!;
  }
}

// ---------------------------------------------------------------------------
// BBox intersection helper
// ---------------------------------------------------------------------------

function bboxIntersects(a: BoundingBox2D, b: BoundingBox2D): boolean {
  return a.minX <= b.maxX && a.maxX >= b.minX && a.minY <= b.maxY && a.maxY >= b.minY;
}

// ---------------------------------------------------------------------------
// Query — find items whose bboxes intersect a query box
// ---------------------------------------------------------------------------

/**
 * Find all item indices whose bounding boxes intersect the query box.
 *
 * @param root  Root of the BVH.
 * @param query Axis-aligned bounding box to test against.
 * @param items The original items array (used for leaf bbox checks).
 * @returns Array of indices into the original items array.
 */
export function bvhQuery(
  root: BVHNode | null,
  query: BoundingBox2D,
  items: SpatialItem[],
): number[] {
  const results: number[] = [];
  if (root === null) return results;
  queryRecursive(root, query, items, results);
  return results;
}

function queryRecursive(
  node: BVHNode,
  query: BoundingBox2D,
  items: SpatialItem[],
  results: number[],
): void {
  if (!bboxIntersects(node.bbox, query)) return;

  if (node.itemIndex >= 0) {
    // Leaf: double-check exact item bbox
    if (bboxIntersects(items[node.itemIndex]!.bbox, query)) {
      results.push(node.itemIndex);
    }
    return;
  }

  if (node.left) queryRecursive(node.left, query, items, results);
  if (node.right) queryRecursive(node.right, query, items, results);
}

// ---------------------------------------------------------------------------
// Raycast — first item hit by a 2D ray
// ---------------------------------------------------------------------------

/**
 * Cast a 2D ray and return the index of the first item hit, or null.
 *
 * The ray is defined by an origin (originX, originY) and direction (dirX, dirY).
 * Direction does not need to be normalized.
 *
 * Uses slab intersection test for AABB-ray intersection and picks the
 * nearest hit along the ray's positive direction.
 *
 * @returns Index into the original items array, or null if no hit.
 */
export function bvhRaycast(
  root: BVHNode | null,
  originX: number,
  originY: number,
  dirX: number,
  dirY: number,
  items: SpatialItem[],
): number | null {
  if (root === null) return null;

  const invDirX = dirX !== 0 ? 1 / dirX : Infinity;
  const invDirY = dirY !== 0 ? 1 / dirY : Infinity;

  const result = { hitIndex: -1, hitT: Infinity };
  raycastRecursive(root, originX, originY, invDirX, invDirY, items, result);

  return result.hitIndex >= 0 ? result.hitIndex : null;
}

function rayBBoxIntersectT(
  bbox: BoundingBox2D,
  ox: number,
  oy: number,
  invDx: number,
  invDy: number,
): number {
  // Slab method for 2D AABB-ray intersection
  let tmin: number;
  let tmax: number;

  if (invDx >= 0) {
    tmin = (bbox.minX - ox) * invDx;
    tmax = (bbox.maxX - ox) * invDx;
  } else {
    tmin = (bbox.maxX - ox) * invDx;
    tmax = (bbox.minX - ox) * invDx;
  }

  let tymin: number;
  let tymax: number;

  if (invDy >= 0) {
    tymin = (bbox.minY - oy) * invDy;
    tymax = (bbox.maxY - oy) * invDy;
  } else {
    tymin = (bbox.maxY - oy) * invDy;
    tymax = (bbox.minY - oy) * invDy;
  }

  if (tmin > tymax || tymin > tmax) return Infinity;

  if (tymin > tmin) tmin = tymin;
  if (tymax < tmax) tmax = tymax;

  // Ray must go forward (tmax >= 0) and entry must be valid
  if (tmax < 0) return Infinity;

  return tmin >= 0 ? tmin : tmax;
}

function raycastRecursive(
  node: BVHNode,
  ox: number,
  oy: number,
  invDx: number,
  invDy: number,
  items: SpatialItem[],
  result: { hitIndex: number; hitT: number },
): void {
  const t = rayBBoxIntersectT(node.bbox, ox, oy, invDx, invDy);
  if (t >= result.hitT) return; // Already found a closer hit

  if (node.itemIndex >= 0) {
    // Leaf node: test actual item bbox
    const itemT = rayBBoxIntersectT(items[node.itemIndex]!.bbox, ox, oy, invDx, invDy);
    if (itemT < result.hitT) {
      result.hitT = itemT;
      result.hitIndex = node.itemIndex;
    }
    return;
  }

  // Visit both children; visit nearer one first for earlier pruning
  if (node.left && node.right) {
    const tLeft = rayBBoxIntersectT(node.left.bbox, ox, oy, invDx, invDy);
    const tRight = rayBBoxIntersectT(node.right.bbox, ox, oy, invDx, invDy);

    if (tLeft < tRight) {
      if (tLeft < result.hitT)
        raycastRecursive(node.left, ox, oy, invDx, invDy, items, result);
      if (tRight < result.hitT)
        raycastRecursive(node.right, ox, oy, invDx, invDy, items, result);
    } else {
      if (tRight < result.hitT)
        raycastRecursive(node.right, ox, oy, invDx, invDy, items, result);
      if (tLeft < result.hitT)
        raycastRecursive(node.left, ox, oy, invDx, invDy, items, result);
    }
  } else if (node.left) {
    raycastRecursive(node.left, ox, oy, invDx, invDy, items, result);
  } else if (node.right) {
    raycastRecursive(node.right, ox, oy, invDx, invDy, items, result);
  }
}

// ---------------------------------------------------------------------------
// Find all pairwise overlaps
// ---------------------------------------------------------------------------

/**
 * Find all pairs of items whose bounding boxes overlap.
 *
 * Uses the BVH to prune the O(n^2) brute-force test. For each leaf,
 * queries the tree for intersecting items and deduplicates pairs.
 *
 * @returns Array of [indexA, indexB] pairs where indexA < indexB.
 */
export function bvhFindOverlaps(
  root: BVHNode | null,
  items: SpatialItem[],
): Array<[number, number]> {
  if (root === null || items.length < 2) return [];

  const pairs: Array<[number, number]> = [];
  const seen = new Set<string>();

  findOverlapsRecursive(root, root, items, pairs, seen);

  return pairs;
}

function findOverlapsRecursive(
  nodeA: BVHNode,
  nodeB: BVHNode,
  items: SpatialItem[],
  pairs: Array<[number, number]>,
  seen: Set<string>,
): void {
  if (!bboxIntersects(nodeA.bbox, nodeB.bbox)) return;

  // Both are leaves
  if (nodeA.itemIndex >= 0 && nodeB.itemIndex >= 0) {
    if (nodeA.itemIndex !== nodeB.itemIndex) {
      const lo = Math.min(nodeA.itemIndex, nodeB.itemIndex);
      const hi = Math.max(nodeA.itemIndex, nodeB.itemIndex);
      const key = `${lo}:${hi}`;
      if (!seen.has(key)) {
        if (bboxIntersects(items[lo]!.bbox, items[hi]!.bbox)) {
          seen.add(key);
          pairs.push([lo, hi]);
        }
      }
    }
    return;
  }

  // A is leaf, B is internal: expand B
  if (nodeA.itemIndex >= 0) {
    if (nodeB.left) findOverlapsRecursive(nodeA, nodeB.left, items, pairs, seen);
    if (nodeB.right) findOverlapsRecursive(nodeA, nodeB.right, items, pairs, seen);
    return;
  }

  // B is leaf, A is internal: expand A
  if (nodeB.itemIndex >= 0) {
    if (nodeA.left) findOverlapsRecursive(nodeA.left, nodeB, items, pairs, seen);
    if (nodeA.right) findOverlapsRecursive(nodeA.right, nodeB, items, pairs, seen);
    return;
  }

  // Both internal: expand the larger one
  if (nodeA.left) {
    if (nodeB.left) findOverlapsRecursive(nodeA.left, nodeB.left, items, pairs, seen);
    if (nodeB.right) findOverlapsRecursive(nodeA.left, nodeB.right, items, pairs, seen);
  }
  if (nodeA.right) {
    if (nodeB.left) findOverlapsRecursive(nodeA.right, nodeB.left, items, pairs, seen);
    if (nodeB.right) findOverlapsRecursive(nodeA.right, nodeB.right, items, pairs, seen);
  }
}

// ---------------------------------------------------------------------------
// Depth
// ---------------------------------------------------------------------------

/**
 * Compute the maximum depth of the BVH.
 */
export function bvhDepth(root: BVHNode | null): number {
  if (root === null) return 0;
  return 1 + Math.max(bvhDepth(root.left), bvhDepth(root.right));
}
