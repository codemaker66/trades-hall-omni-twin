// ---------------------------------------------------------------------------
// HPC-5: Spatial Indexing — k-d Tree
// ---------------------------------------------------------------------------
// Recursive median-split k-d tree for nearest-neighbor and radius queries
// in arbitrary dimensions. Pure TypeScript, zero external dependencies.
// ---------------------------------------------------------------------------

import type { KDTreeNode, NearestResult } from '../types.js';

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

/**
 * Build a k-d tree from a set of points in `dimensions`-dimensional space.
 *
 * Points are split along the median of the current split dimension, cycling
 * through dimensions at each level. This yields a balanced tree for most
 * input distributions.
 *
 * @param points   Array of Float64Arrays, each of length `dimensions`.
 * @param ids      Parallel array of unique string IDs.
 * @param dimensions Number of spatial dimensions.
 * @returns Root of the constructed tree, or null if inputs are empty.
 */
export function buildKDTree(
  points: Float64Array[],
  ids: string[],
  dimensions: number,
): KDTreeNode | null {
  if (points.length === 0) return null;

  // Build index array so we can sort without copying Float64Arrays repeatedly
  const indices = new Array<number>(points.length);
  for (let i = 0; i < points.length; i++) indices[i] = i;

  return buildRecursive(points, ids, indices, 0, indices.length, 0, dimensions);
}

function buildRecursive(
  points: Float64Array[],
  ids: string[],
  indices: number[],
  lo: number,
  hi: number,
  depth: number,
  dimensions: number,
): KDTreeNode | null {
  const count = hi - lo;
  if (count <= 0) return null;

  const splitDim = depth % dimensions;

  if (count === 1) {
    const idx = indices[lo]!;
    return {
      point: points[idx]!,
      id: ids[idx]!,
      splitDimension: splitDim,
      left: null,
      right: null,
    };
  }

  // Partial sort to find median index
  selectMedian(points, indices, lo, hi, splitDim);

  const medianOffset = lo + ((count - 1) >> 1);
  const medianIdx = indices[medianOffset]!;

  return {
    point: points[medianIdx]!,
    id: ids[medianIdx]!,
    splitDimension: splitDim,
    left: buildRecursive(points, ids, indices, lo, medianOffset, depth + 1, dimensions),
    right: buildRecursive(points, ids, indices, medianOffset + 1, hi, depth + 1, dimensions),
  };
}

/**
 * In-place nth_element-style partial sort so that the median element
 * ends up at position lo + floor((hi-lo-1)/2) when comparing along `dim`.
 * Uses the introselect / quickselect algorithm.
 */
function selectMedian(
  points: Float64Array[],
  indices: number[],
  lo: number,
  hi: number,
  dim: number,
): void {
  const target = lo + ((hi - lo - 1) >> 1);
  let left = lo;
  let right = hi - 1;

  while (left < right) {
    // Pick pivot via median-of-three
    const mid = (left + right) >> 1;
    if (valueAt(points, indices, mid, dim) < valueAt(points, indices, left, dim)) {
      swap(indices, left, mid);
    }
    if (valueAt(points, indices, right, dim) < valueAt(points, indices, left, dim)) {
      swap(indices, left, right);
    }
    if (valueAt(points, indices, mid, dim) < valueAt(points, indices, right, dim)) {
      swap(indices, mid, right);
    }
    const pivotVal = valueAt(points, indices, right, dim);

    let i = left;
    let j = right - 1;
    for (;;) {
      while (valueAt(points, indices, i, dim) < pivotVal) i++;
      while (j > i && valueAt(points, indices, j, dim) > pivotVal) j--;
      if (i >= j) break;
      swap(indices, i, j);
      i++;
      j--;
    }
    swap(indices, i, right);

    if (i === target) return;
    if (i < target) left = i + 1;
    else right = i - 1;
  }
}

function valueAt(
  points: Float64Array[],
  indices: number[],
  pos: number,
  dim: number,
): number {
  return points[indices[pos]!]![dim]!;
}

function swap(arr: number[], i: number, j: number): void {
  const tmp = arr[i]!;
  arr[i] = arr[j]!;
  arr[j] = tmp;
}

// ---------------------------------------------------------------------------
// k-Nearest Neighbors
// ---------------------------------------------------------------------------

/**
 * Max-heap of fixed capacity `k` used to track the k closest points.
 * The root is the *farthest* of the current best k, allowing O(1) pruning.
 */
interface MaxHeap {
  readonly items: NearestResult[];
  readonly capacity: number;
}

function heapCreate(k: number): MaxHeap {
  return { items: [], capacity: k };
}

function heapPush(heap: MaxHeap, item: NearestResult): void {
  if (heap.items.length < heap.capacity) {
    heap.items.push(item);
    // Sift up
    let i = heap.items.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (heap.items[parent]!.distance >= heap.items[i]!.distance) break;
      const tmp = heap.items[parent]!;
      heap.items[parent] = heap.items[i]!;
      heap.items[i] = tmp;
      i = parent;
    }
  } else if (item.distance < heap.items[0]!.distance) {
    heap.items[0] = item;
    // Sift down
    heapSiftDown(heap, 0);
  }
}

function heapSiftDown(heap: MaxHeap, i: number): void {
  const n = heap.items.length;
  for (;;) {
    let largest = i;
    const left = 2 * i + 1;
    const right = 2 * i + 2;
    if (left < n && heap.items[left]!.distance > heap.items[largest]!.distance) {
      largest = left;
    }
    if (right < n && heap.items[right]!.distance > heap.items[largest]!.distance) {
      largest = right;
    }
    if (largest === i) break;
    const tmp = heap.items[i]!;
    heap.items[i] = heap.items[largest]!;
    heap.items[largest] = tmp;
    i = largest;
  }
}

function heapWorstDist(heap: MaxHeap): number {
  if (heap.items.length < heap.capacity) return Infinity;
  return heap.items[0]!.distance;
}

/**
 * Find the `k` nearest neighbors to `query` in the k-d tree rooted at `root`.
 *
 * Uses a max-heap of size k and aggressively prunes subtrees whose splitting
 * plane distance exceeds the current worst neighbor distance.
 *
 * @returns Results sorted by distance ascending (closest first).
 */
export function kdTreeNearestN(
  root: KDTreeNode,
  query: Float64Array,
  k: number,
  dimensions: number,
): NearestResult[] {
  const heap = heapCreate(k);
  nearestSearch(root, query, heap, dimensions);

  // Sort ascending by distance
  heap.items.sort((a, b) => a.distance - b.distance);
  return heap.items;
}

function nearestSearch(
  node: KDTreeNode | null,
  query: Float64Array,
  heap: MaxHeap,
  dimensions: number,
): void {
  if (node === null) return;

  // Compute squared Euclidean distance from query to this node's point
  let distSq = 0;
  for (let d = 0; d < dimensions; d++) {
    const diff = query[d]! - node.point[d]!;
    distSq += diff * diff;
  }

  const dist = Math.sqrt(distSq);
  heapPush(heap, { id: node.id, distance: dist, point: node.point });

  // Distance from query to the splitting hyperplane
  const splitDim = node.splitDimension;
  const diff = query[splitDim]! - node.point[splitDim]!;

  // Visit nearer subtree first
  const nearer = diff <= 0 ? node.left : node.right;
  const farther = diff <= 0 ? node.right : node.left;

  nearestSearch(nearer, query, heap, dimensions);

  // Prune: only visit farther subtree if the splitting plane is closer
  // than the current worst distance in the heap
  if (Math.abs(diff) < heapWorstDist(heap)) {
    nearestSearch(farther, query, heap, dimensions);
  }
}

// ---------------------------------------------------------------------------
// Radius Search
// ---------------------------------------------------------------------------

/**
 * Find all points within `radius` Euclidean distance of `query`.
 *
 * @returns Results in no particular order. Each result's `distance` is the
 *          Euclidean distance from the query point.
 */
export function kdTreeRadiusSearch(
  root: KDTreeNode,
  query: Float64Array,
  radius: number,
  dimensions: number,
): NearestResult[] {
  const results: NearestResult[] = [];
  radiusSearchRecursive(root, query, radius, dimensions, results);
  return results;
}

function radiusSearchRecursive(
  node: KDTreeNode | null,
  query: Float64Array,
  radius: number,
  dimensions: number,
  results: NearestResult[],
): void {
  if (node === null) return;

  // Compute squared Euclidean distance
  let distSq = 0;
  for (let d = 0; d < dimensions; d++) {
    const diff = query[d]! - node.point[d]!;
    distSq += diff * diff;
  }

  const dist = Math.sqrt(distSq);
  if (dist <= radius) {
    results.push({ id: node.id, distance: dist, point: node.point });
  }

  // Distance from query to the splitting hyperplane
  const splitDim = node.splitDimension;
  const diff = query[splitDim]! - node.point[splitDim]!;

  // Always visit the side of the hyperplane containing the query
  const nearer = diff <= 0 ? node.left : node.right;
  const farther = diff <= 0 ? node.right : node.left;

  radiusSearchRecursive(nearer, query, radius, dimensions, results);

  // Only visit farther side if the hyperplane is within radius
  if (Math.abs(diff) <= radius) {
    radiusSearchRecursive(farther, query, radius, dimensions, results);
  }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/**
 * Count the total number of nodes in the k-d tree.
 */
export function kdTreeSize(root: KDTreeNode | null): number {
  if (root === null) return 0;
  return 1 + kdTreeSize(root.left) + kdTreeSize(root.right);
}
