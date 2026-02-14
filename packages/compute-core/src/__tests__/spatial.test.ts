import { describe, it, expect } from 'vitest';

import {
  // k-d tree
  buildKDTree,
  kdTreeNearestN,
  kdTreeRadiusSearch,
  kdTreeSize,
  // R-tree
  createRTree,
  rtreeInsert,
  rtreeBulkLoad,
  rtreeSearch,
  rtreeRemove,
  rtreeSize,
  bboxArea,
  bboxIntersects,
  bboxContainsBBox,
  // BVH
  buildBVH,
  bvhQuery,
  bvhRaycast,
  bvhFindOverlaps,
  bvhDepth,
  // Grid hash
  createSpatialHashGrid,
  gridInsert,
  gridRemove,
  gridQuery,
  gridNearby,
  gridClear,
  gridSize,
} from '../spatial/index.js';

import type { SpatialItem, BoundingBox2D } from '../types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makePoint(coords: number[]): Float64Array {
  return new Float64Array(coords);
}

function makeSpatialItem(
  id: string,
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
): SpatialItem {
  return {
    id,
    bbox: { minX, minY, maxX, maxY },
    data: null,
  };
}

// ===================================================================
// k-d Tree
// ===================================================================

describe('buildKDTree / kdTreeSize', () => {
  it('returns null and size 0 for empty input', () => {
    const root = buildKDTree([], [], 2);
    expect(root).toBeNull();
    expect(kdTreeSize(root)).toBe(0);
  });

  it('builds a tree with the correct number of nodes', () => {
    const points = [makePoint([0, 0]), makePoint([1, 1]), makePoint([2, 2])];
    const ids = ['a', 'b', 'c'];
    const root = buildKDTree(points, ids, 2);
    expect(root).not.toBeNull();
    expect(kdTreeSize(root)).toBe(3);
  });

  it('builds a balanced tree for a larger dataset', () => {
    const n = 15;
    const points = Array.from({ length: n }, (_, i) => makePoint([i, i * 2]));
    const ids = Array.from({ length: n }, (_, i) => `p${i}`);
    const root = buildKDTree(points, ids, 2);
    expect(kdTreeSize(root)).toBe(n);
  });
});

describe('kdTreeNearestN', () => {
  const points = [
    makePoint([0, 0]),
    makePoint([3, 4]),
    makePoint([1, 1]),
    makePoint([10, 10]),
    makePoint([5, 5]),
  ];
  const ids = ['origin', 'far1', 'close', 'far2', 'mid'];
  const root = buildKDTree(points, ids, 2)!;

  it('finds the exact match as nearest neighbor', () => {
    const results = kdTreeNearestN(root, makePoint([0, 0]), 1, 2);
    expect(results.length).toBe(1);
    expect(results[0]!.id).toBe('origin');
    expect(results[0]!.distance).toBe(0);
  });

  it('returns k items when k <= n', () => {
    const results = kdTreeNearestN(root, makePoint([0, 0]), 3, 2);
    expect(results.length).toBe(3);
    // Sorted ascending by distance — first must be origin
    expect(results[0]!.id).toBe('origin');
  });

  it('returns all items when k > n', () => {
    const results = kdTreeNearestN(root, makePoint([0, 0]), 100, 2);
    expect(results.length).toBe(5);
  });

  it('returns results sorted by ascending distance', () => {
    const results = kdTreeNearestN(root, makePoint([2, 2]), 5, 2);
    for (let i = 1; i < results.length; i++) {
      expect(results[i]!.distance).toBeGreaterThanOrEqual(results[i - 1]!.distance);
    }
  });
});

describe('kdTreeRadiusSearch', () => {
  const points = [
    makePoint([0, 0]),
    makePoint([1, 0]),
    makePoint([2, 0]),
    makePoint([10, 0]),
  ];
  const ids = ['a', 'b', 'c', 'd'];
  const root = buildKDTree(points, ids, 2)!;

  it('returns only points within the given radius', () => {
    const results = kdTreeRadiusSearch(root, makePoint([0, 0]), 1.5, 2);
    const foundIds = results.map((r) => r.id).sort();
    expect(foundIds).toEqual(['a', 'b']);
  });

  it('returns empty for a radius that contains no points', () => {
    const results = kdTreeRadiusSearch(root, makePoint([5, 5]), 0.1, 2);
    expect(results.length).toBe(0);
  });

  it('returns all points for a very large radius', () => {
    const results = kdTreeRadiusSearch(root, makePoint([0, 0]), 100, 2);
    expect(results.length).toBe(4);
  });
});

// ===================================================================
// R-tree
// ===================================================================

describe('R-tree insert and search', () => {
  it('inserts an item and finds it by search', () => {
    const tree = createRTree(4);
    const item = makeSpatialItem('item1', 0, 0, 10, 10);
    rtreeInsert(tree, item);
    expect(rtreeSize(tree)).toBe(1);

    const results = rtreeSearch(tree, { minX: 5, minY: 5, maxX: 15, maxY: 15 });
    expect(results.length).toBe(1);
    expect(results[0]!.id).toBe('item1');
  });

  it('does not find non-intersecting items', () => {
    const tree = createRTree(4);
    rtreeInsert(tree, makeSpatialItem('a', 0, 0, 1, 1));
    const results = rtreeSearch(tree, { minX: 5, minY: 5, maxX: 6, maxY: 6 });
    expect(results.length).toBe(0);
  });
});

describe('rtreeBulkLoad', () => {
  it('loads many items efficiently and finds them all via search', () => {
    const tree = createRTree(4);
    const items: SpatialItem[] = [];
    for (let i = 0; i < 100; i++) {
      items.push(makeSpatialItem(`item${i}`, i, i, i + 1, i + 1));
    }
    rtreeBulkLoad(tree, items);
    expect(rtreeSize(tree)).toBe(100);

    // Query the entire space
    const results = rtreeSearch(tree, { minX: -1, minY: -1, maxX: 200, maxY: 200 });
    expect(results.length).toBe(100);
  });
});

describe('rtreeRemove', () => {
  it('removes an item and reduces size', () => {
    const tree = createRTree(4);
    rtreeInsert(tree, makeSpatialItem('x', 0, 0, 5, 5));
    rtreeInsert(tree, makeSpatialItem('y', 10, 10, 15, 15));
    expect(rtreeSize(tree)).toBe(2);

    const removed = rtreeRemove(tree, 'x');
    expect(removed).toBe(true);
    expect(rtreeSize(tree)).toBe(1);

    // 'x' should no longer appear in searches
    const results = rtreeSearch(tree, { minX: 0, minY: 0, maxX: 5, maxY: 5 });
    expect(results.length).toBe(0);
  });

  it('returns false for a non-existent id', () => {
    const tree = createRTree();
    expect(rtreeRemove(tree, 'nonexistent')).toBe(false);
  });
});

describe('bbox utility functions', () => {
  it('bboxArea computes width * height', () => {
    expect(bboxArea({ minX: 0, minY: 0, maxX: 10, maxY: 5 })).toBe(50);
  });

  it('bboxIntersects returns true for overlapping boxes', () => {
    const a: BoundingBox2D = { minX: 0, minY: 0, maxX: 10, maxY: 10 };
    const b: BoundingBox2D = { minX: 5, minY: 5, maxX: 15, maxY: 15 };
    expect(bboxIntersects(a, b)).toBe(true);
  });

  it('bboxIntersects returns false for disjoint boxes', () => {
    const a: BoundingBox2D = { minX: 0, minY: 0, maxX: 1, maxY: 1 };
    const b: BoundingBox2D = { minX: 5, minY: 5, maxX: 6, maxY: 6 };
    expect(bboxIntersects(a, b)).toBe(false);
  });

  it('bboxContainsBBox returns true when outer fully contains inner', () => {
    const outer: BoundingBox2D = { minX: 0, minY: 0, maxX: 10, maxY: 10 };
    const inner: BoundingBox2D = { minX: 2, minY: 2, maxX: 8, maxY: 8 };
    expect(bboxContainsBBox(outer, inner)).toBe(true);
    expect(bboxContainsBBox(inner, outer)).toBe(false);
  });
});

// ===================================================================
// BVH
// ===================================================================

describe('buildBVH', () => {
  it('returns null for empty items', () => {
    expect(buildBVH([])).toBeNull();
  });

  it('builds a tree from items', () => {
    const items = [
      makeSpatialItem('a', 0, 0, 1, 1),
      makeSpatialItem('b', 2, 2, 3, 3),
      makeSpatialItem('c', 4, 4, 5, 5),
    ];
    const root = buildBVH(items);
    expect(root).not.toBeNull();
    expect(bvhDepth(root)).toBeGreaterThan(0);
  });
});

describe('bvhQuery', () => {
  const items = [
    makeSpatialItem('a', 0, 0, 2, 2),
    makeSpatialItem('b', 5, 5, 7, 7),
    makeSpatialItem('c', 10, 10, 12, 12),
  ];
  const root = buildBVH(items)!;

  it('finds overlapping items', () => {
    const hits = bvhQuery(root, { minX: 4, minY: 4, maxX: 8, maxY: 8 }, items);
    expect(hits.length).toBe(1);
    expect(items[hits[0]!]!.id).toBe('b');
  });

  it('returns empty for a non-overlapping query', () => {
    const hits = bvhQuery(root, { minX: 20, minY: 20, maxX: 25, maxY: 25 }, items);
    expect(hits.length).toBe(0);
  });
});

describe('bvhRaycast', () => {
  const items = [
    makeSpatialItem('left', 1, -1, 3, 1),
    makeSpatialItem('right', 8, -1, 10, 1),
  ];
  const root = buildBVH(items)!;

  it('finds the nearest hit along a ray', () => {
    // Ray from origin going right (+x direction)
    const hitIdx = bvhRaycast(root, 0, 0, 1, 0, items);
    expect(hitIdx).not.toBeNull();
    expect(items[hitIdx!]!.id).toBe('left');
  });

  it('returns null when ray misses all items', () => {
    // Ray going upward, missing both boxes
    const hitIdx = bvhRaycast(root, 0, 5, 0, 1, items);
    expect(hitIdx).toBeNull();
  });
});

describe('bvhFindOverlaps', () => {
  it('finds pairwise overlapping items', () => {
    const items = [
      makeSpatialItem('a', 0, 0, 5, 5),
      makeSpatialItem('b', 3, 3, 8, 8),
      makeSpatialItem('c', 20, 20, 25, 25),
    ];
    const root = buildBVH(items)!;
    const overlaps = bvhFindOverlaps(root, items);
    // Only a and b overlap
    expect(overlaps.length).toBe(1);
    const [lo, hi] = overlaps[0]!;
    const ids = [items[lo]!.id, items[hi]!.id].sort();
    expect(ids).toEqual(['a', 'b']);
  });

  it('returns empty for non-overlapping items', () => {
    const items = [
      makeSpatialItem('x', 0, 0, 1, 1),
      makeSpatialItem('y', 5, 5, 6, 6),
    ];
    const root = buildBVH(items)!;
    const overlaps = bvhFindOverlaps(root, items);
    expect(overlaps.length).toBe(0);
  });
});

describe('bvhDepth', () => {
  it('returns 0 for null root', () => {
    expect(bvhDepth(null)).toBe(0);
  });

  it('returns depth > 0 for a non-trivial tree', () => {
    const items = Array.from({ length: 10 }, (_, i) =>
      makeSpatialItem(`i${i}`, i * 3, 0, i * 3 + 2, 2),
    );
    const root = buildBVH(items)!;
    expect(bvhDepth(root)).toBeGreaterThan(1);
  });
});

// ===================================================================
// Spatial Hash Grid
// ===================================================================

describe('createSpatialHashGrid / gridInsert / gridQuery', () => {
  it('inserts items and finds them via bbox query', () => {
    const grid = createSpatialHashGrid(10);
    gridInsert(grid, 'a', 5, 5);
    gridInsert(grid, 'b', 55, 55);

    const results = gridQuery(grid, { minX: 0, minY: 0, maxX: 9, maxY: 9 });
    expect(results).toContain('a');
    expect(results).not.toContain('b');
  });
});

describe('gridNearby', () => {
  it('finds items within the given radius', () => {
    const grid = createSpatialHashGrid(5);
    gridInsert(grid, 'close', 1, 1);
    gridInsert(grid, 'far', 100, 100);

    const nearby = gridNearby(grid, 0, 0, 2);
    expect(nearby).toContain('close');
    expect(nearby).not.toContain('far');
  });

  it('returns empty when nothing is nearby', () => {
    const grid = createSpatialHashGrid(5);
    gridInsert(grid, 'a', 50, 50);
    const nearby = gridNearby(grid, 0, 0, 1);
    expect(nearby.length).toBe(0);
  });
});

describe('gridRemove', () => {
  it('removes an item so it no longer appears in queries', () => {
    const grid = createSpatialHashGrid(10);
    gridInsert(grid, 'item', 5, 5);
    expect(gridSize(grid)).toBe(1);

    gridRemove(grid, 'item', 5, 5);
    expect(gridSize(grid)).toBe(0);

    const results = gridQuery(grid, { minX: 0, minY: 0, maxX: 10, maxY: 10 });
    expect(results.length).toBe(0);
  });
});

describe('gridClear', () => {
  it('empties the entire grid', () => {
    const grid = createSpatialHashGrid(10);
    gridInsert(grid, 'a', 1, 1);
    gridInsert(grid, 'b', 2, 2);
    gridInsert(grid, 'c', 3, 3);
    expect(gridSize(grid)).toBe(3);

    gridClear(grid);
    expect(gridSize(grid)).toBe(0);
  });
});

describe('gridSize', () => {
  it('tracks the number of items accurately', () => {
    const grid = createSpatialHashGrid(10);
    expect(gridSize(grid)).toBe(0);

    gridInsert(grid, 'a', 0, 0);
    expect(gridSize(grid)).toBe(1);

    gridInsert(grid, 'b', 1, 1);
    expect(gridSize(grid)).toBe(2);

    gridRemove(grid, 'a', 0, 0);
    expect(gridSize(grid)).toBe(1);
  });
});
