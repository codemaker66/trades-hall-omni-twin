// ---------------------------------------------------------------------------
// HPC-5: Spatial Indexing — Spatial Hash Grid
// ---------------------------------------------------------------------------
// Uniform-grid spatial hash for fast collision detection in scenes with
// roughly uniformly distributed objects. Pure TypeScript, zero dependencies.
// ---------------------------------------------------------------------------

import type { BoundingBox2D } from '../types.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** State of a spatial hash grid. */
export type SpatialHashGrid = {
  cells: Map<number, Set<string>>;
  cellSize: number;
  positions: Map<string, { x: number; y: number }>;
};

// ---------------------------------------------------------------------------
// Cell hashing
// ---------------------------------------------------------------------------

/**
 * Hash a cell coordinate (cx, cy) to a single numeric key.
 * Uses an offset Cantor pairing function to handle negative coordinates.
 * The offset of 16384 allows coordinates in [-16384, +16383] without collision.
 */
function hashCell(cx: number, cy: number): number {
  const a = cx + 16384;
  const b = cy + 16384;
  // Cantor pairing: ((a + b) * (a + b + 1)) / 2 + b
  return (((a + b) * (a + b + 1)) >> 1) + b;
}

// ---------------------------------------------------------------------------
// Create
// ---------------------------------------------------------------------------

/**
 * Create an empty spatial hash grid with the given cell size.
 *
 * @param cellSize Width/height of each grid cell. Should be roughly the
 *                 average radius of objects for best performance.
 */
export function createSpatialHashGrid(cellSize: number): SpatialHashGrid {
  return {
    cells: new Map<number, Set<string>>(),
    cellSize,
    positions: new Map<string, { x: number; y: number }>(),
  };
}

// ---------------------------------------------------------------------------
// Insert
// ---------------------------------------------------------------------------

/**
 * Insert an object at position (x, y) into the grid.
 * If the id already exists, it is first removed from its old cell.
 */
export function gridInsert(grid: SpatialHashGrid, id: string, x: number, y: number): void {
  // Remove from old position if it exists
  const old = grid.positions.get(id);
  if (old !== undefined) {
    gridRemove(grid, id, old.x, old.y);
  }

  const cx = Math.floor(x / grid.cellSize);
  const cy = Math.floor(y / grid.cellSize);
  const key = hashCell(cx, cy);

  let bucket = grid.cells.get(key);
  if (bucket === undefined) {
    bucket = new Set<string>();
    grid.cells.set(key, bucket);
  }
  bucket.add(id);

  grid.positions.set(id, { x, y });
}

// ---------------------------------------------------------------------------
// Remove
// ---------------------------------------------------------------------------

/**
 * Remove an object from the grid at position (x, y).
 * The position is used to determine the correct cell.
 */
export function gridRemove(grid: SpatialHashGrid, id: string, x: number, y: number): void {
  const cx = Math.floor(x / grid.cellSize);
  const cy = Math.floor(y / grid.cellSize);
  const key = hashCell(cx, cy);

  const bucket = grid.cells.get(key);
  if (bucket !== undefined) {
    bucket.delete(id);
    if (bucket.size === 0) {
      grid.cells.delete(key);
    }
  }

  grid.positions.delete(id);
}

// ---------------------------------------------------------------------------
// Query — bounding box
// ---------------------------------------------------------------------------

/**
 * Find all object IDs in cells overlapping the given bounding box.
 *
 * This returns IDs from all cells that the query box touches. It does NOT
 * perform exact distance or containment checks on the objects themselves
 * — use this as a broad-phase filter before fine-grained tests.
 */
export function gridQuery(grid: SpatialHashGrid, bbox: BoundingBox2D): string[] {
  const minCX = Math.floor(bbox.minX / grid.cellSize);
  const minCY = Math.floor(bbox.minY / grid.cellSize);
  const maxCX = Math.floor(bbox.maxX / grid.cellSize);
  const maxCY = Math.floor(bbox.maxY / grid.cellSize);

  const resultSet = new Set<string>();

  for (let cx = minCX; cx <= maxCX; cx++) {
    for (let cy = minCY; cy <= maxCY; cy++) {
      const key = hashCell(cx, cy);
      const bucket = grid.cells.get(key);
      if (bucket !== undefined) {
        for (const id of bucket) {
          resultSet.add(id);
        }
      }
    }
  }

  return Array.from(resultSet);
}

// ---------------------------------------------------------------------------
// Nearby — radius search
// ---------------------------------------------------------------------------

/**
 * Find all object IDs within Euclidean distance `radius` of point (x, y).
 *
 * First determines which cells overlap the bounding circle, then performs
 * an exact distance check against each candidate's stored position.
 */
export function gridNearby(
  grid: SpatialHashGrid,
  x: number,
  y: number,
  radius: number,
): string[] {
  const minCX = Math.floor((x - radius) / grid.cellSize);
  const minCY = Math.floor((y - radius) / grid.cellSize);
  const maxCX = Math.floor((x + radius) / grid.cellSize);
  const maxCY = Math.floor((y + radius) / grid.cellSize);

  const radiusSq = radius * radius;
  const results: string[] = [];
  const seen = new Set<string>();

  for (let cx = minCX; cx <= maxCX; cx++) {
    for (let cy = minCY; cy <= maxCY; cy++) {
      const key = hashCell(cx, cy);
      const bucket = grid.cells.get(key);
      if (bucket !== undefined) {
        for (const id of bucket) {
          if (seen.has(id)) continue;
          seen.add(id);

          const pos = grid.positions.get(id);
          if (pos !== undefined) {
            const dx = pos.x - x;
            const dy = pos.y - y;
            if (dx * dx + dy * dy <= radiusSq) {
              results.push(id);
            }
          }
        }
      }
    }
  }

  return results;
}

// ---------------------------------------------------------------------------
// Clear / Size
// ---------------------------------------------------------------------------

/** Remove all objects from the grid. */
export function gridClear(grid: SpatialHashGrid): void {
  grid.cells.clear();
  grid.positions.clear();
}

/** Return the number of objects in the grid. */
export function gridSize(grid: SpatialHashGrid): number {
  return grid.positions.size;
}
