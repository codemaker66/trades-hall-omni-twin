// ---------------------------------------------------------------------------
// CV-2: Gaussian Splatting — Radix Sort & Tile Assignment
// ---------------------------------------------------------------------------

import type { Vec2, TileAssignment } from '../types.js';

/**
 * Radix sort an array of Gaussian depth values, returning sorted indices
 * for back-to-front rendering order.
 *
 * The algorithm performs an LSB (least-significant-byte first) radix sort
 * on the floating-point depth values. Depths are reinterpreted as sortable
 * 32-bit unsigned integers via a float-to-uint conversion that preserves
 * ordering (IEEE 754 trick: flip sign bit for positives, flip all bits for
 * negatives).
 *
 * @param depths  - Float64Array of per-Gaussian depth values.
 * @param indices - Uint32Array of Gaussian indices to sort (same length as depths).
 * @returns A new Uint32Array of indices sorted by depth in ascending order
 *          (back-to-front: greatest depth first is handled by the renderer).
 */
export function radixSortByDepth(
  depths: Float64Array,
  indices: Uint32Array,
): Uint32Array {
  const n = depths.length;
  if (n === 0) return new Uint32Array(0);

  // --- Convert float depths to sortable uint32 keys ---
  const keys = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    keys[i] = floatToSortableUint(depths[i]!);
  }

  // --- LSB radix sort with 8 bits per pass, 4 passes for 32-bit keys ---
  const RADIX_BITS = 8;
  const RADIX_SIZE = 1 << RADIX_BITS; // 256
  const MASK = RADIX_SIZE - 1;
  const PASSES = 4;

  let srcKeys = keys;
  let srcIdx = new Uint32Array(indices);
  let dstKeys = new Uint32Array(n);
  let dstIdx = new Uint32Array(n);

  const histogram = new Uint32Array(RADIX_SIZE);

  for (let pass = 0; pass < PASSES; pass++) {
    const shift = pass * RADIX_BITS;

    // --- Build histogram ---
    histogram.fill(0);
    for (let i = 0; i < n; i++) {
      const bucket = (srcKeys[i]! >>> shift) & MASK;
      histogram[bucket] = histogram[bucket]! + 1;
    }

    // --- Prefix sum (exclusive scan) ---
    let total = 0;
    for (let i = 0; i < RADIX_SIZE; i++) {
      const count = histogram[i]!;
      histogram[i] = total;
      total += count;
    }

    // --- Scatter ---
    for (let i = 0; i < n; i++) {
      const bucket = (srcKeys[i]! >>> shift) & MASK;
      const dest = histogram[bucket]!;
      histogram[bucket] = dest + 1;
      dstKeys[dest] = srcKeys[i]!;
      dstIdx[dest] = srcIdx[i]!;
    }

    // --- Swap src/dst ---
    const tmpKeys = srcKeys;
    srcKeys = dstKeys;
    dstKeys = tmpKeys;
    const tmpIdx = srcIdx;
    srcIdx = dstIdx;
    dstIdx = tmpIdx;
  }

  return srcIdx;
}

/**
 * Assign Gaussian splats to screen-space tiles for tile-based rasterisation.
 *
 * Each Gaussian's projected 2D centre is mapped to one or more tiles.
 * In this simplified version, each Gaussian is assigned to exactly the tile
 * containing its centre (a more sophisticated approach would assign to all
 * tiles the Gaussian's 2D ellipse overlaps).
 *
 * @param centers  - Array of 2D screen-space centres.
 * @param tileSize - Tile side length in pixels.
 * @param width    - Viewport width in pixels.
 * @param height   - Viewport height in pixels.
 * @returns Array of {@link TileAssignment} entries.
 */
export function assignTiles(
  centers: Vec2[],
  tileSize: number,
  width: number,
  height: number,
): TileAssignment[] {
  if (tileSize <= 0) {
    throw new Error('tileSize must be positive');
  }

  const tilesX = Math.ceil(width / tileSize);
  const tilesY = Math.ceil(height / tileSize);

  const assignments: TileAssignment[] = [];

  for (let i = 0; i < centers.length; i++) {
    const c = centers[i]!;

    // Skip Gaussians outside the viewport
    if (c.x < 0 || c.x >= width || c.y < 0 || c.y >= height) {
      continue;
    }

    const tileX = Math.min(Math.floor(c.x / tileSize), tilesX - 1);
    const tileY = Math.min(Math.floor(c.y / tileSize), tilesY - 1);

    // Use tile-major depth key for stable sorting within tiles
    const depthKey = tileY * tilesX + tileX;

    assignments.push({
      gaussianIndex: i,
      tileX,
      tileY,
      depthKey,
    });
  }

  return assignments;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Convert an IEEE 754 float64 to a uint32 that preserves ordering.
 *
 * We first convert to float32 (for 32-bit radix sort), then reinterpret
 * the bits. For positive floats the bit pattern already sorts correctly;
 * for negative floats we flip all bits so that more-negative values sort
 * before less-negative values.
 */
function floatToSortableUint(f: number): number {
  // Clamp to float32 range
  const buf = new Float32Array(1);
  buf[0] = f;
  const view = new Uint32Array(buf.buffer);
  let bits = view[0]!;

  // If the sign bit is set (negative float), flip all bits.
  // Otherwise, flip only the sign bit so positives sort after negatives.
  if (bits & 0x80000000) {
    bits = ~bits;
  } else {
    bits = bits | 0x80000000;
  }

  return bits >>> 0; // ensure unsigned
}
