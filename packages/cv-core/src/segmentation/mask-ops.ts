// ---------------------------------------------------------------------------
// CV-10: Mask Operations — creation, boolean ops, IoU, contour tracing,
// and connected-component labelling.
// ---------------------------------------------------------------------------

import type { Mask2D, Vec2 } from '../types.js';

// ---------------------------------------------------------------------------
// createMask
// ---------------------------------------------------------------------------

/**
 * Create a 2D mask, optionally initialised with existing data.
 *
 * When no `data` array is supplied the mask values default to 0
 * (background).
 *
 * @param width  Mask width in pixels.
 * @param height Mask height in pixels.
 * @param data   Optional pre-populated label data (row-major, Uint32).
 * @returns A new {@link Mask2D}.
 */
export function createMask(
  width: number,
  height: number,
  data?: Uint8Array,
): Mask2D {
  const count = width * height;
  let buf: Uint32Array;

  if (data) {
    // Widen Uint8 -> Uint32
    buf = new Uint32Array(count);
    for (let i = 0; i < count; i++) {
      buf[i] = data[i]!;
    }
  } else {
    buf = new Uint32Array(count);
  }

  return { data: buf, width, height };
}

// ---------------------------------------------------------------------------
// maskUnion
// ---------------------------------------------------------------------------

/**
 * Pixel-wise OR (union) of two binary masks.
 *
 * A pixel is foreground (1) if it is non-zero in either mask.
 *
 * @param a First mask.
 * @param b Second mask (must have the same dimensions as `a`).
 * @returns A new mask representing the union.
 */
export function maskUnion(a: Mask2D, b: Mask2D): Mask2D {
  const count = a.width * a.height;
  const out = new Uint32Array(count);

  for (let i = 0; i < count; i++) {
    out[i] = (a.data[i]! | b.data[i]!) !== 0 ? 1 : 0;
  }

  return { data: out, width: a.width, height: a.height };
}

// ---------------------------------------------------------------------------
// maskIntersection
// ---------------------------------------------------------------------------

/**
 * Pixel-wise AND (intersection) of two binary masks.
 *
 * A pixel is foreground (1) only if it is non-zero in both masks.
 *
 * @param a First mask.
 * @param b Second mask (must have the same dimensions as `a`).
 * @returns A new mask representing the intersection.
 */
export function maskIntersection(a: Mask2D, b: Mask2D): Mask2D {
  const count = a.width * a.height;
  const out = new Uint32Array(count);

  for (let i = 0; i < count; i++) {
    out[i] = a.data[i]! !== 0 && b.data[i]! !== 0 ? 1 : 0;
  }

  return { data: out, width: a.width, height: a.height };
}

// ---------------------------------------------------------------------------
// computeIoU
// ---------------------------------------------------------------------------

/**
 * Compute the Intersection over Union (IoU) of two binary masks.
 *
 * Any non-zero pixel is considered foreground.  If both masks are
 * entirely background the IoU is defined as 0.
 *
 * @param a First mask.
 * @param b Second mask (must have the same dimensions as `a`).
 * @returns IoU value in [0, 1].
 */
export function computeIoU(a: Mask2D, b: Mask2D): number {
  const count = a.width * a.height;
  let intersection = 0;
  let union = 0;

  for (let i = 0; i < count; i++) {
    const aFg = a.data[i]! !== 0;
    const bFg = b.data[i]! !== 0;

    if (aFg && bFg) {
      intersection++;
      union++;
    } else if (aFg || bFg) {
      union++;
    }
  }

  return union > 0 ? intersection / union : 0;
}

// ---------------------------------------------------------------------------
// maskToPolygon
// ---------------------------------------------------------------------------

/**
 * Trace the boundary of a binary mask to produce an ordered polygon.
 *
 * Uses a simple contour-following algorithm (Moore neighbourhood tracing).
 * Any non-zero pixel is considered foreground.  The function returns the
 * outer boundary of the first connected foreground component encountered
 * in raster-scan order.
 *
 * If the mask contains no foreground pixels the returned array is empty.
 *
 * @param mask Input binary mask.
 * @returns Ordered 2D vertices of the boundary polygon.
 */
export function maskToPolygon(mask: Mask2D): Vec2[] {
  const { width, height, data } = mask;

  /** Read foreground status with bounds checking. */
  const fg = (x: number, y: number): boolean => {
    if (x < 0 || x >= width || y < 0 || y >= height) return false;
    return data[y * width + x]! !== 0;
  };

  // Find the first foreground pixel (raster-scan)
  let startX = -1;
  let startY = -1;
  outer: for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (fg(x, y)) {
        startX = x;
        startY = y;
        break outer;
      }
    }
  }

  if (startX === -1) return [];

  // Moore neighbourhood: 8-connected directions starting from left
  //   0: left, 1: up-left, 2: up, 3: up-right,
  //   4: right, 5: down-right, 6: down, 7: down-left
  const dx = [-1, -1, 0, 1, 1, 1, 0, -1];
  const dy = [0, -1, -1, -1, 0, 1, 1, 1];

  const polygon: Vec2[] = [];
  let cx = startX;
  let cy = startY;
  // Start scanning from the pixel to the left of the start (direction 0's
  // predecessor in backtrack terms is direction 4 — the pixel we came from)
  let dir = 0;

  const maxSteps = width * height * 2; // safety bound
  let steps = 0;

  do {
    polygon.push({ x: cx, y: cy });

    // Backtrack: start scanning from (dir + 5) mod 8 — the next direction
    // after the direction we came from
    let searchDir = (dir + 5) % 8;
    let found = false;

    for (let i = 0; i < 8; i++) {
      const d = (searchDir + i) % 8;
      const nx = cx + dx[d]!;
      const ny = cy + dy[d]!;

      if (fg(nx, ny)) {
        cx = nx;
        cy = ny;
        dir = d;
        found = true;
        break;
      }
    }

    if (!found) break;

    steps++;
  } while ((cx !== startX || cy !== startY) && steps < maxSteps);

  return polygon;
}

// ---------------------------------------------------------------------------
// connectedComponents
// ---------------------------------------------------------------------------

/**
 * Label connected components in a binary mask using flood fill
 * (4-connected).
 *
 * Any non-zero pixel is considered foreground.  Each connected component
 * receives a unique positive label starting at 1.  Background pixels
 * remain 0.
 *
 * @param mask Input binary mask.
 * @returns An object with `labels` (Uint32Array, same dimensions as mask)
 *          and `count` (number of connected components).
 */
export function connectedComponents(
  mask: Mask2D,
): { labels: Uint32Array; count: number } {
  const { width, height, data } = mask;
  const count = width * height;
  const labels = new Uint32Array(count);
  let currentLabel = 0;

  // 4-connected neighbours: right, down, left, up
  const dx = [1, 0, -1, 0];
  const dy = [0, 1, 0, -1];

  // Stack-based flood fill to avoid recursion limits
  const stack: number[] = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;

      // Skip background or already labelled pixels
      if (data[idx]! === 0 || labels[idx]! !== 0) continue;

      currentLabel++;
      stack.push(x, y);

      while (stack.length > 0) {
        const sy = stack.pop()!;
        const sx = stack.pop()!;
        const si = sy * width + sx;

        if (labels[si]! !== 0) continue;
        labels[si] = currentLabel;

        for (let d = 0; d < 4; d++) {
          const nx = sx + dx[d]!;
          const ny = sy + dy[d]!;

          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

          const ni = ny * width + nx;
          if (data[ni]! !== 0 && labels[ni]! === 0) {
            stack.push(nx, ny);
          }
        }
      }
    }
  }

  return { labels, count: currentLabel };
}
