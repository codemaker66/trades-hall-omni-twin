// ---------------------------------------------------------------------------
// CV-9: Depth Processing — depth map creation, hole filling, filtering,
// and gradient computation.
// ---------------------------------------------------------------------------

import type { DepthMap } from '../types.js';

// ---------------------------------------------------------------------------
// createDepthMap
// ---------------------------------------------------------------------------

/**
 * Create a depth map, optionally initialised with existing data.
 *
 * When no `data` array is supplied the depth values default to zero.
 * `minDepth` and `maxDepth` are scanned from the provided data; if no
 * data is given both default to 0.
 *
 * @param width  Map width in pixels.
 * @param height Map height in pixels.
 * @param data   Optional pre-populated depth values (row-major).
 * @returns A new {@link DepthMap}.
 */
export function createDepthMap(
  width: number,
  height: number,
  data?: Float64Array,
): DepthMap {
  const count = width * height;
  const buf = data ? new Float64Array(data) : new Float64Array(count);

  let minDepth = Infinity;
  let maxDepth = -Infinity;
  for (let i = 0; i < buf.length; i++) {
    const v = buf[i]!;
    if (v < minDepth) minDepth = v;
    if (v > maxDepth) maxDepth = v;
  }

  // Handle the empty / zero-length case
  if (!isFinite(minDepth)) minDepth = 0;
  if (!isFinite(maxDepth)) maxDepth = 0;

  return { data: buf, width, height, minDepth, maxDepth };
}

// ---------------------------------------------------------------------------
// fillHoles
// ---------------------------------------------------------------------------

/**
 * Fill small holes (zero-depth pixels) in a depth map by averaging valid
 * neighbours within a square window of radius `maxHoleSize`.
 *
 * A pixel is considered a "hole" if its depth value is exactly 0.  For
 * each hole pixel we search outward from the pixel within a window of
 * `(2 * maxHoleSize + 1)` and compute the mean of all non-zero
 * neighbours. If no valid neighbour is found the pixel stays 0.
 *
 * @param depth       Input depth map.
 * @param maxHoleSize Radius of the search window around each hole pixel.
 * @returns A new depth map with holes filled.
 */
export function fillHoles(depth: DepthMap, maxHoleSize: number): DepthMap {
  const { width, height } = depth;
  const count = width * height;
  const src = depth.data;
  const dst = new Float64Array(count);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const val = src[idx]!;

      if (val !== 0) {
        dst[idx] = val;
        continue;
      }

      // Hole — gather valid neighbours
      let sum = 0;
      let cnt = 0;
      const r = maxHoleSize;
      const yMin = Math.max(0, y - r);
      const yMax = Math.min(height - 1, y + r);
      const xMin = Math.max(0, x - r);
      const xMax = Math.min(width - 1, x + r);

      for (let ny = yMin; ny <= yMax; ny++) {
        for (let nx = xMin; nx <= xMax; nx++) {
          const nv = src[ny * width + nx]!;
          if (nv !== 0) {
            sum += nv;
            cnt++;
          }
        }
      }

      dst[idx] = cnt > 0 ? sum / cnt : 0;
    }
  }

  return createDepthMap(width, height, dst);
}

// ---------------------------------------------------------------------------
// medianFilter
// ---------------------------------------------------------------------------

/**
 * Apply a median filter to a depth map for noise removal.
 *
 * The filter uses a square kernel of size `kernelSize x kernelSize`
 * (must be odd). Boundary pixels are handled by clamping coordinates
 * to the image edge.
 *
 * @param depth      Input depth map.
 * @param kernelSize Side length of the square kernel (must be odd >= 1).
 * @returns A new filtered depth map.
 */
export function medianFilter(depth: DepthMap, kernelSize: number): DepthMap {
  const { width, height } = depth;
  const count = width * height;
  const src = depth.data;
  const dst = new Float64Array(count);
  const half = Math.floor(kernelSize / 2);

  // Reusable buffer for gathering neighbourhood values
  const buf: number[] = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      buf.length = 0;

      for (let ky = -half; ky <= half; ky++) {
        const ny = Math.min(Math.max(y + ky, 0), height - 1);
        for (let kx = -half; kx <= half; kx++) {
          const nx = Math.min(Math.max(x + kx, 0), width - 1);
          buf.push(src[ny * width + nx]!);
        }
      }

      // Sort numerically and pick middle element
      buf.sort((a, b) => a - b);
      dst[y * width + x] = buf[Math.floor(buf.length / 2)]!;
    }
  }

  return createDepthMap(width, height, dst);
}

// ---------------------------------------------------------------------------
// computeDepthGradient
// ---------------------------------------------------------------------------

/**
 * Compute Sobel-like depth gradients in the x and y directions.
 *
 * The 3x3 Sobel kernels are:
 *
 *   Gx = [[-1, 0, 1],    Gy = [[-1, -2, -1],
 *         [-2, 0, 2],          [ 0,  0,  0],
 *         [-1, 0, 1]]          [ 1,  2,  1]]
 *
 * Boundary pixels use clamped (replicated) edge values.
 *
 * @param depth Input depth map.
 * @returns An object with `gx` and `gy` gradient arrays, each of length
 *          `width * height`.
 */
export function computeDepthGradient(
  depth: DepthMap,
): { gx: Float64Array; gy: Float64Array } {
  const { width, height } = depth;
  const count = width * height;
  const src = depth.data;
  const gx = new Float64Array(count);
  const gy = new Float64Array(count);

  /** Clamped pixel read. */
  const px = (x: number, y: number): number => {
    const cx = Math.min(Math.max(x, 0), width - 1);
    const cy = Math.min(Math.max(y, 0), height - 1);
    return src[cy * width + cx]!;
  };

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;

      // Sobel Gx
      gx[idx] =
        -1 * px(x - 1, y - 1) +
         1 * px(x + 1, y - 1) +
        -2 * px(x - 1, y) +
         2 * px(x + 1, y) +
        -1 * px(x - 1, y + 1) +
         1 * px(x + 1, y + 1);

      // Sobel Gy
      gy[idx] =
        -1 * px(x - 1, y - 1) +
        -2 * px(x, y - 1) +
        -1 * px(x + 1, y - 1) +
         1 * px(x - 1, y + 1) +
         2 * px(x, y + 1) +
         1 * px(x + 1, y + 1);
    }
  }

  return { gx, gy };
}
