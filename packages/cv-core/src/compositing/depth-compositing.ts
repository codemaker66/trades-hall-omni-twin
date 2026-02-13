// ---------------------------------------------------------------------------
// CV-3: Depth Compositing — depth buffer creation, unprojection, and
// per-pixel depth-ordering for Gaussian-splat / mesh hybrid rendering.
// ---------------------------------------------------------------------------

import type { CameraIntrinsics, DepthBuffer, Vector3 } from '../types.js';

// ---------------------------------------------------------------------------
// createDepthBuffer
// ---------------------------------------------------------------------------

/**
 * Create a depth buffer initialised to the far plane value.
 *
 * @param width  Buffer width in pixels.
 * @param height Buffer height in pixels.
 * @param near   Near plane distance.
 * @param far    Far plane distance.
 */
export function createDepthBuffer(
  width: number,
  height: number,
  near: number,
  far: number,
): DepthBuffer {
  const data = new Float64Array(width * height);
  data.fill(far);
  return { data, width, height, near, far };
}

// ---------------------------------------------------------------------------
// unprojectPixel
// ---------------------------------------------------------------------------

/**
 * Unproject a pixel coordinate and depth value to a 3D world-space point
 * using the pinhole camera model (no distortion).
 *
 * Uses the standard inverse-pinhole equations:
 *   X = (x - cx) * depth / fx
 *   Y = (y - cy) * depth / fy
 *   Z = depth
 *
 * @param x          Pixel column.
 * @param y          Pixel row.
 * @param depth      Linear depth at the pixel.
 * @param intrinsics Camera intrinsics (pinhole model).
 * @returns          The 3D point in camera space.
 */
export function unprojectPixel(
  x: number,
  y: number,
  depth: number,
  intrinsics: CameraIntrinsics,
): Vector3 {
  return {
    x: ((x - intrinsics.cx) * depth) / intrinsics.fx,
    y: ((y - intrinsics.cy) * depth) / intrinsics.fy,
    z: depth,
  };
}

// ---------------------------------------------------------------------------
// compositeDepthOrder
// ---------------------------------------------------------------------------

/**
 * Per-pixel comparison of two depth buffers (Gaussian splats vs triangle mesh).
 *
 * For every pixel the function writes:
 *   0 — Gaussian depth is in front (nearer or equal).
 *   1 — Mesh depth is in front.
 *
 * Both input arrays are assumed to be row-major with dimensions
 * `width x height`.
 *
 * @param gaussianDepths Depth values from the Gaussian-splat pass.
 * @param meshDepths     Depth values from the triangle-mesh pass.
 * @param width          Image width in pixels.
 * @param height         Image height in pixels.
 */
export function compositeDepthOrder(
  gaussianDepths: Float64Array,
  meshDepths: Float64Array,
  width: number,
  height: number,
): Uint8Array {
  const count = width * height;
  const result = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    // Gaussian in front when its depth is less than or equal to mesh depth.
    result[i] = gaussianDepths[i]! <= meshDepths[i]! ? 0 : 1;
  }
  return result;
}

// ---------------------------------------------------------------------------
// linearizeDepth
// ---------------------------------------------------------------------------

/**
 * Convert a raw normalised-device-coordinate (NDC) depth value to linear
 * (eye-space) depth.
 *
 * Assuming a standard perspective projection where the NDC depth in [0, 1]
 * maps the near plane to 0 and the far plane to 1:
 *
 *   z_linear = (near * far) / (far - rawDepth * (far - near))
 *
 * @param rawDepth NDC depth in [0, 1].
 * @param near     Near plane distance.
 * @param far      Far plane distance.
 * @returns        Linear depth in world units.
 */
export function linearizeDepth(
  rawDepth: number,
  near: number,
  far: number,
): number {
  return (near * far) / (far - rawDepth * (far - near));
}
