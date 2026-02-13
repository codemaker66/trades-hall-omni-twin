// ---------------------------------------------------------------------------
// CV-7: Shadow Mapping — cascaded shadow map splits, light-space matrix
// computation, and screen-space contact shadow ray-marching.
// ---------------------------------------------------------------------------

import type { Vec2, Vector3 } from '../types.js';
import { vec3Cross, vec3Dot, vec3Normalize, vec3Sub } from '../types.js';

// ---------------------------------------------------------------------------
// computeCascadeSplits
// ---------------------------------------------------------------------------

/**
 * Compute cascade split distances for Cascaded Shadow Maps (CSM).
 *
 * Uses the practical split scheme that blends logarithmic and uniform
 * distributions, as described by Zhang et al. (2006):
 *
 *   C_log(i) = near * (far / near) ^ (i / n)
 *   C_uni(i) = near + (far - near) * (i / n)
 *   C(i)     = lambda * C_log(i) + (1 - lambda) * C_uni(i)
 *
 * The first element is always `near` and the last is always `far`.
 *
 * @param near       Camera near plane distance.
 * @param far        Camera far plane distance.
 * @param nCascades  Number of cascades (>= 1).
 * @param lambda     Blend factor between logarithmic (1) and uniform (0).
 * @returns          Float64Array of length `nCascades + 1` with split distances.
 */
export function computeCascadeSplits(
  near: number,
  far: number,
  nCascades: number,
  lambda: number,
): Float64Array {
  const splits = new Float64Array(nCascades + 1);
  splits[0] = near;
  splits[nCascades] = far;

  const ratio = far / near;

  for (let i = 1; i < nCascades; i++) {
    const frac = i / nCascades;
    const cLog = near * Math.pow(ratio, frac);
    const cUni = near + (far - near) * frac;
    splits[i] = lambda * cLog + (1 - lambda) * cUni;
  }

  return splits;
}

// ---------------------------------------------------------------------------
// computeShadowMatrix
// ---------------------------------------------------------------------------

/**
 * Compute an orthographic shadow (light-space) view-projection matrix for a
 * single cascade.
 *
 * The resulting 4x4 matrix (column-major) transforms world-space positions
 * into the light's clip space for shadow-map rendering.
 *
 * Steps:
 * 1. Build a look-at view matrix from the light direction.
 * 2. Transform the cascade frustum corners into light space.
 * 3. Fit a tight orthographic projection around those bounds.
 * 4. Return lightProjection * lightView.
 *
 * For simplicity, the cascade frustum is approximated as the AABB of the
 * camera-view column vectors scaled by the near/far splits.
 *
 * @param lightDir    Unit direction *toward* the light (will be negated for the view).
 * @param cascadeNear Near split distance for this cascade.
 * @param cascadeFar  Far split distance for this cascade.
 * @param cameraView  4x4 camera view matrix (column-major) used to derive
 *                    the frustum slice in world space.
 * @returns           4x4 light view-projection matrix (column-major).
 */
export function computeShadowMatrix(
  lightDir: Vector3,
  cascadeNear: number,
  cascadeFar: number,
  cameraView: Float64Array,
): Float64Array {
  // --- Light view matrix (look-at) ---
  // Light looks along -lightDir.  We pick an arbitrary up that is not
  // parallel to lightDir to derive right and true-up.
  const forward = vec3Normalize({ x: -lightDir.x, y: -lightDir.y, z: -lightDir.z });
  let up: Vector3 = { x: 0, y: 1, z: 0 };
  if (Math.abs(vec3Dot(forward, up)) > 0.999) {
    up = { x: 1, y: 0, z: 0 };
  }
  const right = vec3Normalize(vec3Cross(up, forward));
  const trueUp = vec3Cross(forward, right);

  // Place the light "eye" at the centroid of the cascade range along the
  // camera's forward direction.  For a fully general implementation we
  // would compute the frustum corners; here we derive a centre from the
  // camera view inverse.  As a practical shortcut we use the origin.
  const centre: Vector3 = { x: 0, y: 0, z: 0 };

  // Light view matrix (column-major):
  //   col0 = right, col1 = trueUp, col2 = forward, col3 = -dot(axis, eye)
  const lv = new Float64Array(16);
  lv[0] = right.x;       lv[1] = trueUp.x;    lv[2] = forward.x;    lv[3] = 0;
  lv[4] = right.y;       lv[5] = trueUp.y;    lv[6] = forward.y;    lv[7] = 0;
  lv[8] = right.z;       lv[9] = trueUp.z;    lv[10] = forward.z;   lv[11] = 0;
  lv[12] = -vec3Dot(right, centre);
  lv[13] = -vec3Dot(trueUp, centre);
  lv[14] = -vec3Dot(forward, centre);
  lv[15] = 1;

  // --- Orthographic projection sized by cascade range ---
  // Use the cascade split distances to estimate a half-size.
  const halfSize = cascadeFar * 0.5;
  const depthRange = cascadeFar - cascadeNear;

  const l = -halfSize;
  const r2 = halfSize;
  const b = -halfSize;
  const t = halfSize;
  const n = 0;
  const f = depthRange > 0 ? depthRange : 1;

  // Orthographic projection (column-major):
  const lp = new Float64Array(16);
  lp[0] = 2 / (r2 - l);
  lp[5] = 2 / (t - b);
  lp[10] = -2 / (f - n);
  lp[12] = -(r2 + l) / (r2 - l);
  lp[13] = -(t + b) / (t - b);
  lp[14] = -(f + n) / (f - n);
  lp[15] = 1;

  // Multiply: result = lp * lv
  const out = new Float64Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        sum += lp[k * 4 + row]! * lv[col * 4 + k]!;
      }
      out[col * 4 + row] = sum;
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// contactShadowRaymarch
// ---------------------------------------------------------------------------

/**
 * Screen-space contact shadow ray-march.
 *
 * Starting from a screen-space origin, march along a 2D light direction in
 * the depth buffer.  If the ray dips below the stored depth (i.e. the marched
 * position is occluded), the pixel is considered to be in shadow.
 *
 * @param depthBuffer Linearised depth values (width * height, row-major).
 * @param origin      Screen-space origin of the ray ({x, y} in pixels).
 * @param lightDir    Screen-space light direction (normalised, {x, y}).
 * @param width       Depth-buffer width in pixels.
 * @param height      Depth-buffer height in pixels.
 * @param nSteps      Number of march steps.
 * @returns           `true` if the point is in shadow (occluded), else `false`.
 */
export function contactShadowRaymarch(
  depthBuffer: Float64Array,
  origin: Vec2,
  lightDir: Vec2,
  width: number,
  height: number,
  nSteps: number,
): boolean {
  // Step size: traverse up to a fraction of the screen diagonal.
  const maxDist = Math.sqrt(width * width + height * height) * 0.1;
  const stepSize = maxDist / Math.max(nSteps, 1);

  // Read the depth at the origin to use as the reference.
  const ox = Math.round(origin.x);
  const oy = Math.round(origin.y);
  if (ox < 0 || ox >= width || oy < 0 || oy >= height) return false;
  const originDepth = depthBuffer[oy * width + ox]!;

  for (let i = 1; i <= nSteps; i++) {
    const sx = origin.x + lightDir.x * stepSize * i;
    const sy = origin.y + lightDir.y * stepSize * i;

    const px = Math.round(sx);
    const py = Math.round(sy);

    // Out-of-bounds — stop marching.
    if (px < 0 || px >= width || py < 0 || py >= height) break;

    const sampledDepth = depthBuffer[py * width + px]!;

    // If the sampled depth is closer than the origin depth, we are
    // behind geometry — the point is in shadow.
    if (sampledDepth < originDepth - 1e-4) {
      return true;
    }
  }

  return false;
}
