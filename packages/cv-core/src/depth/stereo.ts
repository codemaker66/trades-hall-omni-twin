// ---------------------------------------------------------------------------
// CV-9: Stereo Depth — block-matching disparity, disparity-to-depth
// conversion, and baseline estimation.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// computeDisparity
// ---------------------------------------------------------------------------

/**
 * Compute a disparity map from a rectified stereo pair using block matching
 * with the Sum of Absolute Differences (SAD) cost function.
 *
 * For each pixel in the left image the function searches up to
 * `maxDisparity` pixels to the left in the right image to find the
 * best-matching block.
 *
 * @param left          Left image intensities (row-major, width * height).
 * @param right         Right image intensities (row-major, width * height).
 * @param width         Image width in pixels.
 * @param height        Image height in pixels.
 * @param maxDisparity  Maximum disparity to search (pixels).
 * @param blockSize     Side length of the matching block (should be odd).
 * @returns Disparity map (Float64Array, width * height, row-major).
 */
export function computeDisparity(
  left: Float64Array,
  right: Float64Array,
  width: number,
  height: number,
  maxDisparity: number,
  blockSize: number,
): Float64Array {
  const count = width * height;
  const disparity = new Float64Array(count);
  const half = Math.floor(blockSize / 2);

  for (let y = half; y < height - half; y++) {
    for (let x = half; x < width - half; x++) {
      let bestCost = Infinity;
      let bestD = 0;

      const dMax = Math.min(maxDisparity, x - half);

      for (let d = 0; d <= dMax; d++) {
        let sad = 0;

        for (let ky = -half; ky <= half; ky++) {
          for (let kx = -half; kx <= half; kx++) {
            const ly = y + ky;
            const lx = x + kx;
            const rx = lx - d;
            const lVal = left[ly * width + lx]!;
            const rVal = right[ly * width + rx]!;
            sad += Math.abs(lVal - rVal);
          }
        }

        if (sad < bestCost) {
          bestCost = sad;
          bestD = d;
        }
      }

      disparity[y * width + x] = bestD;
    }
  }

  return disparity;
}

// ---------------------------------------------------------------------------
// disparityToDepth
// ---------------------------------------------------------------------------

/**
 * Convert a disparity map to a depth map using the stereo depth equation:
 *
 *   Z = focalLength * baseline / disparity
 *
 * Pixels with zero disparity are assigned a depth of 0 (invalid).
 *
 * @param disparity   Disparity values (row-major, same dimensions as images).
 * @param baseline    Baseline distance between cameras in metres.
 * @param focalLength Focal length in pixels.
 * @returns Depth values in metres (Float64Array, same length as disparity).
 */
export function disparityToDepth(
  disparity: Float64Array,
  baseline: number,
  focalLength: number,
): Float64Array {
  const count = disparity.length;
  const depth = new Float64Array(count);
  const fb = focalLength * baseline;

  for (let i = 0; i < count; i++) {
    const d = disparity[i]!;
    depth[i] = d > 0 ? fb / d : 0;
  }

  return depth;
}

// ---------------------------------------------------------------------------
// estimateBaseline
// ---------------------------------------------------------------------------

/**
 * Estimate the baseline distance between two cameras given two depth maps
 * and a known rigid transform between them.
 *
 * The baseline is approximated as the Euclidean norm of the translation
 * component extracted from the 4x4 column-major transform matrix.
 *
 * The two depth maps are provided for potential refinement but the primary
 * estimate comes from the translation vector in the transform.
 *
 * @param _depth1   Depth map from camera 1 (reserved for future scale refinement).
 * @param _depth2   Depth map from camera 2 (reserved for future scale refinement).
 * @param transform 4x4 column-major rigid transform from camera 1 to camera 2.
 * @returns Estimated baseline in metres.
 */
export function estimateBaseline(
  _depth1: Float64Array,
  _depth2: Float64Array,
  transform: Float64Array,
): number {
  // Extract translation from column-major 4x4 matrix:
  //   tx = transform[12], ty = transform[13], tz = transform[14]
  const tx = transform[12]!;
  const ty = transform[13]!;
  const tz = transform[14]!;
  return Math.sqrt(tx * tx + ty * ty + tz * tz);
}
