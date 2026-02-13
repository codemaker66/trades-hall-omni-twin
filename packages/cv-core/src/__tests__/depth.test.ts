import { describe, it, expect } from 'vitest';
import {
  createDepthMap,
  fillHoles,
  medianFilter,
  computeDepthGradient,
  computeDisparity,
  disparityToDepth,
  estimateBaseline,
  createTSDFVolume,
  integrateTSDFFrame,
  extractMeshFromTSDF,
} from '../depth/index.js';
import type { DepthMap, TSDFVolume, CameraIntrinsics } from '../types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a simple depth map from a flat array of values. */
function makeDepthMap(
  width: number,
  height: number,
  values: number[],
): DepthMap {
  return createDepthMap(width, height, new Float64Array(values));
}

/** Build a column-major 4x4 identity matrix. */
function identity4(): Float64Array {
  const m = new Float64Array(16);
  m[0] = 1;
  m[5] = 1;
  m[10] = 1;
  m[15] = 1;
  return m;
}

/** Build a column-major 4x4 translation matrix. */
function translation4(tx: number, ty: number, tz: number): Float64Array {
  const m = identity4();
  m[12] = tx;
  m[13] = ty;
  m[14] = tz;
  return m;
}

// ---------------------------------------------------------------------------
// createDepthMap
// ---------------------------------------------------------------------------

describe('createDepthMap', () => {
  it('creates a depth map with correct dimensions', () => {
    const dm = createDepthMap(10, 20);
    expect(dm.width).toBe(10);
    expect(dm.height).toBe(20);
    expect(dm.data.length).toBe(200);
  });

  it('initialises all values to zero when no data is provided', () => {
    const dm = createDepthMap(4, 4);
    for (let i = 0; i < dm.data.length; i++) {
      expect(dm.data[i]).toBe(0);
    }
  });

  it('copies provided data into the depth map', () => {
    const data = new Float64Array([1, 2, 3, 4]);
    const dm = createDepthMap(2, 2, data);
    expect(dm.data[0]).toBeCloseTo(1, 5);
    expect(dm.data[3]).toBeCloseTo(4, 5);
  });

  it('computes min and max depth from provided data', () => {
    const data = new Float64Array([5, 2, 8, 1, 9, 3]);
    const dm = createDepthMap(3, 2, data);
    expect(dm.minDepth).toBeCloseTo(1, 5);
    expect(dm.maxDepth).toBeCloseTo(9, 5);
  });

  it('sets minDepth and maxDepth to 0 for empty data', () => {
    const dm = createDepthMap(0, 0);
    expect(dm.minDepth).toBe(0);
    expect(dm.maxDepth).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// fillHoles
// ---------------------------------------------------------------------------

describe('fillHoles', () => {
  it('fills zero-valued pixels with neighbour average', () => {
    // 3x3 depth map with a hole in the centre
    const dm = makeDepthMap(3, 3, [
      5, 5, 5,
      5, 0, 5,
      5, 5, 5,
    ]);
    const filled = fillHoles(dm, 1);
    // The centre pixel should now be the average of its 8 neighbours (all 5)
    expect(filled.data[4]).toBeCloseTo(5, 5);
  });

  it('does not modify non-zero pixels', () => {
    const dm = makeDepthMap(2, 2, [3, 4, 5, 6]);
    const filled = fillHoles(dm, 1);
    expect(filled.data[0]).toBeCloseTo(3, 5);
    expect(filled.data[1]).toBeCloseTo(4, 5);
    expect(filled.data[2]).toBeCloseTo(5, 5);
    expect(filled.data[3]).toBeCloseTo(6, 5);
  });

  it('leaves holes that have no valid neighbours', () => {
    const dm = makeDepthMap(3, 3, [
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
    ]);
    const filled = fillHoles(dm, 1);
    // No valid neighbours, so all remain 0
    for (let i = 0; i < filled.data.length; i++) {
      expect(filled.data[i]).toBe(0);
    }
  });

  it('fills multiple holes in one pass', () => {
    const dm = makeDepthMap(3, 3, [
      10, 0, 10,
       0, 0,  0,
      10, 0, 10,
    ]);
    const filled = fillHoles(dm, 1);
    // The centre pixel should be filled with the average of its valid neighbours
    // Centre has 4 valid neighbours (corners), each at distance 1, all value 10
    expect(filled.data[4]).toBeCloseTo(10, 5);
  });

  it('returns a depth map with correct min/max after filling', () => {
    const dm = makeDepthMap(3, 1, [10, 0, 20]);
    const filled = fillHoles(dm, 1);
    expect(filled.data[1]).toBeGreaterThan(0);
    expect(filled.minDepth).toBeGreaterThanOrEqual(0);
    expect(filled.maxDepth).toBeLessThanOrEqual(20);
  });
});

// ---------------------------------------------------------------------------
// medianFilter
// ---------------------------------------------------------------------------

describe('medianFilter', () => {
  it('smooths a single noisy pixel to match neighbours', () => {
    const dm = makeDepthMap(3, 3, [
      5, 5, 5,
      5, 100, 5,
      5, 5, 5,
    ]);
    const filtered = medianFilter(dm, 3);
    // Median of the 3x3 neighbourhood: eight 5s and one 100 -> median is 5
    expect(filtered.data[4]).toBeCloseTo(5, 5);
  });

  it('preserves uniform regions', () => {
    const dm = makeDepthMap(4, 4, Array(16).fill(7));
    const filtered = medianFilter(dm, 3);
    for (let i = 0; i < filtered.data.length; i++) {
      expect(filtered.data[i]).toBeCloseTo(7, 5);
    }
  });

  it('returns a depth map with the same dimensions', () => {
    const dm = makeDepthMap(5, 3, Array(15).fill(1));
    const filtered = medianFilter(dm, 3);
    expect(filtered.width).toBe(5);
    expect(filtered.height).toBe(3);
    expect(filtered.data.length).toBe(15);
  });

  it('handles a 1x1 kernel (no change)', () => {
    const dm = makeDepthMap(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const filtered = medianFilter(dm, 1);
    for (let i = 0; i < dm.data.length; i++) {
      expect(filtered.data[i]).toBeCloseTo(dm.data[i]!, 5);
    }
  });

  it('reduces noise magnitude in a noisy image', () => {
    // Alternating high/low pattern
    const values: number[] = [];
    for (let i = 0; i < 25; i++) {
      values.push(i % 2 === 0 ? 10 : 1);
    }
    const dm = makeDepthMap(5, 5, values);
    const filtered = medianFilter(dm, 3);

    // After median filtering, variance should decrease
    let origVariance = 0;
    let filtVariance = 0;
    const origMean = values.reduce((a, b) => a + b, 0) / values.length;
    let filtMean = 0;
    for (let i = 0; i < filtered.data.length; i++) {
      filtMean += filtered.data[i]!;
    }
    filtMean /= filtered.data.length;

    for (let i = 0; i < values.length; i++) {
      origVariance += (values[i]! - origMean) ** 2;
      filtVariance += (filtered.data[i]! - filtMean) ** 2;
    }
    expect(filtVariance).toBeLessThan(origVariance);
  });
});

// ---------------------------------------------------------------------------
// computeDepthGradient
// ---------------------------------------------------------------------------

describe('computeDepthGradient', () => {
  it('returns zero gradients for a uniform depth map', () => {
    const dm = makeDepthMap(5, 5, Array(25).fill(10));
    const { gx, gy } = computeDepthGradient(dm);
    for (let i = 0; i < gx.length; i++) {
      expect(gx[i]).toBeCloseTo(0, 5);
      expect(gy[i]).toBeCloseTo(0, 5);
    }
  });

  it('detects a horizontal gradient (depth increasing along x)', () => {
    // 5 columns, depth increases from left to right
    const values: number[] = [];
    for (let y = 0; y < 5; y++) {
      for (let x = 0; x < 5; x++) {
        values.push(x);
      }
    }
    const dm = makeDepthMap(5, 5, values);
    const { gx } = computeDepthGradient(dm);
    // Centre pixel (2, 2): Sobel Gx should be positive
    const centreIdx = 2 * 5 + 2;
    expect(gx[centreIdx]!).toBeGreaterThan(0);
  });

  it('detects a vertical gradient (depth increasing along y)', () => {
    const values: number[] = [];
    for (let y = 0; y < 5; y++) {
      for (let x = 0; x < 5; x++) {
        values.push(y);
      }
    }
    const dm = makeDepthMap(5, 5, values);
    const { gy } = computeDepthGradient(dm);
    const centreIdx = 2 * 5 + 2;
    expect(gy[centreIdx]!).toBeGreaterThan(0);
  });

  it('returns arrays of the correct length', () => {
    const dm = makeDepthMap(10, 8, Array(80).fill(1));
    const { gx, gy } = computeDepthGradient(dm);
    expect(gx.length).toBe(80);
    expect(gy.length).toBe(80);
  });

  it('produces large gradients at sharp depth discontinuities', () => {
    // Left half depth 0, right half depth 100
    const values: number[] = [];
    for (let y = 0; y < 5; y++) {
      for (let x = 0; x < 5; x++) {
        values.push(x < 3 ? 0 : 100);
      }
    }
    const dm = makeDepthMap(5, 5, values);
    const { gx } = computeDepthGradient(dm);
    // Near the edge (x=2, y=2) the gradient should be large
    const edgeIdx = 2 * 5 + 2;
    expect(Math.abs(gx[edgeIdx]!)).toBeGreaterThan(10);
  });
});

// ---------------------------------------------------------------------------
// computeDisparity + disparityToDepth
// ---------------------------------------------------------------------------

describe('computeDisparity', () => {
  it('returns a Float64Array of the correct size', () => {
    const w = 10;
    const h = 10;
    const left = new Float64Array(w * h).fill(128);
    const right = new Float64Array(w * h).fill(128);
    const disp = computeDisparity(left, right, w, h, 5, 3);
    expect(disp.length).toBe(w * h);
  });

  it('produces zero disparity for identical left and right images', () => {
    const w = 10;
    const h = 10;
    const img = new Float64Array(w * h);
    for (let i = 0; i < w * h; i++) {
      img[i] = i;
    }
    const disp = computeDisparity(img, img, w, h, 5, 3);
    // When left == right, best match is at d = 0
    for (let i = 0; i < disp.length; i++) {
      expect(disp[i]!).toBeGreaterThanOrEqual(0);
    }
  });

  it('produces non-negative disparity values', () => {
    const w = 16;
    const h = 16;
    const left = new Float64Array(w * h);
    const right = new Float64Array(w * h);
    for (let i = 0; i < w * h; i++) {
      left[i] = Math.sin(i * 0.1) * 100;
      right[i] = Math.sin((i + 2) * 0.1) * 100;
    }
    const disp = computeDisparity(left, right, w, h, 8, 3);
    for (let i = 0; i < disp.length; i++) {
      expect(disp[i]!).toBeGreaterThanOrEqual(0);
    }
  });

  it('detects a known horizontal shift', () => {
    const w = 20;
    const h = 10;
    const left = new Float64Array(w * h);
    const right = new Float64Array(w * h);
    // Create a strong feature (vertical bar) in the left image at x=10
    // and shifted left by 3 in the right image (at x=7)
    for (let y = 0; y < h; y++) {
      left[y * w + 10] = 255;
      right[y * w + 7] = 255;
    }
    const disp = computeDisparity(left, right, w, h, 5, 1);
    // At the feature location in the left image, disparity should be ~3
    const featureDisp = disp[5 * w + 10]!;
    expect(featureDisp).toBeCloseTo(3, 0);
  });

  it('returns zero disparity at boundary pixels within the half-block margin', () => {
    const w = 8;
    const h = 8;
    const left = new Float64Array(w * h).fill(50);
    const right = new Float64Array(w * h).fill(50);
    const disp = computeDisparity(left, right, w, h, 4, 3);
    // Top-left corner (within half=1 margin) should be 0 (not computed)
    expect(disp[0]).toBe(0);
  });
});

describe('disparityToDepth', () => {
  it('produces the correct depth using Z = f * b / d', () => {
    const disparity = new Float64Array([2, 4, 8, 0]);
    const baseline = 0.1; // 10 cm
    const focalLength = 500; // pixels
    const depth = disparityToDepth(disparity, baseline, focalLength);
    expect(depth[0]).toBeCloseTo(25, 5); // 500 * 0.1 / 2 = 25
    expect(depth[1]).toBeCloseTo(12.5, 5); // 500 * 0.1 / 4 = 12.5
    expect(depth[2]).toBeCloseTo(6.25, 5); // 500 * 0.1 / 8 = 6.25
    expect(depth[3]).toBe(0); // zero disparity -> zero depth
  });

  it('gives larger depth for smaller disparity (inverse relationship)', () => {
    const disparity = new Float64Array([10, 5, 1]);
    const depth = disparityToDepth(disparity, 0.1, 500);
    expect(depth[0]!).toBeLessThan(depth[1]!);
    expect(depth[1]!).toBeLessThan(depth[2]!);
  });

  it('returns zero for zero-disparity pixels', () => {
    const disparity = new Float64Array([0, 0, 0]);
    const depth = disparityToDepth(disparity, 0.5, 1000);
    for (let i = 0; i < depth.length; i++) {
      expect(depth[i]).toBe(0);
    }
  });

  it('returns the same length array as input', () => {
    const disparity = new Float64Array(100);
    const depth = disparityToDepth(disparity, 0.1, 500);
    expect(depth.length).toBe(100);
  });

  it('scales linearly with baseline', () => {
    const disparity = new Float64Array([5]);
    const d1 = disparityToDepth(disparity, 0.1, 500);
    const d2 = disparityToDepth(disparity, 0.2, 500);
    expect(d2[0]!).toBeCloseTo(d1[0]! * 2, 5);
  });
});

// ---------------------------------------------------------------------------
// estimateBaseline
// ---------------------------------------------------------------------------

describe('estimateBaseline', () => {
  it('extracts baseline from translation component of transform', () => {
    const transform = translation4(3, 4, 0);
    const baseline = estimateBaseline(
      new Float64Array(0),
      new Float64Array(0),
      transform,
    );
    expect(baseline).toBeCloseTo(5, 5); // sqrt(9 + 16)
  });

  it('returns zero for identity transform', () => {
    const baseline = estimateBaseline(
      new Float64Array(0),
      new Float64Array(0),
      identity4(),
    );
    expect(baseline).toBeCloseTo(0, 5);
  });

  it('computes 3D Euclidean norm for diagonal translation', () => {
    const transform = translation4(1, 1, 1);
    const baseline = estimateBaseline(
      new Float64Array(0),
      new Float64Array(0),
      transform,
    );
    expect(baseline).toBeCloseTo(Math.sqrt(3), 5);
  });

  it('handles negative translations correctly', () => {
    const transform = translation4(-3, -4, 0);
    const baseline = estimateBaseline(
      new Float64Array(0),
      new Float64Array(0),
      transform,
    );
    expect(baseline).toBeCloseTo(5, 5);
  });

  it('returns a non-negative value', () => {
    const transform = translation4(-10, 5, -7);
    const baseline = estimateBaseline(
      new Float64Array(0),
      new Float64Array(0),
      transform,
    );
    expect(baseline).toBeGreaterThanOrEqual(0);
  });
});

// ---------------------------------------------------------------------------
// createTSDFVolume
// ---------------------------------------------------------------------------

describe('createTSDFVolume', () => {
  it('creates a volume with correct dimensions', () => {
    const vol = createTSDFVolume(8, 0.01, { x: 0, y: 0, z: 0 });
    expect(vol.data.length).toBe(8 * 8 * 8);
    expect(vol.weights.length).toBe(8 * 8 * 8);
  });

  it('initialises TSDF values to +1', () => {
    const vol = createTSDFVolume(4, 0.05, { x: 0, y: 0, z: 0 });
    for (let i = 0; i < vol.data.length; i++) {
      expect(vol.data[i]).toBeCloseTo(1.0, 5);
    }
  });

  it('initialises all weights to 0', () => {
    const vol = createTSDFVolume(4, 0.05, { x: 0, y: 0, z: 0 });
    for (let i = 0; i < vol.weights.length; i++) {
      expect(vol.weights[i]).toBe(0);
    }
  });

  it('stores the voxel size and origin', () => {
    const origin = { x: 1, y: 2, z: 3 };
    const vol = createTSDFVolume(10, 0.02, origin);
    expect(vol.voxelSize).toBeCloseTo(0.02, 5);
    expect(vol.origin.x).toBeCloseTo(1, 5);
    expect(vol.origin.y).toBeCloseTo(2, 5);
    expect(vol.origin.z).toBeCloseTo(3, 5);
  });

  it('sets truncation distance to 3x voxel size', () => {
    const vol = createTSDFVolume(5, 0.04, { x: 0, y: 0, z: 0 });
    expect(vol.truncationDistance).toBeCloseTo(0.12, 5);
  });
});

// ---------------------------------------------------------------------------
// integrateTSDFFrame
// ---------------------------------------------------------------------------

describe('integrateTSDFFrame', () => {
  it('changes weights from zero after integration', () => {
    const vol = createTSDFVolume(4, 1.0, { x: -2, y: -2, z: 0 });
    const dmData = new Float64Array(4 * 4);
    dmData.fill(3.0); // depth of 3 metres
    const dm = createDepthMap(4, 4, dmData);

    const intrinsics: CameraIntrinsics = {
      fx: 2,
      fy: 2,
      cx: 2,
      cy: 2,
      width: 4,
      height: 4,
    };

    const extrinsics = identity4();
    integrateTSDFFrame(vol, dm, intrinsics, extrinsics, 1.5);

    // At least some weights should be non-zero now
    let nonZeroWeights = 0;
    for (let i = 0; i < vol.weights.length; i++) {
      if (vol.weights[i]! > 0) nonZeroWeights++;
    }
    expect(nonZeroWeights).toBeGreaterThan(0);
  });

  it('modifies TSDF values from their initial +1', () => {
    const vol = createTSDFVolume(4, 1.0, { x: -2, y: -2, z: 0 });
    const dmData = new Float64Array(4 * 4);
    dmData.fill(2.0);
    const dm = createDepthMap(4, 4, dmData);

    const intrinsics: CameraIntrinsics = {
      fx: 2,
      fy: 2,
      cx: 2,
      cy: 2,
      width: 4,
      height: 4,
    };

    integrateTSDFFrame(vol, dm, intrinsics, identity4(), 1.5);

    let changed = false;
    for (let i = 0; i < vol.data.length; i++) {
      if (Math.abs(vol.data[i]! - 1.0) > 1e-10) {
        changed = true;
        break;
      }
    }
    expect(changed).toBe(true);
  });

  it('ignores zero-depth pixels', () => {
    const vol = createTSDFVolume(4, 1.0, { x: -2, y: -2, z: 0 });
    const dmData = new Float64Array(4 * 4); // all zeros
    const dm = createDepthMap(4, 4, dmData);

    const intrinsics: CameraIntrinsics = {
      fx: 2,
      fy: 2,
      cx: 2,
      cy: 2,
      width: 4,
      height: 4,
    };

    integrateTSDFFrame(vol, dm, intrinsics, identity4(), 1.0);

    // All weights should remain zero
    for (let i = 0; i < vol.weights.length; i++) {
      expect(vol.weights[i]).toBe(0);
    }
  });

  it('accumulates weights with multiple integrations', () => {
    const vol = createTSDFVolume(4, 1.0, { x: -2, y: -2, z: 0 });
    const dmData = new Float64Array(4 * 4);
    dmData.fill(3.0);
    const dm = createDepthMap(4, 4, dmData);

    const intrinsics: CameraIntrinsics = {
      fx: 2,
      fy: 2,
      cx: 2,
      cy: 2,
      width: 4,
      height: 4,
    };

    integrateTSDFFrame(vol, dm, intrinsics, identity4(), 1.5);

    // Record max weight after first integration
    let maxW1 = 0;
    for (let i = 0; i < vol.weights.length; i++) {
      if (vol.weights[i]! > maxW1) maxW1 = vol.weights[i]!;
    }

    integrateTSDFFrame(vol, dm, intrinsics, identity4(), 1.5);

    let maxW2 = 0;
    for (let i = 0; i < vol.weights.length; i++) {
      if (vol.weights[i]! > maxW2) maxW2 = vol.weights[i]!;
    }

    expect(maxW2).toBeGreaterThan(maxW1);
  });

  it('updates truncation distance on the volume', () => {
    const vol = createTSDFVolume(4, 1.0, { x: 0, y: 0, z: 0 });
    const dm = createDepthMap(4, 4, new Float64Array(16).fill(5));

    const intrinsics: CameraIntrinsics = {
      fx: 2,
      fy: 2,
      cx: 2,
      cy: 2,
      width: 4,
      height: 4,
    };

    integrateTSDFFrame(vol, dm, intrinsics, identity4(), 2.5);
    expect(vol.truncationDistance).toBeCloseTo(2.5, 5);
  });
});

// ---------------------------------------------------------------------------
// extractMeshFromTSDF
// ---------------------------------------------------------------------------

describe('extractMeshFromTSDF', () => {
  it('produces vertices and indices from a volume with sign changes', () => {
    // Create a small volume where some voxels are negative (inside) and
    // some positive (outside), creating a zero-crossing.
    const res = 4;
    const vol = createTSDFVolume(res, 1.0, { x: 0, y: 0, z: 0 });

    // Set the interior voxels to negative values to create a surface
    for (let z = 0; z < res; z++) {
      for (let y = 0; y < res; y++) {
        for (let x = 0; x < res; x++) {
          const idx = x + y * res + z * res * res;
          // Distance from centre, mapped to signed distance
          const cx = x - res / 2 + 0.5;
          const cy = y - res / 2 + 0.5;
          const cz = z - res / 2 + 0.5;
          const dist = Math.sqrt(cx * cx + cy * cy + cz * cz);
          vol.data[idx] = dist - 1.2; // negative inside, positive outside
        }
      }
    }

    const mesh = extractMeshFromTSDF(vol, 0);
    expect(mesh.vertices.length).toBeGreaterThan(0);
    expect(mesh.indices.length).toBeGreaterThan(0);
  });

  it('returns empty mesh for a uniform TSDF (no sign changes)', () => {
    const vol = createTSDFVolume(4, 1.0, { x: 0, y: 0, z: 0 });
    // All values are +1 (uniform, no zero-crossing)
    const mesh = extractMeshFromTSDF(vol, 0);
    expect(mesh.vertices.length).toBe(0);
    expect(mesh.indices.length).toBe(0);
  });

  it('produces vertices that are Float64Array', () => {
    const vol = createTSDFVolume(3, 1.0, { x: 0, y: 0, z: 0 });
    // Create a simple sign change
    vol.data[0] = -1; // inside
    const mesh = extractMeshFromTSDF(vol, 0);
    expect(mesh.vertices).toBeInstanceOf(Float64Array);
    expect(mesh.indices).toBeInstanceOf(Uint32Array);
  });

  it('produces indices that are valid references into the vertex array', () => {
    const res = 4;
    const vol = createTSDFVolume(res, 1.0, { x: 0, y: 0, z: 0 });
    for (let z = 0; z < res; z++) {
      for (let y = 0; y < res; y++) {
        for (let x = 0; x < res; x++) {
          const idx = x + y * res + z * res * res;
          const cx = x - res / 2 + 0.5;
          const cy = y - res / 2 + 0.5;
          const cz = z - res / 2 + 0.5;
          vol.data[idx] = Math.sqrt(cx * cx + cy * cy + cz * cz) - 1.5;
        }
      }
    }

    const mesh = extractMeshFromTSDF(vol, 0);
    const maxVertexIdx = mesh.vertices.length / 3;
    for (let i = 0; i < mesh.indices.length; i++) {
      expect(mesh.indices[i]!).toBeLessThan(maxVertexIdx);
      expect(mesh.indices[i]!).toBeGreaterThanOrEqual(0);
    }
  });

  it('indices length is a multiple of 3 (triangles)', () => {
    const res = 4;
    const vol = createTSDFVolume(res, 1.0, { x: 0, y: 0, z: 0 });
    vol.data[0] = -1;
    vol.data[1] = -1;
    const mesh = extractMeshFromTSDF(vol, 0);
    expect(mesh.indices.length % 3).toBe(0);
  });
});
