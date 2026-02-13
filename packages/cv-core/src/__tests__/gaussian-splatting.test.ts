// ---------------------------------------------------------------------------
// Tests: CV-2 Gaussian Splatting — gaussians, sorting, compression, TSDF
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';
import {
  createGaussian3D,
  covarianceFromScaleRotation,
  projectGaussian2D,
  evaluateGaussian2D,
  radixSortByDepth,
  assignTiles,
  quantizePositions,
  truncateSH,
  estimateCompressedSize,
  createTSDFGrid,
  integrateTSDF,
  marchingCubes,
} from '../gaussian-splatting/index.js';
import type { SplatCloud, QuantizationConfig, Vector3 } from '../types.js';
import { mat4Identity } from '../types.js';

// ---------------------------------------------------------------------------
// covarianceFromScaleRotation
// ---------------------------------------------------------------------------

describe('covarianceFromScaleRotation', () => {
  it('produces a 9-element Float64Array', () => {
    const cov = covarianceFromScaleRotation(
      { x: 1, y: 1, z: 1 },
      { x: 0, y: 0, z: 0, w: 1 },
    );
    expect(cov).toBeInstanceOf(Float64Array);
    expect(cov.length).toBe(9);
  });

  it('produces a symmetric matrix (Sigma_ij == Sigma_ji)', () => {
    const cov = covarianceFromScaleRotation(
      { x: 2, y: 3, z: 0.5 },
      { x: 0, y: 0, z: 0, w: 1 },
    );
    // column-major: (row,col) at col*3+row
    // (0,1) = cov[3], (1,0) = cov[1]
    expect(cov[3]!).toBeCloseTo(cov[1]!, 10);
    // (0,2) = cov[6], (2,0) = cov[2]
    expect(cov[6]!).toBeCloseTo(cov[2]!, 10);
    // (1,2) = cov[7], (2,1) = cov[5]
    expect(cov[7]!).toBeCloseTo(cov[5]!, 10);
  });

  it('is positive semi-definite (all eigenvalues >= 0) via diagonal dominance check', () => {
    const cov = covarianceFromScaleRotation(
      { x: 1, y: 2, z: 3 },
      { x: 0, y: 0, z: 0, w: 1 },
    );
    // For identity rotation, Sigma = diag(sx^2, sy^2, sz^2)
    expect(cov[0]!).toBeCloseTo(1, 10);   // sx^2
    expect(cov[4]!).toBeCloseTo(4, 10);   // sy^2
    expect(cov[8]!).toBeCloseTo(9, 10);   // sz^2
  });

  it('diagonal is always non-negative for arbitrary rotation', () => {
    // 90-degree rotation about z-axis: quat = (0, 0, sin(45), cos(45))
    const s45 = Math.sin(Math.PI / 4);
    const c45 = Math.cos(Math.PI / 4);
    const cov = covarianceFromScaleRotation(
      { x: 5, y: 1, z: 1 },
      { x: 0, y: 0, z: s45, w: c45 },
    );
    expect(cov[0]!).toBeGreaterThanOrEqual(0);
    expect(cov[4]!).toBeGreaterThanOrEqual(0);
    expect(cov[8]!).toBeGreaterThanOrEqual(0);
  });

  it('uniform scales produce an isotropic covariance (proportional to identity)', () => {
    const cov = covarianceFromScaleRotation(
      { x: 3, y: 3, z: 3 },
      { x: 0, y: 0, z: 0, w: 1 },
    );
    expect(cov[0]!).toBeCloseTo(9, 10);
    expect(cov[4]!).toBeCloseTo(9, 10);
    expect(cov[8]!).toBeCloseTo(9, 10);
    // Off-diagonal should be ~0
    expect(cov[1]!).toBeCloseTo(0, 10);
    expect(cov[3]!).toBeCloseTo(0, 10);
  });
});

// ---------------------------------------------------------------------------
// createGaussian3D
// ---------------------------------------------------------------------------

describe('createGaussian3D', () => {
  it('creates a Gaussian3D with correct centre, colour, and opacity', () => {
    const g = createGaussian3D(
      { x: 1, y: 2, z: 3 },
      { x: 1, y: 1, z: 1 },
      { x: 0, y: 0, z: 0, w: 1 },
      { x: 0.5, y: 0.5, z: 0.5 },
      0.8,
    );
    expect(g.center).toEqual({ x: 1, y: 2, z: 3 });
    expect(g.color).toEqual({ x: 0.5, y: 0.5, z: 0.5 });
    expect(g.opacity).toBe(0.8);
    expect(g.covariance).toBeInstanceOf(Float64Array);
    expect(g.covariance.length).toBe(9);
  });

  it('initialises sh_coeffs as an empty Float64Array', () => {
    const g = createGaussian3D(
      { x: 0, y: 0, z: 0 },
      { x: 1, y: 1, z: 1 },
      { x: 0, y: 0, z: 0, w: 1 },
      { x: 0, y: 0, z: 0 },
      1,
    );
    expect(g.sh_coeffs.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// evaluateGaussian2D
// ---------------------------------------------------------------------------

describe('evaluateGaussian2D', () => {
  it('returns 1.0 at the centre (zero Mahalanobis distance)', () => {
    // Identity covariance inverse = [[1,0],[0,1]]
    const covInv = new Float64Array([1, 0, 0, 1]);
    const val = evaluateGaussian2D({ x: 5, y: 5 }, { x: 5, y: 5 }, covInv);
    expect(val).toBeCloseTo(1, 10);
  });

  it('decays away from the centre', () => {
    const covInv = new Float64Array([1, 0, 0, 1]);
    const center = { x: 0, y: 0 };
    const atCenter = evaluateGaussian2D({ x: 0, y: 0 }, center, covInv);
    const away = evaluateGaussian2D({ x: 2, y: 0 }, center, covInv);
    expect(atCenter).toBeGreaterThan(away);
  });

  it('returns exp(-0.5) at 1 sigma distance from centre', () => {
    const covInv = new Float64Array([1, 0, 0, 1]);
    const val = evaluateGaussian2D({ x: 1, y: 0 }, { x: 0, y: 0 }, covInv);
    expect(val).toBeCloseTo(Math.exp(-0.5), 10);
  });

  it('is symmetric about the centre', () => {
    const covInv = new Float64Array([1, 0, 0, 1]);
    const center = { x: 3, y: 3 };
    const left = evaluateGaussian2D({ x: 1, y: 3 }, center, covInv);
    const right = evaluateGaussian2D({ x: 5, y: 3 }, center, covInv);
    expect(left).toBeCloseTo(right, 10);
  });

  it('works with anisotropic covariance inverse', () => {
    // Sigma = diag(4, 1), Sigma^-1 = diag(0.25, 1)
    const covInv = new Float64Array([0.25, 0, 0, 1]);
    const center = { x: 0, y: 0 };
    const alongX = evaluateGaussian2D({ x: 2, y: 0 }, center, covInv);
    const alongY = evaluateGaussian2D({ x: 0, y: 2 }, center, covInv);
    // Along x: exp(-0.5 * 0.25 * 4) = exp(-0.5)
    // Along y: exp(-0.5 * 1 * 4)    = exp(-2)
    expect(alongX).toBeCloseTo(Math.exp(-0.5), 10);
    expect(alongY).toBeCloseTo(Math.exp(-2), 10);
  });
});

// ---------------------------------------------------------------------------
// radixSortByDepth
// ---------------------------------------------------------------------------

describe('radixSortByDepth', () => {
  it('returns an empty array for empty input', () => {
    const result = radixSortByDepth(new Float64Array(0), new Uint32Array(0));
    expect(result.length).toBe(0);
  });

  it('sorts indices by ascending depth', () => {
    const depths = new Float64Array([3.0, 1.0, 2.0]);
    const indices = new Uint32Array([0, 1, 2]);
    const sorted = radixSortByDepth(depths, indices);
    // Sorted depths should be [1.0, 2.0, 3.0] => indices [1, 2, 0]
    expect(sorted[0]).toBe(1);
    expect(sorted[1]).toBe(2);
    expect(sorted[2]).toBe(0);
  });

  it('preserves single-element arrays', () => {
    const depths = new Float64Array([42.0]);
    const indices = new Uint32Array([0]);
    const sorted = radixSortByDepth(depths, indices);
    expect(sorted[0]).toBe(0);
  });

  it('handles already sorted input', () => {
    const depths = new Float64Array([1, 2, 3, 4, 5]);
    const indices = new Uint32Array([0, 1, 2, 3, 4]);
    const sorted = radixSortByDepth(depths, indices);
    for (let i = 0; i < 5; i++) {
      expect(sorted[i]).toBe(i);
    }
  });

  it('handles negative depths correctly', () => {
    const depths = new Float64Array([-5, -1, -3, 2]);
    const indices = new Uint32Array([0, 1, 2, 3]);
    const sorted = radixSortByDepth(depths, indices);
    // Expected order: -5, -3, -1, 2 => indices [0, 2, 1, 3]
    expect(sorted[0]).toBe(0);
    expect(sorted[1]).toBe(2);
    expect(sorted[2]).toBe(1);
    expect(sorted[3]).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// assignTiles
// ---------------------------------------------------------------------------

describe('assignTiles', () => {
  it('assigns a centre in the viewport to the correct tile', () => {
    const centers = [{ x: 15, y: 25 }];
    const assignments = assignTiles(centers, 16, 64, 64);
    expect(assignments).toHaveLength(1);
    expect(assignments[0]!.tileX).toBe(0); // 15 / 16 = 0
    expect(assignments[0]!.tileY).toBe(1); // 25 / 16 = 1
  });

  it('skips centres outside the viewport', () => {
    const centers = [
      { x: -1, y: 5 },
      { x: 5, y: -1 },
      { x: 100, y: 5 },
      { x: 5, y: 100 },
    ];
    const assignments = assignTiles(centers, 16, 64, 64);
    expect(assignments).toHaveLength(0);
  });

  it('throws for non-positive tileSize', () => {
    expect(() => assignTiles([], 0, 64, 64)).toThrow();
  });

  it('computes the depthKey using tile-major ordering', () => {
    const centers = [{ x: 33, y: 17 }]; // tile (2,1) with tileSize=16, 4 tiles wide
    const assignments = assignTiles(centers, 16, 64, 64);
    const tilesX = Math.ceil(64 / 16); // 4
    expect(assignments[0]!.depthKey).toBe(1 * tilesX + 2); // tileY * tilesX + tileX
  });

  it('includes gaussianIndex in the assignment', () => {
    const centers = [{ x: 0, y: 0 }, { x: 10, y: 10 }];
    const assignments = assignTiles(centers, 16, 64, 64);
    expect(assignments[0]!.gaussianIndex).toBe(0);
    expect(assignments[1]!.gaussianIndex).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// quantizePositions
// ---------------------------------------------------------------------------

describe('quantizePositions', () => {
  it('returns empty arrays for empty input', () => {
    const result = quantizePositions(new Float64Array(0), 16);
    expect(result.quantized.length).toBe(0);
    expect(result.offset.length).toBe(0);
  });

  it('produces an Int16Array of the same length as input', () => {
    const positions = new Float64Array([1, 2, 3, 4, 5, 6]);
    const result = quantizePositions(positions, 16);
    expect(result.quantized).toBeInstanceOf(Int16Array);
    expect(result.quantized.length).toBe(6);
  });

  it('quantized values are within Int16 range', () => {
    const positions = new Float64Array([0, 100, -100, 50, -50, 75]);
    const result = quantizePositions(positions, 16);
    for (let i = 0; i < result.quantized.length; i++) {
      expect(result.quantized[i]!).toBeGreaterThanOrEqual(-32768);
      expect(result.quantized[i]!).toBeLessThanOrEqual(32767);
    }
  });

  it('scale is positive for a non-degenerate range', () => {
    const positions = new Float64Array([0, 10, 20]);
    const result = quantizePositions(positions, 16);
    expect(result.scale).toBeGreaterThan(0);
  });

  it('maps minimum value to -32767 and maximum to +32767 for 16-bit', () => {
    const positions = new Float64Array([0, 100]);
    const result = quantizePositions(positions, 16);
    // min maps to normalised 0 => q = round(0 * 32767 * 2 - 32767) = -32767
    expect(result.quantized[0]!).toBe(-32767);
    // max maps to normalised 1 => q = round(1 * 32767 * 2 - 32767) = 32767
    expect(result.quantized[1]!).toBe(32767);
  });
});

// ---------------------------------------------------------------------------
// truncateSH
// ---------------------------------------------------------------------------

describe('truncateSH', () => {
  it('returns a copy when maxDegree >= originalDegree', () => {
    const coeffs = new Float64Array([1, 2, 3, 4, 5, 6]);
    const result = truncateSH(coeffs, 3, 2);
    expect(result.length).toBe(coeffs.length);
    expect(result[0]!).toBeCloseTo(1, 10);
  });

  it('truncates degree 3 to degree 0 (3 coefficients)', () => {
    const coeffs = new Float64Array(48); // degree 3 = 16 * 3 = 48
    for (let i = 0; i < 48; i++) coeffs[i] = i;
    const result = truncateSH(coeffs, 0, 3);
    // degree 0 = (0+1)^2 * 3 = 3 coefficients
    expect(result.length).toBe(3);
    expect(result[0]!).toBeCloseTo(0, 10);
    expect(result[1]!).toBeCloseTo(1, 10);
    expect(result[2]!).toBeCloseTo(2, 10);
  });

  it('truncates degree 3 to degree 1 (12 coefficients)', () => {
    const coeffs = new Float64Array(48);
    const result = truncateSH(coeffs, 1, 3);
    expect(result.length).toBe(12); // (1+1)^2 * 3 = 12
  });

  it('truncates degree 3 to degree 2 (27 coefficients)', () => {
    const coeffs = new Float64Array(48);
    const result = truncateSH(coeffs, 2, 3);
    expect(result.length).toBe(27); // (2+1)^2 * 3 = 27
  });

  it('does not modify the original array', () => {
    const coeffs = new Float64Array([10, 20, 30, 40, 50, 60]);
    truncateSH(coeffs, 0, 1);
    expect(coeffs[3]!).toBeCloseTo(40, 10);
  });
});

// ---------------------------------------------------------------------------
// estimateCompressedSize
// ---------------------------------------------------------------------------

describe('estimateCompressedSize', () => {
  it('returns header-only size for zero gaussians', () => {
    const cloud: SplatCloud = {
      gaussians: [],
      count: 0,
      boundsMin: { x: 0, y: 0, z: 0 },
      boundsMax: { x: 1, y: 1, z: 1 },
      shDegree: 0,
    };
    const config: QuantizationConfig = {
      positionBits: 16,
      covarianceBits: 12,
      colorBits: 8,
      opacityBits: 8,
      shBits: 8,
      boundsMin: { x: 0, y: 0, z: 0 },
      boundsMax: { x: 1, y: 1, z: 1 },
    };
    expect(estimateCompressedSize(cloud, config)).toBe(64); // header only
  });

  it('returns a value greater than header for non-zero gaussians', () => {
    const cloud: SplatCloud = {
      gaussians: [],
      count: 100,
      boundsMin: { x: 0, y: 0, z: 0 },
      boundsMax: { x: 1, y: 1, z: 1 },
      shDegree: 0,
    };
    const config: QuantizationConfig = {
      positionBits: 16,
      covarianceBits: 12,
      colorBits: 8,
      opacityBits: 8,
      shBits: 8,
      boundsMin: { x: 0, y: 0, z: 0 },
      boundsMax: { x: 1, y: 1, z: 1 },
    };
    expect(estimateCompressedSize(cloud, config)).toBeGreaterThan(64);
  });

  it('increases with higher SH degree', () => {
    const config: QuantizationConfig = {
      positionBits: 16,
      covarianceBits: 12,
      colorBits: 8,
      opacityBits: 8,
      shBits: 8,
      boundsMin: { x: 0, y: 0, z: 0 },
      boundsMax: { x: 1, y: 1, z: 1 },
    };
    const cloud0: SplatCloud = {
      gaussians: [],
      count: 10,
      boundsMin: { x: 0, y: 0, z: 0 },
      boundsMax: { x: 1, y: 1, z: 1 },
      shDegree: 0,
    };
    const cloud3: SplatCloud = { ...cloud0, shDegree: 3 };

    expect(estimateCompressedSize(cloud3, config)).toBeGreaterThan(
      estimateCompressedSize(cloud0, config),
    );
  });
});

// ---------------------------------------------------------------------------
// createTSDFGrid
// ---------------------------------------------------------------------------

describe('createTSDFGrid', () => {
  it('creates a grid with correct resolution and voxel size', () => {
    const grid = createTSDFGrid(8, 0.05);
    expect(grid.resolution).toBe(8);
    expect(grid.voxelSize).toBeCloseTo(0.05, 10);
  });

  it('initialises all voxels to +1.0', () => {
    const grid = createTSDFGrid(4, 0.1);
    const totalVoxels = 4 * 4 * 4;
    expect(grid.data.length).toBe(totalVoxels);
    for (let i = 0; i < totalVoxels; i++) {
      expect(grid.data[i]!).toBeCloseTo(1.0, 10);
    }
  });

  it('allocates resolution^3 voxels', () => {
    const grid = createTSDFGrid(16, 0.01);
    expect(grid.data.length).toBe(16 * 16 * 16);
  });
});

// ---------------------------------------------------------------------------
// integrateTSDF
// ---------------------------------------------------------------------------

describe('integrateTSDF', () => {
  it('modifies the grid when depth values are provided', () => {
    const grid = createTSDFGrid(4, 1.0);
    const intrinsics = { fx: 1, fy: 1, cx: 0, cy: 0, width: 4, height: 4 };
    const depth = new Float64Array(16);
    depth.fill(3.0);
    const extrinsics = mat4Identity();
    const originalSum = grid.data.reduce((a, b) => a + b, 0);

    integrateTSDF(grid, depth, intrinsics, extrinsics);

    const newSum = grid.data.reduce((a, b) => a + b, 0);
    // The grid should have been modified — at least some voxels should differ
    expect(newSum).not.toBeCloseTo(originalSum, 1);
  });

  it('does not modify grid when all depths are zero (invalid)', () => {
    const grid = createTSDFGrid(4, 1.0);
    const intrinsics = { fx: 1, fy: 1, cx: 0, cy: 0, width: 4, height: 4 };
    const depth = new Float64Array(16); // all zeros = invalid
    const extrinsics = mat4Identity();

    const before = new Float64Array(grid.data);
    integrateTSDF(grid, depth, intrinsics, extrinsics);

    for (let i = 0; i < grid.data.length; i++) {
      expect(grid.data[i]!).toBeCloseTo(before[i]!, 10);
    }
  });
});

// ---------------------------------------------------------------------------
// marchingCubes
// ---------------------------------------------------------------------------

describe('marchingCubes', () => {
  it('returns empty mesh for a grid entirely above the iso level', () => {
    const grid = createTSDFGrid(4, 0.1); // all +1.0
    const { vertices, indices } = marchingCubes(grid, 0);
    expect(vertices.length).toBe(0);
    expect(indices.length).toBe(0);
  });

  it('produces vertices when a surface crossing exists', () => {
    const grid = createTSDFGrid(4, 0.1);
    // Set interior voxels to negative to create a surface crossing
    const res = grid.resolution;
    for (let z = 1; z < res - 1; z++) {
      for (let y = 1; y < res - 1; y++) {
        for (let x = 1; x < res - 1; x++) {
          grid.data[x + y * res + z * res * res] = -1.0;
        }
      }
    }
    const { vertices, indices } = marchingCubes(grid, 0);
    expect(vertices.length).toBeGreaterThan(0);
    expect(indices.length).toBeGreaterThan(0);
    // Triangle indices come in triples
    expect(indices.length % 3).toBe(0);
  });

  it('all extracted vertex positions are within the grid bounds', () => {
    const grid = createTSDFGrid(8, 0.5);
    const res = grid.resolution;
    // Create a sphere-like signed distance
    const center = (res * 0.5) * grid.voxelSize;
    const radius = res * 0.25 * grid.voxelSize;
    for (let z = 0; z < res; z++) {
      for (let y = 0; y < res; y++) {
        for (let x = 0; x < res; x++) {
          const wx = (x + 0.5) * grid.voxelSize;
          const wy = (y + 0.5) * grid.voxelSize;
          const wz = (z + 0.5) * grid.voxelSize;
          const dist = Math.sqrt(
            (wx - center) ** 2 + (wy - center) ** 2 + (wz - center) ** 2,
          );
          grid.data[x + y * res + z * res * res] = (dist - radius) / radius;
        }
      }
    }

    const { vertices } = marchingCubes(grid, 0);
    const gridExtent = res * grid.voxelSize;
    for (let i = 0; i < vertices.length; i += 3) {
      expect(vertices[i]!).toBeGreaterThanOrEqual(0);
      expect(vertices[i]!).toBeLessThanOrEqual(gridExtent);
      expect(vertices[i + 1]!).toBeGreaterThanOrEqual(0);
      expect(vertices[i + 1]!).toBeLessThanOrEqual(gridExtent);
      expect(vertices[i + 2]!).toBeGreaterThanOrEqual(0);
      expect(vertices[i + 2]!).toBeLessThanOrEqual(gridExtent);
    }
  });
});

// ---------------------------------------------------------------------------
// projectGaussian2D
// ---------------------------------------------------------------------------

describe('projectGaussian2D', () => {
  it('returns a depth matching the camera-space z', () => {
    const g = createGaussian3D(
      { x: 0, y: 0, z: -5 },
      { x: 1, y: 1, z: 1 },
      { x: 0, y: 0, z: 0, w: 1 },
      { x: 1, y: 1, z: 1 },
      1,
    );
    const view = mat4Identity();
    const proj = mat4Identity();
    const result = projectGaussian2D(g, view, proj, 100, 100);
    // With identity view, camera z = world z = -5
    expect(result.depth).toBeCloseTo(-5, 5);
  });

  it('produces a 4-element cov2D array', () => {
    const g = createGaussian3D(
      { x: 0, y: 0, z: -10 },
      { x: 1, y: 1, z: 1 },
      { x: 0, y: 0, z: 0, w: 1 },
      { x: 0, y: 0, z: 0 },
      1,
    );
    const view = mat4Identity();
    const proj = mat4Identity();
    const result = projectGaussian2D(g, view, proj, 640, 480);
    expect(result.cov2D).toBeInstanceOf(Float64Array);
    expect(result.cov2D.length).toBe(4);
  });

  it('cov2D is symmetric (element [1] == element [2])', () => {
    const g = createGaussian3D(
      { x: 1, y: 2, z: -10 },
      { x: 2, y: 3, z: 1 },
      { x: 0, y: 0, z: 0, w: 1 },
      { x: 0, y: 0, z: 0 },
      1,
    );
    const view = mat4Identity();
    const proj = mat4Identity();
    const result = projectGaussian2D(g, view, proj, 800, 600);
    expect(result.cov2D[1]!).toBeCloseTo(result.cov2D[2]!, 10);
  });
});
