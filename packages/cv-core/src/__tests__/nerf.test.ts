import { describe, it, expect } from 'vitest';
import {
  sampleRay,
  alphaComposite,
  computeTransmittance,
  renderRay,
  sphereTrace,
  estimateNormal,
  sdfToMesh,
  createHashEncoding,
  hashVertex,
  trilinearInterpolate,
} from '../nerf/index.js';
import type {
  Vector3,
  VolumeSample,
  VolumeRenderConfig,
  HashEncodingConfig,
} from '../types.js';
import { createPRNG, vec3Length } from '../types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Sphere SDF: distance from point to unit sphere centred at origin. */
function sphereSDF(p: Vector3): number {
  return Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z) - 1.0;
}

// ---------------------------------------------------------------------------
// Tests: sampleRay
// ---------------------------------------------------------------------------

describe('sampleRay', () => {
  it('returns the correct number of samples', () => {
    const origin: Vector3 = { x: 0, y: 0, z: 0 };
    const dir: Vector3 = { x: 0, y: 0, z: 1 };
    const samples = sampleRay(origin, dir, 1.0, 10.0, 64);

    expect(samples).toBeInstanceOf(Float64Array);
    expect(samples.length).toBe(64);
  });

  it('all samples are between near and far', () => {
    const origin: Vector3 = { x: 0, y: 0, z: 0 };
    const dir: Vector3 = { x: 0, y: 0, z: 1 };
    const near = 2.0;
    const far = 8.0;
    const samples = sampleRay(origin, dir, near, far, 32);

    for (let i = 0; i < samples.length; i++) {
      expect(samples[i]!).toBeGreaterThanOrEqual(near);
      expect(samples[i]!).toBeLessThanOrEqual(far);
    }
  });

  it('samples are monotonically non-decreasing (stratified)', () => {
    const origin: Vector3 = { x: 0, y: 0, z: 0 };
    const dir: Vector3 = { x: 1, y: 0, z: 0 };
    const samples = sampleRay(origin, dir, 0, 100, 50);

    for (let i = 1; i < samples.length; i++) {
      expect(samples[i]!).toBeGreaterThanOrEqual(samples[i - 1]!);
    }
  });

  it('produces deterministic results with the same PRNG seed', () => {
    const origin: Vector3 = { x: 0, y: 0, z: 0 };
    const dir: Vector3 = { x: 0, y: 0, z: 1 };
    const rng1 = createPRNG(123);
    const rng2 = createPRNG(123);

    const s1 = sampleRay(origin, dir, 0, 10, 16, rng1);
    const s2 = sampleRay(origin, dir, 0, 10, 16, rng2);

    for (let i = 0; i < s1.length; i++) {
      expect(s1[i]!).toBeCloseTo(s2[i]!, 15);
    }
  });
});

// ---------------------------------------------------------------------------
// Tests: computeTransmittance
// ---------------------------------------------------------------------------

describe('computeTransmittance', () => {
  it('starts at 1.0 for the first sample', () => {
    const sigmas = new Float64Array([0.5, 1.0, 0.3]);
    const deltas = new Float64Array([0.1, 0.1, 0.1]);
    const T = computeTransmittance(sigmas, deltas);

    expect(T[0]!).toBeCloseTo(1.0, 10);
  });

  it('transmittance is monotonically non-increasing for positive densities', () => {
    const sigmas = new Float64Array([1.0, 2.0, 0.5, 3.0, 1.5]);
    const deltas = new Float64Array([0.5, 0.5, 0.5, 0.5, 0.5]);
    const T = computeTransmittance(sigmas, deltas);

    for (let i = 1; i < T.length; i++) {
      expect(T[i]!).toBeLessThanOrEqual(T[i - 1]! + 1e-10);
    }
  });

  it('all transmittance values are in [0, 1]', () => {
    const sigmas = new Float64Array([5.0, 10.0, 2.0]);
    const deltas = new Float64Array([1.0, 1.0, 1.0]);
    const T = computeTransmittance(sigmas, deltas);

    for (let i = 0; i < T.length; i++) {
      expect(T[i]!).toBeGreaterThanOrEqual(0);
      expect(T[i]!).toBeLessThanOrEqual(1.0 + 1e-10);
    }
  });

  it('is 1.0 everywhere when all densities are zero', () => {
    const sigmas = new Float64Array([0, 0, 0, 0]);
    const deltas = new Float64Array([1, 1, 1, 1]);
    const T = computeTransmittance(sigmas, deltas);

    for (let i = 0; i < T.length; i++) {
      expect(T[i]!).toBeCloseTo(1.0, 10);
    }
  });
});

// ---------------------------------------------------------------------------
// Tests: alphaComposite
// ---------------------------------------------------------------------------

describe('alphaComposite', () => {
  it('opacity sums to <= 1 for moderate densities', () => {
    const samples: VolumeSample[] = [
      { t: 1, sigma: 0.5, rgb: { x: 1, y: 0, z: 0 } },
      { t: 2, sigma: 0.5, rgb: { x: 0, y: 1, z: 0 } },
      { t: 3, sigma: 0.5, rgb: { x: 0, y: 0, z: 1 } },
      { t: 4, sigma: 0.5, rgb: { x: 1, y: 1, z: 0 } },
      { t: 5, sigma: 0.5, rgb: { x: 0, y: 1, z: 1 } },
    ];
    const result = alphaComposite(samples);

    expect(result.opacity).toBeGreaterThanOrEqual(0);
    expect(result.opacity).toBeLessThanOrEqual(1.0 + 1e-6);
  });

  it('rgb components are non-negative', () => {
    const samples: VolumeSample[] = [
      { t: 0.5, sigma: 1.0, rgb: { x: 0.8, y: 0.2, z: 0.5 } },
      { t: 1.5, sigma: 2.0, rgb: { x: 0.3, y: 0.7, z: 0.1 } },
      { t: 2.5, sigma: 0.5, rgb: { x: 0.6, y: 0.6, z: 0.6 } },
    ];
    const result = alphaComposite(samples);

    expect(result.rgb.x).toBeGreaterThanOrEqual(0);
    expect(result.rgb.y).toBeGreaterThanOrEqual(0);
    expect(result.rgb.z).toBeGreaterThanOrEqual(0);
  });

  it('returns zero opacity for zero-density samples', () => {
    const samples: VolumeSample[] = [
      { t: 1, sigma: 0, rgb: { x: 1, y: 1, z: 1 } },
      { t: 2, sigma: 0, rgb: { x: 1, y: 1, z: 1 } },
    ];
    const result = alphaComposite(samples);

    expect(result.opacity).toBeCloseTo(0, 10);
  });

  it('depth is a weighted combination of t values', () => {
    const samples: VolumeSample[] = [
      { t: 2.0, sigma: 10.0, rgb: { x: 1, y: 0, z: 0 } },
      { t: 5.0, sigma: 0.01, rgb: { x: 0, y: 1, z: 0 } },
    ];
    const result = alphaComposite(samples);

    // With high sigma at t=2.0, the expected depth should be close to 2.0
    expect(result.depth).toBeGreaterThan(1.5);
    expect(result.depth).toBeLessThan(5.5);
  });
});

// ---------------------------------------------------------------------------
// Tests: renderRay
// ---------------------------------------------------------------------------

describe('renderRay', () => {
  it('produces valid colour and opacity values', () => {
    const config: VolumeRenderConfig = {
      near: 0.1,
      far: 5.0,
      nCoarseSamples: 32,
      nFineSamples: 0,
      hierarchical: false,
      whiteBackground: false,
      densityNoise: 0,
      chunkSize: 1024,
    };
    const origin: Vector3 = { x: 0, y: 0, z: -3 };
    const direction: Vector3 = { x: 0, y: 0, z: 1 };

    const result = renderRay(config, origin, direction);

    expect(result.opacity).toBeGreaterThanOrEqual(0);
    expect(result.opacity).toBeLessThanOrEqual(1.0 + 1e-6);
    expect(result.weights.length).toBe(32);
    expect(result.rgb.x).toBeGreaterThanOrEqual(0);
  });

  it('weights sum to the total opacity', () => {
    const config: VolumeRenderConfig = {
      near: 0.5,
      far: 4.0,
      nCoarseSamples: 16,
      nFineSamples: 0,
      hierarchical: false,
      whiteBackground: false,
      densityNoise: 0,
      chunkSize: 1024,
    };
    const origin: Vector3 = { x: 0, y: 0, z: -2 };
    const direction: Vector3 = { x: 0, y: 0, z: 1 };

    const result = renderRay(config, origin, direction);

    let weightSum = 0;
    for (let i = 0; i < result.weights.length; i++) {
      weightSum += result.weights[i]!;
    }
    expect(weightSum).toBeCloseTo(result.opacity, 5);
  });
});

// ---------------------------------------------------------------------------
// Tests: sphereTrace
// ---------------------------------------------------------------------------

describe('sphereTrace', () => {
  it('converges to the surface of a unit sphere', () => {
    const origin: Vector3 = { x: 0, y: 0, z: -5 };
    const direction: Vector3 = { x: 0, y: 0, z: 1 };

    const result = sphereTrace(sphereSDF, origin, direction, 100, 1e-4);

    expect(result.hit).toBe(true);
    // The hit point should be on the sphere surface (distance from origin ~ 1)
    const dist = Math.sqrt(
      result.point.x * result.point.x +
      result.point.y * result.point.y +
      result.point.z * result.point.z,
    );
    expect(dist).toBeCloseTo(1.0, 2);
  });

  it('misses when the ray does not intersect the sphere', () => {
    const origin: Vector3 = { x: 5, y: 5, z: -5 };
    const direction: Vector3 = { x: 0, y: 0, z: 1 };

    const result = sphereTrace(sphereSDF, origin, direction, 200, 1e-4);

    // Ray at (5, 5, z) is far from the unit sphere: should not hit
    expect(result.hit).toBe(false);
  });

  it('reports a positive distance to the hit point', () => {
    const origin: Vector3 = { x: 3, y: 0, z: 0 };
    const direction: Vector3 = { x: -1, y: 0, z: 0 };

    const result = sphereTrace(sphereSDF, origin, direction, 100, 1e-4);

    expect(result.hit).toBe(true);
    expect(result.distance).toBeGreaterThan(0);
    // Should hit at approximately x = 1 (surface at +X)
    expect(result.point.x).toBeCloseTo(1.0, 2);
  });
});

// ---------------------------------------------------------------------------
// Tests: estimateNormal
// ---------------------------------------------------------------------------

describe('estimateNormal', () => {
  it('estimates unit-length normal on a sphere', () => {
    const point: Vector3 = { x: 1, y: 0, z: 0 };
    const normal = estimateNormal(sphereSDF, point);

    const len = vec3Length(normal);
    expect(len).toBeCloseTo(1.0, 3);
  });

  it('normal points outward on the sphere surface', () => {
    const point: Vector3 = { x: 0, y: 1, z: 0 };
    const normal = estimateNormal(sphereSDF, point);

    // For a sphere at the origin, the outward normal at (0,1,0) should be ~ (0,1,0)
    expect(normal.x).toBeCloseTo(0, 2);
    expect(normal.y).toBeCloseTo(1, 2);
    expect(normal.z).toBeCloseTo(0, 2);
  });

  it('normal at (0,0,1) points in z direction', () => {
    const point: Vector3 = { x: 0, y: 0, z: 1 };
    const normal = estimateNormal(sphereSDF, point);

    expect(normal.z).toBeCloseTo(1, 2);
    expect(Math.abs(normal.x)).toBeLessThan(0.05);
    expect(Math.abs(normal.y)).toBeLessThan(0.05);
  });
});

// ---------------------------------------------------------------------------
// Tests: sdfToMesh
// ---------------------------------------------------------------------------

describe('sdfToMesh', () => {
  it('produces vertices and indices for a sphere SDF', () => {
    const bounds = {
      min: { x: -2, y: -2, z: -2 },
      max: { x: 2, y: 2, z: 2 },
    };
    const result = sdfToMesh(sphereSDF, bounds, 8);

    expect(result.vertices.length).toBeGreaterThan(0);
    expect(result.indices.length).toBeGreaterThan(0);
    // Indices should be multiples of 3 (triangles)
    expect(result.indices.length % 3).toBe(0);
  });

  it('mesh vertices lie near the sphere surface', () => {
    const bounds = {
      min: { x: -2, y: -2, z: -2 },
      max: { x: 2, y: 2, z: 2 },
    };
    const result = sdfToMesh(sphereSDF, bounds, 10);

    const nVerts = result.vertices.length / 3;
    for (let i = 0; i < nVerts; i++) {
      const x = result.vertices[i * 3]!;
      const y = result.vertices[i * 3 + 1]!;
      const z = result.vertices[i * 3 + 2]!;
      const dist = Math.sqrt(x * x + y * y + z * z);
      // Vertex should be within a reasonable tolerance of the unit sphere
      expect(dist).toBeGreaterThan(0.3);
      expect(dist).toBeLessThan(2.0);
    }
  });
});

// ---------------------------------------------------------------------------
// Tests: hashVertex
// ---------------------------------------------------------------------------

describe('hashVertex', () => {
  it('returns an index within table size', () => {
    const tableSize = 1024;
    for (let i = 0; i < 50; i++) {
      const idx = hashVertex(i, i * 7, i * 13, tableSize);
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(tableSize);
    }
  });

  it('different inputs generally produce different hashes', () => {
    const tableSize = 65536;
    const hashes = new Set<number>();
    for (let x = 0; x < 10; x++) {
      for (let y = 0; y < 10; y++) {
        hashes.add(hashVertex(x, y, 0, tableSize));
      }
    }
    // With 100 inputs and a 64k table, collisions should be rare
    expect(hashes.size).toBeGreaterThan(50);
  });
});

// ---------------------------------------------------------------------------
// Tests: trilinearInterpolate
// ---------------------------------------------------------------------------

describe('trilinearInterpolate', () => {
  it('returns exact corner values at integer coordinates', () => {
    const resolution = 4;
    const featureDim = 2;
    const grid = new Float64Array(resolution * resolution * resolution * featureDim);

    // Set a known value at corner (1, 1, 1)
    const idx = (1 + 1 * resolution + 1 * resolution * resolution) * featureDim;
    grid[idx] = 3.0;
    grid[idx + 1] = 7.0;

    const result = trilinearInterpolate(grid, 1, 1, 1, resolution, featureDim);
    expect(result[0]!).toBeCloseTo(3.0, 5);
    expect(result[1]!).toBeCloseTo(7.0, 5);
  });

  it('interpolates midpoint as average of 8 corners', () => {
    const resolution = 2;
    const featureDim = 1;
    // All 8 corners of a 2^3 grid filled with the same value
    const grid = new Float64Array(resolution * resolution * resolution * featureDim);
    grid.fill(4.0);

    const result = trilinearInterpolate(grid, 0.5, 0.5, 0.5, resolution, featureDim);
    expect(result[0]!).toBeCloseTo(4.0, 5);
  });

  it('result length matches featureDim', () => {
    const resolution = 3;
    const featureDim = 5;
    const grid = new Float64Array(resolution * resolution * resolution * featureDim);
    const result = trilinearInterpolate(grid, 1, 1, 1, resolution, featureDim);
    expect(result.length).toBe(featureDim);
  });
});

// ---------------------------------------------------------------------------
// Tests: createHashEncoding
// ---------------------------------------------------------------------------

describe('createHashEncoding', () => {
  it('creates an encoder that returns correct feature dimension', () => {
    const config: HashEncodingConfig = {
      nLevels: 4,
      nFeaturesPerLevel: 2,
      log2HashTableSize: 10,
      baseResolution: 4,
      perLevelScale: 2.0,
      boundsMin: { x: 0, y: 0, z: 0 },
      boundsMax: { x: 1, y: 1, z: 1 },
    };
    const encoder = createHashEncoding(config);

    const features = encoder.encode({ x: 0.5, y: 0.5, z: 0.5 });
    expect(features.length).toBe(config.nLevels * config.nFeaturesPerLevel);
  });

  it('produces deterministic output for the same point', () => {
    const config: HashEncodingConfig = {
      nLevels: 3,
      nFeaturesPerLevel: 2,
      log2HashTableSize: 8,
      baseResolution: 4,
      perLevelScale: 1.5,
      boundsMin: { x: -1, y: -1, z: -1 },
      boundsMax: { x: 1, y: 1, z: 1 },
    };
    const enc1 = createHashEncoding(config);
    const enc2 = createHashEncoding(config);

    const f1 = enc1.encode({ x: 0.25, y: -0.5, z: 0.1 });
    const f2 = enc2.encode({ x: 0.25, y: -0.5, z: 0.1 });

    for (let i = 0; i < f1.length; i++) {
      expect(f1[i]!).toBeCloseTo(f2[i]!, 10);
    }
  });

  it('different points produce different feature vectors', () => {
    const config: HashEncodingConfig = {
      nLevels: 4,
      nFeaturesPerLevel: 2,
      log2HashTableSize: 12,
      baseResolution: 8,
      perLevelScale: 2.0,
      boundsMin: { x: 0, y: 0, z: 0 },
      boundsMax: { x: 10, y: 10, z: 10 },
    };
    const encoder = createHashEncoding(config);

    const f1 = encoder.encode({ x: 1, y: 1, z: 1 });
    const f2 = encoder.encode({ x: 9, y: 9, z: 9 });

    let allSame = true;
    for (let i = 0; i < f1.length; i++) {
      if (Math.abs(f1[i]! - f2[i]!) > 1e-10) {
        allSame = false;
        break;
      }
    }
    expect(allSame).toBe(false);
  });
});
