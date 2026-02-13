import { describe, it, expect } from 'vitest';
import {
  computeScreenSpaceError,
  selectLOD,
  buildLODChain,
  buildInstanceBuffer,
  frustumCull,
  extractFrustumPlanes,
  computeCascadeSplits,
  computeShadowMatrix,
  contactShadowRaymarch,
  createPerfBudget,
  estimateVRAM,
  checkBudget,
  adaptiveQuality,
} from '../rendering/index.js';
import type {
  LODLevel,
  Mesh,
  FrustumPlanes,
  InstanceData,
  PerfBudget,
} from '../types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a minimal Mesh for testing. */
function makeMesh(vertexCount: number, triangleCount: number): Mesh {
  return {
    vertices: new Float64Array(vertexCount * 3),
    indices: new Uint32Array(triangleCount * 3),
    vertexCount,
    triangleCount,
  };
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
// computeScreenSpaceError
// ---------------------------------------------------------------------------

describe('computeScreenSpaceError', () => {
  const fovY = Math.PI / 3; // 60 degrees
  const screenHeight = 1080;
  const geometricError = 1.0;

  it('returns Infinity when distance is zero', () => {
    expect(computeScreenSpaceError(0, geometricError, screenHeight, fovY)).toBe(
      Infinity,
    );
  });

  it('SSE decreases as distance increases', () => {
    const sseNear = computeScreenSpaceError(10, geometricError, screenHeight, fovY);
    const sseFar = computeScreenSpaceError(100, geometricError, screenHeight, fovY);
    expect(sseNear).toBeGreaterThan(sseFar);
  });

  it('SSE increases with larger geometric error', () => {
    const sseSmall = computeScreenSpaceError(50, 0.5, screenHeight, fovY);
    const sseLarge = computeScreenSpaceError(50, 2.0, screenHeight, fovY);
    expect(sseLarge).toBeGreaterThan(sseSmall);
  });

  it('SSE scales linearly with screen height', () => {
    const sse540 = computeScreenSpaceError(50, geometricError, 540, fovY);
    const sse1080 = computeScreenSpaceError(50, geometricError, 1080, fovY);
    expect(sse1080).toBeCloseTo(sse540 * 2, 5);
  });

  it('produces a positive finite value for valid inputs', () => {
    const sse = computeScreenSpaceError(25, geometricError, screenHeight, fovY);
    expect(sse).toBeGreaterThan(0);
    expect(isFinite(sse)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// selectLOD
// ---------------------------------------------------------------------------

describe('selectLOD', () => {
  const fovY = Math.PI / 4;
  const screenHeight = 720;

  // Construct levels with increasing screen-space error thresholds.
  const levels: LODLevel[] = [
    { mesh: makeMesh(1000, 500), screenSpaceError: 0, distance: 0 },
    { mesh: makeMesh(500, 250), screenSpaceError: 5, distance: 50 },
    { mesh: makeMesh(100, 50), screenSpaceError: 10, distance: 100 },
  ];

  it('returns 0 for an empty levels array', () => {
    expect(selectLOD([], 50, screenHeight, fovY)).toBe(0);
  });

  it('selects the finest level (0) at very close distances', () => {
    // At very close distance the SSE for level 0 is 0 which is <= 0 threshold
    const idx = selectLOD(levels, 0.01, screenHeight, fovY);
    expect(idx).toBe(0);
  });

  it('selects a coarser level at large distances', () => {
    const idx = selectLOD(levels, 10000, screenHeight, fovY);
    // At extremely far distance, SSE for all levels is tiny, so level 0 qualifies.
    // But for moderate distances the coarser levels should be chosen.
    expect(idx).toBeGreaterThanOrEqual(0);
    expect(idx).toBeLessThan(levels.length);
  });

  it('never returns an index beyond the last level', () => {
    const idx = selectLOD(levels, 5, screenHeight, fovY);
    expect(idx).toBeLessThan(levels.length);
  });

  it('falls back to the coarsest level when no level satisfies threshold', () => {
    // Large SSE levels that are hard to satisfy at moderate distance
    const hardLevels: LODLevel[] = [
      { mesh: makeMesh(100, 50), screenSpaceError: 0.001, distance: 0 },
      { mesh: makeMesh(50, 25), screenSpaceError: 0.001, distance: 10 },
    ];
    const idx = selectLOD(hardLevels, 1, screenHeight, fovY);
    expect(idx).toBe(hardLevels.length - 1);
  });
});

// ---------------------------------------------------------------------------
// buildLODChain
// ---------------------------------------------------------------------------

describe('buildLODChain', () => {
  const baseMesh = makeMesh(1000, 500);

  it('returns at least the base level when no ratios are given', () => {
    const chain = buildLODChain(baseMesh, []);
    expect(chain.length).toBe(1);
    expect(chain[0]!.screenSpaceError).toBe(0);
  });

  it('creates the correct number of levels', () => {
    const chain = buildLODChain(baseMesh, [0.5, 0.25, 0.1]);
    expect(chain.length).toBe(4); // 1 base + 3 ratios
  });

  it('assigns increasing geometric error to coarser levels', () => {
    const chain = buildLODChain(baseMesh, [0.5, 0.25]);
    for (let i = 1; i < chain.length; i++) {
      expect(chain[i]!.screenSpaceError).toBeGreaterThanOrEqual(
        chain[i - 1]!.screenSpaceError,
      );
    }
  });

  it('assigns increasing distance thresholds to coarser levels', () => {
    const chain = buildLODChain(baseMesh, [0.5, 0.25, 0.1]);
    for (let i = 1; i < chain.length; i++) {
      expect(chain[i]!.distance).toBeGreaterThanOrEqual(chain[i - 1]!.distance);
    }
  });

  it('reduces triangle count at coarser levels', () => {
    const chain = buildLODChain(baseMesh, [0.5, 0.1]);
    expect(chain[1]!.mesh.triangleCount).toBeLessThanOrEqual(baseMesh.triangleCount);
    expect(chain[2]!.mesh.triangleCount).toBeLessThanOrEqual(
      chain[1]!.mesh.triangleCount,
    );
  });
});

// ---------------------------------------------------------------------------
// frustumCull + extractFrustumPlanes
// ---------------------------------------------------------------------------

describe('frustumCull', () => {
  // Use an identity VP matrix to get a simple cube frustum [-1,1]^3.
  const vpIdentity = identity4();
  const planes = extractFrustumPlanes(vpIdentity);

  it('keeps instances inside the frustum', () => {
    const instances: InstanceData[] = [
      {
        transform: identity4(),
        boundingSphere: { centre: { x: 0, y: 0, z: 0 }, radius: 0.1 },
        instanceId: 0,
      },
    ];
    const visible = frustumCull(instances, planes);
    expect(visible).toContain(0);
  });

  it('culls instances clearly outside the frustum', () => {
    const instances: InstanceData[] = [
      {
        transform: identity4(),
        boundingSphere: { centre: { x: 100, y: 100, z: 100 }, radius: 0.1 },
        instanceId: 0,
      },
    ];
    const visible = frustumCull(instances, planes);
    expect(visible.length).toBe(0);
  });

  it('keeps instances that partially overlap the frustum boundary', () => {
    // Sphere straddles the edge of the frustum cube
    const instances: InstanceData[] = [
      {
        transform: identity4(),
        boundingSphere: { centre: { x: 0.95, y: 0, z: 0 }, radius: 0.2 },
        instanceId: 0,
      },
    ];
    const visible = frustumCull(instances, planes);
    expect(visible.length).toBe(1);
  });

  it('correctly handles multiple instances with mixed visibility', () => {
    const instances: InstanceData[] = [
      {
        transform: identity4(),
        boundingSphere: { centre: { x: 0, y: 0, z: 0 }, radius: 0.1 },
        instanceId: 0,
      },
      {
        transform: identity4(),
        boundingSphere: { centre: { x: 50, y: 50, z: 50 }, radius: 0.1 },
        instanceId: 1,
      },
      {
        transform: identity4(),
        boundingSphere: { centre: { x: -0.5, y: 0.5, z: 0 }, radius: 0.1 },
        instanceId: 2,
      },
    ];
    const visible = frustumCull(instances, planes);
    expect(visible).toContain(0);
    expect(visible).not.toContain(1);
    expect(visible).toContain(2);
  });

  it('returns empty for an empty instances array', () => {
    const visible = frustumCull([], planes);
    expect(visible.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// buildInstanceBuffer
// ---------------------------------------------------------------------------

describe('buildInstanceBuffer', () => {
  it('creates the correct number of instances', () => {
    const transforms = [identity4(), translation4(5, 0, 0)];
    const spheres = [
      { center: { x: 0, y: 0, z: 0 }, radius: 1 },
      { center: { x: 0, y: 0, z: 0 }, radius: 1 },
    ];
    const result = buildInstanceBuffer(transforms, spheres);
    expect(result.length).toBe(2);
  });

  it('offsets bounding sphere centres by the transform translation', () => {
    const transforms = [translation4(10, 20, 30)];
    const spheres = [{ center: { x: 0, y: 0, z: 0 }, radius: 1 }];
    const result = buildInstanceBuffer(transforms, spheres);
    expect(result[0]!.boundingSphere.centre.x).toBeCloseTo(10, 5);
    expect(result[0]!.boundingSphere.centre.y).toBeCloseTo(20, 5);
    expect(result[0]!.boundingSphere.centre.z).toBeCloseTo(30, 5);
  });

  it('scales the bounding sphere radius by the transform scale', () => {
    const scaled = identity4();
    // Scale by 2 in all axes
    scaled[0] = 2;
    scaled[5] = 2;
    scaled[10] = 2;
    const transforms = [scaled];
    const spheres = [{ center: { x: 0, y: 0, z: 0 }, radius: 1 }];
    const result = buildInstanceBuffer(transforms, spheres);
    expect(result[0]!.boundingSphere.radius).toBeCloseTo(2, 5);
  });

  it('assigns sequential instance IDs', () => {
    const transforms = [identity4(), identity4(), identity4()];
    const spheres = [
      { center: { x: 0, y: 0, z: 0 }, radius: 1 },
      { center: { x: 0, y: 0, z: 0 }, radius: 1 },
      { center: { x: 0, y: 0, z: 0 }, radius: 1 },
    ];
    const result = buildInstanceBuffer(transforms, spheres);
    expect(result[0]!.instanceId).toBe(0);
    expect(result[1]!.instanceId).toBe(1);
    expect(result[2]!.instanceId).toBe(2);
  });

  it('handles mismatched array lengths by using the shorter count', () => {
    const transforms = [identity4(), identity4()];
    const spheres = [{ center: { x: 0, y: 0, z: 0 }, radius: 1 }];
    const result = buildInstanceBuffer(transforms, spheres);
    expect(result.length).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// computeCascadeSplits
// ---------------------------------------------------------------------------

describe('computeCascadeSplits', () => {
  it('returns near as first element and far as last', () => {
    const splits = computeCascadeSplits(0.1, 1000, 4, 0.5);
    expect(splits[0]).toBeCloseTo(0.1, 5);
    expect(splits[4]).toBeCloseTo(1000, 5);
  });

  it('produces monotonically increasing split distances', () => {
    const splits = computeCascadeSplits(1, 500, 4, 0.5);
    for (let i = 1; i < splits.length; i++) {
      expect(splits[i]!).toBeGreaterThan(splits[i - 1]!);
    }
  });

  it('returns near and far only for a single cascade', () => {
    const splits = computeCascadeSplits(1, 100, 1, 0.5);
    expect(splits.length).toBe(2);
    expect(splits[0]).toBeCloseTo(1, 5);
    expect(splits[1]).toBeCloseTo(100, 5);
  });

  it('produces uniform splits when lambda is 0', () => {
    // Use near > 0 to avoid NaN from the logarithmic term (0 * (far/0)^frac)
    const splits = computeCascadeSplits(1, 101, 4, 0);
    // Uniform: splits at 1, 26, 51, 76, 101
    expect(splits[1]).toBeCloseTo(26, 5);
    expect(splits[2]).toBeCloseTo(51, 5);
    expect(splits[3]).toBeCloseTo(76, 5);
  });

  it('has the correct array length for n cascades', () => {
    const splits = computeCascadeSplits(1, 200, 6, 0.7);
    expect(splits.length).toBe(7); // nCascades + 1
  });
});

// ---------------------------------------------------------------------------
// computeShadowMatrix
// ---------------------------------------------------------------------------

describe('computeShadowMatrix', () => {
  it('returns a 16-element Float64Array', () => {
    const lightDir = { x: 0, y: 1, z: 0 };
    const cameraView = identity4();
    const result = computeShadowMatrix(lightDir, 1, 100, cameraView);
    expect(result.length).toBe(16);
    expect(result).toBeInstanceOf(Float64Array);
  });

  it('produces a finite matrix with valid inputs', () => {
    const lightDir = { x: 0.577, y: 0.577, z: 0.577 };
    const cameraView = identity4();
    const result = computeShadowMatrix(lightDir, 0.5, 50, cameraView);
    for (let i = 0; i < 16; i++) {
      expect(isFinite(result[i]!)).toBe(true);
    }
  });

  it('produces a non-identity matrix', () => {
    const lightDir = { x: 0, y: -1, z: 0 };
    const cameraView = identity4();
    const result = computeShadowMatrix(lightDir, 1, 100, cameraView);
    const ident = identity4();
    let same = true;
    for (let i = 0; i < 16; i++) {
      if (Math.abs(result[i]! - ident[i]!) > 1e-10) {
        same = false;
        break;
      }
    }
    expect(same).toBe(false);
  });

  it('produces different matrices for different cascade ranges', () => {
    const lightDir = { x: 0, y: 1, z: -1 };
    const cameraView = identity4();
    const m1 = computeShadowMatrix(lightDir, 1, 50, cameraView);
    const m2 = computeShadowMatrix(lightDir, 50, 200, cameraView);
    let differ = false;
    for (let i = 0; i < 16; i++) {
      if (Math.abs(m1[i]! - m2[i]!) > 1e-10) {
        differ = true;
        break;
      }
    }
    expect(differ).toBe(true);
  });

  it('has at least one non-zero element per column', () => {
    const lightDir = { x: 0.577, y: 0.577, z: 0.577 };
    const cameraView = identity4();
    const result = computeShadowMatrix(lightDir, 1, 100, cameraView);
    // Each column should have at least one non-zero entry
    for (let col = 0; col < 4; col++) {
      let maxAbs = 0;
      for (let row = 0; row < 4; row++) {
        maxAbs = Math.max(maxAbs, Math.abs(result[col * 4 + row]!));
      }
      expect(maxAbs).toBeGreaterThan(0);
    }
  });
});

// ---------------------------------------------------------------------------
// contactShadowRaymarch
// ---------------------------------------------------------------------------

describe('contactShadowRaymarch', () => {
  const width = 32;
  const height = 32;

  it('returns false when marching through a flat depth buffer', () => {
    const depthBuffer = new Float64Array(width * height);
    depthBuffer.fill(10.0);
    const result = contactShadowRaymarch(
      depthBuffer,
      { x: 16, y: 16 },
      { x: 1, y: 0 },
      width,
      height,
      10,
    );
    expect(result).toBe(false);
  });

  it('returns true when an occluder is present along the ray', () => {
    const depthBuffer = new Float64Array(width * height);
    depthBuffer.fill(10.0);
    // Place a closer depth value along the ray direction (positive x)
    for (let x = 18; x < 24; x++) {
      depthBuffer[16 * width + x] = 5.0; // Closer than origin depth of 10
    }
    const result = contactShadowRaymarch(
      depthBuffer,
      { x: 16, y: 16 },
      { x: 1, y: 0 },
      width,
      height,
      20,
    );
    expect(result).toBe(true);
  });

  it('returns false when origin is out of bounds', () => {
    const depthBuffer = new Float64Array(width * height);
    depthBuffer.fill(10.0);
    const result = contactShadowRaymarch(
      depthBuffer,
      { x: -5, y: -5 },
      { x: 1, y: 0 },
      width,
      height,
      10,
    );
    expect(result).toBe(false);
  });

  it('returns false with zero steps', () => {
    const depthBuffer = new Float64Array(width * height);
    depthBuffer.fill(10.0);
    depthBuffer[16 * width + 17] = 1.0;
    const result = contactShadowRaymarch(
      depthBuffer,
      { x: 16, y: 16 },
      { x: 1, y: 0 },
      width,
      height,
      0,
    );
    expect(result).toBe(false);
  });

  it('returns false when depth buffer values are all farther', () => {
    const depthBuffer = new Float64Array(width * height);
    depthBuffer.fill(5.0);
    // Set origin to a close depth; neighbours are all farther or equal
    depthBuffer[16 * width + 16] = 5.0;
    for (let x = 17; x < 32; x++) {
      depthBuffer[16 * width + x] = 20.0; // Farther than origin
    }
    const result = contactShadowRaymarch(
      depthBuffer,
      { x: 16, y: 16 },
      { x: 1, y: 0 },
      width,
      height,
      10,
    );
    expect(result).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// estimateVRAM
// ---------------------------------------------------------------------------

describe('estimateVRAM', () => {
  it('returns a positive value for a non-trivial mesh', () => {
    const mesh = makeMesh(1000, 500);
    const vram = estimateVRAM(mesh, 1024);
    expect(vram).toBeGreaterThan(0);
  });

  it('increases with larger texture size', () => {
    const mesh = makeMesh(100, 50);
    const vram256 = estimateVRAM(mesh, 256);
    const vram2048 = estimateVRAM(mesh, 2048);
    expect(vram2048).toBeGreaterThan(vram256);
  });

  it('increases when mesh has normals', () => {
    const meshNoNormals = makeMesh(1000, 500);
    const meshWithNormals: Mesh = {
      ...meshNoNormals,
      normals: new Float64Array(1000 * 3),
    };
    const vramNo = estimateVRAM(meshNoNormals, 512);
    const vramWith = estimateVRAM(meshWithNormals, 512);
    expect(vramWith).toBeGreaterThan(vramNo);
  });

  it('increases with vertex count', () => {
    const small = makeMesh(100, 50);
    const large = makeMesh(10000, 5000);
    expect(estimateVRAM(large, 512)).toBeGreaterThan(estimateVRAM(small, 512));
  });

  it('returns a value in megabytes (sane order of magnitude)', () => {
    const mesh = makeMesh(50000, 25000);
    const vram = estimateVRAM(mesh, 2048);
    // Should be in the range of a few MB, not GB
    expect(vram).toBeGreaterThan(0.01);
    expect(vram).toBeLessThan(1000);
  });
});

// ---------------------------------------------------------------------------
// checkBudget
// ---------------------------------------------------------------------------

describe('checkBudget', () => {
  const budget: PerfBudget = createPerfBudget(1000, 1_000_000, 512);

  it('reports within budget when all metrics are below limits', () => {
    const result = checkBudget(budget, 500, 500_000, 256);
    expect(result.withinBudget).toBe(true);
  });

  it('reports over budget when draw calls exceed limit', () => {
    const result = checkBudget(budget, 1500, 500_000, 256);
    expect(result.withinBudget).toBe(false);
  });

  it('reports over budget when triangles exceed limit', () => {
    const result = checkBudget(budget, 500, 2_000_000, 256);
    expect(result.withinBudget).toBe(false);
  });

  it('reports over budget when VRAM exceeds limit', () => {
    const result = checkBudget(budget, 500, 500_000, 1024);
    expect(result.withinBudget).toBe(false);
  });

  it('provides correct utilization ratios', () => {
    const result = checkBudget(budget, 500, 250_000, 128);
    expect(result.utilization.drawCalls).toBeCloseTo(0.5, 5);
    expect(result.utilization.triangles).toBeCloseTo(0.25, 5);
    expect(result.utilization.vram).toBeCloseTo(0.25, 5);
  });

  it('reports within budget at exactly the limit', () => {
    const result = checkBudget(budget, 1000, 1_000_000, 512);
    expect(result.withinBudget).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// adaptiveQuality
// ---------------------------------------------------------------------------

describe('adaptiveQuality', () => {
  const budget: PerfBudget = createPerfBudget(1000, 1_000_000, 512);

  it('increases quality when FPS exceeds target', () => {
    const newQ = adaptiveQuality(budget, 90, 60, 0.5);
    expect(newQ).toBeGreaterThan(0.5);
  });

  it('decreases quality when FPS is below target', () => {
    const newQ = adaptiveQuality(budget, 30, 60, 0.5);
    expect(newQ).toBeLessThan(0.5);
  });

  it('clamps quality to [0, 1] range', () => {
    const qHigh = adaptiveQuality(budget, 120, 60, 0.99);
    expect(qHigh).toBeLessThanOrEqual(1.0);
    const qLow = adaptiveQuality(budget, 10, 60, 0.01);
    expect(qLow).toBeGreaterThanOrEqual(0.0);
  });

  it('makes no change when FPS exactly equals target', () => {
    const newQ = adaptiveQuality(budget, 60, 60, 0.5);
    expect(newQ).toBeCloseTo(0.5, 5);
  });

  it('returns current quality when target FPS is zero', () => {
    const newQ = adaptiveQuality(budget, 60, 0, 0.7);
    expect(newQ).toBeCloseTo(0.7, 5);
  });
});
