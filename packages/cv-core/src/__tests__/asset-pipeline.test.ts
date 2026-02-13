import { describe, it, expect } from 'vitest';
import {
  validateTriangleCount,
  validateUVBounds,
  validateScale,
  validateManifold,
  createAssetMetadata,
  computePhysicalDimensions,
  computeMeshComplexity,
  createQualityGate,
  scoreMeshQuality,
  checkAllGates,
  planLODChain,
  estimateLODSizes,
  selectLODForBandwidth,
} from '../asset-pipeline/index.js';
import type { Mesh, QualityGateConfig, AssetValidationConfig } from '../types.js';

// ---------------------------------------------------------------------------
// Helper: build a simple unit-cube-like mesh for testing
// ---------------------------------------------------------------------------

/** Build a minimal valid mesh. A single triangle with known vertices. */
function makeTriangleMesh(): Mesh {
  // A single triangle in the XY plane: (0,0,0), (1,0,0), (0,1,0)
  const vertices = new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const indices = new Uint32Array([0, 1, 2]);
  return {
    vertices,
    indices,
    vertexCount: 3,
    triangleCount: 1,
  };
}

/**
 * Build a small manifold tetrahedron mesh (4 triangles sharing 4 vertices).
 * Every edge is shared by exactly 2 triangles.
 */
function makeTetrahedronMesh(): Mesh {
  const vertices = new Float64Array([
    0, 0, 0,      // v0
    1, 0, 0,      // v1
    0.5, 1, 0,    // v2
    0.5, 0.5, 1,  // v3
  ]);
  // 4 faces — each edge shared by exactly 2 faces
  const indices = new Uint32Array([
    0, 1, 2, // bottom
    0, 1, 3,
    1, 2, 3,
    0, 2, 3,
  ]);
  return {
    vertices,
    indices,
    vertexCount: 4,
    triangleCount: 4,
  };
}

/**
 * Build a mesh with a known bounding box: [0,2] x [0,3] x [0,4].
 * Uses 8 corner vertices and 2 triangles (just so triangleCount > 0).
 */
function makeKnownDimsMesh(): Mesh {
  const vertices = new Float64Array([
    0, 0, 0,
    2, 0, 0,
    2, 3, 0,
    0, 3, 0,
    0, 0, 4,
    2, 0, 4,
    2, 3, 4,
    0, 3, 4,
  ]);
  const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);
  return {
    vertices,
    indices,
    vertexCount: 8,
    triangleCount: 2,
  };
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

describe('validateTriangleCount', () => {
  it('should pass when triangle count is within the limit', () => {
    const mesh = makeTriangleMesh();
    const result = validateTriangleCount(mesh, 100);
    expect(result.pass).toBe(true);
    expect(result.actual).toBe(1);
    expect(result.max).toBe(100);
  });

  it('should fail when triangle count exceeds the limit', () => {
    const mesh = makeTetrahedronMesh();
    const result = validateTriangleCount(mesh, 2);
    expect(result.pass).toBe(false);
    expect(result.actual).toBe(4);
  });
});

describe('validateUVBounds', () => {
  it('should pass when all UVs are in [0, 1]', () => {
    const uvs = new Float64Array([0, 0, 0.5, 0.5, 1, 1]);
    const result = validateUVBounds(uvs);
    expect(result.pass).toBe(true);
    expect(result.outOfBounds).toBe(0);
  });

  it('should fail when UVs are out of bounds', () => {
    const uvs = new Float64Array([0, 0, 1.5, -0.2, 0.5, 0.5]);
    const result = validateUVBounds(uvs);
    expect(result.pass).toBe(false);
    expect(result.outOfBounds).toBe(2); // 1.5 and -0.2
  });
});

describe('validateScale', () => {
  it('should pass when the mesh fits within expected size', () => {
    // knownDims mesh spans 4 units in Z (largest extent)
    const mesh = makeKnownDimsMesh();
    const result = validateScale(mesh.vertices, { min: 1, max: 10 });
    expect(result.pass).toBe(true);
    expect(result.actualSize).toBeCloseTo(4, 10);
  });

  it('should fail when the mesh is too large', () => {
    const mesh = makeKnownDimsMesh();
    const result = validateScale(mesh.vertices, { min: 0, max: 3 });
    expect(result.pass).toBe(false);
    expect(result.actualSize).toBeCloseTo(4, 10);
  });

  it('should fail when the mesh is too small', () => {
    const mesh = makeKnownDimsMesh();
    const result = validateScale(mesh.vertices, { min: 5, max: 100 });
    expect(result.pass).toBe(false);
  });
});

describe('validateManifold', () => {
  it('should pass for a manifold tetrahedron', () => {
    const mesh = makeTetrahedronMesh();
    const result = validateManifold(mesh.vertices, mesh.indices);
    expect(result.pass).toBe(true);
    expect(result.openEdges).toBe(0);
  });

  it('should detect open edges in a single triangle', () => {
    const mesh = makeTriangleMesh();
    const result = validateManifold(mesh.vertices, mesh.indices);
    // A single triangle has 3 edges, each shared by only 1 face => 3 open edges
    expect(result.pass).toBe(false);
    expect(result.openEdges).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

describe('createAssetMetadata', () => {
  it('should produce metadata with correct triangle and vertex counts', () => {
    const mesh = makeTetrahedronMesh();
    const meta = createAssetMetadata(mesh);
    expect(meta.triangleCount).toBe(4);
    expect(meta.vertexCount).toBe(4);
    expect(meta.isManifold).toBe(true);
  });

  it('should report missing UVs and normals', () => {
    const mesh = makeTriangleMesh();
    const meta = createAssetMetadata(mesh);
    expect(meta.hasUVs).toBe(false);
    expect(meta.hasNormals).toBe(false);
  });
});

describe('computePhysicalDimensions', () => {
  it('should compute correct width, height, depth for known vertices', () => {
    const mesh = makeKnownDimsMesh();
    const dims = computePhysicalDimensions(mesh.vertices);
    expect(dims.width).toBeCloseTo(2, 10);
    expect(dims.height).toBeCloseTo(3, 10);
    expect(dims.depth).toBeCloseTo(4, 10);
  });

  it('should return zero dimensions for insufficient vertex data', () => {
    const dims = computePhysicalDimensions(new Float64Array([]));
    expect(dims.width).toBe(0);
    expect(dims.height).toBe(0);
    expect(dims.depth).toBe(0);
  });
});

describe('computeMeshComplexity', () => {
  it('should report increasing vertex count for larger meshes', () => {
    const small = makeTriangleMesh();
    const large = makeTetrahedronMesh();
    const cSmall = computeMeshComplexity(small);
    const cLarge = computeMeshComplexity(large);
    expect(cLarge.triangleCount).toBeGreaterThan(cSmall.triangleCount);
    expect(cLarge.vertexCount).toBeGreaterThan(cSmall.vertexCount);
  });

  it('should compute a positive average edge length for a valid mesh', () => {
    const mesh = makeTetrahedronMesh();
    const c = computeMeshComplexity(mesh);
    expect(c.avgEdgeLength).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// Quality gates
// ---------------------------------------------------------------------------

describe('createQualityGate', () => {
  const permissiveValidation: AssetValidationConfig = {
    maxTriangles: 100_000,
    maxTextureResolution: 4096,
    maxFileSize: 50 * 1024 * 1024,
    maxTextureFileSize: 10 * 1024 * 1024,
    requireUVs: false,
    requireNormals: false,
    checkDegenerateTriangles: false,
    checkManifold: false,
    minEdgeLength: 0,
    maxPhysicalDimension: 0,
    allowedFormats: [],
  };

  const strictValidation: AssetValidationConfig = {
    maxTriangles: 2,
    maxTextureResolution: 512,
    maxFileSize: 1024,
    maxTextureFileSize: 512,
    requireUVs: true,
    requireNormals: true,
    checkDegenerateTriangles: true,
    checkManifold: true,
    minEdgeLength: 0.001,
    maxPhysicalDimension: 1,
    allowedFormats: ['glb'],
  };

  it('should pass a simple mesh through a permissive gate', () => {
    const config: QualityGateConfig = {
      tiers: [{ name: 'standard', validation: permissiveValidation }],
      autoAssignTier: false,
      blockOnFailure: true,
    };
    const gate = createQualityGate(config);
    const result = gate(makeTriangleMesh());
    expect(result.pass).toBe(true);
    expect(result.violations.length).toBe(0);
  });

  it('should fail a simple mesh through a strict gate', () => {
    const config: QualityGateConfig = {
      tiers: [{ name: 'hero', validation: strictValidation }],
      autoAssignTier: false,
      blockOnFailure: true,
    };
    const gate = createQualityGate(config);
    const result = gate(makeTetrahedronMesh());
    expect(result.pass).toBe(false);
    expect(result.violations.length).toBeGreaterThan(0);
  });
});

describe('scoreMeshQuality', () => {
  it('should return a score between 0 and 100', () => {
    const mesh = makeTetrahedronMesh();
    const score = scoreMeshQuality(mesh);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(100);
  });

  it('should give a higher score to a manifold mesh with normals', () => {
    const plain = makeTetrahedronMesh();

    const withNormals: Mesh = {
      ...makeTetrahedronMesh(),
      normals: new Float64Array([
        0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,
      ]),
    };

    const scorePlain = scoreMeshQuality(plain);
    const scoreWithNormals = scoreMeshQuality(withNormals);
    expect(scoreWithNormals).toBeGreaterThan(scorePlain);
  });
});

describe('checkAllGates', () => {
  it('should aggregate violations from multiple gates', () => {
    const strictConfig: QualityGateConfig = {
      tiers: [{
        name: 'hero',
        validation: {
          maxTriangles: 1,
          maxTextureResolution: 512,
          maxFileSize: 50 * 1024 * 1024,
          maxTextureFileSize: 10 * 1024 * 1024,
          requireUVs: true,
          requireNormals: false,
          checkDegenerateTriangles: false,
          checkManifold: false,
          minEdgeLength: 0,
          maxPhysicalDimension: 0,
          allowedFormats: [],
        },
      }],
      autoAssignTier: false,
      blockOnFailure: true,
    };

    const result = checkAllGates(makeTetrahedronMesh(), [strictConfig]);
    expect(result.pass).toBe(false);
    expect(result.violationCount).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// Progressive LOD
// ---------------------------------------------------------------------------

describe('planLODChain', () => {
  it('should produce face counts that decrease with each level', () => {
    const baseMesh: Mesh = {
      vertices: new Float64Array(3000), // 1000 verts
      indices: new Uint32Array(3000),   // 1000 triangles
      vertexCount: 1000,
      triangleCount: 1000,
    };

    const configs = planLODChain(baseMesh, 4);
    expect(configs.length).toBe(4);

    // Extract target triangle counts per level
    const triCounts = configs.map((c) => c.levels[0]!.targetTriangles);

    // Each level should have fewer or equal triangles than the previous
    for (let i = 1; i < triCounts.length; i++) {
      expect(triCounts[i]!).toBeLessThanOrEqual(triCounts[i - 1]!);
    }

    // First level should match the base mesh
    expect(triCounts[0]!).toBe(1000);
  });
});

describe('estimateLODSizes', () => {
  it('should produce decreasing byte sizes for decreasing ratios', () => {
    const baseMesh: Mesh = {
      vertices: new Float64Array(30_000),
      indices: new Uint32Array(30_000),
      vertexCount: 10_000,
      triangleCount: 10_000,
    };

    const ratios = [1.0, 0.5, 0.25, 0.125];
    const { triangles, estimatedBytes } = estimateLODSizes(baseMesh, ratios);

    expect(triangles.length).toBe(4);
    expect(estimatedBytes.length).toBe(4);

    // Triangle counts should decrease
    for (let i = 1; i < triangles.length; i++) {
      expect(triangles[i]!).toBeLessThanOrEqual(triangles[i - 1]!);
    }

    // Byte sizes should decrease
    for (let i = 1; i < estimatedBytes.length; i++) {
      expect(estimatedBytes[i]!).toBeLessThanOrEqual(estimatedBytes[i - 1]!);
    }
  });
});

describe('selectLODForBandwidth', () => {
  it('should pick the finest LOD that fits the bandwidth budget', () => {
    // Sizes in descending order (finest first)
    const sizes = [10_000_000, 5_000_000, 1_000_000, 200_000];

    // Budget: 6 MB/s * 1 s = 6 MB
    const idx = selectLODForBandwidth(sizes, 6_000_000, 1);
    // The first LOD (10 MB) does not fit, second (5 MB) fits
    expect(idx).toBe(1);
  });

  it('should return the coarsest LOD when nothing fits', () => {
    const sizes = [10_000_000, 5_000_000, 1_000_000];
    // Very low bandwidth: 100 bytes/s * 1 s = 100 bytes
    const idx = selectLODForBandwidth(sizes, 100, 1);
    expect(idx).toBe(2); // last index
  });

  it('should return the finest LOD when budget is very large', () => {
    const sizes = [10_000_000, 5_000_000, 1_000_000];
    const idx = selectLODForBandwidth(sizes, 1_000_000_000, 10);
    expect(idx).toBe(0); // finest
  });
});
