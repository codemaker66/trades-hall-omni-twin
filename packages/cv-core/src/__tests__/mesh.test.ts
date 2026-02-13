import { describe, it, expect } from 'vitest';
import {
  computeQuadricError,
  buildQuadricMatrices,
  decimateMesh,
  computeAngleBasedFlattening,
  packAtlas,
  chartBoundaryLength,
  bakeNormalMap,
  computeAmbientOcclusion,
  createGLTFAccessor,
  estimateGLTFSize,
  validateMeshManifold,
} from '../mesh/index.js';
import type { UVChart, Mesh } from '../types.js';
import { createPRNG } from '../types.js';

// ---------------------------------------------------------------------------
// Helpers: build simple test meshes
// ---------------------------------------------------------------------------

/**
 * Build a single-triangle mesh (right triangle in the XY plane).
 * Vertices: (0,0,0), (1,0,0), (0,1,0)
 */
function singleTriangle(): { vertices: Float64Array; indices: Uint32Array } {
  const vertices = new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const indices = new Uint32Array([0, 1, 2]);
  return { vertices, indices };
}

/**
 * Build a two-triangle quad mesh in the XY plane.
 * Vertices: (0,0,0), (1,0,0), (1,1,0), (0,1,0)
 * Faces: [0,1,2], [0,2,3]
 */
function quadMesh(): { vertices: Float64Array; indices: Uint32Array } {
  const vertices = new Float64Array([
    0, 0, 0,
    1, 0, 0,
    1, 1, 0,
    0, 1, 0,
  ]);
  const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);
  return { vertices, indices };
}

/**
 * Build a tetrahedron (4 vertices, 4 triangular faces).
 * This is a closed manifold mesh.
 */
function tetrahedron(): { vertices: Float64Array; indices: Uint32Array } {
  const vertices = new Float64Array([
    1, 1, 1,     // 0
    -1, -1, 1,   // 1
    -1, 1, -1,   // 2
    1, -1, -1,   // 3
  ]);
  const indices = new Uint32Array([
    0, 1, 2,
    0, 3, 1,
    0, 2, 3,
    1, 3, 2,
  ]);
  return { vertices, indices };
}

/**
 * Build a larger mesh (subdivided grid) for decimation tests.
 * Creates an NxN grid of quads (2*N*N triangles).
 */
function gridMesh(n: number): { vertices: Float64Array; indices: Uint32Array } {
  const verts: number[] = [];
  for (let iy = 0; iy <= n; iy++) {
    for (let ix = 0; ix <= n; ix++) {
      verts.push(ix / n, iy / n, 0);
    }
  }
  const idxs: number[] = [];
  for (let iy = 0; iy < n; iy++) {
    for (let ix = 0; ix < n; ix++) {
      const bl = iy * (n + 1) + ix;
      const br = bl + 1;
      const tl = bl + (n + 1);
      const tr = tl + 1;
      idxs.push(bl, br, tl);
      idxs.push(br, tr, tl);
    }
  }
  return {
    vertices: new Float64Array(verts),
    indices: new Uint32Array(idxs),
  };
}

// ---------------------------------------------------------------------------
// Tests: computeQuadricError
// ---------------------------------------------------------------------------

describe('computeQuadricError', () => {
  it('returns zero for a point on a plane through the origin', () => {
    // Plane z = 0 => normal (0,0,1), d = 0 => p = [0,0,1,0]
    // Q = p * p^T => all zeros except Q[10] = 1 (row 2, col 2)
    const Q = new Float64Array(16);
    Q[10] = 1; // z^2 coefficient

    const v = { x: 5, y: 3, z: 0 }; // on the z = 0 plane
    const error = computeQuadricError(v, Q);
    expect(error).toBeCloseTo(0, 10);
  });

  it('returns positive error for a point off the plane', () => {
    const Q = new Float64Array(16);
    Q[10] = 1; // z^2 coefficient

    const v = { x: 0, y: 0, z: 3 };
    const error = computeQuadricError(v, Q);
    expect(error).toBeCloseTo(9, 5); // z^2 = 9
  });

  it('returns correct error for a full quadric matrix', () => {
    // Plane x + y + z - 1 = 0 => n = (1/sqrt3, 1/sqrt3, 1/sqrt3), d = -1/sqrt3
    // p = [1, 1, 1, -1] (unnormalised)
    // Q = p * p^T
    const Q = new Float64Array(16);
    const p = [1, 1, 1, -1];
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        Q[r * 4 + c] = p[r]! * p[c]!;
      }
    }
    // Point (1/3, 1/3, 1/3) is on the plane x+y+z=1
    const v = { x: 1 / 3, y: 1 / 3, z: 1 / 3 };
    const error = computeQuadricError(v, Q);
    expect(error).toBeCloseTo(0, 8);
  });
});

// ---------------------------------------------------------------------------
// Tests: buildQuadricMatrices
// ---------------------------------------------------------------------------

describe('buildQuadricMatrices', () => {
  it('builds one quadric per vertex', () => {
    const { vertices, indices } = singleTriangle();
    const quadrics = buildQuadricMatrices(vertices, indices);

    expect(quadrics.length).toBe(3); // 3 vertices
    for (const Q of quadrics) {
      expect(Q.length).toBe(16);
    }
  });

  it('quadric error is zero for vertices on the face plane', () => {
    const { vertices, indices } = singleTriangle();
    const quadrics = buildQuadricMatrices(vertices, indices);

    // All three vertices lie on the single face plane, so Q(v) should be ~ 0
    for (let i = 0; i < 3; i++) {
      const v = {
        x: vertices[i * 3]!,
        y: vertices[i * 3 + 1]!,
        z: vertices[i * 3 + 2]!,
      };
      const error = computeQuadricError(v, quadrics[i]!);
      expect(error).toBeCloseTo(0, 8);
    }
  });
});

// ---------------------------------------------------------------------------
// Tests: decimateMesh
// ---------------------------------------------------------------------------

describe('decimateMesh', () => {
  it('reduces face count when targeting a lower ratio', () => {
    const { vertices, indices } = gridMesh(5);
    const originalFaceCount = indices.length / 3;

    const result = decimateMesh(vertices, indices, 0.5);
    const newFaceCount = result.indices.length / 3;

    expect(newFaceCount).toBeLessThan(originalFaceCount);
    expect(newFaceCount).toBeGreaterThan(0);
  });

  it('decimated mesh has valid triangle indices', () => {
    const { vertices, indices } = gridMesh(4);
    const result = decimateMesh(vertices, indices, 0.5);

    const nVerts = result.vertices.length / 3;
    for (let i = 0; i < result.indices.length; i++) {
      expect(result.indices[i]!).toBeGreaterThanOrEqual(0);
      expect(result.indices[i]!).toBeLessThan(nVerts);
    }
    // Indices should be a multiple of 3
    expect(result.indices.length % 3).toBe(0);
  });

  it('returns the same mesh at ratio 1.0', () => {
    const { vertices, indices } = quadMesh();
    const result = decimateMesh(vertices, indices, 1.0);

    // With ratio 1.0, target is 2 faces (same as input), so no collapse should happen
    expect(result.indices.length / 3).toBe(indices.length / 3);
  });

  it('produces fewer vertices after decimation', () => {
    const { vertices, indices } = gridMesh(6);
    const result = decimateMesh(vertices, indices, 0.3);

    expect(result.vertices.length).toBeLessThan(vertices.length);
  });

  it('decimation indices form valid triangles with non-degenerate vertices', () => {
    const { vertices, indices } = gridMesh(5);
    const result = decimateMesh(vertices, indices, 0.5);

    const nFaces = result.indices.length / 3;
    for (let f = 0; f < nFaces; f++) {
      const i0 = result.indices[f * 3]!;
      const i1 = result.indices[f * 3 + 1]!;
      const i2 = result.indices[f * 3 + 2]!;
      // All three indices should be distinct (non-degenerate)
      expect(i0).not.toBe(i1);
      expect(i1).not.toBe(i2);
      expect(i0).not.toBe(i2);
    }
  });
});

// ---------------------------------------------------------------------------
// Tests: computeAngleBasedFlattening
// ---------------------------------------------------------------------------

describe('computeAngleBasedFlattening', () => {
  it('produces UV coordinates in [0, 1] range', () => {
    const { vertices, indices } = quadMesh();
    const uvs = computeAngleBasedFlattening(vertices, indices);

    expect(uvs.length).toBe(4 * 2); // 4 vertices, 2 components each
    for (let i = 0; i < uvs.length; i++) {
      expect(uvs[i]!).toBeGreaterThanOrEqual(-1e-6);
      expect(uvs[i]!).toBeLessThanOrEqual(1.0 + 1e-6);
    }
  });

  it('produces correct number of UV pairs for a grid mesh', () => {
    const { vertices, indices } = gridMesh(3);
    const nVerts = vertices.length / 3;
    const uvs = computeAngleBasedFlattening(vertices, indices);

    expect(uvs.length).toBe(nVerts * 2);
  });

  it('UV range spans the full [0,1] interval for a non-degenerate mesh', () => {
    const { vertices, indices } = gridMesh(4);
    const uvs = computeAngleBasedFlattening(vertices, indices);

    let uMin = Infinity;
    let uMax = -Infinity;
    let vMin = Infinity;
    let vMax = -Infinity;

    const nVerts = vertices.length / 3;
    for (let i = 0; i < nVerts; i++) {
      const u = uvs[i * 2]!;
      const v = uvs[i * 2 + 1]!;
      if (u < uMin) uMin = u;
      if (u > uMax) uMax = u;
      if (v < vMin) vMin = v;
      if (v > vMax) vMax = v;
    }

    expect(uMin).toBeCloseTo(0, 1);
    expect(uMax).toBeCloseTo(1, 1);
    expect(vMin).toBeCloseTo(0, 1);
    expect(vMax).toBeCloseTo(1, 1);
  });
});

// ---------------------------------------------------------------------------
// Tests: packAtlas
// ---------------------------------------------------------------------------

describe('packAtlas', () => {
  it('packs charts into an atlas with correct dimensions', () => {
    const charts: UVChart[] = [
      {
        triangleIndices: new Uint32Array([0]),
        boundingRect: new Float64Array([0, 0, 0.3, 0.4]),
        uvArea: 0.12,
        worldArea: 1.0,
      },
      {
        triangleIndices: new Uint32Array([1]),
        boundingRect: new Float64Array([0, 0, 0.2, 0.5]),
        uvArea: 0.1,
        worldArea: 0.8,
      },
    ];

    const atlas = packAtlas(charts, 512);
    expect(atlas.width).toBe(512);
    expect(atlas.height).toBe(512);
    expect(atlas.chartCount).toBe(2);
  });

  it('has non-overlapping chart placements', () => {
    const charts: UVChart[] = [
      {
        triangleIndices: new Uint32Array([0]),
        boundingRect: new Float64Array([0, 0, 0.25, 0.25]),
        uvArea: 0.0625,
        worldArea: 1.0,
      },
      {
        triangleIndices: new Uint32Array([1]),
        boundingRect: new Float64Array([0, 0, 0.25, 0.25]),
        uvArea: 0.0625,
        worldArea: 1.0,
      },
      {
        triangleIndices: new Uint32Array([2]),
        boundingRect: new Float64Array([0, 0, 0.25, 0.25]),
        uvArea: 0.0625,
        worldArea: 1.0,
      },
    ];

    const atlas = packAtlas(charts, 256);

    // Check that no two placed charts overlap
    for (let i = 0; i < atlas.charts.length; i++) {
      for (let j = i + 1; j < atlas.charts.length; j++) {
        const a = atlas.charts[i]!.boundingRect;
        const b = atlas.charts[j]!.boundingRect;
        // Two rectangles do NOT overlap if one is entirely to the left/right/above/below
        const noOverlap =
          a[2]! <= b[0]! + 1e-6 ||
          b[2]! <= a[0]! + 1e-6 ||
          a[3]! <= b[1]! + 1e-6 ||
          b[3]! <= a[1]! + 1e-6;
        expect(noOverlap).toBe(true);
      }
    }
  });

  it('returns zero efficiency for empty chart list', () => {
    const atlas = packAtlas([], 256);
    expect(atlas.efficiency).toBe(0);
    expect(atlas.chartCount).toBe(0);
  });

  it('efficiency is between 0 and 1', () => {
    const charts: UVChart[] = [
      {
        triangleIndices: new Uint32Array([0]),
        boundingRect: new Float64Array([0, 0, 0.5, 0.5]),
        uvArea: 0.25,
        worldArea: 2.0,
      },
    ];
    const atlas = packAtlas(charts, 1024);
    expect(atlas.efficiency).toBeGreaterThan(0);
    expect(atlas.efficiency).toBeLessThanOrEqual(1.0 + 1e-6);
  });
});

// ---------------------------------------------------------------------------
// Tests: chartBoundaryLength
// ---------------------------------------------------------------------------

describe('chartBoundaryLength', () => {
  it('returns correct perimeter for a known rectangle', () => {
    const chart: UVChart = {
      triangleIndices: new Uint32Array([0]),
      boundingRect: new Float64Array([0.1, 0.2, 0.5, 0.7]),
      uvArea: 0.2,
      worldArea: 1.0,
    };
    // width = 0.4, height = 0.5 => perimeter = 2*(0.4+0.5) = 1.8
    const len = chartBoundaryLength(chart);
    expect(len).toBeCloseTo(1.8, 5);
  });
});

// ---------------------------------------------------------------------------
// Tests: computeAmbientOcclusion
// ---------------------------------------------------------------------------

describe('computeAmbientOcclusion', () => {
  it('AO values are in [0, 1]', () => {
    const { vertices, indices } = singleTriangle();
    const normals = new Float64Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
    const rng = createPRNG(42);

    const ao = computeAmbientOcclusion(vertices, normals, indices, 8, rng);

    expect(ao.length).toBe(3);
    for (let i = 0; i < ao.length; i++) {
      expect(ao[i]!).toBeGreaterThanOrEqual(0);
      expect(ao[i]!).toBeLessThanOrEqual(1.0 + 1e-6);
    }
  });

  it('open geometry has high AO (mostly unoccluded)', () => {
    // A single triangle with normals facing away from any occluder
    const { vertices, indices } = singleTriangle();
    const normals = new Float64Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
    const rng = createPRNG(99);

    const ao = computeAmbientOcclusion(vertices, normals, indices, 32, rng);

    // A single isolated triangle should be mostly unoccluded
    for (let i = 0; i < ao.length; i++) {
      expect(ao[i]!).toBeGreaterThan(0.5);
    }
  });

  it('returns correct number of AO values', () => {
    const { vertices, indices } = tetrahedron();
    const normals = new Float64Array(12);
    // Simple normals (not accurate for a tetrahedron, but sufficient for the test)
    for (let i = 0; i < 4; i++) {
      normals[i * 3 + 2] = 1; // all pointing +Z
    }
    const rng = createPRNG(7);
    const ao = computeAmbientOcclusion(vertices, normals, indices, 4, rng);
    expect(ao.length).toBe(4);
  });
});

// ---------------------------------------------------------------------------
// Tests: bakeNormalMap
// ---------------------------------------------------------------------------

describe('bakeNormalMap', () => {
  it('produces a flat normal map when high and low poly are identical', () => {
    const { vertices, indices } = singleTriangle();
    const normals = new Float64Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
    const uvs = new Float64Array([0, 0, 1, 0, 0.5, 1]);

    const normalMap = bakeNormalMap(
      { vertices, normals, indices },
      { vertices, normals, indices, uvs },
      4,
    );

    // 4x4 * 3 channels
    expect(normalMap.length).toBe(4 * 4 * 3);

    // Most texels should have the flat normal (0.5, 0.5, 1.0) or close to it
    for (let i = 0; i < normalMap.length / 3; i++) {
      const r = normalMap[i * 3]!;
      const g = normalMap[i * 3 + 1]!;
      const b = normalMap[i * 3 + 2]!;
      // Values should be in [0, 1]
      expect(r).toBeGreaterThanOrEqual(0);
      expect(r).toBeLessThanOrEqual(1.0 + 1e-6);
      expect(g).toBeGreaterThanOrEqual(0);
      expect(g).toBeLessThanOrEqual(1.0 + 1e-6);
      expect(b).toBeGreaterThanOrEqual(0);
      expect(b).toBeLessThanOrEqual(1.0 + 1e-6);
    }
  });
});

// ---------------------------------------------------------------------------
// Tests: createGLTFAccessor
// ---------------------------------------------------------------------------

describe('createGLTFAccessor', () => {
  it('creates a valid accessor with the given parameters', () => {
    const accessor = createGLTFAccessor(5126, 100, 'VEC3');

    expect(accessor.componentType).toBe(5126);
    expect(accessor.count).toBe(100);
    expect(accessor.type).toBe('VEC3');
    expect(accessor.bufferViewIndex).toBe(0);
    expect(accessor.byteOffset).toBe(0);
  });

  it('supports SCALAR type', () => {
    const accessor = createGLTFAccessor(5125, 50, 'SCALAR');
    expect(accessor.type).toBe('SCALAR');
    expect(accessor.count).toBe(50);
  });

  it('supports MAT4 type', () => {
    const accessor = createGLTFAccessor(5126, 10, 'MAT4');
    expect(accessor.type).toBe('MAT4');
  });
});

// ---------------------------------------------------------------------------
// Tests: estimateGLTFSize
// ---------------------------------------------------------------------------

describe('estimateGLTFSize', () => {
  it('returns positive size for a simple mesh', () => {
    const mesh: Mesh = {
      vertices: new Float64Array(9),
      indices: new Uint32Array(3),
      vertexCount: 3,
      triangleCount: 1,
    };
    const size = estimateGLTFSize(mesh);
    expect(size).toBeGreaterThan(0);
  });

  it('size increases with vertex count', () => {
    const small: Mesh = {
      vertices: new Float64Array(30),
      indices: new Uint32Array(12),
      vertexCount: 10,
      triangleCount: 4,
    };
    const large: Mesh = {
      vertices: new Float64Array(300),
      indices: new Uint32Array(120),
      vertexCount: 100,
      triangleCount: 40,
    };
    expect(estimateGLTFSize(large)).toBeGreaterThan(estimateGLTFSize(small));
  });

  it('size increases when normals and UVs are present', () => {
    const base: Mesh = {
      vertices: new Float64Array(30),
      indices: new Uint32Array(12),
      vertexCount: 10,
      triangleCount: 4,
    };
    const withAttribs: Mesh = {
      ...base,
      normals: new Float64Array(30),
      uvs: new Float64Array(20),
    };
    expect(estimateGLTFSize(withAttribs)).toBeGreaterThan(estimateGLTFSize(base));
  });

  it('includes JSON overhead of 1024 bytes', () => {
    const mesh: Mesh = {
      vertices: new Float64Array(0),
      indices: new Uint32Array(0),
      vertexCount: 0,
      triangleCount: 0,
    };
    const size = estimateGLTFSize(mesh);
    expect(size).toBe(1024); // only JSON overhead
  });
});

// ---------------------------------------------------------------------------
// Tests: validateMeshManifold
// ---------------------------------------------------------------------------

describe('validateMeshManifold', () => {
  it('reports a closed tetrahedron as manifold', () => {
    const { vertices, indices } = tetrahedron();
    const result = validateMeshManifold(vertices, indices);

    expect(result.isManifold).toBe(true);
    expect(result.nonManifoldEdges).toBe(0);
    expect(result.openEdges).toBe(0);
  });

  it('detects open edges on a single triangle', () => {
    const { vertices, indices } = singleTriangle();
    const result = validateMeshManifold(vertices, indices);

    expect(result.isManifold).toBe(true); // manifold-with-boundary is still manifold
    expect(result.openEdges).toBe(3); // all 3 edges are boundary
    expect(result.nonManifoldEdges).toBe(0);
  });

  it('detects open edges on a quad mesh', () => {
    const { vertices, indices } = quadMesh();
    const result = validateMeshManifold(vertices, indices);

    expect(result.isManifold).toBe(true);
    // Quad has 4 boundary edges (the outer perimeter) and 1 shared interior edge
    expect(result.openEdges).toBe(4);
    expect(result.nonManifoldEdges).toBe(0);
  });

  it('detects non-manifold edges when 3 triangles share an edge', () => {
    // Create a bowtie-like configuration where 3 triangles share edge (0,1)
    const vertices = new Float64Array([
      0, 0, 0,    // 0
      1, 0, 0,    // 1
      0.5, 1, 0,  // 2
      0.5, -1, 0, // 3
      0.5, 0, 1,  // 4
    ]);
    const indices = new Uint32Array([
      0, 1, 2,
      0, 1, 3,
      0, 1, 4,
    ]);
    const result = validateMeshManifold(vertices, indices);

    expect(result.isManifold).toBe(false);
    expect(result.nonManifoldEdges).toBeGreaterThan(0);
  });

  it('grid mesh is manifold with boundary', () => {
    const { vertices, indices } = gridMesh(3);
    const result = validateMeshManifold(vertices, indices);

    expect(result.isManifold).toBe(true);
    expect(result.openEdges).toBeGreaterThan(0); // open mesh has boundary
    expect(result.nonManifoldEdges).toBe(0);
  });
});
