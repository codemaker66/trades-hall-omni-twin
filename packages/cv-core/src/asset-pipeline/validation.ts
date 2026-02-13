// ---------------------------------------------------------------------------
// CV-12: Asset Pipeline — Mesh Validation
// Triangle count, UV bounds, physical scale, and manifold checks.
// ---------------------------------------------------------------------------

import type { Mesh } from '../types.js';

// ---------------------------------------------------------------------------
// validateTriangleCount
// ---------------------------------------------------------------------------

/**
 * Validate that a mesh does not exceed a maximum triangle count.
 *
 * @param mesh         The mesh to check.
 * @param maxTriangles Maximum allowed triangle count.
 * @returns            An object with `pass`, `actual`, and `max` fields.
 */
export function validateTriangleCount(
  mesh: Mesh,
  maxTriangles: number,
): { pass: boolean; actual: number; max: number } {
  return {
    pass: mesh.triangleCount <= maxTriangles,
    actual: mesh.triangleCount,
    max: maxTriangles,
  };
}

// ---------------------------------------------------------------------------
// validateUVBounds
// ---------------------------------------------------------------------------

/**
 * Check that all UV coordinates lie within the [0, 1] range.
 *
 * UV data is packed as [u0, v0, u1, v1, ...].  Each component is tested
 * individually and any value outside [0, 1] (with a small epsilon tolerance)
 * is counted as out-of-bounds.
 *
 * @param uvs UV coordinate array (interleaved u, v pairs).
 * @returns   An object with `pass` (true if all UVs valid) and
 *            `outOfBounds` (number of individual components outside range).
 */
export function validateUVBounds(
  uvs: Float64Array,
): { pass: boolean; outOfBounds: number } {
  const eps = 1e-6;
  let outOfBounds = 0;

  for (let i = 0; i < uvs.length; i++) {
    const v = uvs[i]!;
    if (v < -eps || v > 1 + eps) {
      outOfBounds++;
    }
  }

  return { pass: outOfBounds === 0, outOfBounds };
}

// ---------------------------------------------------------------------------
// validateScale
// ---------------------------------------------------------------------------

/**
 * Validate that a mesh's physical size falls within an expected range.
 *
 * The "size" is the maximum extent of the axis-aligned bounding box computed
 * from the vertex positions.
 *
 * @param vertices     Vertex positions packed as [x0,y0,z0, x1,y1,z1, ...].
 * @param expectedSize Object with `min` and `max` extents in metres.
 * @returns            `pass` is true when actualSize is in [min, max].
 */
export function validateScale(
  vertices: Float64Array,
  expectedSize: { min: number; max: number },
): { pass: boolean; actualSize: number } {
  if (vertices.length < 3) {
    return { pass: false, actualSize: 0 };
  }

  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;

  const vertexCount = Math.floor(vertices.length / 3);
  for (let i = 0; i < vertexCount; i++) {
    const x = vertices[i * 3]!;
    const y = vertices[i * 3 + 1]!;
    const z = vertices[i * 3 + 2]!;

    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }

  const dx = maxX - minX;
  const dy = maxY - minY;
  const dz = maxZ - minZ;
  const actualSize = Math.max(dx, dy, dz);

  return {
    pass: actualSize >= expectedSize.min && actualSize <= expectedSize.max,
    actualSize,
  };
}

// ---------------------------------------------------------------------------
// validateManifold
// ---------------------------------------------------------------------------

/**
 * Check whether a triangle mesh is manifold (watertight).
 *
 * A mesh is manifold if every edge is shared by exactly two triangles. An
 * edge shared by only one triangle is "open" (boundary); an edge shared by
 * more than two is non-manifold. This function counts open edges.
 *
 * @param vertices Vertex positions packed as [x0,y0,z0, ...] (used only for
 *                 count validation — the topology is defined by `indices`).
 * @param indices  Triangle index buffer (3 indices per triangle).
 * @returns        `pass` is true when `openEdges` is 0.
 */
export function validateManifold(
  vertices: Float64Array,
  indices: Uint32Array,
): { pass: boolean; openEdges: number } {
  // Build a map of edge -> use count.
  // Canonical edge key: smaller index first.
  const edgeCounts = new Map<string, number>();

  const triCount = Math.floor(indices.length / 3);
  for (let t = 0; t < triCount; t++) {
    const i0 = indices[t * 3]!;
    const i1 = indices[t * 3 + 1]!;
    const i2 = indices[t * 3 + 2]!;

    addEdge(edgeCounts, i0, i1);
    addEdge(edgeCounts, i1, i2);
    addEdge(edgeCounts, i2, i0);
  }

  let openEdges = 0;
  for (const count of edgeCounts.values()) {
    if (count !== 2) {
      openEdges++;
    }
  }

  return { pass: openEdges === 0, openEdges };
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function addEdge(map: Map<string, number>, a: number, b: number): void {
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  const key = `${lo}:${hi}`;
  const prev = map.get(key);
  map.set(key, prev === undefined ? 1 : prev + 1);
}
