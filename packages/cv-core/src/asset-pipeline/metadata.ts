// ---------------------------------------------------------------------------
// CV-12: Asset Pipeline — Asset Metadata
// Extract metadata, compute physical dimensions, and analyse mesh complexity.
// ---------------------------------------------------------------------------

import type { Mesh, AssetMetadata, Vector3 } from '../types.js';

// ---------------------------------------------------------------------------
// createAssetMetadata
// ---------------------------------------------------------------------------

/**
 * Extract comprehensive metadata from a mesh.
 *
 * This function inspects vertex buffers, index buffers, and optional
 * attribute arrays to build a full `AssetMetadata` descriptor.  Manifold
 * checking is performed via an edge-count analysis identical to
 * `validateManifold`.
 *
 * @param mesh        The source mesh.
 * @param textureSize Optional maximum texture dimension (pixels). Defaults to 0.
 * @returns           An `AssetMetadata` descriptor.
 */
export function createAssetMetadata(
  mesh: Mesh,
  textureSize?: number,
): AssetMetadata {
  const dims = computePhysicalDimensions(mesh.vertices);
  const complexity = computeMeshComplexity(mesh);
  const texSz = textureSize ?? 0;

  // Estimate file size: vertex data + index data + texture (RGBA8 compressed 4:1).
  const vertexBytes = mesh.vertexCount * 32; // pos(12) + normal(12) + uv(8)
  const indexBytes = mesh.triangleCount * 3 * 4;
  const texBytes = texSz > 0 ? (texSz * texSz * 4) / 4 : 0;
  const fileSize = Math.ceil((vertexBytes + indexBytes + texBytes) * 1.1);

  // Manifold check: count edges used by != 2 triangles.
  const { openEdges } = countOpenEdges(mesh.indices);

  // Degenerate triangle check.
  const degenerateCount = countDegenerateTriangles(mesh);

  return {
    triangleCount: mesh.triangleCount,
    vertexCount: mesh.vertexCount,
    textureSize: texSz,
    fileSize,
    physicalDimensions: {
      x: dims.width,
      y: dims.height,
      z: dims.depth,
    },
    materialCount: 1,
    textureCount: texSz > 0 ? 1 : 0,
    textureMemoryBytes: texSz > 0 ? texSz * texSz * 4 : 0,
    hasUVs: mesh.uvs !== undefined && mesh.uvs.length > 0,
    hasNormals: mesh.normals !== undefined && mesh.normals.length > 0,
    isManifold: openEdges === 0,
    degenerateTriangleCount: degenerateCount,
    nonManifoldEdgeCount: openEdges,
  };
}

// ---------------------------------------------------------------------------
// computePhysicalDimensions
// ---------------------------------------------------------------------------

/**
 * Compute the axis-aligned bounding box dimensions from packed vertex data.
 *
 * @param vertices Vertex positions packed as [x0,y0,z0, x1,y1,z1, ...].
 * @returns        Width (X), height (Y), and depth (Z) extents in metres.
 */
export function computePhysicalDimensions(
  vertices: Float64Array,
): { width: number; height: number; depth: number } {
  if (vertices.length < 3) {
    return { width: 0, height: 0, depth: 0 };
  }

  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;

  const count = Math.floor(vertices.length / 3);
  for (let i = 0; i < count; i++) {
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

  return {
    width: maxX - minX,
    height: maxY - minY,
    depth: maxZ - minZ,
  };
}

// ---------------------------------------------------------------------------
// computeMeshComplexity
// ---------------------------------------------------------------------------

/**
 * Compute summary complexity metrics for a mesh.
 *
 * The average edge length is computed by iterating all triangle edges and
 * averaging their Euclidean lengths.
 *
 * @param mesh The mesh to analyse.
 * @returns    Triangle count, vertex count, and average edge length.
 */
export function computeMeshComplexity(
  mesh: Mesh,
): { triangleCount: number; vertexCount: number; avgEdgeLength: number } {
  let totalEdgeLength = 0;
  let edgeCount = 0;

  const verts = mesh.vertices;
  const indices = mesh.indices;
  const triCount = Math.floor(indices.length / 3);

  for (let t = 0; t < triCount; t++) {
    const i0 = indices[t * 3]!;
    const i1 = indices[t * 3 + 1]!;
    const i2 = indices[t * 3 + 2]!;

    totalEdgeLength += edgeLength(verts, i0, i1);
    totalEdgeLength += edgeLength(verts, i1, i2);
    totalEdgeLength += edgeLength(verts, i2, i0);
    edgeCount += 3;
  }

  return {
    triangleCount: mesh.triangleCount,
    vertexCount: mesh.vertexCount,
    avgEdgeLength: edgeCount > 0 ? totalEdgeLength / edgeCount : 0,
  };
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function edgeLength(verts: Float64Array, a: number, b: number): number {
  const ax = verts[a * 3]!;
  const ay = verts[a * 3 + 1]!;
  const az = verts[a * 3 + 2]!;
  const bx = verts[b * 3]!;
  const by = verts[b * 3 + 1]!;
  const bz = verts[b * 3 + 2]!;
  const dx = bx - ax;
  const dy = by - ay;
  const dz = bz - az;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function countOpenEdges(indices: Uint32Array): { openEdges: number } {
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
    if (count !== 2) openEdges++;
  }

  return { openEdges };
}

function addEdge(map: Map<string, number>, a: number, b: number): void {
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  const key = `${lo}:${hi}`;
  const prev = map.get(key);
  map.set(key, prev === undefined ? 1 : prev + 1);
}

function countDegenerateTriangles(mesh: Mesh): number {
  const verts = mesh.vertices;
  const indices = mesh.indices;
  const triCount = Math.floor(indices.length / 3);
  let degenerate = 0;

  for (let t = 0; t < triCount; t++) {
    const i0 = indices[t * 3]!;
    const i1 = indices[t * 3 + 1]!;
    const i2 = indices[t * 3 + 2]!;

    // Compute triangle area via cross product of two edges.
    const ax = verts[i1 * 3]! - verts[i0 * 3]!;
    const ay = verts[i1 * 3 + 1]! - verts[i0 * 3 + 1]!;
    const az = verts[i1 * 3 + 2]! - verts[i0 * 3 + 2]!;
    const bx = verts[i2 * 3]! - verts[i0 * 3]!;
    const by = verts[i2 * 3 + 1]! - verts[i0 * 3 + 1]!;
    const bz = verts[i2 * 3 + 2]! - verts[i0 * 3 + 2]!;

    const cx = ay * bz - az * by;
    const cy = az * bx - ax * bz;
    const cz = ax * by - ay * bx;

    const area = 0.5 * Math.sqrt(cx * cx + cy * cy + cz * cz);
    if (area < 1e-12) {
      degenerate++;
    }
  }

  return degenerate;
}
