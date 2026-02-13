// ---------------------------------------------------------------------------
// CV-6: Mesh Processing — glTF Utilities & Mesh Validation
// ---------------------------------------------------------------------------

import type { Mesh, GLTFAccessor } from '../types.js';

// ---------------------------------------------------------------------------
// glTF accessor creation
// ---------------------------------------------------------------------------

/**
 * Create a glTF accessor descriptor with the given parameters.
 *
 * This produces the metadata object that would appear in a glTF JSON;
 * `bufferViewIndex` and `byteOffset` default to 0 (to be patched during
 * serialisation).
 *
 * Common `componentType` values:
 *  - 5120 = BYTE
 *  - 5121 = UNSIGNED_BYTE
 *  - 5122 = SHORT
 *  - 5123 = UNSIGNED_SHORT
 *  - 5125 = UNSIGNED_INT
 *  - 5126 = FLOAT
 *
 * @param componentType - WebGL component type constant.
 * @param count         - Number of elements (e.g. number of vertices).
 * @param type          - Element type string ('SCALAR', 'VEC2', 'VEC3', etc.).
 * @returns A {@link GLTFAccessor} with defaults for bufferViewIndex and byteOffset.
 */
export function createGLTFAccessor(
  componentType: number,
  count: number,
  type: 'SCALAR' | 'VEC2' | 'VEC3' | 'VEC4' | 'MAT2' | 'MAT3' | 'MAT4',
): GLTFAccessor {
  return {
    bufferViewIndex: 0,
    byteOffset: 0,
    componentType,
    count,
    type,
  };
}

// ---------------------------------------------------------------------------
// glTF size estimation
// ---------------------------------------------------------------------------

/** Bytes per component for standard glTF component types. */
const COMPONENT_BYTES: Record<number, number> = {
  5120: 1, // BYTE
  5121: 1, // UNSIGNED_BYTE
  5122: 2, // SHORT
  5123: 2, // UNSIGNED_SHORT
  5125: 4, // UNSIGNED_INT
  5126: 4, // FLOAT
};

/** Number of components per accessor type. */
const TYPE_COMPONENTS: Record<string, number> = {
  SCALAR: 1,
  VEC2: 2,
  VEC3: 3,
  VEC4: 4,
  MAT2: 4,
  MAT3: 9,
  MAT4: 16,
};

/**
 * Estimate the total byte size of a mesh when encoded as glTF binary.
 *
 * Accounts for:
 *  - Vertex positions (VEC3, FLOAT)
 *  - Index buffer (SCALAR, UNSIGNED_INT)
 *  - Normals (VEC3, FLOAT) if present
 *  - UVs (VEC2, FLOAT) if present
 *  - Colours (VEC3, FLOAT) if present
 *  - Approximate JSON overhead (fixed 1024 bytes)
 *
 * @param mesh - The mesh to estimate size for.
 * @returns Estimated byte count for glTF binary encoding.
 */
export function estimateGLTFSize(mesh: Mesh): number {
  const floatSize = COMPONENT_BYTES[5126]!;
  const uintSize = COMPONENT_BYTES[5125]!;

  // Positions: vertexCount * VEC3 * FLOAT
  let totalBytes = mesh.vertexCount * TYPE_COMPONENTS['VEC3']! * floatSize;

  // Indices: triangleCount * 3 * UNSIGNED_INT
  totalBytes += mesh.triangleCount * 3 * uintSize;

  // Normals (optional)
  if (mesh.normals) {
    totalBytes += mesh.vertexCount * TYPE_COMPONENTS['VEC3']! * floatSize;
  }

  // UVs (optional)
  if (mesh.uvs) {
    totalBytes += mesh.vertexCount * TYPE_COMPONENTS['VEC2']! * floatSize;
  }

  // Colours (optional)
  if (mesh.colors) {
    totalBytes += mesh.vertexCount * TYPE_COMPONENTS['VEC3']! * floatSize;
  }

  // JSON overhead estimate
  totalBytes += 1024;

  return totalBytes;
}

// ---------------------------------------------------------------------------
// Manifold validation
// ---------------------------------------------------------------------------

/**
 * Validate mesh topology by checking for manifold properties.
 *
 * A manifold mesh has the property that every edge is shared by exactly
 * two triangles (interior edge) or exactly one triangle (boundary edge).
 * Edges shared by 0 or more than 2 triangles indicate non-manifold geometry.
 *
 * @param vertices - Packed vertex positions (used for vertex count only).
 * @param indices  - Triangle index buffer.
 * @returns Object reporting manifold status, open edge count, and non-manifold edge count.
 */
export function validateMeshManifold(
  vertices: Float64Array,
  indices: Uint32Array,
): { isManifold: boolean; openEdges: number; nonManifoldEdges: number } {
  const nTriangles = indices.length / 3;

  // Count how many triangles share each directed edge
  // We use undirected edges: always store the smaller index first
  const edgeCounts = new Map<string, number>();

  for (let f = 0; f < nTriangles; f++) {
    const i0 = indices[f * 3]!;
    const i1 = indices[f * 3 + 1]!;
    const i2 = indices[f * 3 + 2]!;

    const triVerts = [i0, i1, i2];
    for (let e = 0; e < 3; e++) {
      const a = triVerts[e]!;
      const b = triVerts[(e + 1) % 3]!;
      const key = a < b ? `${a}_${b}` : `${b}_${a}`;
      edgeCounts.set(key, (edgeCounts.get(key) ?? 0) + 1);
    }
  }

  let openEdges = 0;
  let nonManifoldEdges = 0;

  for (const count of edgeCounts.values()) {
    if (count === 1) {
      // Boundary edge — acceptable in open meshes
      openEdges++;
    } else if (count > 2) {
      // Non-manifold: more than 2 triangles share this edge
      nonManifoldEdges++;
    }
    // count === 2 is the normal manifold case
  }

  // A closed manifold has zero open edges and zero non-manifold edges.
  // We consider a mesh "manifold" if it has no non-manifold edges
  // (open meshes with boundary are still valid manifolds-with-boundary).
  const isManifold = nonManifoldEdges === 0;

  return { isManifold, openEdges, nonManifoldEdges };
}
