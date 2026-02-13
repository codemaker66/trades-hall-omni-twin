// ---------------------------------------------------------------------------
// CV-6: Mesh Processing — Normal Map Baking & Ambient Occlusion
// ---------------------------------------------------------------------------

import type { Vector3, PRNG } from '../types.js';
import {
  vec3Sub,
  vec3Cross,
  vec3Dot,
  vec3Normalize,
  vec3Add,
  vec3Scale,
  vec3Length,
} from '../types.js';

// ---------------------------------------------------------------------------
// Normal map baking
// ---------------------------------------------------------------------------

/**
 * Bake a tangent-space normal map by ray-casting from the low-poly mesh
 * to the high-poly mesh and recording normal differences.
 *
 * For each texel on the low-poly surface (determined by its UV mapping):
 *  1. Compute the world-space position and normal on the low-poly surface.
 *  2. Cast a ray from that position (along the low-poly normal) against
 *     the high-poly mesh.
 *  3. Record the high-poly normal relative to the low-poly tangent frame.
 *
 * The output is a flat Float64Array of size (resolution * resolution * 3)
 * with RGB normal-map values in [0, 1] (0.5 = zero offset).
 *
 * @param highPoly   - High-poly mesh with positions, normals, and indices.
 * @param lowPoly    - Low-poly mesh with positions, normals, indices, and UVs.
 * @param resolution - Output normal map resolution (width = height).
 * @returns Float64Array of length resolution * resolution * 3 (RGB normal map).
 */
export function bakeNormalMap(
  highPoly: {
    vertices: Float64Array;
    normals: Float64Array;
    indices: Uint32Array;
  },
  lowPoly: {
    vertices: Float64Array;
    normals: Float64Array;
    indices: Uint32Array;
    uvs: Float64Array;
  },
  resolution: number,
): Float64Array {
  const texels = resolution * resolution;
  const normalMap = new Float64Array(texels * 3);

  // Initialise to flat normal (0.5, 0.5, 1.0) in tangent space
  for (let i = 0; i < texels; i++) {
    normalMap[i * 3] = 0.5;
    normalMap[i * 3 + 1] = 0.5;
    normalMap[i * 3 + 2] = 1.0;
  }

  const nLowTris = lowPoly.indices.length / 3;
  const nHighTris = highPoly.indices.length / 3;

  // For each low-poly triangle, rasterise its UV footprint into the normal map
  for (let lf = 0; lf < nLowTris; lf++) {
    const li0 = lowPoly.indices[lf * 3]!;
    const li1 = lowPoly.indices[lf * 3 + 1]!;
    const li2 = lowPoly.indices[lf * 3 + 2]!;

    // UV coordinates
    const u0 = lowPoly.uvs[li0 * 2]!;
    const v0 = lowPoly.uvs[li0 * 2 + 1]!;
    const u1 = lowPoly.uvs[li1 * 2]!;
    const v1 = lowPoly.uvs[li1 * 2 + 1]!;
    const u2 = lowPoly.uvs[li2 * 2]!;
    const v2 = lowPoly.uvs[li2 * 2 + 1]!;

    // World-space positions
    const p0: Vector3 = {
      x: lowPoly.vertices[li0 * 3]!,
      y: lowPoly.vertices[li0 * 3 + 1]!,
      z: lowPoly.vertices[li0 * 3 + 2]!,
    };
    const p1: Vector3 = {
      x: lowPoly.vertices[li1 * 3]!,
      y: lowPoly.vertices[li1 * 3 + 1]!,
      z: lowPoly.vertices[li1 * 3 + 2]!,
    };
    const p2: Vector3 = {
      x: lowPoly.vertices[li2 * 3]!,
      y: lowPoly.vertices[li2 * 3 + 1]!,
      z: lowPoly.vertices[li2 * 3 + 2]!,
    };

    // Normals
    const n0: Vector3 = {
      x: lowPoly.normals[li0 * 3]!,
      y: lowPoly.normals[li0 * 3 + 1]!,
      z: lowPoly.normals[li0 * 3 + 2]!,
    };
    const n1: Vector3 = {
      x: lowPoly.normals[li1 * 3]!,
      y: lowPoly.normals[li1 * 3 + 1]!,
      z: lowPoly.normals[li1 * 3 + 2]!,
    };
    const n2: Vector3 = {
      x: lowPoly.normals[li2 * 3]!,
      y: lowPoly.normals[li2 * 3 + 1]!,
      z: lowPoly.normals[li2 * 3 + 2]!,
    };

    // Bounding box of UV triangle in texel space
    const minU = Math.floor(Math.min(u0, u1, u2) * resolution);
    const maxU = Math.ceil(Math.max(u0, u1, u2) * resolution);
    const minV = Math.floor(Math.min(v0, v1, v2) * resolution);
    const maxV = Math.ceil(Math.max(v0, v1, v2) * resolution);

    // Rasterise: for each texel in the bounding box, check if it's inside
    // the UV triangle using barycentric coordinates
    for (let ty = Math.max(0, minV); ty < Math.min(resolution, maxV); ty++) {
      for (let tx = Math.max(0, minU); tx < Math.min(resolution, maxU); tx++) {
        // Texel centre in UV space
        const tu = (tx + 0.5) / resolution;
        const tv = (ty + 0.5) / resolution;

        // Barycentric coordinates
        const denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2);
        if (Math.abs(denom) < 1e-12) continue;

        const bary0 = ((v1 - v2) * (tu - u2) + (u2 - u1) * (tv - v2)) / denom;
        const bary1 = ((v2 - v0) * (tu - u2) + (u0 - u2) * (tv - v2)) / denom;
        const bary2 = 1 - bary0 - bary1;

        if (bary0 < -1e-4 || bary1 < -1e-4 || bary2 < -1e-4) continue;

        // Interpolate world-space position and normal on the low-poly surface
        const worldPos: Vector3 = {
          x: bary0 * p0.x + bary1 * p1.x + bary2 * p2.x,
          y: bary0 * p0.y + bary1 * p1.y + bary2 * p2.y,
          z: bary0 * p0.z + bary1 * p1.z + bary2 * p2.z,
        };
        const lowNormal = vec3Normalize({
          x: bary0 * n0.x + bary1 * n1.x + bary2 * n2.x,
          y: bary0 * n0.y + bary1 * n1.y + bary2 * n2.y,
          z: bary0 * n0.z + bary1 * n1.z + bary2 * n2.z,
        });

        // Ray-cast against high-poly mesh (brute-force for simplicity)
        let bestT = Infinity;
        let hitNormal: Vector3 | null = null;

        for (let hf = 0; hf < nHighTris; hf++) {
          const hi0 = highPoly.indices[hf * 3]!;
          const hi1 = highPoly.indices[hf * 3 + 1]!;
          const hi2 = highPoly.indices[hf * 3 + 2]!;

          const hp0: Vector3 = {
            x: highPoly.vertices[hi0 * 3]!,
            y: highPoly.vertices[hi0 * 3 + 1]!,
            z: highPoly.vertices[hi0 * 3 + 2]!,
          };
          const hp1: Vector3 = {
            x: highPoly.vertices[hi1 * 3]!,
            y: highPoly.vertices[hi1 * 3 + 1]!,
            z: highPoly.vertices[hi1 * 3 + 2]!,
          };
          const hp2: Vector3 = {
            x: highPoly.vertices[hi2 * 3]!,
            y: highPoly.vertices[hi2 * 3 + 1]!,
            z: highPoly.vertices[hi2 * 3 + 2]!,
          };

          // Moller-Trumbore ray-triangle intersection
          const edge1 = vec3Sub(hp1, hp0);
          const edge2 = vec3Sub(hp2, hp0);
          const h = vec3Cross(lowNormal, edge2);
          const det = vec3Dot(edge1, h);

          if (Math.abs(det) < 1e-12) continue;

          const invDet = 1.0 / det;
          const s = vec3Sub(worldPos, hp0);
          const u = vec3Dot(s, h) * invDet;
          if (u < 0 || u > 1) continue;

          const q = vec3Cross(s, edge1);
          const vv = vec3Dot(lowNormal, q) * invDet;
          if (vv < 0 || u + vv > 1) continue;

          const t = vec3Dot(edge2, q) * invDet;
          if (Math.abs(t) < bestT) {
            bestT = Math.abs(t);

            // Interpolate high-poly normal at the hit point
            const hb0 = 1 - u - vv;
            const hb1 = u;
            const hb2 = vv;
            hitNormal = vec3Normalize({
              x:
                hb0 * highPoly.normals[hi0 * 3]! +
                hb1 * highPoly.normals[hi1 * 3]! +
                hb2 * highPoly.normals[hi2 * 3]!,
              y:
                hb0 * highPoly.normals[hi0 * 3 + 1]! +
                hb1 * highPoly.normals[hi1 * 3 + 1]! +
                hb2 * highPoly.normals[hi2 * 3 + 1]!,
              z:
                hb0 * highPoly.normals[hi0 * 3 + 2]! +
                hb1 * highPoly.normals[hi1 * 3 + 2]! +
                hb2 * highPoly.normals[hi2 * 3 + 2]!,
            });
          }
        }

        if (hitNormal) {
          // Build tangent frame from the low-poly normal
          const tangent = buildTangent(lowNormal);
          const bitangent = vec3Cross(lowNormal, tangent);

          // Transform high-poly normal into tangent space
          const tsX = vec3Dot(hitNormal, tangent);
          const tsY = vec3Dot(hitNormal, bitangent);
          const tsZ = vec3Dot(hitNormal, lowNormal);

          // Encode to [0, 1]: tangent-space normal of (0,0,1) maps to (0.5, 0.5, 1.0)
          const idx = (ty * resolution + tx) * 3;
          normalMap[idx] = tsX * 0.5 + 0.5;
          normalMap[idx + 1] = tsY * 0.5 + 0.5;
          normalMap[idx + 2] = tsZ * 0.5 + 0.5;
        }
      }
    }
  }

  return normalMap;
}

/**
 * Build an arbitrary tangent vector perpendicular to the given normal.
 */
function buildTangent(normal: Vector3): Vector3 {
  const absX = Math.abs(normal.x);
  const absY = Math.abs(normal.y);
  const up: Vector3 = absX < 0.9 ? { x: 1, y: 0, z: 0 } : { x: 0, y: 1, z: 0 };
  // Suppress unused variable warning — absY is used for readability but the
  // branch logic only checks absX for simplicity.
  void absY;
  return vec3Normalize(vec3Cross(normal, up));
}

// ---------------------------------------------------------------------------
// Ambient occlusion
// ---------------------------------------------------------------------------

/**
 * Compute per-vertex ambient occlusion by hemisphere sampling.
 *
 * For each vertex, `nSamples` random rays are cast in the hemisphere
 * oriented along the vertex normal. The AO value is the fraction of
 * rays that are NOT occluded by the mesh.
 *
 * Occlusion is tested via brute-force ray-triangle intersection against
 * all triangles. In production this would use a BVH or spatial hash.
 *
 * @param vertices - Packed vertex positions.
 * @param normals  - Packed vertex normals.
 * @param indices  - Triangle index buffer.
 * @param nSamples - Number of hemisphere rays per vertex.
 * @param rng      - Seeded PRNG.
 * @returns Float64Array of per-vertex AO values in [0, 1] (1 = fully unoccluded).
 */
export function computeAmbientOcclusion(
  vertices: Float64Array,
  normals: Float64Array,
  indices: Uint32Array,
  nSamples: number,
  rng: PRNG,
): Float64Array {
  const nVertices = vertices.length / 3;
  const nTriangles = indices.length / 3;
  const ao = new Float64Array(nVertices);

  // Pre-extract triangle data for faster intersection
  const triVerts: Vector3[][] = [];
  for (let f = 0; f < nTriangles; f++) {
    const i0 = indices[f * 3]!;
    const i1 = indices[f * 3 + 1]!;
    const i2 = indices[f * 3 + 2]!;
    triVerts.push([
      { x: vertices[i0 * 3]!, y: vertices[i0 * 3 + 1]!, z: vertices[i0 * 3 + 2]! },
      { x: vertices[i1 * 3]!, y: vertices[i1 * 3 + 1]!, z: vertices[i1 * 3 + 2]! },
      { x: vertices[i2 * 3]!, y: vertices[i2 * 3 + 1]!, z: vertices[i2 * 3 + 2]! },
    ]);
  }

  for (let vi = 0; vi < nVertices; vi++) {
    const origin: Vector3 = {
      x: vertices[vi * 3]!,
      y: vertices[vi * 3 + 1]!,
      z: vertices[vi * 3 + 2]!,
    };
    const normal = vec3Normalize({
      x: normals[vi * 3]!,
      y: normals[vi * 3 + 1]!,
      z: normals[vi * 3 + 2]!,
    });

    // Build a local tangent frame
    const tangent = buildTangent(normal);
    const bitangent = vec3Cross(normal, tangent);

    let unoccluded = 0;

    for (let s = 0; s < nSamples; s++) {
      // Generate random direction in the hemisphere (cosine-weighted)
      const u1 = rng();
      const u2 = rng();
      const r = Math.sqrt(u1);
      const theta = 2 * Math.PI * u2;
      const dx = r * Math.cos(theta);
      const dy = r * Math.sin(theta);
      const dz = Math.sqrt(Math.max(0, 1 - u1));

      // Transform to world space
      const dir = vec3Normalize(
        vec3Add(
          vec3Add(vec3Scale(tangent, dx), vec3Scale(bitangent, dy)),
          vec3Scale(normal, dz),
        ),
      );

      // Offset ray origin slightly to avoid self-intersection
      const rayOrigin = vec3Add(origin, vec3Scale(normal, 1e-4));

      // Test intersection against all triangles
      let occluded = false;
      for (let f = 0; f < nTriangles; f++) {
        const tri = triVerts[f]!;
        if (rayTriangleIntersect(rayOrigin, dir, tri[0]!, tri[1]!, tri[2]!)) {
          occluded = true;
          break;
        }
      }

      if (!occluded) unoccluded++;
    }

    ao[vi] = nSamples > 0 ? unoccluded / nSamples : 1.0;
  }

  return ao;
}

/**
 * Moller-Trumbore ray-triangle intersection test.
 *
 * @returns true if the ray intersects the triangle at a positive t.
 */
function rayTriangleIntersect(
  origin: Vector3,
  direction: Vector3,
  v0: Vector3,
  v1: Vector3,
  v2: Vector3,
): boolean {
  const edge1 = vec3Sub(v1, v0);
  const edge2 = vec3Sub(v2, v0);
  const h = vec3Cross(direction, edge2);
  const det = vec3Dot(edge1, h);

  if (Math.abs(det) < 1e-12) return false;

  const invDet = 1.0 / det;
  const s = vec3Sub(origin, v0);
  const u = vec3Dot(s, h) * invDet;
  if (u < 0 || u > 1) return false;

  const q = vec3Cross(s, edge1);
  const v = vec3Dot(direction, q) * invDet;
  if (v < 0 || u + v > 1) return false;

  const t = vec3Dot(edge2, q) * invDet;
  return t > 1e-6;
}
