// ---------------------------------------------------------------------------
// CV-5: NeRF — Signed Distance Functions
// ---------------------------------------------------------------------------

import type { Vector3 } from '../types.js';
import { vec3Add, vec3Scale, vec3Normalize } from '../types.js';

// ---------------------------------------------------------------------------
// Sphere tracing / ray marching
// ---------------------------------------------------------------------------

/**
 * Sphere tracing (ray marching) against an arbitrary signed distance function.
 *
 * Starting from `origin` along `direction`, the algorithm advances by the
 * SDF value at the current point until the distance drops below `tolerance`
 * (hit) or `maxSteps` is exceeded (miss).
 *
 * @param sdfFn     - Signed distance function: positive outside, negative inside.
 * @param origin    - Ray origin.
 * @param direction - Unit-length ray direction.
 * @param maxSteps  - Maximum iteration count.
 * @param tolerance - Distance threshold for a hit.
 * @returns Object with `hit`, surface `point`, `steps` taken, and total `distance`.
 */
export function sphereTrace(
  sdfFn: (p: Vector3) => number,
  origin: Vector3,
  direction: Vector3,
  maxSteps: number,
  tolerance: number,
): { hit: boolean; point: Vector3; steps: number; distance: number } {
  let totalDistance = 0;
  let point: Vector3 = { x: origin.x, y: origin.y, z: origin.z };

  for (let step = 0; step < maxSteps; step++) {
    const d = sdfFn(point);

    if (Math.abs(d) < tolerance) {
      return { hit: true, point, steps: step + 1, distance: totalDistance };
    }

    totalDistance += d;
    point = vec3Add(origin, vec3Scale(direction, totalDistance));

    // Divergence guard — if we march too far, abort
    if (totalDistance > 1e6) {
      return { hit: false, point, steps: step + 1, distance: totalDistance };
    }
  }

  return { hit: false, point, steps: maxSteps, distance: totalDistance };
}

// ---------------------------------------------------------------------------
// Normal estimation
// ---------------------------------------------------------------------------

/**
 * Estimate the surface normal of an SDF at a point using central differences.
 *
 *   n_x = sdf(p + (eps,0,0)) - sdf(p - (eps,0,0))
 *   n_y = sdf(p + (0,eps,0)) - sdf(p - (0,eps,0))
 *   n_z = sdf(p + (0,0,eps)) - sdf(p - (0,0,eps))
 *
 * The result is normalised to unit length.
 *
 * @param sdfFn - Signed distance function.
 * @param point - Surface point (or near-surface).
 * @param eps   - Finite-difference step (default 1e-4).
 * @returns Normalised gradient vector approximating the outward surface normal.
 */
export function estimateNormal(
  sdfFn: (p: Vector3) => number,
  point: Vector3,
  eps: number = 1e-4,
): Vector3 {
  const nx =
    sdfFn({ x: point.x + eps, y: point.y, z: point.z }) -
    sdfFn({ x: point.x - eps, y: point.y, z: point.z });
  const ny =
    sdfFn({ x: point.x, y: point.y + eps, z: point.z }) -
    sdfFn({ x: point.x, y: point.y - eps, z: point.z });
  const nz =
    sdfFn({ x: point.x, y: point.y, z: point.z + eps }) -
    sdfFn({ x: point.x, y: point.y, z: point.z - eps });

  return vec3Normalize({ x: nx, y: ny, z: nz });
}

// ---------------------------------------------------------------------------
// Marching cubes — SDF to mesh
// ---------------------------------------------------------------------------

/**
 * Convert a signed distance function to a triangle mesh using marching cubes.
 *
 * The SDF is evaluated on a regular 3D grid within `bounds` at the given
 * `resolution` (number of cells per axis). Surface crossings (sign changes)
 * are identified and triangulated using the classic marching cubes algorithm.
 *
 * This is a simplified implementation that handles the most common cube
 * configurations via a mid-edge vertex interpolation scheme.
 *
 * @param sdfFn      - Signed distance function.
 * @param bounds     - Axis-aligned bounding box for evaluation.
 * @param resolution - Number of cells along each axis.
 * @returns Object with `vertices` (packed xyz Float64Array) and triangle `indices`.
 */
export function sdfToMesh(
  sdfFn: (p: Vector3) => number,
  bounds: { min: Vector3; max: Vector3 },
  resolution: number,
): { vertices: Float64Array; indices: Uint32Array } {
  const res = resolution;
  const dx = (bounds.max.x - bounds.min.x) / res;
  const dy = (bounds.max.y - bounds.min.y) / res;
  const dz = (bounds.max.z - bounds.min.z) / res;

  // Pre-evaluate the SDF on the grid
  const nx = res + 1;
  const ny = res + 1;
  const nz = res + 1;
  const grid = new Float64Array(nx * ny * nz);

  for (let iz = 0; iz < nz; iz++) {
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const p: Vector3 = {
          x: bounds.min.x + ix * dx,
          y: bounds.min.y + iy * dy,
          z: bounds.min.z + iz * dz,
        };
        grid[ix + iy * nx + iz * nx * ny] = sdfFn(p);
      }
    }
  }

  // Edge interpolation helper — finds zero-crossing between two grid points.
  function interpolateEdge(
    p1: Vector3,
    p2: Vector3,
    v1: number,
    v2: number,
  ): Vector3 {
    if (Math.abs(v1) < 1e-10) return { x: p1.x, y: p1.y, z: p1.z };
    if (Math.abs(v2) < 1e-10) return { x: p2.x, y: p2.y, z: p2.z };
    if (Math.abs(v1 - v2) < 1e-10) return { x: p1.x, y: p1.y, z: p1.z };
    const t = v1 / (v1 - v2);
    return {
      x: p1.x + t * (p2.x - p1.x),
      y: p1.y + t * (p2.y - p1.y),
      z: p1.z + t * (p2.z - p1.z),
    };
  }

  // Vertex deduplication map: key -> index
  const vertexMap = new Map<string, number>();
  const vertexList: number[] = [];
  const indexList: number[] = [];

  function addVertex(v: Vector3): number {
    // Quantise to avoid floating-point key collisions
    const key = `${Math.round(v.x * 1e6)},${Math.round(v.y * 1e6)},${Math.round(v.z * 1e6)}`;
    const existing = vertexMap.get(key);
    if (existing !== undefined) return existing;
    const idx = vertexList.length / 3;
    vertexList.push(v.x, v.y, v.z);
    vertexMap.set(key, idx);
    return idx;
  }

  // Cube corner offsets (local ix, iy, iz)
  const cornerOffsets: [number, number, number][] = [
    [0, 0, 0], // 0
    [1, 0, 0], // 1
    [1, 1, 0], // 2
    [0, 1, 0], // 3
    [0, 0, 1], // 4
    [1, 0, 1], // 5
    [1, 1, 1], // 6
    [0, 1, 1], // 7
  ];

  // 12 edges of the cube, each defined by two corner indices
  const edgeCorners: [number, number][] = [
    [0, 1], [1, 2], [2, 3], [3, 0], // bottom face
    [4, 5], [5, 6], [6, 7], [7, 4], // top face
    [0, 4], [1, 5], [2, 6], [3, 7], // vertical edges
  ];

  // March through each cube
  for (let iz = 0; iz < res; iz++) {
    for (let iy = 0; iy < res; iy++) {
      for (let ix = 0; ix < res; ix++) {
        // Evaluate the 8 corners
        const cornerValues: number[] = [];
        const cornerPositions: Vector3[] = [];
        let cubeIndex = 0;

        for (let c = 0; c < 8; c++) {
          const off = cornerOffsets[c]!;
          const ci = ix + off[0];
          const cj = iy + off[1];
          const ck = iz + off[2];
          const val = grid[ci + cj * nx + ck * nx * ny]!;
          cornerValues.push(val);
          cornerPositions.push({
            x: bounds.min.x + ci * dx,
            y: bounds.min.y + cj * dy,
            z: bounds.min.z + ck * dz,
          });
          if (val < 0) cubeIndex |= 1 << c;
        }

        // Skip if entirely inside or outside
        if (cubeIndex === 0 || cubeIndex === 255) continue;

        // Find interpolated edge vertices for active edges
        const edgeVertices: (Vector3 | null)[] = new Array(12).fill(null);
        for (let e = 0; e < 12; e++) {
          const [c0, c1] = edgeCorners[e]!;
          const v0 = cornerValues[c0]!;
          const v1 = cornerValues[c1]!;
          // Edge is active if the two corners have different signs
          if ((v0 < 0) !== (v1 < 0)) {
            edgeVertices[e] = interpolateEdge(
              cornerPositions[c0]!,
              cornerPositions[c1]!,
              v0,
              v1,
            );
          }
        }

        // Build triangles by connecting active edge midpoints.
        // This is a simplified approach: for each pair of adjacent active
        // edges around a face, create triangles fan-style from the first
        // active edge vertex. A full implementation would use the 256-entry
        // marching cubes lookup table; here we use a centroid-fan method
        // that produces correct topology for the most common cases.
        const activeEdgeIndices: number[] = [];
        for (let e = 0; e < 12; e++) {
          if (edgeVertices[e] !== null) activeEdgeIndices.push(e);
        }

        if (activeEdgeIndices.length >= 3) {
          // Compute centroid of active edge vertices for fan triangulation
          let cx = 0;
          let cy = 0;
          let cz = 0;
          for (let a = 0; a < activeEdgeIndices.length; a++) {
            const v = edgeVertices[activeEdgeIndices[a]!]!;
            cx += v.x;
            cy += v.y;
            cz += v.z;
          }
          cx /= activeEdgeIndices.length;
          cy /= activeEdgeIndices.length;
          cz /= activeEdgeIndices.length;
          const centroidIdx = addVertex({ x: cx, y: cy, z: cz });

          // Fan from centroid to each pair of adjacent active edges
          for (let a = 0; a < activeEdgeIndices.length; a++) {
            const next = (a + 1) % activeEdgeIndices.length;
            const v0 = edgeVertices[activeEdgeIndices[a]!]!;
            const v1 = edgeVertices[activeEdgeIndices[next]!]!;
            const i0 = addVertex(v0);
            const i1 = addVertex(v1);
            indexList.push(centroidIdx, i0, i1);
          }
        }
      }
    }
  }

  return {
    vertices: new Float64Array(vertexList),
    indices: new Uint32Array(indexList),
  };
}
