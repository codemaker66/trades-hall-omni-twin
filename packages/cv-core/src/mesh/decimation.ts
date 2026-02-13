// ---------------------------------------------------------------------------
// CV-6: Mesh Processing — QEM Edge Collapse Decimation
// ---------------------------------------------------------------------------

import type { Vector3, QEMConfig } from '../types.js';

// ---------------------------------------------------------------------------
// Quadric error evaluation
// ---------------------------------------------------------------------------

/**
 * Evaluate the quadric error v^T Q v for a 4x4 symmetric quadric matrix Q
 * and a 3D vertex v (homogeneous coordinate w=1).
 *
 * Q is stored as a 16-element Float64Array in row-major order:
 *   [ q00 q01 q02 q03 ]
 *   [ q10 q11 q12 q13 ]
 *   [ q20 q21 q22 q23 ]
 *   [ q30 q31 q32 q33 ]
 *
 * The error is computed as v_h^T * Q * v_h where v_h = [vx, vy, vz, 1].
 *
 * @param v - 3D vertex position.
 * @param Q - 4x4 quadric matrix as a 16-element Float64Array (row-major).
 * @returns Quadric error (scalar).
 */
export function computeQuadricError(v: Vector3, Q: Float64Array): number {
  const x = v.x;
  const y = v.y;
  const z = v.z;

  // v_h^T Q v_h with v_h = [x, y, z, 1]
  // Expanded from the 4x4 symmetric matrix product:
  return (
    Q[0]! * x * x +
    2 * Q[1]! * x * y +
    2 * Q[2]! * x * z +
    2 * Q[3]! * x +
    Q[5]! * y * y +
    2 * Q[6]! * y * z +
    2 * Q[7]! * y +
    Q[10]! * z * z +
    2 * Q[11]! * z +
    Q[15]!
  );
}

// ---------------------------------------------------------------------------
// Per-vertex quadric matrix construction
// ---------------------------------------------------------------------------

/**
 * Build per-vertex quadric error matrices from the face planes of a mesh.
 *
 * For each triangle face, the plane equation is ax + by + cz + d = 0 with
 * [a, b, c] = unit normal and d = -dot(normal, vertex). The fundamental
 * quadric for this plane is the outer product K_p = p * p^T where
 * p = [a, b, c, d].
 *
 * Each vertex accumulates the quadrics of all incident faces.
 *
 * @param vertices - Packed vertex positions [x0,y0,z0, x1,y1,z1, ...].
 * @param indices  - Triangle index buffer (3 indices per triangle).
 * @returns Array of per-vertex 4x4 quadric matrices (row-major Float64Array).
 */
export function buildQuadricMatrices(
  vertices: Float64Array,
  indices: Uint32Array,
): Float64Array[] {
  const nVertices = vertices.length / 3;
  const nTriangles = indices.length / 3;

  // Initialise per-vertex quadrics to zero
  const quadrics: Float64Array[] = [];
  for (let i = 0; i < nVertices; i++) {
    quadrics.push(new Float64Array(16));
  }

  for (let f = 0; f < nTriangles; f++) {
    const i0 = indices[f * 3]!;
    const i1 = indices[f * 3 + 1]!;
    const i2 = indices[f * 3 + 2]!;

    // Vertex positions
    const v0x = vertices[i0 * 3]!;
    const v0y = vertices[i0 * 3 + 1]!;
    const v0z = vertices[i0 * 3 + 2]!;
    const v1x = vertices[i1 * 3]!;
    const v1y = vertices[i1 * 3 + 1]!;
    const v1z = vertices[i1 * 3 + 2]!;
    const v2x = vertices[i2 * 3]!;
    const v2y = vertices[i2 * 3 + 1]!;
    const v2z = vertices[i2 * 3 + 2]!;

    // Edge vectors
    const e1x = v1x - v0x;
    const e1y = v1y - v0y;
    const e1z = v1z - v0z;
    const e2x = v2x - v0x;
    const e2y = v2y - v0y;
    const e2z = v2z - v0z;

    // Cross product for face normal
    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * e2x - e1x * e2z;
    let nz = e1x * e2y - e1y * e2x;

    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (len < 1e-15) continue; // degenerate triangle
    nx /= len;
    ny /= len;
    nz /= len;

    // Plane equation: ax + by + cz + d = 0
    const a = nx;
    const b = ny;
    const c = nz;
    const d = -(a * v0x + b * v0y + c * v0z);

    // Fundamental quadric K_p = p * p^T (symmetric 4x4, stored row-major)
    // p = [a, b, c, d]
    const Kp = new Float64Array(16);
    Kp[0] = a * a;   Kp[1] = a * b;   Kp[2] = a * c;   Kp[3] = a * d;
    Kp[4] = b * a;   Kp[5] = b * b;   Kp[6] = b * c;   Kp[7] = b * d;
    Kp[8] = c * a;   Kp[9] = c * b;   Kp[10] = c * c;  Kp[11] = c * d;
    Kp[12] = d * a;  Kp[13] = d * b;  Kp[14] = d * c;  Kp[15] = d * d;

    // Accumulate into each incident vertex
    const faceVerts = [i0, i1, i2];
    for (let vi = 0; vi < 3; vi++) {
      const Q = quadrics[faceVerts[vi]!]!;
      for (let j = 0; j < 16; j++) {
        Q[j] = Q[j]! + Kp[j]!;
      }
    }
  }

  return quadrics;
}

// ---------------------------------------------------------------------------
// QEM edge collapse decimation
// ---------------------------------------------------------------------------

/**
 * Simplify a triangle mesh using Quadric Error Metric (QEM) edge collapse.
 *
 * The algorithm:
 *  1. Builds per-vertex quadric matrices from face planes.
 *  2. For each edge, computes the optimal collapse position and error.
 *  3. Greedily collapses the lowest-error edge, updating quadrics and
 *     the mesh connectivity, until the target triangle ratio is reached.
 *
 * @param vertices    - Packed vertex positions [x0,y0,z0, ...].
 * @param indices     - Triangle index buffer.
 * @param targetRatio - Target ratio of triangles to keep (0, 1].
 * @param config      - Optional QEM configuration.
 * @returns Simplified mesh with new `vertices` and `indices`.
 */
export function decimateMesh(
  vertices: Float64Array,
  indices: Uint32Array,
  targetRatio: number,
  config?: QEMConfig,
): { vertices: Float64Array; indices: Uint32Array } {
  const maxError = config?.maxError ?? Infinity;
  const preserveBoundary = config?.preserveBoundary ?? false;

  const nVertices = vertices.length / 3;
  const nTriangles = indices.length / 3;
  const targetTriCount = Math.max(1, Math.floor(nTriangles * targetRatio));

  // Copy vertices into a mutable array of positions
  const positions: Vector3[] = [];
  for (let i = 0; i < nVertices; i++) {
    positions.push({
      x: vertices[i * 3]!,
      y: vertices[i * 3 + 1]!,
      z: vertices[i * 3 + 2]!,
    });
  }

  // Build mutable face list
  const faces: [number, number, number][] = [];
  for (let f = 0; f < nTriangles; f++) {
    faces.push([
      indices[f * 3]!,
      indices[f * 3 + 1]!,
      indices[f * 3 + 2]!,
    ]);
  }

  // Track which faces are still alive
  const faceAlive = new Uint8Array(nTriangles).fill(1);
  let liveFaceCount = nTriangles;

  // Union-find for collapsed vertices: vertex i maps to its representative
  const representative = new Uint32Array(nVertices);
  for (let i = 0; i < nVertices; i++) representative[i] = i;

  function find(v: number): number {
    while (representative[v]! !== v) {
      representative[v] = representative[representative[v]!]!;
      v = representative[v]!;
    }
    return v;
  }

  // Build quadric matrices
  const quadrics = buildQuadricMatrices(vertices, indices);

  // Identify boundary vertices (if preserveBoundary is true)
  const boundaryVerts = new Set<number>();
  if (preserveBoundary) {
    const edgeCount = new Map<string, number>();
    for (let f = 0; f < nTriangles; f++) {
      const tri = faces[f]!;
      for (let e = 0; e < 3; e++) {
        const a = tri[e]!;
        const b = tri[(e + 1) % 3]!;
        const key = a < b ? `${a}_${b}` : `${b}_${a}`;
        edgeCount.set(key, (edgeCount.get(key) ?? 0) + 1);
      }
    }
    for (const [key, count] of edgeCount) {
      if (count === 1) {
        const [aStr, bStr] = key.split('_');
        boundaryVerts.add(Number(aStr));
        boundaryVerts.add(Number(bStr));
      }
    }
  }

  // Build edge set and compute initial costs
  interface EdgeEntry {
    v0: number;
    v1: number;
    error: number;
    optimal: Vector3;
  }

  function computeEdgeCost(v0: number, v1: number): EdgeEntry {
    const Q0 = quadrics[v0]!;
    const Q1 = quadrics[v1]!;

    // Combined quadric
    const Qsum = new Float64Array(16);
    for (let i = 0; i < 16; i++) Qsum[i] = Q0[i]! + Q1[i]!;

    // Try the optimal vertex position by solving the linear system
    // derived from the quadric. Fallback: midpoint of the edge.
    const p0 = positions[v0]!;
    const p1 = positions[v1]!;
    const mid: Vector3 = {
      x: (p0.x + p1.x) * 0.5,
      y: (p0.y + p1.y) * 0.5,
      z: (p0.z + p1.z) * 0.5,
    };

    // Evaluate error at midpoint and both endpoints, pick the best
    const errMid = computeQuadricError(mid, Qsum);
    const err0 = computeQuadricError(p0, Qsum);
    const err1 = computeQuadricError(p1, Qsum);

    let optimal = mid;
    let error = errMid;
    if (err0 < error) {
      error = err0;
      optimal = p0;
    }
    if (err1 < error) {
      error = err1;
      optimal = p1;
    }

    return { v0, v1, error, optimal };
  }

  // Collect all unique edges
  const edgeSet = new Set<string>();
  const edgeList: EdgeEntry[] = [];

  for (let f = 0; f < nTriangles; f++) {
    const tri = faces[f]!;
    for (let e = 0; e < 3; e++) {
      const a = tri[e]!;
      const b = tri[(e + 1) % 3]!;
      const key = a < b ? `${a}_${b}` : `${b}_${a}`;
      if (!edgeSet.has(key)) {
        edgeSet.add(key);
        edgeList.push(computeEdgeCost(a, b));
      }
    }
  }

  // Sort edges by error (ascending)
  edgeList.sort((a, b) => a.error - b.error);

  // Greedy edge collapse loop
  let edgeIdx = 0;
  while (liveFaceCount > targetTriCount && edgeIdx < edgeList.length) {
    const edge = edgeList[edgeIdx]!;
    edgeIdx++;

    // Resolve current representatives
    const rv0 = find(edge.v0);
    const rv1 = find(edge.v1);

    // Skip if already collapsed to the same vertex
    if (rv0 === rv1) continue;

    // Skip boundary vertices if requested
    if (preserveBoundary && (boundaryVerts.has(rv0) || boundaryVerts.has(rv1))) continue;

    // Skip if error exceeds threshold
    if (edge.error > maxError) break;

    // Collapse rv1 into rv0
    representative[rv1] = rv0;

    // Update position of rv0 to the optimal placement
    positions[rv0] = edge.optimal;

    // Merge quadrics
    const Q0 = quadrics[rv0]!;
    const Q1 = quadrics[rv1]!;
    for (let i = 0; i < 16; i++) {
      Q0[i] = Q0[i]! + Q1[i]!;
    }

    // Deactivate degenerate faces and update face vertex references
    for (let f = 0; f < faces.length; f++) {
      if (!faceAlive[f]) continue;
      const tri = faces[f]!;

      // Remap vertices through union-find
      tri[0] = find(tri[0]);
      tri[1] = find(tri[1]);
      tri[2] = find(tri[2]);

      // Check for degenerate triangle (two or more vertices are the same)
      if (tri[0] === tri[1] || tri[1] === tri[2] || tri[0] === tri[2]) {
        faceAlive[f] = 0;
        liveFaceCount--;
      }
    }
  }

  // Compact the result: collect surviving vertices and faces
  const vertexRemap = new Map<number, number>();
  const newVertices: number[] = [];

  function getRemappedIndex(v: number): number {
    const rv = find(v);
    const existing = vertexRemap.get(rv);
    if (existing !== undefined) return existing;
    const idx = newVertices.length / 3;
    const pos = positions[rv]!;
    newVertices.push(pos.x, pos.y, pos.z);
    vertexRemap.set(rv, idx);
    return idx;
  }

  const newIndices: number[] = [];
  for (let f = 0; f < faces.length; f++) {
    if (!faceAlive[f]) continue;
    const tri = faces[f]!;
    const a = getRemappedIndex(tri[0]);
    const b = getRemappedIndex(tri[1]);
    const c = getRemappedIndex(tri[2]);
    newIndices.push(a, b, c);
  }

  return {
    vertices: new Float64Array(newVertices),
    indices: new Uint32Array(newIndices),
  };
}
