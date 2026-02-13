// ---------------------------------------------------------------------------
// CV-6: Mesh Processing — UV Unwrapping & Atlas Packing
// ---------------------------------------------------------------------------

import type { UVChart, UVAtlas } from '../types.js';

// ---------------------------------------------------------------------------
// Angle-based flattening
// ---------------------------------------------------------------------------

/**
 * Compute UV coordinates using a simplified angle-based flattening approach.
 *
 * For each triangle, this maps 3D vertex positions into a local 2D coordinate
 * frame that preserves angles (conformal-like). The resulting UVs are a
 * single-chart parameterisation suitable for simple meshes; for complex
 * meshes this should be combined with chart segmentation.
 *
 * Algorithm:
 *  1. For each triangle, compute local 2D coordinates by flattening the
 *     triangle into its own plane using edge lengths and the cosine rule.
 *  2. Average the per-triangle UV contributions at each vertex.
 *
 * @param vertices - Packed vertex positions [x0,y0,z0, x1,y1,z1, ...].
 * @param indices  - Triangle index buffer (3 indices per triangle).
 * @returns Float64Array of UV coordinates [u0,v0, u1,v1, ...], length = (vertices.length / 3) * 2.
 */
export function computeAngleBasedFlattening(
  vertices: Float64Array,
  indices: Uint32Array,
): Float64Array {
  const nVertices = vertices.length / 3;
  const nTriangles = indices.length / 3;

  // Accumulate UV per vertex (weighted by triangle area)
  const uvSum = new Float64Array(nVertices * 2);
  const weightSum = new Float64Array(nVertices);

  for (let f = 0; f < nTriangles; f++) {
    const i0 = indices[f * 3]!;
    const i1 = indices[f * 3 + 1]!;
    const i2 = indices[f * 3 + 2]!;

    // 3D positions
    const ax = vertices[i0 * 3]!;
    const ay = vertices[i0 * 3 + 1]!;
    const az = vertices[i0 * 3 + 2]!;
    const bx = vertices[i1 * 3]!;
    const by = vertices[i1 * 3 + 1]!;
    const bz = vertices[i1 * 3 + 2]!;
    const cx = vertices[i2 * 3]!;
    const cy = vertices[i2 * 3 + 1]!;
    const cz = vertices[i2 * 3 + 2]!;

    // Edge vectors
    const e0x = bx - ax;
    const e0y = by - ay;
    const e0z = bz - az;
    const e1x = cx - ax;
    const e1y = cy - ay;
    const e1z = cz - az;

    // Edge lengths
    const lenE0 = Math.sqrt(e0x * e0x + e0y * e0y + e0z * e0z);
    const lenE1 = Math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z);

    if (lenE0 < 1e-12 || lenE1 < 1e-12) continue;

    // Angle at vertex A using dot product
    const dot = e0x * e1x + e0y * e1y + e0z * e1z;
    const cosAngle = Math.max(-1, Math.min(1, dot / (lenE0 * lenE1)));
    const angle = Math.acos(cosAngle);

    // Flatten triangle into 2D:
    //   A = (0, 0)
    //   B = (lenE0, 0)
    //   C = (lenE1 * cos(angle), lenE1 * sin(angle))
    const u0 = 0;
    const v0 = 0;
    const u1 = lenE0;
    const v1 = 0;
    const u2 = lenE1 * Math.cos(angle);
    const v2 = lenE1 * Math.sin(angle);

    // Triangle area (2D) as weight
    const area = 0.5 * Math.abs(u1 * v2 - u2 * v1);
    const w = Math.max(area, 1e-12);

    // Accumulate
    uvSum[i0 * 2] = uvSum[i0 * 2]! + u0 * w;
    uvSum[i0 * 2 + 1] = uvSum[i0 * 2 + 1]! + v0 * w;
    weightSum[i0] = weightSum[i0]! + w;

    uvSum[i1 * 2] = uvSum[i1 * 2]! + u1 * w;
    uvSum[i1 * 2 + 1] = uvSum[i1 * 2 + 1]! + v1 * w;
    weightSum[i1] = weightSum[i1]! + w;

    uvSum[i2 * 2] = uvSum[i2 * 2]! + u2 * w;
    uvSum[i2 * 2 + 1] = uvSum[i2 * 2 + 1]! + v2 * w;
    weightSum[i2] = weightSum[i2]! + w;
  }

  // Normalise accumulated UVs
  const uvs = new Float64Array(nVertices * 2);
  for (let i = 0; i < nVertices; i++) {
    const w = weightSum[i]!;
    if (w > 0) {
      uvs[i * 2] = uvSum[i * 2]! / w;
      uvs[i * 2 + 1] = uvSum[i * 2 + 1]! / w;
    }
  }

  // Normalise UVs into [0, 1] range
  let uMin = Infinity;
  let uMax = -Infinity;
  let vMin = Infinity;
  let vMax = -Infinity;

  for (let i = 0; i < nVertices; i++) {
    const u = uvs[i * 2]!;
    const v = uvs[i * 2 + 1]!;
    if (u < uMin) uMin = u;
    if (u > uMax) uMax = u;
    if (v < vMin) vMin = v;
    if (v > vMax) vMax = v;
  }

  const uRange = uMax - uMin;
  const vRange = vMax - vMin;
  if (uRange > 1e-12 && vRange > 1e-12) {
    for (let i = 0; i < nVertices; i++) {
      uvs[i * 2] = (uvs[i * 2]! - uMin) / uRange;
      uvs[i * 2 + 1] = (uvs[i * 2 + 1]! - vMin) / vRange;
    }
  }

  return uvs;
}

// ---------------------------------------------------------------------------
// Atlas packing (shelf packing)
// ---------------------------------------------------------------------------

/**
 * Pack UV charts into a square atlas using a shelf-packing algorithm.
 *
 * Charts are sorted by descending height, then placed left-to-right on
 * shelves. When a chart does not fit on the current shelf, a new shelf is
 * opened. Each chart's UV bounding rect is used for placement.
 *
 * @param charts    - Array of UV charts with bounding rectangles.
 * @param atlasSize - Output atlas resolution (width = height, in pixels).
 * @returns {@link UVAtlas} with packed chart positions and efficiency metric.
 */
export function packAtlas(charts: UVChart[], atlasSize: number): UVAtlas {
  if (charts.length === 0) {
    return {
      charts: [],
      width: atlasSize,
      height: atlasSize,
      uvs: new Float64Array(0),
      efficiency: 0,
      chartCount: 0,
    };
  }

  // Sort charts by descending bounding-rect height for better shelf utilisation
  const sortedIndices = charts.map((_, i) => i);
  sortedIndices.sort((a, b) => {
    const hA = charts[a]!.boundingRect[3]! - charts[a]!.boundingRect[1]!;
    const hB = charts[b]!.boundingRect[3]! - charts[b]!.boundingRect[1]!;
    return hB - hA;
  });

  // Shelf packing state
  let shelfX = 0;
  let shelfY = 0;
  let shelfHeight = 0;
  let totalUsedArea = 0;

  // Track placed chart bounding rects (normalised to [0,1])
  const placedCharts: UVChart[] = [];

  for (let si = 0; si < sortedIndices.length; si++) {
    const idx = sortedIndices[si]!;
    const chart = charts[idx]!;
    const br = chart.boundingRect;
    const chartW = br[2]! - br[0]!;
    const chartH = br[3]! - br[1]!;

    // Scale chart dimensions from UV space to pixel space
    const pixelW = chartW * atlasSize;
    const pixelH = chartH * atlasSize;

    // Check if chart fits on current shelf
    if (shelfX + pixelW > atlasSize) {
      // Start a new shelf
      shelfY += shelfHeight;
      shelfX = 0;
      shelfHeight = 0;
    }

    // Place the chart
    const placedUMin = shelfX / atlasSize;
    const placedVMin = shelfY / atlasSize;
    const placedUMax = (shelfX + pixelW) / atlasSize;
    const placedVMax = (shelfY + pixelH) / atlasSize;

    const placedRect = new Float64Array(4);
    placedRect[0] = placedUMin;
    placedRect[1] = placedVMin;
    placedRect[2] = placedUMax;
    placedRect[3] = placedVMax;

    placedCharts.push({
      triangleIndices: chart.triangleIndices,
      boundingRect: placedRect,
      uvArea: chart.uvArea,
      worldArea: chart.worldArea,
    });

    totalUsedArea += pixelW * pixelH;
    shelfX += pixelW;
    if (pixelH > shelfHeight) shelfHeight = pixelH;
  }

  const totalAtlasArea = atlasSize * atlasSize;
  const efficiency = totalAtlasArea > 0 ? totalUsedArea / totalAtlasArea : 0;

  return {
    charts: placedCharts,
    width: atlasSize,
    height: atlasSize,
    uvs: new Float64Array(0), // Caller should remap per-vertex UVs using placed bounding rects
    efficiency,
    chartCount: placedCharts.length,
  };
}

// ---------------------------------------------------------------------------
// Chart boundary length
// ---------------------------------------------------------------------------

/**
 * Compute the perimeter (boundary length) of a UV chart's bounding rectangle.
 *
 * @param chart - UV chart with bounding rectangle [uMin, vMin, uMax, vMax].
 * @returns Perimeter length in UV space.
 */
export function chartBoundaryLength(chart: UVChart): number {
  const br = chart.boundingRect;
  const width = br[2]! - br[0]!;
  const height = br[3]! - br[1]!;
  return 2 * (width + height);
}
