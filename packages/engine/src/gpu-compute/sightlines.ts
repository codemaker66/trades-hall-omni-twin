/**
 * Sightline analysis: per-seat line-of-sight to a focal point.
 *
 * GPU: Cast rays in parallel for each seat toward the focal point.
 * CPU: Sequential ray-AABB intersection tests.
 *
 * Produces:
 * - Per-chair score (0 = fully blocked, 1 = clear)
 * - Overall coverage percentage
 * - Heatmap (grid-based sightline quality for visualization)
 */

import type { AnalysisItem, Point2D, RoomGeometry, SightlineResult, AABB2D } from './types'

// ─── WGSL Shader Source ─────────────────────────────────────────────────────

export const SIGHTLINE_SHADER = /* wgsl */`
  struct AABB {
    minX: f32,
    minZ: f32,
    maxX: f32,
    maxZ: f32,
  }

  struct Params {
    focalX: f32,
    focalZ: f32,
    chairCount: u32,
    obstacleCount: u32,
    gridWidth: u32,
    gridHeight: u32,
    cellSize: f32,
  }

  @group(0) @binding(0) var<uniform> params: Params;
  @group(0) @binding(1) var<storage, read> chairs: array<vec2<f32>>;
  @group(0) @binding(2) var<storage, read> obstacles: array<AABB>;
  @group(0) @binding(3) var<storage, read_write> results: array<f32>;
  @group(0) @binding(4) var<storage, read_write> heatmap: array<f32>;

  fn rayIntersectsAABB(ox: f32, oz: f32, dx: f32, dz: f32, box_: AABB) -> bool {
    var tmin: f32 = 0.0;
    var tmax: f32 = 1.0;

    if (abs(dx) > 1e-10) {
      let tx1 = (box_.minX - ox) / dx;
      let tx2 = (box_.maxX - ox) / dx;
      tmin = max(tmin, min(tx1, tx2));
      tmax = min(tmax, max(tx1, tx2));
    } else {
      if (ox < box_.minX || ox > box_.maxX) { return false; }
    }

    if (abs(dz) > 1e-10) {
      let tz1 = (box_.minZ - oz) / dz;
      let tz2 = (box_.maxZ - oz) / dz;
      tmin = max(tmin, min(tz1, tz2));
      tmax = min(tmax, max(tz1, tz2));
    } else {
      if (oz < box_.minZ || oz > box_.maxZ) { return false; }
    }

    return tmin <= tmax;
  }

  @compute @workgroup_size(64)
  fn chairSightlines(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.chairCount) { return; }

    let chairPos = chairs[i];
    let dx = params.focalX - chairPos.x;
    let dz = params.focalZ - chairPos.y;

    var blocked: bool = false;
    for (var j: u32 = 0u; j < params.obstacleCount; j++) {
      if (rayIntersectsAABB(chairPos.x, chairPos.y, dx, dz, obstacles[j])) {
        blocked = true;
        break;
      }
    }

    results[i] = select(1.0, 0.0, blocked);
  }

  @compute @workgroup_size(8, 8)
  fn heatmapSightlines(@builtin(global_invocation_id) id: vec3<u32>) {
    let col = id.x;
    let row = id.y;
    if (col >= params.gridWidth || row >= params.gridHeight) { return; }

    let cellX = (f32(col) + 0.5) * params.cellSize;
    let cellZ = (f32(row) + 0.5) * params.cellSize;
    let dx = params.focalX - cellX;
    let dz = params.focalZ - cellZ;

    var blocked: bool = false;
    for (var j: u32 = 0u; j < params.obstacleCount; j++) {
      if (rayIntersectsAABB(cellX, cellZ, dx, dz, obstacles[j])) {
        blocked = true;
        break;
      }
    }

    let idx = row * params.gridWidth + col;
    heatmap[idx] = select(1.0, 0.0, blocked);
  }
`

// ─── CPU Fallback ───────────────────────────────────────────────────────────

function rayIntersectsAABB(
  ox: number, oz: number,
  dx: number, dz: number,
  box: AABB2D,
): boolean {
  let tmin = 0
  let tmax = 1

  if (Math.abs(dx) > 1e-10) {
    const tx1 = (box.minX - ox) / dx
    const tx2 = (box.maxX - ox) / dx
    tmin = Math.max(tmin, Math.min(tx1, tx2))
    tmax = Math.min(tmax, Math.max(tx1, tx2))
  } else {
    if (ox < box.minX || ox > box.maxX) return false
  }

  if (Math.abs(dz) > 1e-10) {
    const tz1 = (box.minZ - oz) / dz
    const tz2 = (box.maxZ - oz) / dz
    tmin = Math.max(tmin, Math.min(tz1, tz2))
    tmax = Math.min(tmax, Math.max(tz1, tz2))
  } else {
    if (oz < box.minZ || oz > box.maxZ) return false
  }

  return tmin <= tmax
}

function itemToAABB(item: AnalysisItem): AABB2D {
  return {
    minX: item.x - item.halfWidth,
    minZ: item.z - item.halfDepth,
    maxX: item.x + item.halfWidth,
    maxZ: item.z + item.halfDepth,
  }
}

/**
 * CPU fallback: sightline analysis via sequential raycasting.
 */
export function analyzeSightlinesCPU(
  items: AnalysisItem[],
  focalPoint: Point2D,
  room: RoomGeometry,
  cellSize = 0.5,
): SightlineResult {
  const chairs = items.filter(i => i.isChair)
  const obstacles = items.filter(i => !i.isChair).map(itemToAABB)

  // Per-chair sightlines
  const perChair: number[] = []
  for (const chair of chairs) {
    const dx = focalPoint.x - chair.x
    const dz = focalPoint.z - chair.z
    let blocked = false
    for (const obs of obstacles) {
      if (rayIntersectsAABB(chair.x, chair.z, dx, dz, obs)) {
        blocked = true
        break
      }
    }
    perChair.push(blocked ? 0 : 1)
  }

  const coverage = chairs.length > 0
    ? perChair.reduce((sum, v) => sum + v, 0) / chairs.length
    : 1

  // Heatmap
  const heatmapWidth = Math.ceil(room.width / cellSize)
  const heatmapHeight = Math.ceil(room.depth / cellSize)
  const heatmap = new Float32Array(heatmapWidth * heatmapHeight)

  for (let row = 0; row < heatmapHeight; row++) {
    for (let col = 0; col < heatmapWidth; col++) {
      const cellX = (col + 0.5) * cellSize
      const cellZ = (row + 0.5) * cellSize
      const dx = focalPoint.x - cellX
      const dz = focalPoint.z - cellZ

      let blocked = false
      for (const obs of obstacles) {
        if (rayIntersectsAABB(cellX, cellZ, dx, dz, obs)) {
          blocked = true
          break
        }
      }
      heatmap[row * heatmapWidth + col] = blocked ? 0 : 1
    }
  }

  return { perChair, coverage, heatmap, heatmapWidth, heatmapHeight }
}

/**
 * Analyze sightlines using GPU if available, otherwise CPU fallback.
 */
export function analyzeSightlines(
  items: AnalysisItem[],
  focalPoint: Point2D,
  room: RoomGeometry,
  cellSize = 0.5,
  _gpuDevice?: unknown,
): SightlineResult {
  return analyzeSightlinesCPU(items, focalPoint, room, cellSize)
}
