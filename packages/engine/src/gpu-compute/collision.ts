/**
 * Parallel collision detection.
 *
 * GPU: Check all N*(N-1)/2 furniture pairs simultaneously.
 * CPU: Sequential O(N^2) pair check (same results, slower for large N).
 *
 * Both use AABB intersection — sufficient for axis-aligned furniture.
 */

import type { AABB2D, AnalysisItem, CollisionResult } from './types'

// ─── WGSL Shader Source ─────────────────────────────────────────────────────

export const COLLISION_SHADER = /* wgsl */`
  struct AABB {
    minX: f32,
    minZ: f32,
    maxX: f32,
    maxZ: f32,
  }

  @group(0) @binding(0) var<storage, read> boxes: array<AABB>;
  @group(0) @binding(1) var<storage, read_write> collisions: array<atomic<u32>>;
  @group(0) @binding(2) var<uniform> count: u32;

  fn aabbOverlap(a: AABB, b: AABB) -> bool {
    return a.minX < b.maxX && a.maxX > b.minX
        && a.minZ < b.maxZ && a.maxZ > b.minZ;
  }

  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= count) { return; }

    let boxI = boxes[i];
    for (var j = i + 1u; j < count; j++) {
      if (aabbOverlap(boxI, boxes[j])) {
        // Set bit j in collisions[i] and bit i in collisions[j]
        atomicOr(&collisions[i], 1u << (j % 32u));
        atomicOr(&collisions[j], 1u << (i % 32u));
      }
    }
  }
`

// ─── CPU Fallback ───────────────────────────────────────────────────────────

function itemToAABB(item: AnalysisItem): AABB2D {
  return {
    minX: item.x - item.halfWidth,
    minZ: item.z - item.halfDepth,
    maxX: item.x + item.halfWidth,
    maxZ: item.z + item.halfDepth,
  }
}

function aabbOverlap(a: AABB2D, b: AABB2D): boolean {
  return a.minX < b.maxX && a.maxX > b.minX
      && a.minZ < b.maxZ && a.maxZ > b.minZ
}

/**
 * CPU fallback: check all pairs for AABB overlap.
 * O(N^2) but correct reference implementation.
 */
export function detectCollisionsCPU(items: AnalysisItem[]): CollisionResult {
  const n = items.length
  const boxes = items.map(itemToAABB)
  const pairs: Array<[number, number]> = []
  const colliding = new Array<boolean>(n).fill(false)

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (aabbOverlap(boxes[i]!, boxes[j]!)) {
        pairs.push([i, j])
        colliding[i] = true
        colliding[j] = true
      }
    }
  }

  return { pairs, count: pairs.length, colliding }
}

/**
 * Detect collisions using GPU if available, otherwise CPU fallback.
 * The GPU path is a placeholder — actual WebGPU dispatch requires
 * a GPUDevice instance (browser-only).
 */
export function detectCollisions(
  items: AnalysisItem[],
  _gpuDevice?: unknown,
): CollisionResult {
  // GPU path would dispatch COLLISION_SHADER here
  // For now, always use CPU fallback
  return detectCollisionsCPU(items)
}
