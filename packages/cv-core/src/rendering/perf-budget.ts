// ---------------------------------------------------------------------------
// CV-7: Performance Budget — VRAM estimation, budget checking, and
// adaptive quality control for real-time rendering.
// ---------------------------------------------------------------------------

import type { Mesh, PerfBudget } from '../types.js';

// ---------------------------------------------------------------------------
// createPerfBudget
// ---------------------------------------------------------------------------

/**
 * Create a {@link PerfBudget} with the given hard limits.
 *
 * @param maxDrawCalls Maximum draw calls per frame.
 * @param maxTriangles Maximum triangle count per frame.
 * @param maxVRAM_MB   Maximum VRAM usage in megabytes.
 * @returns            A new {@link PerfBudget}.
 */
export function createPerfBudget(
  maxDrawCalls: number,
  maxTriangles: number,
  maxVRAM_MB: number,
): PerfBudget {
  return { maxDrawCalls, maxTriangles, maxVRAM_MB };
}

// ---------------------------------------------------------------------------
// estimateVRAM
// ---------------------------------------------------------------------------

/**
 * Estimate the VRAM footprint of a mesh plus a square RGBA texture.
 *
 * Geometry:
 *   - Vertex positions: vertexCount * 3 * 4 bytes (Float32 on GPU).
 *   - Normals:          vertexCount * 3 * 4 bytes (if present).
 *   - UVs:              vertexCount * 2 * 4 bytes (if present).
 *   - Index buffer:     triangleCount * 3 * 4 bytes (Uint32).
 *
 * Texture:
 *   - textureSize * textureSize * 4 bytes (RGBA8) plus ~33 % for mip-chain.
 *
 * @param mesh        The mesh to estimate.
 * @param textureSize Texture edge resolution in pixels (assumes square RGBA8).
 * @returns           Estimated VRAM in megabytes.
 */
export function estimateVRAM(mesh: Mesh, textureSize: number): number {
  const BYTES_PER_FLOAT32 = 4;
  const BYTES_PER_UINT32 = 4;
  const MB = 1024 * 1024;

  let geometryBytes = 0;

  // Positions (always present).
  geometryBytes += mesh.vertexCount * 3 * BYTES_PER_FLOAT32;
  // Normals.
  if (mesh.normals) {
    geometryBytes += mesh.vertexCount * 3 * BYTES_PER_FLOAT32;
  }
  // UVs.
  if (mesh.uvs) {
    geometryBytes += mesh.vertexCount * 2 * BYTES_PER_FLOAT32;
  }
  // Index buffer.
  geometryBytes += mesh.triangleCount * 3 * BYTES_PER_UINT32;

  // Texture: RGBA8 base + ~33% for full mip-chain.
  const baseTextureBytes = textureSize * textureSize * 4;
  const textureBytes = baseTextureBytes * 1.33;

  return (geometryBytes + textureBytes) / MB;
}

// ---------------------------------------------------------------------------
// checkBudget
// ---------------------------------------------------------------------------

/**
 * Check current resource usage against a performance budget.
 *
 * Returns whether all three metrics (draw calls, triangles, VRAM) are within
 * their limits, plus per-metric utilisation ratios in [0, 1+].
 *
 * @param budget     The {@link PerfBudget} to check against.
 * @param drawCalls  Current draw-call count.
 * @param triangles  Current triangle count.
 * @param vramMB     Current VRAM usage in megabytes.
 * @returns          Budget check result with utilisation ratios.
 */
export function checkBudget(
  budget: PerfBudget,
  drawCalls: number,
  triangles: number,
  vramMB: number,
): {
  withinBudget: boolean;
  utilization: { drawCalls: number; triangles: number; vram: number };
} {
  const utilDrawCalls =
    budget.maxDrawCalls > 0 ? drawCalls / budget.maxDrawCalls : 0;
  const utilTriangles =
    budget.maxTriangles > 0 ? triangles / budget.maxTriangles : 0;
  const utilVram =
    budget.maxVRAM_MB > 0 ? vramMB / budget.maxVRAM_MB : 0;

  const withinBudget =
    utilDrawCalls <= 1 && utilTriangles <= 1 && utilVram <= 1;

  return {
    withinBudget,
    utilization: {
      drawCalls: utilDrawCalls,
      triangles: utilTriangles,
      vram: utilVram,
    },
  };
}

// ---------------------------------------------------------------------------
// adaptiveQuality
// ---------------------------------------------------------------------------

/**
 * Compute a new quality level based on current FPS relative to a target.
 *
 * Uses a simple proportional controller:
 *   - If currentFPS < targetFPS, reduce quality (clamp to 0).
 *   - If currentFPS >= targetFPS, increase quality (clamp to 1).
 *
 * The step size is proportional to the relative deviation, clamped to
 * +/- 0.1 per call to avoid oscillation.
 *
 * @param budget         Performance budget (used for its optional `targetFPS`).
 * @param currentFPS     Current measured frame rate.
 * @param targetFPS      Target frame rate (overrides budget.targetFPS).
 * @param currentQuality Current quality level in [0, 1].
 * @returns              Adjusted quality level in [0, 1].
 */
export function adaptiveQuality(
  _budget: PerfBudget,
  currentFPS: number,
  targetFPS: number,
  currentQuality: number,
): number {
  if (targetFPS <= 0) return currentQuality;

  // Relative error: positive when we have headroom, negative when too slow.
  const relativeError = (currentFPS - targetFPS) / targetFPS;

  // Proportional gain.
  const gain = 0.25;
  let step = relativeError * gain;

  // Clamp step to avoid large jumps.
  const maxStep = 0.1;
  step = Math.max(-maxStep, Math.min(maxStep, step));

  // Apply and clamp to [0, 1].
  const newQuality = currentQuality + step;
  return Math.max(0, Math.min(1, newQuality));
}
