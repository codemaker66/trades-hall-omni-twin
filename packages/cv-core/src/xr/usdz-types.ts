// ---------------------------------------------------------------------------
// CV-11: XR — USDZ Asset Metadata & Validation
// Helpers for creating, validating, and estimating USDZ assets used by
// Apple AR Quick Look and similar XR viewers.
// ---------------------------------------------------------------------------

import type { USDZAssetMeta } from '../types.js';

// ---------------------------------------------------------------------------
// Constants — Apple AR Quick Look recommended limits
// ---------------------------------------------------------------------------

/** Maximum recommended triangle count for AR Quick Look. */
const MAX_TRIANGLES = 100_000;

/** Maximum recommended texture dimension (width or height) in pixels. */
const MAX_TEXTURE_SIZE = 2048;

/** Maximum recommended USDZ file size in bytes (25 MB). */
const MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024;

// ---------------------------------------------------------------------------
// createUSDZMeta
// ---------------------------------------------------------------------------

/**
 * Create a `USDZAssetMeta` descriptor from basic mesh stats.
 *
 * File size, texture memory, and physical dimensions are estimated from the
 * provided triangle count and texture size.  This is a lightweight factory
 * intended for quick prototyping and pre-validation before an actual USDZ
 * export.
 *
 * @param name          Human-readable asset name (used as fileName).
 * @param triangleCount Total number of triangles in the mesh.
 * @param textureSize   Maximum texture dimension in pixels.
 * @returns             A `USDZAssetMeta` with estimated values.
 */
export function createUSDZMeta(
  name: string,
  triangleCount: number,
  textureSize: number,
): USDZAssetMeta {
  const estimatedFileSize = estimateUSDZFileSize(
    triangleCount,
    textureSize,
    false,
  );

  // Estimate texture memory: assume RGBA8 for each texel.
  const textureMemory = textureSize * textureSize * 4;

  return {
    fileName: `${name}.usdz`,
    fileSize: estimatedFileSize,
    meshCount: 1,
    materialCount: 1,
    triangleCount,
    textureMemory,
    physicalDimensions: { x: 1, y: 1, z: 1 },
    hasAnimations: false,
    schemaVersion: '1.0',
  };
}

// ---------------------------------------------------------------------------
// validateUSDZConstraints
// ---------------------------------------------------------------------------

/**
 * Validate a USDZ asset descriptor against Apple AR Quick Look recommended
 * constraints.
 *
 * Checks:
 * - Triangle count <= 100 000
 * - Texture dimensions <= 2048 px
 * - File size <= 25 MB
 *
 * @param meta The USDZ asset metadata to validate.
 * @returns    An object with a `valid` flag and a list of human-readable
 *             `issues` (empty when valid).
 */
export function validateUSDZConstraints(
  meta: USDZAssetMeta,
): { valid: boolean; issues: string[] } {
  const issues: string[] = [];

  if (meta.triangleCount > MAX_TRIANGLES) {
    issues.push(
      `Triangle count ${meta.triangleCount} exceeds maximum ${MAX_TRIANGLES}`,
    );
  }

  // Derive the texture dimension from texture memory (assume RGBA8).
  const texDim = Math.sqrt(meta.textureMemory / 4);
  if (texDim > MAX_TEXTURE_SIZE) {
    issues.push(
      `Texture dimension ~${Math.round(texDim)}px exceeds maximum ${MAX_TEXTURE_SIZE}px`,
    );
  }

  if (meta.fileSize > MAX_FILE_SIZE_BYTES) {
    const sizeMB = (meta.fileSize / (1024 * 1024)).toFixed(1);
    issues.push(
      `File size ${sizeMB} MB exceeds maximum ${MAX_FILE_SIZE_BYTES / (1024 * 1024)} MB`,
    );
  }

  return { valid: issues.length === 0, issues };
}

// ---------------------------------------------------------------------------
// estimateUSDZFileSize
// ---------------------------------------------------------------------------

/**
 * Estimate the USDZ file size in bytes from mesh and texture parameters.
 *
 * The estimate is based on:
 * - Vertex data: ~32 bytes per vertex (position + normal + UV).
 * - Index data: 4 bytes per index (3 per triangle).
 * - Texture: RGBA8 compressed with a 4:1 ratio heuristic.
 * - Animation overhead: +20 % if animations are present.
 * - USDZ container overhead: +10 %.
 *
 * The result is intentionally conservative (over-estimates) so that assets
 * which pass validation are virtually certain to meet real-world limits.
 *
 * @param triangleCount Number of triangles.
 * @param textureSize   Maximum texture dimension (width or height, pixels).
 * @param hasAnimation  Whether the asset includes animation data.
 * @returns             Estimated file size in bytes.
 */
export function estimateUSDZFileSize(
  triangleCount: number,
  textureSize: number,
  hasAnimation: boolean,
): number {
  // Heuristic: each triangle has ~3 unique vertices on average (no sharing).
  // Real meshes share vertices so this is conservative.
  const estimatedVertices = triangleCount * 3;

  // Vertex data: position (12) + normal (12) + UV (8) = 32 bytes.
  const vertexBytes = estimatedVertices * 32;

  // Index data: 3 uint32 indices per triangle.
  const indexBytes = triangleCount * 3 * 4;

  // Texture: RGBA8 = 4 bytes/texel, compressed ~4:1.
  const rawTexBytes = textureSize * textureSize * 4;
  const compressedTexBytes = rawTexBytes / 4;

  let total = vertexBytes + indexBytes + compressedTexBytes;

  // Animation overhead.
  if (hasAnimation) {
    total *= 1.2;
  }

  // USDZ container overhead (~10 %).
  total *= 1.1;

  return Math.ceil(total);
}
