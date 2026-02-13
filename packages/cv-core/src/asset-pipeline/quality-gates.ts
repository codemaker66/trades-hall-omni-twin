// ---------------------------------------------------------------------------
// CV-12: Asset Pipeline — Quality Gates
// Factory-based quality checkers, mesh quality scoring, and multi-gate
// validation for the asset pipeline.
// ---------------------------------------------------------------------------

import type {
  Mesh,
  AssetValidationResult,
  QualityGateConfig,
} from '../types.js';
import { createAssetMetadata } from './metadata.js';
import {
  validateTriangleCount,
  validateUVBounds,
  validateManifold,
  validateScale,
} from './validation.js';

// ---------------------------------------------------------------------------
// createQualityGate
// ---------------------------------------------------------------------------

/**
 * Create a quality-gate function from a `QualityGateConfig`.
 *
 * The returned function accepts a mesh and runs it against the first tier
 * whose validation constraints it satisfies (if `autoAssignTier` is true) or
 * against all tiers and reports violations for the best matching one.
 *
 * @param config Quality gate configuration describing one or more tiers.
 * @returns      A function `(mesh: Mesh) => AssetValidationResult`.
 */
export function createQualityGate(
  config: QualityGateConfig,
): (mesh: Mesh) => AssetValidationResult {
  return (mesh: Mesh): AssetValidationResult => {
    const metadata = createAssetMetadata(mesh);
    const violations: string[] = [];

    if (config.tiers.length === 0) {
      return {
        pass: true,
        violations: [],
        violationCount: 0,
        metrics: metadata,
        validatedAt: new Date().toISOString(),
      };
    }

    // If auto-assign, try each tier from highest quality to lowest.
    // Return the result for the first tier that passes — or if none pass,
    // return the violations from the most permissive (last) tier.
    if (config.autoAssignTier) {
      for (let i = 0; i < config.tiers.length; i++) {
        const tier = config.tiers[i]!;
        const result = validateAgainstTier(mesh, tier.validation, metadata);
        if (result.pass) return result;
      }

      // Fall back to the last (most permissive) tier.
      const lastTier = config.tiers[config.tiers.length - 1]!;
      return validateAgainstTier(mesh, lastTier.validation, metadata);
    }

    // Non-auto: validate against all tiers and report violations from the
    // first (strictest) tier.
    const firstTier = config.tiers[0]!;
    return validateAgainstTier(mesh, firstTier.validation, metadata);
  };
}

// ---------------------------------------------------------------------------
// scoreMeshQuality
// ---------------------------------------------------------------------------

/**
 * Score a mesh's overall quality on a 0-100 scale.
 *
 * The score is a weighted combination of:
 * - Triangle quality (aspect ratios): 40 %
 * - UV coverage: 20 %
 * - Manifoldness: 20 %
 * - Normal consistency: 20 %
 *
 * Higher scores indicate a healthier mesh that is more likely to pass
 * quality gates.
 *
 * @param mesh The mesh to score.
 * @returns    A quality score in [0, 100].
 */
export function scoreMeshQuality(mesh: Mesh): number {
  let score = 0;

  // --- Triangle quality (40 pts) ---
  // Score based on minimum aspect ratio across triangles.
  const triangleScore = computeTriangleQualityScore(mesh);
  score += triangleScore * 40;

  // --- UV coverage (20 pts) ---
  if (mesh.uvs && mesh.uvs.length > 0) {
    const { pass } = validateUVBounds(mesh.uvs);
    score += pass ? 20 : 10; // Partial credit if UVs exist but are out of range.
  }
  // No UVs => 0 for this component.

  // --- Manifoldness (20 pts) ---
  const { openEdges } = validateManifold(mesh.vertices, mesh.indices);
  if (openEdges === 0) {
    score += 20;
  } else {
    // Degrade linearly; clamp at 0.
    const totalEdges = mesh.triangleCount * 3; // upper bound on unique edges
    const ratio = totalEdges > 0 ? 1 - openEdges / totalEdges : 0;
    score += Math.max(0, ratio * 20);
  }

  // --- Normal consistency (20 pts) ---
  if (mesh.normals && mesh.normals.length > 0) {
    score += 20;
  }

  return Math.max(0, Math.min(100, Math.round(score)));
}

// ---------------------------------------------------------------------------
// checkAllGates
// ---------------------------------------------------------------------------

/**
 * Run a mesh through multiple quality gate configurations and produce a
 * combined validation result.
 *
 * The mesh passes only if it passes every gate.  All violations across
 * gates are aggregated into a single result.
 *
 * @param mesh  The mesh to validate.
 * @param gates Array of quality gate configurations to check.
 * @returns     A combined `AssetValidationResult`.
 */
export function checkAllGates(
  mesh: Mesh,
  gates: QualityGateConfig[],
): AssetValidationResult {
  const metadata = createAssetMetadata(mesh);
  const allViolations: string[] = [];
  let allPassed = true;

  for (let i = 0; i < gates.length; i++) {
    const gate = gates[i]!;
    const checker = createQualityGate(gate);
    const result = checker(mesh);
    if (!result.pass) {
      allPassed = false;
      for (const v of result.violations) {
        allViolations.push(v);
      }
    }
  }

  return {
    pass: allPassed,
    violations: allViolations,
    violationCount: allViolations.length,
    metrics: metadata,
    validatedAt: new Date().toISOString(),
  };
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

import type { AssetValidationConfig, AssetMetadata } from '../types.js';

function validateAgainstTier(
  mesh: Mesh,
  config: AssetValidationConfig,
  metadata: AssetMetadata,
): AssetValidationResult {
  const violations: string[] = [];

  // Triangle count.
  const triResult = validateTriangleCount(mesh, config.maxTriangles);
  if (!triResult.pass) {
    violations.push(
      `Triangle count ${triResult.actual} exceeds limit ${triResult.max}`,
    );
  }

  // UV check.
  if (config.requireUVs) {
    if (!mesh.uvs || mesh.uvs.length === 0) {
      violations.push('Mesh is missing UV coordinates');
    } else {
      const uvResult = validateUVBounds(mesh.uvs);
      if (!uvResult.pass) {
        violations.push(
          `${uvResult.outOfBounds} UV component(s) out of [0,1] range`,
        );
      }
    }
  }

  // Normals.
  if (config.requireNormals) {
    if (!mesh.normals || mesh.normals.length === 0) {
      violations.push('Mesh is missing vertex normals');
    }
  }

  // Manifold.
  if (config.checkManifold) {
    const mResult = validateManifold(mesh.vertices, mesh.indices);
    if (!mResult.pass) {
      violations.push(`Mesh has ${mResult.openEdges} non-manifold/open edge(s)`);
    }
  }

  // Physical scale.
  if (config.maxPhysicalDimension > 0) {
    const sResult = validateScale(mesh.vertices, {
      min: 0,
      max: config.maxPhysicalDimension,
    });
    if (!sResult.pass) {
      violations.push(
        `Mesh max dimension ${sResult.actualSize.toFixed(2)}m exceeds limit ${config.maxPhysicalDimension}m`,
      );
    }
  }

  return {
    pass: violations.length === 0,
    violations,
    violationCount: violations.length,
    metrics: metadata,
    validatedAt: new Date().toISOString(),
  };
}

/**
 * Compute a normalised [0,1] triangle quality score for the mesh.
 *
 * Quality is measured by the average "aspect ratio quality" of each triangle,
 * defined as the ratio of the shortest edge to the longest edge. A perfectly
 * equilateral triangle scores 1.0.
 */
function computeTriangleQualityScore(mesh: Mesh): number {
  const verts = mesh.vertices;
  const indices = mesh.indices;
  const triCount = Math.floor(indices.length / 3);

  if (triCount === 0) return 0;

  let totalQuality = 0;

  for (let t = 0; t < triCount; t++) {
    const i0 = indices[t * 3]!;
    const i1 = indices[t * 3 + 1]!;
    const i2 = indices[t * 3 + 2]!;

    const e0 = edgeLengthSq(verts, i0, i1);
    const e1 = edgeLengthSq(verts, i1, i2);
    const e2 = edgeLengthSq(verts, i2, i0);

    const longest = Math.sqrt(Math.max(e0, e1, e2));
    const shortest = Math.sqrt(Math.min(e0, e1, e2));

    if (longest > 1e-15) {
      totalQuality += shortest / longest;
    }
  }

  return totalQuality / triCount;
}

function edgeLengthSq(verts: Float64Array, a: number, b: number): number {
  const dx = verts[b * 3]! - verts[a * 3]!;
  const dy = verts[b * 3 + 1]! - verts[a * 3 + 1]!;
  const dz = verts[b * 3 + 2]! - verts[a * 3 + 2]!;
  return dx * dx + dy * dy + dz * dz;
}
