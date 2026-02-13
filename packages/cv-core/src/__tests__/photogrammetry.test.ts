import { describe, it, expect } from 'vitest';
import {
  computeHomography,
  ransacFundamental,
  triangulatePoint,
  eightPointAlgorithm,
  bundleAdjustment,
  computeReprojectionError,
  projectPoint,
  planCapturePositions,
  computeCoverage,
  optimizeCapturePath,
} from '../photogrammetry/index.js';
import type {
  Vec2,
  Vector3,
  CameraIntrinsics,
  CameraExtrinsics,
  Match,
  BundleAdjustmentConfig,
} from '../types.js';
import { createPRNG } from '../types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Apply a 3x3 column-major homography to a 2D point (homogeneous). */
function applyHomography(H: Float64Array, p: Vec2): Vec2 {
  const x = H[0]! * p.x + H[3]! * p.y + H[6]!;
  const y = H[1]! * p.x + H[4]! * p.y + H[7]!;
  const w = H[2]! * p.x + H[5]! * p.y + H[8]!;
  return { x: x / w, y: y / w };
}

/** Identity 3x3 rotation (column-major). */
function identityR(): Float64Array {
  const R = new Float64Array(9);
  R[0] = 1;
  R[4] = 1;
  R[8] = 1;
  return R;
}

/** Build a 3x4 row-major projection matrix from K * [R | t]. */
function buildProjectionMatrix(
  intrinsics: CameraIntrinsics,
  extrinsics: CameraExtrinsics,
): Float64Array {
  const { fx, fy, cx, cy } = intrinsics;
  const { R, t } = extrinsics;

  // K is 3x3 row-major:
  //  fx  0  cx
  //   0 fy  cy
  //   0  0   1
  // R is column-major 3x3: element(row,col) = R[col*3+row]
  // [R|t] is 3x4 row-major

  // KR row i, col j = sum_k K[i][k] * R(k,j) where R(k,j) = R[j*3+k]
  const P = new Float64Array(12);

  // Row 0 of P = [fx, 0, cx] * [R|t]
  for (let j = 0; j < 3; j++) {
    P[j] = fx * R[j * 3]! + cx * R[j * 3 + 2]!;
  }
  P[3] = fx * t.x + cx * t.z;

  // Row 1 of P = [0, fy, cy] * [R|t]
  for (let j = 0; j < 3; j++) {
    P[4 + j] = fy * R[j * 3 + 1]! + cy * R[j * 3 + 2]!;
  }
  P[7] = fy * t.y + cy * t.z;

  // Row 2 of P = [0, 0, 1] * [R|t]
  for (let j = 0; j < 3; j++) {
    P[8 + j] = R[j * 3 + 2]!;
  }
  P[11] = t.z;

  return P;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('computeHomography', () => {
  it('maps source points to destination with low error', () => {
    // Simple known homography: a translation of (10, 20)
    const src: Vec2[] = [
      { x: 0, y: 0 },
      { x: 100, y: 0 },
      { x: 100, y: 100 },
      { x: 0, y: 100 },
    ];
    const dst: Vec2[] = src.map((p) => ({ x: p.x + 10, y: p.y + 20 }));

    const H = computeHomography(src, dst);
    expect(H).toBeInstanceOf(Float64Array);
    expect(H.length).toBe(9);

    for (let i = 0; i < src.length; i++) {
      const mapped = applyHomography(H, src[i]!);
      expect(mapped.x).toBeCloseTo(dst[i]!.x, 1);
      expect(mapped.y).toBeCloseTo(dst[i]!.y, 1);
    }
  });

  it('recovers a scaling + translation homography', () => {
    const src: Vec2[] = [
      { x: 0, y: 0 },
      { x: 50, y: 0 },
      { x: 50, y: 50 },
      { x: 0, y: 50 },
    ];
    // Scale by 2, shift by (5, 10)
    const dst: Vec2[] = src.map((p) => ({ x: p.x * 2 + 5, y: p.y * 2 + 10 }));

    const H = computeHomography(src, dst);
    for (let i = 0; i < src.length; i++) {
      const mapped = applyHomography(H, src[i]!);
      expect(mapped.x).toBeCloseTo(dst[i]!.x, 1);
      expect(mapped.y).toBeCloseTo(dst[i]!.y, 1);
    }
  });

  it('throws with fewer than 4 correspondences', () => {
    const src: Vec2[] = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
    ];
    expect(() => computeHomography(src, src)).toThrow();
  });
});

describe('triangulatePoint', () => {
  it('recovers a known 3D point from two views', () => {
    // Camera 1: identity at the origin, Camera 2: translated 1 unit along X
    const intrinsics: CameraIntrinsics = {
      fx: 500, fy: 500, cx: 320, cy: 240, width: 640, height: 480,
    };

    const cam1: CameraExtrinsics = { R: identityR(), t: { x: 0, y: 0, z: 0 } };
    const cam2: CameraExtrinsics = { R: identityR(), t: { x: -1, y: 0, z: 0 } };

    const P1 = buildProjectionMatrix(intrinsics, cam1);
    const P2 = buildProjectionMatrix(intrinsics, cam2);

    // Known 3D point
    const point3D: Vector3 = { x: 0.5, y: 0.3, z: 5.0 };

    // Project to each image
    const proj1 = projectPoint(point3D, intrinsics, cam1);
    const proj2 = projectPoint(point3D, intrinsics, cam2);

    const recovered = triangulatePoint(P1, P2, proj1, proj2);

    expect(recovered.x).toBeCloseTo(point3D.x, 1);
    expect(recovered.y).toBeCloseTo(point3D.y, 1);
    expect(recovered.z).toBeCloseTo(point3D.z, 1);
  });

  it('works for a point along the camera axis', () => {
    const intrinsics: CameraIntrinsics = {
      fx: 500, fy: 500, cx: 320, cy: 240, width: 640, height: 480,
    };
    const cam1: CameraExtrinsics = { R: identityR(), t: { x: 0, y: 0, z: 0 } };
    const cam2: CameraExtrinsics = { R: identityR(), t: { x: -2, y: 0, z: 0 } };

    const P1 = buildProjectionMatrix(intrinsics, cam1);
    const P2 = buildProjectionMatrix(intrinsics, cam2);

    const point3D: Vector3 = { x: 1, y: 0, z: 10 };
    const proj1 = projectPoint(point3D, intrinsics, cam1);
    const proj2 = projectPoint(point3D, intrinsics, cam2);

    const recovered = triangulatePoint(P1, P2, proj1, proj2);

    expect(recovered.x).toBeCloseTo(point3D.x, 0);
    expect(recovered.y).toBeCloseTo(point3D.y, 0);
    expect(recovered.z).toBeCloseTo(point3D.z, 0);
  });
});

describe('projectPoint + computeReprojectionError', () => {
  it('projects a point in front of the camera to valid pixel coords', () => {
    const intrinsics: CameraIntrinsics = {
      fx: 500, fy: 500, cx: 320, cy: 240, width: 640, height: 480,
    };
    const extrinsics: CameraExtrinsics = { R: identityR(), t: { x: 0, y: 0, z: 0 } };

    const p3d: Vector3 = { x: 0, y: 0, z: 5 };
    const result = projectPoint(p3d, intrinsics, extrinsics);

    // Should project to the principal point since point is on the optical axis
    expect(result.x).toBeCloseTo(320, 5);
    expect(result.y).toBeCloseTo(240, 5);
  });

  it('reprojection error is zero when observed matches projection', () => {
    const intrinsics: CameraIntrinsics = {
      fx: 500, fy: 500, cx: 320, cy: 240, width: 640, height: 480,
    };
    const extrinsics: CameraExtrinsics = { R: identityR(), t: { x: 0, y: 0, z: 0 } };

    const p3d: Vector3 = { x: 1, y: 2, z: 10 };
    const projected = projectPoint(p3d, intrinsics, extrinsics);
    const error = computeReprojectionError(p3d, { intrinsics, extrinsics }, projected);

    expect(error).toBeCloseTo(0, 10);
  });

  it('reprojection error is non-zero for perturbed observation', () => {
    const intrinsics: CameraIntrinsics = {
      fx: 500, fy: 500, cx: 320, cy: 240, width: 640, height: 480,
    };
    const extrinsics: CameraExtrinsics = { R: identityR(), t: { x: 0, y: 0, z: 0 } };
    const p3d: Vector3 = { x: 1, y: 2, z: 10 };
    const projected = projectPoint(p3d, intrinsics, extrinsics);
    const observed: Vec2 = { x: projected.x + 5, y: projected.y - 3 };
    const error = computeReprojectionError(p3d, { intrinsics, extrinsics }, observed);

    expect(error).toBeCloseTo(Math.sqrt(25 + 9), 5);
  });
});

describe('eightPointAlgorithm', () => {
  it('produces a 3x3 matrix from 8+ correspondences', () => {
    // Generate synthetic corresponding points from two cameras
    const intrinsics: CameraIntrinsics = {
      fx: 500, fy: 500, cx: 320, cy: 240, width: 640, height: 480,
    };
    const cam1: CameraExtrinsics = { R: identityR(), t: { x: 0, y: 0, z: 0 } };
    const cam2: CameraExtrinsics = { R: identityR(), t: { x: -1, y: 0, z: 0 } };

    const points3D: Vector3[] = [
      { x: -2, y: -1, z: 8 },
      { x: 2, y: -1, z: 10 },
      { x: 2, y: 1, z: 12 },
      { x: -2, y: 1, z: 9 },
      { x: 0, y: 0, z: 7 },
      { x: 1, y: -2, z: 11 },
      { x: -1, y: 2, z: 6 },
      { x: 3, y: 0, z: 15 },
    ];

    const pts1: Vec2[] = points3D.map((p) => projectPoint(p, intrinsics, cam1));
    const pts2: Vec2[] = points3D.map((p) => projectPoint(p, intrinsics, cam2));

    const F = eightPointAlgorithm(pts1, pts2);

    expect(F).toBeInstanceOf(Float64Array);
    expect(F.length).toBe(9);

    // F should satisfy x2^T F x1 ~ 0 for all correspondences
    for (let i = 0; i < pts1.length; i++) {
      const p1 = pts1[i]!;
      const p2 = pts2[i]!;
      // x2^T F x1 in column-major: F is column-major so F[col*3+row]
      const val =
        p2.x * (F[0]! * p1.x + F[3]! * p1.y + F[6]!) +
        p2.y * (F[1]! * p1.x + F[4]! * p1.y + F[7]!) +
        (F[2]! * p1.x + F[5]! * p1.y + F[8]!);
      // The epipolar constraint should be close to zero (normalised by F norm)
      let fNorm = 0;
      for (let j = 0; j < 9; j++) fNorm += F[j]! * F[j]!;
      fNorm = Math.sqrt(fNorm);
      expect(Math.abs(val) / fNorm).toBeLessThan(5);
    }
  });
});

describe('ransacFundamental', () => {
  it('finds inliers among clean correspondences', () => {
    const intrinsics: CameraIntrinsics = {
      fx: 500, fy: 500, cx: 320, cy: 240, width: 640, height: 480,
    };
    const cam1: CameraExtrinsics = { R: identityR(), t: { x: 0, y: 0, z: 0 } };
    const cam2: CameraExtrinsics = { R: identityR(), t: { x: -1, y: 0, z: 0 } };

    const points3D: Vector3[] = [];
    const rng = createPRNG(99);
    for (let i = 0; i < 20; i++) {
      points3D.push({
        x: (rng() - 0.5) * 6,
        y: (rng() - 0.5) * 4,
        z: 5 + rng() * 10,
      });
    }

    const pts1: Vec2[] = points3D.map((p) => projectPoint(p, intrinsics, cam1));
    const pts2: Vec2[] = points3D.map((p) => projectPoint(p, intrinsics, cam2));

    const matches: Match[] = pts1.map((_, i) => ({
      queryIdx: i,
      trainIdx: i,
      distance: 0,
    }));

    const ransacRng = createPRNG(42);
    const result = ransacFundamental(matches, pts1, pts2, 3.0, 100, ransacRng);

    expect(result.F).toBeInstanceOf(Float64Array);
    expect(result.F.length).toBe(9);
    expect(result.inlierCount).toBeGreaterThan(0);
    expect(result.inlierMask.length).toBe(matches.length);
  });
});

describe('bundleAdjustment', () => {
  it('converges on the trivial zero-observation config', () => {
    const config: BundleAdjustmentConfig = {
      nCameras: 0,
      nPoints: 0,
      nObservations: 0,
      maxIterations: 10,
      parameterTolerance: 1e-6,
      functionTolerance: 1e-6,
      initialLambda: 1e-3,
      optimiseIntrinsics: false,
      lossFunction: 'trivial',
      lossParameter: 1.0,
    };

    const result = bundleAdjustment(config);
    expect(result.converged).toBe(true);
    expect(result.finalCost).toBeCloseTo(0, 10);
    expect(result.meanReprojError).toBeCloseTo(0, 10);
  });

  it('runs LM iterations for a non-trivial config', () => {
    const config: BundleAdjustmentConfig = {
      nCameras: 2,
      nPoints: 3,
      nObservations: 6,
      maxIterations: 5,
      parameterTolerance: 1e-8,
      functionTolerance: 1e-8,
      initialLambda: 1e-3,
      optimiseIntrinsics: false,
      lossFunction: 'trivial',
      lossParameter: 1.0,
    };

    const result = bundleAdjustment(config);
    expect(result.cameras.length).toBe(2);
    expect(result.points.length).toBe(3);
    expect(result.iterations).toBeGreaterThan(0);
    expect(result.finalCost).toBeGreaterThanOrEqual(0);
  });
});

describe('planCapturePositions', () => {
  it('generates positions that cover the bounding box', () => {
    const bounds = {
      min: { x: 0, y: 0, z: 0 },
      max: { x: 10, y: 2, z: 10 },
    };
    const positions = planCapturePositions(bounds, 0.3, Math.PI / 3);

    expect(positions.length).toBeGreaterThan(0);

    // All positions should be above the bounding box
    for (const pos of positions) {
      expect(pos.y).toBeGreaterThan(bounds.max.y);
    }
  });

  it('increases position count with higher overlap', () => {
    const bounds = {
      min: { x: 0, y: 0, z: 0 },
      max: { x: 10, y: 2, z: 10 },
    };
    const lowOverlap = planCapturePositions(bounds, 0.1, Math.PI / 3);
    const highOverlap = planCapturePositions(bounds, 0.8, Math.PI / 3);

    expect(highOverlap.length).toBeGreaterThanOrEqual(lowOverlap.length);
  });
});

describe('computeCoverage', () => {
  it('returns > 0 for cameras positioned above target points', () => {
    const cameras: Vector3[] = [{ x: 5, y: 10, z: 5 }];
    const targets: Vector3[] = [
      { x: 5, y: 0, z: 5 },
      { x: 6, y: 0, z: 5 },
    ];
    const coverage = computeCoverage(cameras, targets, Math.PI / 2, 20);

    expect(coverage).toBeGreaterThan(0);
    expect(coverage).toBeLessThanOrEqual(1);
  });

  it('returns 0 with no cameras', () => {
    const targets: Vector3[] = [{ x: 0, y: 0, z: 0 }];
    const coverage = computeCoverage([], targets, Math.PI / 2, 20);
    expect(coverage).toBe(0);
  });

  it('returns 1 with empty target array', () => {
    const cameras: Vector3[] = [{ x: 0, y: 10, z: 0 }];
    const coverage = computeCoverage(cameras, [], Math.PI / 2, 20);
    expect(coverage).toBe(1);
  });
});

describe('optimizeCapturePath', () => {
  it('visits all positions exactly once', () => {
    const positions: Vector3[] = [
      { x: 0, y: 0, z: 0 },
      { x: 10, y: 0, z: 0 },
      { x: 5, y: 0, z: 5 },
      { x: 3, y: 0, z: 1 },
      { x: 8, y: 0, z: 4 },
    ];
    const path = optimizeCapturePath(positions);

    expect(path.length).toBe(positions.length);

    // Every original position should appear in the path
    for (const orig of positions) {
      const found = path.some(
        (p) =>
          Math.abs(p.x - orig.x) < 1e-10 &&
          Math.abs(p.y - orig.y) < 1e-10 &&
          Math.abs(p.z - orig.z) < 1e-10,
      );
      expect(found).toBe(true);
    }
  });

  it('starts from the first input position', () => {
    const positions: Vector3[] = [
      { x: 100, y: 0, z: 100 },
      { x: 0, y: 0, z: 0 },
      { x: 50, y: 0, z: 50 },
    ];
    const path = optimizeCapturePath(positions);
    expect(path[0]!.x).toBeCloseTo(100, 5);
    expect(path[0]!.z).toBeCloseTo(100, 5);
  });

  it('returns empty array for empty input', () => {
    const path = optimizeCapturePath([]);
    expect(path.length).toBe(0);
  });
});
