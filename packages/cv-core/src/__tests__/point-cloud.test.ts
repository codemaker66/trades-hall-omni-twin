import { describe, it, expect } from 'vitest';
import {
  statisticalOutlierRemoval,
  voxelDownsample,
  icpPointToPoint,
  findClosestPoint,
  ransacPlaneDetection,
  douglasPeucker,
  extractFloorBoundary,
  estimateNormals,
  orientNormals,
} from '../point-cloud/index.js';
import type { PointCloud, Vec2 } from '../types.js';
import { createPRNG } from '../types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a point cloud from an array of [x, y, z] tuples. */
function makeCloud(pts: [number, number, number][]): PointCloud {
  const positions = new Float64Array(pts.length * 3);
  for (let i = 0; i < pts.length; i++) {
    positions[i * 3] = pts[i]![0];
    positions[i * 3 + 1] = pts[i]![1];
    positions[i * 3 + 2] = pts[i]![2];
  }
  return { positions, count: pts.length };
}

// ---------------------------------------------------------------------------
// statisticalOutlierRemoval
// ---------------------------------------------------------------------------

describe('statisticalOutlierRemoval', () => {
  it('removes distant outliers from a tight cluster', () => {
    // 10 points clustered near origin, 1 outlier far away
    const pts: [number, number, number][] = [];
    for (let i = 0; i < 10; i++) {
      pts.push([i * 0.01, 0, 0]);
    }
    pts.push([100, 100, 100]); // outlier

    const cloud = makeCloud(pts);
    const result = statisticalOutlierRemoval(cloud, 5, 1.0);
    expect(result.count).toBeLessThan(cloud.count);
    expect(result.count).toBe(10); // the 10 inliers survive
  });

  it('preserves all points when there are no outliers', () => {
    const pts: [number, number, number][] = [];
    for (let i = 0; i < 20; i++) {
      pts.push([i * 0.1, 0, 0]);
    }
    const cloud = makeCloud(pts);
    const result = statisticalOutlierRemoval(cloud, 5, 3.0);
    // With a generous stdRatio, all points should be inliers
    expect(result.count).toBe(cloud.count);
  });

  it('returns the cloud unchanged when count <= k', () => {
    const cloud = makeCloud([
      [0, 0, 0],
      [1, 1, 1],
    ]);
    const result = statisticalOutlierRemoval(cloud, 5, 1.0);
    expect(result.count).toBe(cloud.count);
  });

  it('returns a valid PointCloud with correct position data', () => {
    const pts: [number, number, number][] = [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 1, 1],
      [50, 50, 50],
    ];
    const cloud = makeCloud(pts);
    const result = statisticalOutlierRemoval(cloud, 3, 1.0);
    expect(result.positions.length).toBe(result.count * 3);
    expect(result.count).toBeLessThan(cloud.count);
  });

  it('removes multiple outliers', () => {
    const pts: [number, number, number][] = [];
    for (let i = 0; i < 15; i++) {
      pts.push([i * 0.01, 0, 0]);
    }
    // Add 3 outliers
    pts.push([200, 0, 0]);
    pts.push([0, 200, 0]);
    pts.push([0, 0, 200]);

    const cloud = makeCloud(pts);
    const result = statisticalOutlierRemoval(cloud, 5, 1.0);
    expect(result.count).toBeLessThanOrEqual(15);
  });
});

// ---------------------------------------------------------------------------
// voxelDownsample
// ---------------------------------------------------------------------------

describe('voxelDownsample', () => {
  it('reduces point count for a dense cloud', () => {
    // Create a grid of 1000 points in a 1m cube
    const pts: [number, number, number][] = [];
    for (let i = 0; i < 10; i++) {
      for (let j = 0; j < 10; j++) {
        for (let k = 0; k < 10; k++) {
          pts.push([i * 0.1, j * 0.1, k * 0.1]);
        }
      }
    }
    const cloud = makeCloud(pts);
    const result = voxelDownsample(cloud, 0.5);
    expect(result.count).toBeLessThan(cloud.count);
    expect(result.count).toBeGreaterThan(0);
  });

  it('returns the same cloud when voxel size is zero', () => {
    const cloud = makeCloud([
      [0, 0, 0],
      [1, 1, 1],
    ]);
    const result = voxelDownsample(cloud, 0);
    expect(result.count).toBe(cloud.count);
  });

  it('reduces to a single point when voxel is very large', () => {
    const pts: [number, number, number][] = [
      [0, 0, 0],
      [0.1, 0.1, 0.1],
      [0.2, 0.2, 0.2],
    ];
    const cloud = makeCloud(pts);
    const result = voxelDownsample(cloud, 100);
    expect(result.count).toBe(1);
  });

  it('averages positions within a voxel', () => {
    const pts: [number, number, number][] = [
      [0, 0, 0],
      [0.1, 0, 0],
    ];
    const cloud = makeCloud(pts);
    const result = voxelDownsample(cloud, 1.0);
    // Both points fall in the same voxel; result should be their average
    expect(result.count).toBe(1);
    expect(result.positions[0]!).toBeCloseTo(0.05, 5);
  });

  it('returns correct count for empty cloud', () => {
    const cloud: PointCloud = { positions: new Float64Array(0), count: 0 };
    const result = voxelDownsample(cloud, 1.0);
    expect(result.count).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// findClosestPoint
// ---------------------------------------------------------------------------

describe('findClosestPoint', () => {
  const cloud = makeCloud([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [5, 5, 5],
  ]);

  it('finds the exact point when queried with an existing point', () => {
    const result = findClosestPoint({ x: 1, y: 0, z: 0 }, cloud);
    expect(result.index).toBe(1);
    expect(result.distance).toBeCloseTo(0, 5);
  });

  it('finds the closest point for a nearby query', () => {
    const result = findClosestPoint({ x: 0.01, y: 0.01, z: 0.01 }, cloud);
    expect(result.index).toBe(0);
    expect(result.distance).toBeGreaterThan(0);
  });

  it('returns the farthest point when query is distant', () => {
    const result = findClosestPoint({ x: 10, y: 10, z: 10 }, cloud);
    expect(result.index).toBe(4); // (5,5,5) is closest to (10,10,10)
  });

  it('returns a non-negative distance', () => {
    const result = findClosestPoint({ x: -3, y: -3, z: -3 }, cloud);
    expect(result.distance).toBeGreaterThanOrEqual(0);
  });

  it('handles a single-point cloud', () => {
    const singleCloud = makeCloud([[7, 8, 9]]);
    const result = findClosestPoint({ x: 0, y: 0, z: 0 }, singleCloud);
    expect(result.index).toBe(0);
    expect(result.distance).toBeCloseTo(Math.sqrt(7 * 7 + 8 * 8 + 9 * 9), 5);
  });
});

// ---------------------------------------------------------------------------
// icpPointToPoint
// ---------------------------------------------------------------------------

describe('icpPointToPoint', () => {
  it('converges to identity for two identical clouds', () => {
    const cloud = makeCloud([
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 1, 1],
    ]);
    const result = icpPointToPoint(cloud, cloud, 50, 1e-8);
    expect(result.mse).toBeCloseTo(0, 3);
    expect(result.converged).toBe(true);
  });

  it('recovers a known pure translation', () => {
    const target = makeCloud([
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 1, 0],
      [1, 0, 1],
      [0, 1, 1],
      [1, 1, 1],
    ]);
    // Source is target shifted by (0.1, 0, 0)
    const sourcePts: [number, number, number][] = [
      [0.1, 0, 0],
      [1.1, 0, 0],
      [0.1, 1, 0],
      [0.1, 0, 1],
      [1.1, 1, 0],
      [1.1, 0, 1],
      [0.1, 1, 1],
      [1.1, 1, 1],
    ];
    const source = makeCloud(sourcePts);
    const result = icpPointToPoint(source, target, 100, 1e-10);
    expect(result.mse).toBeLessThan(0.01);
  });

  it('returns a 16-element transform matrix', () => {
    const cloud = makeCloud([
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
    ]);
    const result = icpPointToPoint(cloud, cloud, 10, 1e-6);
    expect(result.transform.length).toBe(16);
  });

  it('handles empty source cloud gracefully', () => {
    const empty: PointCloud = { positions: new Float64Array(0), count: 0 };
    const target = makeCloud([[0, 0, 0]]);
    const result = icpPointToPoint(empty, target, 10, 1e-6);
    expect(result.converged).toBe(false);
    expect(result.mse).toBe(Infinity);
  });

  it('reports the number of iterations used', () => {
    const cloud = makeCloud([
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]);
    const result = icpPointToPoint(cloud, cloud, 50, 1e-8);
    expect(result.iterations).toBeGreaterThanOrEqual(1);
    expect(result.iterations).toBeLessThanOrEqual(50);
  });
});

// ---------------------------------------------------------------------------
// ransacPlaneDetection
// ---------------------------------------------------------------------------

describe('ransacPlaneDetection', () => {
  const rng = createPRNG(42);

  it('detects a known horizontal plane (z = 0)', () => {
    // Points on the z = 0 plane
    const pts: [number, number, number][] = [];
    for (let i = 0; i < 50; i++) {
      pts.push([i * 0.1, (i % 7) * 0.1, 0]);
    }
    const cloud = makeCloud(pts);
    const result = ransacPlaneDetection(cloud, 0.01, 100, rng);
    // Normal should be close to (0, 0, +/-1)
    expect(Math.abs(result.normal.z)).toBeCloseTo(1, 1);
    expect(result.inlierCount).toBeGreaterThan(40);
  });

  it('reports a high inlier ratio for a perfectly planar cloud', () => {
    const pts: [number, number, number][] = [];
    for (let i = 0; i < 30; i++) {
      pts.push([i * 0.5, i * 0.3, 0]);
    }
    const cloud = makeCloud(pts);
    const localRng = createPRNG(123);
    const result = ransacPlaneDetection(cloud, 0.01, 200, localRng);
    expect(result.inlierRatio).toBeCloseTo(1.0, 1);
  });

  it('returns zero inliers for fewer than 3 points', () => {
    const cloud = makeCloud([
      [0, 0, 0],
      [1, 0, 0],
    ]);
    const localRng = createPRNG(99);
    const result = ransacPlaneDetection(cloud, 0.01, 100, localRng);
    expect(result.inlierCount).toBe(0);
  });

  it('separates plane inliers from off-plane outliers', () => {
    const pts: [number, number, number][] = [];
    // 20 points on z = 0
    for (let i = 0; i < 20; i++) {
      pts.push([i * 0.1, 0, 0]);
    }
    // 5 points far off the plane
    for (let i = 0; i < 5; i++) {
      pts.push([0, 0, 10 + i]);
    }
    const cloud = makeCloud(pts);
    const localRng = createPRNG(7);
    const result = ransacPlaneDetection(cloud, 0.1, 200, localRng);
    expect(result.inlierCount).toBeGreaterThanOrEqual(15);
    expect(result.inlierCount).toBeLessThanOrEqual(25);
  });

  it('returns a unit normal vector', () => {
    const pts: [number, number, number][] = [];
    for (let i = 0; i < 20; i++) {
      pts.push([i, 0, 0]);
    }
    for (let i = 0; i < 20; i++) {
      pts.push([0, i, 0]);
    }
    const cloud = makeCloud(pts);
    const localRng = createPRNG(55);
    const result = ransacPlaneDetection(cloud, 0.01, 200, localRng);
    const len = Math.sqrt(
      result.normal.x ** 2 + result.normal.y ** 2 + result.normal.z ** 2,
    );
    expect(len).toBeCloseTo(1.0, 5);
  });
});

// ---------------------------------------------------------------------------
// douglasPeucker
// ---------------------------------------------------------------------------

describe('douglasPeucker', () => {
  it('returns endpoints for a straight line', () => {
    const pts: Vec2[] = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 2, y: 0 },
      { x: 3, y: 0 },
      { x: 4, y: 0 },
    ];
    const result = douglasPeucker(pts, 0.1);
    expect(result.length).toBe(2);
    expect(result[0]!.x).toBeCloseTo(0, 5);
    expect(result[1]!.x).toBeCloseTo(4, 5);
  });

  it('preserves sharp corners', () => {
    const pts: Vec2[] = [
      { x: 0, y: 0 },
      { x: 5, y: 0 },
      { x: 5, y: 5 },
    ];
    const result = douglasPeucker(pts, 0.1);
    expect(result.length).toBe(3); // corner is preserved
  });

  it('returns fewer points than input for a noisy curve', () => {
    const pts: Vec2[] = [];
    for (let i = 0; i < 50; i++) {
      pts.push({ x: i, y: Math.sin(i * 0.1) * 0.01 });
    }
    const result = douglasPeucker(pts, 0.1);
    expect(result.length).toBeLessThan(pts.length);
  });

  it('returns the input unchanged when it has 2 or fewer points', () => {
    const pts: Vec2[] = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
    ];
    const result = douglasPeucker(pts, 0.01);
    expect(result.length).toBe(2);
  });

  it('preserves all points when epsilon is very small', () => {
    const pts: Vec2[] = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
      { x: 2, y: 0 },
      { x: 3, y: 1 },
    ];
    const result = douglasPeucker(pts, 0);
    // With epsilon 0, every deviation matters, so all points should be kept
    expect(result.length).toBe(pts.length);
  });
});

// ---------------------------------------------------------------------------
// extractFloorBoundary
// ---------------------------------------------------------------------------

describe('extractFloorBoundary', () => {
  it('extracts a convex hull from floor-height points', () => {
    // Points at z = 0 forming a square
    const pts: [number, number, number][] = [
      [0, 0, 0],
      [10, 0, 0],
      [10, 10, 0],
      [0, 10, 0],
      [5, 5, 0],
    ];
    const cloud = makeCloud(pts);
    const boundary = extractFloorBoundary(cloud, 0, 0.1);
    expect(boundary.length).toBeGreaterThanOrEqual(4); // at least the 4 corners
  });

  it('ignores points far from the floor height', () => {
    const pts: [number, number, number][] = [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 10], // way above floor
      [1, 1, 10], // way above floor
    ];
    const cloud = makeCloud(pts);
    const boundary = extractFloorBoundary(cloud, 0, 0.5);
    // Only the 3 floor-level points are used
    expect(boundary.length).toBe(3);
  });

  it('returns empty for a cloud with no points near the floor', () => {
    const pts: [number, number, number][] = [
      [0, 0, 10],
      [1, 0, 10],
    ];
    const cloud = makeCloud(pts);
    const boundary = extractFloorBoundary(cloud, 0, 0.1);
    expect(boundary.length).toBeLessThanOrEqual(2);
  });

  it('returns 2D Vec2 points (no z component)', () => {
    const pts: [number, number, number][] = [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [1, 1, 0],
    ];
    const cloud = makeCloud(pts);
    const boundary = extractFloorBoundary(cloud, 0, 0.1);
    for (const p of boundary) {
      expect(typeof p.x).toBe('number');
      expect(typeof p.y).toBe('number');
      expect((p as unknown as { z?: number }).z).toBeUndefined();
    }
  });

  it('handles a non-zero floor height', () => {
    const pts: [number, number, number][] = [
      [0, 0, 3],
      [5, 0, 3],
      [5, 5, 3],
      [0, 5, 3],
      [2.5, 2.5, 0], // below floor
    ];
    const cloud = makeCloud(pts);
    const boundary = extractFloorBoundary(cloud, 3, 0.5);
    expect(boundary.length).toBeGreaterThanOrEqual(4);
  });
});

// ---------------------------------------------------------------------------
// estimateNormals
// ---------------------------------------------------------------------------

describe('estimateNormals', () => {
  it('produces unit-length normals for a planar set', () => {
    // Points on the z = 0 plane in a grid
    const pts: [number, number, number][] = [];
    for (let i = 0; i < 5; i++) {
      for (let j = 0; j < 5; j++) {
        pts.push([i, j, 0]);
      }
    }
    const cloud = makeCloud(pts);
    const normals = estimateNormals(cloud, 5);
    expect(normals.length).toBe(cloud.count * 3);

    for (let i = 0; i < cloud.count; i++) {
      const nx = normals[i * 3]!;
      const ny = normals[i * 3 + 1]!;
      const nz = normals[i * 3 + 2]!;
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
      expect(len).toBeCloseTo(1.0, 2);
    }
  });

  it('estimates normals close to z-axis for a horizontal plane', () => {
    const pts: [number, number, number][] = [];
    for (let i = 0; i < 5; i++) {
      for (let j = 0; j < 5; j++) {
        pts.push([i, j, 0]);
      }
    }
    const cloud = makeCloud(pts);
    const normals = estimateNormals(cloud, 8);

    // Check a central point (not edge) for z-dominance
    const midIdx = 12; // center of 5x5 grid
    const nz = Math.abs(normals[midIdx * 3 + 2]!);
    expect(nz).toBeGreaterThan(0.8);
  });

  it('returns zero normals when cloud has fewer than 3 points', () => {
    const cloud = makeCloud([
      [0, 0, 0],
      [1, 0, 0],
    ]);
    const normals = estimateNormals(cloud, 5);
    for (let i = 0; i < normals.length; i++) {
      expect(normals[i]).toBe(0);
    }
  });

  it('returns the correct array length', () => {
    const cloud = makeCloud([
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 1, 1],
    ]);
    const normals = estimateNormals(cloud, 3);
    expect(normals.length).toBe(5 * 3);
  });

  it('produces different normals for non-coplanar neighbourhoods', () => {
    // Estimate normals for XY plane (z = 0): normal should be z-dominant
    const xyPts: [number, number, number][] = [];
    for (let i = 0; i < 5; i++) {
      for (let j = 0; j < 5; j++) {
        xyPts.push([i, j, 0]);
      }
    }
    const xyCloud = makeCloud(xyPts);
    const xyNormals = estimateNormals(xyCloud, 8);
    const midXY = 12; // center of 5x5 grid
    const xyNz = Math.abs(xyNormals[midXY * 3 + 2]!);
    expect(xyNz).toBeGreaterThan(0.5);

    // Estimate normals for YZ plane (x = 0): normal should be x-dominant
    const yzPts: [number, number, number][] = [];
    for (let i = 0; i < 5; i++) {
      for (let j = 0; j < 5; j++) {
        yzPts.push([0, i, j]);
      }
    }
    const yzCloud = makeCloud(yzPts);
    const yzNormals = estimateNormals(yzCloud, 8);
    const midYZ = 12;
    const yzNx = Math.abs(yzNormals[midYZ * 3]!);
    expect(yzNx).toBeGreaterThan(0.5);

    // The two normals should be significantly different
    const dotProduct = Math.abs(
      xyNormals[midXY * 3]! * yzNormals[midYZ * 3]! +
      xyNormals[midXY * 3 + 1]! * yzNormals[midYZ * 3 + 1]! +
      xyNormals[midXY * 3 + 2]! * yzNormals[midYZ * 3 + 2]!,
    );
    expect(dotProduct).toBeLessThan(0.5); // nearly perpendicular
  });
});

// ---------------------------------------------------------------------------
// orientNormals
// ---------------------------------------------------------------------------

describe('orientNormals', () => {
  it('flips normals to be consistent with the seed direction', () => {
    // Simple planar cloud with deliberately flipped normals
    const pts: [number, number, number][] = [];
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        pts.push([i, j, 0]);
      }
    }
    const cloud = makeCloud(pts);

    // Start with normals pointing in alternating z directions
    const normals = new Float64Array(cloud.count * 3);
    for (let i = 0; i < cloud.count; i++) {
      normals[i * 3 + 2] = i % 2 === 0 ? 1 : -1;
    }

    const oriented = orientNormals(cloud, normals);

    // After orientation, all normals should point in the same z direction
    const firstZ = oriented[2]!;
    for (let i = 0; i < cloud.count; i++) {
      const nz = oriented[i * 3 + 2]!;
      expect(nz * firstZ).toBeGreaterThanOrEqual(0);
    }
  });

  it('returns a Float64Array of the same length', () => {
    const cloud = makeCloud([
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
    ]);
    const normals = new Float64Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
    const oriented = orientNormals(cloud, normals);
    expect(oriented.length).toBe(normals.length);
    expect(oriented).toBeInstanceOf(Float64Array);
  });

  it('does not modify normals that are already consistently oriented', () => {
    const cloud = makeCloud([
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [1, 1, 0],
    ]);
    const normals = new Float64Array(cloud.count * 3);
    for (let i = 0; i < cloud.count; i++) {
      normals[i * 3 + 2] = 1; // all pointing up
    }
    const oriented = orientNormals(cloud, normals);
    for (let i = 0; i < cloud.count; i++) {
      expect(oriented[i * 3 + 2]!).toBeCloseTo(1, 5);
    }
  });

  it('ensures the seed normal points up (positive z)', () => {
    const cloud = makeCloud([
      [0, 0, 5],
      [1, 0, 0],
      [0, 1, 0],
    ]);
    // Seed will be index 0 (highest z). Normal points down.
    const normals = new Float64Array([0, 0, -1, 0, 0, -1, 0, 0, -1]);
    const oriented = orientNormals(cloud, normals);
    // The seed normal should be flipped to positive z
    expect(oriented[2]!).toBeGreaterThan(0);
  });

  it('returns the input unchanged for a single-point cloud', () => {
    const cloud = makeCloud([[0, 0, 0]]);
    const normals = new Float64Array([0, 0, 1]);
    const oriented = orientNormals(cloud, normals);
    expect(oriented[0]).toBeCloseTo(0, 5);
    expect(oriented[1]).toBeCloseTo(0, 5);
    expect(oriented[2]).toBeCloseTo(1, 5);
  });
});
