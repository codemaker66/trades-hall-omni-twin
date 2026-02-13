import { describe, it, expect } from 'vitest';
import {
  rayPlaneIntersection,
  snapToSurface,
  hitTestFloor,
  createPose6DoF,
  interpolatePose,
  poseToMatrix,
  matrixToPose,
  composePoses,
  inversePose,
  createUSDZMeta,
  validateUSDZConstraints,
  estimateUSDZFileSize,
} from '../xr/index.js';
import type { Vector3, Quaternion, Pose6DoF } from '../types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Compare two Vector3 values with approximate equality. */
function expectVec3Close(actual: Vector3, expected: Vector3, digits = 6): void {
  expect(actual.x).toBeCloseTo(expected.x, digits);
  expect(actual.y).toBeCloseTo(expected.y, digits);
  expect(actual.z).toBeCloseTo(expected.z, digits);
}

/** Compare two Quaternion values with approximate equality (sign-agnostic). */
function expectQuatClose(actual: Quaternion, expected: Quaternion, digits = 6): void {
  // Quaternions q and -q represent the same rotation.
  const sign = Math.sign(actual.w * expected.w + actual.x * expected.x +
    actual.y * expected.y + actual.z * expected.z) || 1;
  expect(actual.x * sign).toBeCloseTo(expected.x, digits);
  expect(actual.y * sign).toBeCloseTo(expected.y, digits);
  expect(actual.z * sign).toBeCloseTo(expected.z, digits);
  expect(actual.w * sign).toBeCloseTo(expected.w, digits);
}

const IDENTITY_QUAT: Quaternion = { x: 0, y: 0, z: 0, w: 1 };

// ---------------------------------------------------------------------------
// Hit testing
// ---------------------------------------------------------------------------

describe('rayPlaneIntersection', () => {
  it('should find the intersection of a ray with a known XZ plane', () => {
    // Plane: y = 0 => normal = (0,1,0), planeD = 0
    const origin: Vector3 = { x: 0, y: 5, z: 0 };
    const dir: Vector3 = { x: 0, y: -1, z: 0 };
    const planeNormal: Vector3 = { x: 0, y: 1, z: 0 };

    const hit = rayPlaneIntersection(origin, dir, planeNormal, 0);
    expect(hit).not.toBeNull();
    expectVec3Close(hit!, { x: 0, y: 0, z: 0 });
  });

  it('should return null when the ray is parallel to the plane', () => {
    const origin: Vector3 = { x: 0, y: 5, z: 0 };
    const dir: Vector3 = { x: 1, y: 0, z: 0 }; // parallel to XZ plane
    const planeNormal: Vector3 = { x: 0, y: 1, z: 0 };

    const hit = rayPlaneIntersection(origin, dir, planeNormal, 0);
    expect(hit).toBeNull();
  });

  it('should return null when intersection is behind the ray origin', () => {
    // Ray points upward, plane is below
    const origin: Vector3 = { x: 0, y: 5, z: 0 };
    const dir: Vector3 = { x: 0, y: 1, z: 0 }; // pointing up
    const planeNormal: Vector3 = { x: 0, y: 1, z: 0 };

    const hit = rayPlaneIntersection(origin, dir, planeNormal, 0);
    expect(hit).toBeNull();
  });
});

describe('hitTestFloor', () => {
  it('should hit the floor at the correct y coordinate', () => {
    const origin: Vector3 = { x: 0, y: 10, z: 0 };
    const dir: Vector3 = { x: 0, y: -1, z: 0 };

    const result = hitTestFloor(origin, dir, 2.0);
    expect(result).not.toBeNull();
    expect(result!.pose.position.y).toBeCloseTo(2.0, 10);
    expect(result!.distance).toBeCloseTo(8.0, 10);
    expect(result!.type).toBe('plane');
    expect(result!.planeType).toBe('floor');
  });

  it('should return null when ray points away from the floor', () => {
    const origin: Vector3 = { x: 0, y: 10, z: 0 };
    const dir: Vector3 = { x: 0, y: 1, z: 0 }; // pointing up

    const result = hitTestFloor(origin, dir, 0);
    expect(result).toBeNull();
  });
});

describe('snapToSurface', () => {
  it('should project a point onto the nearest surface', () => {
    // Surface: horizontal plane at y=3
    const surfaceNormal: Vector3 = { x: 0, y: 1, z: 0 };
    const surfacePoint: Vector3 = { x: 0, y: 3, z: 0 };
    const point: Vector3 = { x: 5, y: 10, z: 7 };

    const snapped = snapToSurface(point, surfaceNormal, surfacePoint);
    expect(snapped.y).toBeCloseTo(3.0, 10);
    // x and z should be unchanged
    expect(snapped.x).toBeCloseTo(5.0, 10);
    expect(snapped.z).toBeCloseTo(7.0, 10);
  });
});

// ---------------------------------------------------------------------------
// Pose utilities
// ---------------------------------------------------------------------------

describe('poseToMatrix / matrixToPose roundtrip', () => {
  it('should recover the original pose from its matrix representation', () => {
    // A 90-degree rotation about the Y axis
    const halfAngle = Math.PI / 4; // half of 90 degrees
    const pose = createPose6DoF(
      { x: 1, y: 2, z: 3 },
      { x: 0, y: Math.sin(halfAngle), z: 0, w: Math.cos(halfAngle) },
    );

    const matrix = poseToMatrix(pose);
    const recovered = matrixToPose(matrix);

    expectVec3Close(recovered.position, pose.position);
    expectQuatClose(recovered.orientation, pose.orientation);
  });

  it('should roundtrip the identity pose', () => {
    const pose = createPose6DoF({ x: 0, y: 0, z: 0 }, IDENTITY_QUAT);
    const matrix = poseToMatrix(pose);
    const recovered = matrixToPose(matrix);

    expectVec3Close(recovered.position, { x: 0, y: 0, z: 0 });
    expectQuatClose(recovered.orientation, IDENTITY_QUAT);
  });
});

describe('composePoses / inversePose', () => {
  it('should produce identity when composing a pose with its inverse', () => {
    const halfAngle = Math.PI / 6;
    const pose = createPose6DoF(
      { x: 3, y: -1, z: 7 },
      { x: 0, y: Math.sin(halfAngle), z: 0, w: Math.cos(halfAngle) },
    );

    const inv = inversePose(pose);
    const composed = composePoses(pose, inv);

    expectVec3Close(composed.position, { x: 0, y: 0, z: 0 }, 5);
    expectQuatClose(composed.orientation, IDENTITY_QUAT, 5);
  });

  it('should produce identity when composing inverse first', () => {
    const halfAngle = Math.PI / 3;
    const pose = createPose6DoF(
      { x: -2, y: 4, z: 1 },
      { x: Math.sin(halfAngle) * 0.577, y: Math.sin(halfAngle) * 0.577, z: Math.sin(halfAngle) * 0.577, w: Math.cos(halfAngle) },
    );

    const inv = inversePose(pose);
    const composed = composePoses(inv, pose);

    expectVec3Close(composed.position, { x: 0, y: 0, z: 0 }, 5);
    expectQuatClose(composed.orientation, IDENTITY_QUAT, 5);
  });
});

describe('interpolatePose', () => {
  const poseA: Pose6DoF = createPose6DoF(
    { x: 0, y: 0, z: 0 },
    IDENTITY_QUAT,
  );

  const halfAngle = Math.PI / 4;
  const poseB: Pose6DoF = createPose6DoF(
    { x: 10, y: 0, z: 0 },
    { x: 0, y: Math.sin(halfAngle), z: 0, w: Math.cos(halfAngle) },
  );

  it('should return pose A at t=0', () => {
    const result = interpolatePose(poseA, poseB, 0);
    expectVec3Close(result.position, poseA.position);
    expectQuatClose(result.orientation, poseA.orientation);
  });

  it('should return pose B at t=1', () => {
    const result = interpolatePose(poseA, poseB, 1);
    expectVec3Close(result.position, poseB.position);
    expectQuatClose(result.orientation, poseB.orientation);
  });

  it('should return a midpoint position at t=0.5', () => {
    const result = interpolatePose(poseA, poseB, 0.5);
    expect(result.position.x).toBeCloseTo(5, 5);
    expect(result.position.y).toBeCloseTo(0, 5);
    expect(result.position.z).toBeCloseTo(0, 5);
  });
});

// ---------------------------------------------------------------------------
// USDZ utilities
// ---------------------------------------------------------------------------

describe('validateUSDZConstraints', () => {
  it('should pass for a small asset within all limits', () => {
    const meta = createUSDZMeta('small-object', 5000, 512);
    const result = validateUSDZConstraints(meta);
    expect(result.valid).toBe(true);
    expect(result.issues.length).toBe(0);
  });

  it('should fail when triangle count exceeds the limit', () => {
    const meta = createUSDZMeta('huge-mesh', 200_000, 512);
    const result = validateUSDZConstraints(meta);
    expect(result.valid).toBe(false);
    expect(result.issues.length).toBeGreaterThan(0);
    expect(result.issues.some((s) => s.includes('Triangle count'))).toBe(true);
  });

  it('should fail when texture size exceeds the limit', () => {
    const meta = createUSDZMeta('big-tex', 1000, 8192);
    const result = validateUSDZConstraints(meta);
    expect(result.valid).toBe(false);
    expect(result.issues.some((s) => s.includes('Texture dimension'))).toBe(true);
  });
});

describe('estimateUSDZFileSize', () => {
  it('should return a positive number for valid inputs', () => {
    const size = estimateUSDZFileSize(10_000, 1024, false);
    expect(size).toBeGreaterThan(0);
  });

  it('should return a larger size when animations are present', () => {
    const withoutAnim = estimateUSDZFileSize(10_000, 1024, false);
    const withAnim = estimateUSDZFileSize(10_000, 1024, true);
    expect(withAnim).toBeGreaterThan(withoutAnim);
  });
});
