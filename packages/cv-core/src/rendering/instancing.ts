// ---------------------------------------------------------------------------
// CV-7: Instanced Rendering — instance buffer construction, frustum
// culling, and frustum-plane extraction for high-throughput draw calls.
// ---------------------------------------------------------------------------

import type { FrustumPlanes, InstanceData, Vec4, Vector3 } from '../types.js';
import { mat4Identity } from '../types.js';

// ---------------------------------------------------------------------------
// buildInstanceBuffer
// ---------------------------------------------------------------------------

/**
 * Create an array of {@link InstanceData} from per-instance 4x4 transforms
 * and bounding spheres.
 *
 * Each `transforms[i]` is a 16-element {@link Float64Array} in column-major
 * order.  The corresponding `boundingSpheres[i]` defines the object-space
 * bounding sphere (centre + radius).
 *
 * @param transforms      Per-instance 4x4 transform matrices (column-major).
 * @param boundingSpheres Per-instance bounding spheres.
 * @returns               Array of {@link InstanceData}.
 */
export function buildInstanceBuffer(
  transforms: Float64Array[],
  boundingSpheres: { center: Vector3; radius: number }[],
): InstanceData[] {
  const count = Math.min(transforms.length, boundingSpheres.length);
  const instances: InstanceData[] = [];

  for (let i = 0; i < count; i++) {
    const xf = transforms[i]!;
    const bs = boundingSpheres[i]!;

    // Transform the bounding-sphere centre into world space using the
    // translation column of the instance matrix (column 3, rows 0-2).
    const cx = bs.center.x + xf[12]!;
    const cy = bs.center.y + xf[13]!;
    const cz = bs.center.z + xf[14]!;

    // Approximate uniform scale from the length of the first basis column.
    const sx = xf[0]!;
    const sy = xf[1]!;
    const sz = xf[2]!;
    const scale = Math.sqrt(sx * sx + sy * sy + sz * sz);

    instances.push({
      transform: xf,
      boundingSphere: {
        centre: { x: cx, y: cy, z: cz },
        radius: bs.radius * scale,
      },
      instanceId: i,
    });
  }

  return instances;
}

// ---------------------------------------------------------------------------
// frustumCull  (sphere-vs-6-planes test)
// ---------------------------------------------------------------------------

/**
 * Test a bounding sphere against a single frustum plane.
 *
 * Plane stored as Vec4 {x,y,z,w} where (x,y,z) is the inward normal and
 * w is the signed distance.  A point is inside if dot(n, p) + w >= 0.
 *
 * @returns true if the sphere is **entirely outside** the half-space.
 */
function sphereOutsidePlane(
  cx: number,
  cy: number,
  cz: number,
  radius: number,
  plane: Vec4,
): boolean {
  const dist = plane.x * cx + plane.y * cy + plane.z * cz + plane.w;
  return dist < -radius;
}

/**
 * Frustum-cull an array of instances and return the indices of those that are
 * at least partially inside the frustum.
 *
 * Uses a conservative sphere-vs-6-planes test: an instance is culled only if
 * its bounding sphere is entirely outside **any** frustum plane.
 *
 * @param instances Array of {@link InstanceData} with world-space bounding spheres.
 * @param planes    Six frustum planes extracted from the view-projection matrix.
 * @returns         Array of instance indices that survive the cull.
 */
export function frustumCull(
  instances: InstanceData[],
  planes: FrustumPlanes,
): number[] {
  const planeArray: Vec4[] = [
    planes.left,
    planes.right,
    planes.bottom,
    planes.top,
    planes.near,
    planes.far,
  ];

  const visible: number[] = [];

  for (let i = 0; i < instances.length; i++) {
    const inst = instances[i]!;
    const { centre, radius } = inst.boundingSphere;
    let inside = true;

    for (let p = 0; p < 6; p++) {
      if (sphereOutsidePlane(centre.x, centre.y, centre.z, radius, planeArray[p]!)) {
        inside = false;
        break;
      }
    }

    if (inside) {
      visible.push(i);
    }
  }

  return visible;
}

// ---------------------------------------------------------------------------
// extractFrustumPlanes
// ---------------------------------------------------------------------------

/**
 * Extract the six frustum planes from a 4x4 view-projection matrix.
 *
 * Uses the Gribb/Hartmann method: each plane is derived from the sum or
 * difference of two rows of the **row-major** interpretation of the matrix.
 * Since our matrices are stored **column-major**, we read rows accordingly:
 *   row i = [ m[i], m[4+i], m[8+i], m[12+i] ]
 *
 * Each resulting plane is normalised so that (x,y,z) is a unit vector.
 *
 * @param viewProjection 16-element column-major VP matrix.
 * @returns              Six frustum planes.
 */
export function extractFrustumPlanes(
  viewProjection: Float64Array,
): FrustumPlanes {
  const m = viewProjection;

  // Helper to read row r of the column-major matrix.
  const row = (r: number): [number, number, number, number] => [
    m[r]!,
    m[4 + r]!,
    m[8 + r]!,
    m[12 + r]!,
  ];

  const r0 = row(0);
  const r1 = row(1);
  const r2 = row(2);
  const r3 = row(3);

  /** Build a Vec4 from a + b (element-wise) and normalise. */
  function makePlane(
    a: [number, number, number, number],
    b: [number, number, number, number],
    sign: 1 | -1,
  ): Vec4 {
    let nx = a[0] + sign * b[0];
    let ny = a[1] + sign * b[1];
    let nz = a[2] + sign * b[2];
    let nw = a[3] + sign * b[3];

    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 1e-15) {
      nx /= len;
      ny /= len;
      nz /= len;
      nw /= len;
    }

    return { x: nx, y: ny, z: nz, w: nw };
  }

  return {
    left: makePlane(r3, r0, 1),    // row3 + row0
    right: makePlane(r3, r0, -1),   // row3 - row0
    bottom: makePlane(r3, r1, 1),   // row3 + row1
    top: makePlane(r3, r1, -1),     // row3 - row1
    near: makePlane(r3, r2, 1),     // row3 + row2
    far: makePlane(r3, r2, -1),     // row3 - row2
  };
}
