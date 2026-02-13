// ---------------------------------------------------------------------------
// Tests: CV-3 Compositing — depth buffers, unprojection, raycasting
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';
import {
  createDepthBuffer,
  unprojectPixel,
  compositeDepthOrder,
  linearizeDepth,
  rayEllipsoidIntersection,
  rayTriangleIntersection,
  rayAABBIntersection,
} from '../compositing/index.js';
import type { Vector3 } from '../types.js';

// ---------------------------------------------------------------------------
// createDepthBuffer
// ---------------------------------------------------------------------------

describe('createDepthBuffer', () => {
  it('creates a buffer with correct dimensions', () => {
    const buf = createDepthBuffer(4, 3, 0.1, 100);
    expect(buf.width).toBe(4);
    expect(buf.height).toBe(3);
    expect(buf.near).toBeCloseTo(0.1, 10);
    expect(buf.far).toBeCloseTo(100, 10);
  });

  it('initialises all pixels to the far value', () => {
    const buf = createDepthBuffer(8, 8, 0.1, 1000);
    expect(buf.data.length).toBe(64);
    for (let i = 0; i < 64; i++) {
      expect(buf.data[i]!).toBeCloseTo(1000, 10);
    }
  });

  it('data is a Float64Array', () => {
    const buf = createDepthBuffer(2, 2, 0.5, 50);
    expect(buf.data).toBeInstanceOf(Float64Array);
  });

  it('stores near and far plane distances', () => {
    const buf = createDepthBuffer(1, 1, 0.01, 500);
    expect(buf.near).toBeCloseTo(0.01, 10);
    expect(buf.far).toBeCloseTo(500, 10);
  });

  it('handles 1x1 buffer', () => {
    const buf = createDepthBuffer(1, 1, 0.1, 10);
    expect(buf.data.length).toBe(1);
    expect(buf.data[0]!).toBeCloseTo(10, 10);
  });
});

// ---------------------------------------------------------------------------
// unprojectPixel
// ---------------------------------------------------------------------------

describe('unprojectPixel', () => {
  const intrinsics = {
    fx: 500,
    fy: 500,
    cx: 320,
    cy: 240,
    width: 640,
    height: 480,
  };

  it('unprojecting the principal point returns (0, 0, depth)', () => {
    const p = unprojectPixel(320, 240, 5.0, intrinsics);
    expect(p.x).toBeCloseTo(0, 10);
    expect(p.y).toBeCloseTo(0, 10);
    expect(p.z).toBeCloseTo(5, 10);
  });

  it('x and y scale linearly with depth', () => {
    const p1 = unprojectPixel(420, 240, 1.0, intrinsics);
    const p2 = unprojectPixel(420, 240, 2.0, intrinsics);
    expect(p2.x).toBeCloseTo(p1.x * 2, 10);
  });

  it('off-center pixel has correct camera-space offset', () => {
    // pixel (420, 340), depth 10
    // X = (420 - 320) * 10 / 500 = 100 * 10 / 500 = 2
    // Y = (340 - 240) * 10 / 500 = 100 * 10 / 500 = 2
    const p = unprojectPixel(420, 340, 10, intrinsics);
    expect(p.x).toBeCloseTo(2, 10);
    expect(p.y).toBeCloseTo(2, 10);
    expect(p.z).toBeCloseTo(10, 10);
  });

  it('unproject then reproject recovers the original pixel', () => {
    const px = 200;
    const py = 150;
    const depth = 7.5;
    const pt = unprojectPixel(px, py, depth, intrinsics);
    // Re-project: pixel_x = fx * X / Z + cx
    const reprojX = intrinsics.fx * pt.x / pt.z + intrinsics.cx;
    const reprojY = intrinsics.fy * pt.y / pt.z + intrinsics.cy;
    expect(reprojX).toBeCloseTo(px, 8);
    expect(reprojY).toBeCloseTo(py, 8);
  });

  it('returns negative x for pixels left of principal point', () => {
    const p = unprojectPixel(100, 240, 5, intrinsics);
    expect(p.x).toBeLessThan(0);
  });
});

// ---------------------------------------------------------------------------
// compositeDepthOrder
// ---------------------------------------------------------------------------

describe('compositeDepthOrder', () => {
  it('returns 0 where gaussian depth <= mesh depth', () => {
    const gauss = new Float64Array([1, 5, 3]);
    const mesh = new Float64Array([2, 4, 3]);
    const result = compositeDepthOrder(gauss, mesh, 3, 1);
    expect(result[0]!).toBe(0); // 1 <= 2
    expect(result[1]!).toBe(1); // 5 > 4
    expect(result[2]!).toBe(0); // 3 <= 3
  });

  it('returns all zeros when gaussian is always in front', () => {
    const gauss = new Float64Array([1, 1, 1, 1]);
    const mesh = new Float64Array([2, 2, 2, 2]);
    const result = compositeDepthOrder(gauss, mesh, 2, 2);
    for (let i = 0; i < 4; i++) {
      expect(result[i]!).toBe(0);
    }
  });

  it('returns all ones when mesh is always in front', () => {
    const gauss = new Float64Array([10, 10]);
    const mesh = new Float64Array([1, 1]);
    const result = compositeDepthOrder(gauss, mesh, 1, 2);
    for (let i = 0; i < 2; i++) {
      expect(result[i]!).toBe(1);
    }
  });

  it('result is a Uint8Array of size width*height', () => {
    const gauss = new Float64Array(12);
    const mesh = new Float64Array(12);
    const result = compositeDepthOrder(gauss, mesh, 4, 3);
    expect(result).toBeInstanceOf(Uint8Array);
    expect(result.length).toBe(12);
  });

  it('equal depths are assigned to gaussian (value 0)', () => {
    const gauss = new Float64Array([5]);
    const mesh = new Float64Array([5]);
    const result = compositeDepthOrder(gauss, mesh, 1, 1);
    expect(result[0]!).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// linearizeDepth
// ---------------------------------------------------------------------------

describe('linearizeDepth', () => {
  it('returns near when rawDepth = 0', () => {
    const result = linearizeDepth(0, 0.1, 100);
    expect(result).toBeCloseTo(0.1, 10);
  });

  it('returns far when rawDepth = 1', () => {
    const result = linearizeDepth(1, 0.1, 100);
    expect(result).toBeCloseTo(100, 10);
  });

  it('returns a value between near and far for rawDepth = 0.5', () => {
    const result = linearizeDepth(0.5, 1, 100);
    expect(result).toBeGreaterThan(1);
    expect(result).toBeLessThan(100);
  });

  it('is monotonically increasing with rawDepth', () => {
    const near = 0.1;
    const far = 1000;
    let prev = linearizeDepth(0, near, far);
    for (let d = 0.1; d <= 1.0; d += 0.1) {
      const curr = linearizeDepth(d, near, far);
      expect(curr).toBeGreaterThanOrEqual(prev);
      prev = curr;
    }
  });

  it('returns the correct value for a known intermediate depth', () => {
    // near=1, far=10, rawDepth=0.5 => (1*10) / (10 - 0.5*9) = 10 / 5.5 ~ 1.818
    const result = linearizeDepth(0.5, 1, 10);
    expect(result).toBeCloseTo(10 / 5.5, 8);
  });
});

// ---------------------------------------------------------------------------
// rayTriangleIntersection
// ---------------------------------------------------------------------------

describe('rayTriangleIntersection', () => {
  // Triangle in the XY plane at z=0
  const v0: Vector3 = { x: 0, y: 0, z: 0 };
  const v1: Vector3 = { x: 2, y: 0, z: 0 };
  const v2: Vector3 = { x: 1, y: 2, z: 0 };

  it('hits a triangle with a ray aimed at its interior', () => {
    const origin: Vector3 = { x: 1, y: 0.5, z: -5 };
    const dir: Vector3 = { x: 0, y: 0, z: 1 };
    const hit = rayTriangleIntersection(origin, dir, v0, v1, v2);
    expect(hit).not.toBeNull();
    expect(hit!.distance).toBeCloseTo(5, 8);
    expect(hit!.point.z).toBeCloseTo(0, 8);
  });

  it('misses when ray is aimed away from the triangle', () => {
    const origin: Vector3 = { x: 1, y: 0.5, z: -5 };
    const dir: Vector3 = { x: 0, y: 0, z: -1 }; // pointing away
    const hit = rayTriangleIntersection(origin, dir, v0, v1, v2);
    expect(hit).toBeNull();
  });

  it('misses when ray is parallel to the triangle plane', () => {
    const origin: Vector3 = { x: 0, y: 0, z: 1 };
    const dir: Vector3 = { x: 1, y: 0, z: 0 }; // parallel to XY
    const hit = rayTriangleIntersection(origin, dir, v0, v1, v2);
    expect(hit).toBeNull();
  });

  it('misses when ray goes outside the triangle boundary', () => {
    const origin: Vector3 = { x: 5, y: 5, z: -1 };
    const dir: Vector3 = { x: 0, y: 0, z: 1 };
    const hit = rayTriangleIntersection(origin, dir, v0, v1, v2);
    expect(hit).toBeNull();
  });

  it('returns a unit-length normal vector', () => {
    const origin: Vector3 = { x: 1, y: 0.5, z: -1 };
    const dir: Vector3 = { x: 0, y: 0, z: 1 };
    const hit = rayTriangleIntersection(origin, dir, v0, v1, v2);
    expect(hit).not.toBeNull();
    const n = hit!.normal;
    const len = Math.sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    expect(len).toBeCloseTo(1, 8);
  });

  it('normal points in z-direction for an XY-plane triangle', () => {
    const origin: Vector3 = { x: 1, y: 0.5, z: -1 };
    const dir: Vector3 = { x: 0, y: 0, z: 1 };
    const hit = rayTriangleIntersection(origin, dir, v0, v1, v2);
    expect(hit).not.toBeNull();
    // Normal should be (0, 0, +/-1)
    expect(Math.abs(hit!.normal.z)).toBeCloseTo(1, 8);
  });
});

// ---------------------------------------------------------------------------
// rayEllipsoidIntersection
// ---------------------------------------------------------------------------

describe('rayEllipsoidIntersection', () => {
  it('hits a unit sphere at the origin', () => {
    const origin: Vector3 = { x: 0, y: 0, z: -5 };
    const dir: Vector3 = { x: 0, y: 0, z: 1 };
    const center: Vector3 = { x: 0, y: 0, z: 0 };
    const radii: Vector3 = { x: 1, y: 1, z: 1 };
    const t = rayEllipsoidIntersection(origin, dir, center, radii);
    expect(t).not.toBeNull();
    expect(t!).toBeCloseTo(4, 8); // enters at z = -1, origin.z = -5, t = 4
  });

  it('returns null when ray misses the ellipsoid', () => {
    const origin: Vector3 = { x: 10, y: 10, z: -5 };
    const dir: Vector3 = { x: 0, y: 0, z: 1 };
    const center: Vector3 = { x: 0, y: 0, z: 0 };
    const radii: Vector3 = { x: 1, y: 1, z: 1 };
    const t = rayEllipsoidIntersection(origin, dir, center, radii);
    expect(t).toBeNull();
  });

  it('returns null for degenerate (zero) radii', () => {
    const origin: Vector3 = { x: 0, y: 0, z: -5 };
    const dir: Vector3 = { x: 0, y: 0, z: 1 };
    const center: Vector3 = { x: 0, y: 0, z: 0 };
    const radii: Vector3 = { x: 0, y: 1, z: 1 };
    const t = rayEllipsoidIntersection(origin, dir, center, radii);
    expect(t).toBeNull();
  });

  it('hits an elongated ellipsoid', () => {
    const origin: Vector3 = { x: 0, y: 0, z: -10 };
    const dir: Vector3 = { x: 0, y: 0, z: 1 };
    const center: Vector3 = { x: 0, y: 0, z: 0 };
    const radii: Vector3 = { x: 0.5, y: 0.5, z: 5 };
    const t = rayEllipsoidIntersection(origin, dir, center, radii);
    expect(t).not.toBeNull();
    // Enters at z = -5, so t = -10 - (-5) = 5
    expect(t!).toBeCloseTo(5, 8);
  });

  it('returns 0 or small positive for ray originating inside the sphere', () => {
    const origin: Vector3 = { x: 0, y: 0, z: 0 }; // inside the sphere
    const dir: Vector3 = { x: 0, y: 0, z: 1 };
    const center: Vector3 = { x: 0, y: 0, z: 0 };
    const radii: Vector3 = { x: 2, y: 2, z: 2 };
    const t = rayEllipsoidIntersection(origin, dir, center, radii);
    // t0 should be negative (behind), t1 positive (the exit hit)
    expect(t).not.toBeNull();
    expect(t!).toBeGreaterThan(0);
    expect(t!).toBeCloseTo(2, 8); // exits at z = 2
  });
});

// ---------------------------------------------------------------------------
// rayAABBIntersection
// ---------------------------------------------------------------------------

describe('rayAABBIntersection', () => {
  const boxMin: Vector3 = { x: -1, y: -1, z: -1 };
  const boxMax: Vector3 = { x: 1, y: 1, z: 1 };

  it('hits a unit box from the front', () => {
    const origin: Vector3 = { x: 0, y: 0, z: -5 };
    const dir: Vector3 = { x: 0, y: 0, z: 1 };
    const t = rayAABBIntersection(origin, dir, boxMin, boxMax);
    expect(t).not.toBeNull();
    expect(t!).toBeCloseTo(4, 8); // enters at z = -1, origin.z = -5, t = 4
  });

  it('returns null when ray misses the box', () => {
    const origin: Vector3 = { x: 5, y: 5, z: -5 };
    const dir: Vector3 = { x: 0, y: 0, z: 1 };
    const t = rayAABBIntersection(origin, dir, boxMin, boxMax);
    expect(t).toBeNull();
  });

  it('returns 0 for ray originating inside the box', () => {
    const origin: Vector3 = { x: 0, y: 0, z: 0 };
    const dir: Vector3 = { x: 1, y: 0, z: 0 };
    const t = rayAABBIntersection(origin, dir, boxMin, boxMax);
    expect(t).toBe(0);
  });

  it('hits along the x-axis', () => {
    const origin: Vector3 = { x: -10, y: 0, z: 0 };
    const dir: Vector3 = { x: 1, y: 0, z: 0 };
    const t = rayAABBIntersection(origin, dir, boxMin, boxMax);
    expect(t).not.toBeNull();
    expect(t!).toBeCloseTo(9, 8); // enters at x = -1
  });

  it('returns null for ray pointing away from the box', () => {
    const origin: Vector3 = { x: 0, y: 0, z: 5 };
    const dir: Vector3 = { x: 0, y: 0, z: 1 }; // pointing away
    const t = rayAABBIntersection(origin, dir, boxMin, boxMax);
    expect(t).toBeNull();
  });

  it('handles ray parallel to one axis slab that is within the slab', () => {
    // Ray parallel to X, origin within Y and Z slabs
    const origin: Vector3 = { x: -5, y: 0, z: 0 };
    const dir: Vector3 = { x: 1, y: 0, z: 0 };
    const t = rayAABBIntersection(origin, dir, boxMin, boxMax);
    expect(t).not.toBeNull();
  });

  it('handles ray parallel to one axis slab that is outside the slab', () => {
    // Ray parallel to X, origin outside Y slab
    const origin: Vector3 = { x: -5, y: 5, z: 0 };
    const dir: Vector3 = { x: 1, y: 0, z: 0 };
    const t = rayAABBIntersection(origin, dir, boxMin, boxMax);
    expect(t).toBeNull();
  });
});
