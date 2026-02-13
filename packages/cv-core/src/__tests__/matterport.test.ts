// ---------------------------------------------------------------------------
// Tests: CV-1 Matterport — sweeps, floor plans, E57 parsing, transforms
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';
import {
  createSweep,
  createFloorPlan,
  sweepDistance,
  computeRoomArea,
  extractRoomBoundaries,
  isPointInRoom,
  computeFloorPlanBounds,
  parseE57Header,
  transformPoint,
  computeTransformFromPoints,
} from '../matterport/index.js';
import type { Room, Vec2, Vector3 } from '../types.js';
import { mat4Identity } from '../types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a minimal Room value from a polygon and pre-computed area. */
function makeRoom(name: string, polygon: Vec2[], area: number): Room {
  return { name, polygon, area };
}

// ---------------------------------------------------------------------------
// createSweep
// ---------------------------------------------------------------------------

describe('createSweep', () => {
  it('stores the provided id, position, and rotation', () => {
    const sweep = createSweep(
      'sweep-1',
      { x: 1, y: 2, z: 3 },
      { x: 0, y: 0, z: 0, w: 1 },
    );
    expect(sweep.id).toBe('sweep-1');
    expect(sweep.position).toEqual({ x: 1, y: 2, z: 3 });
    expect(sweep.rotation).toEqual({ x: 0, y: 0, z: 0, w: 1 });
  });

  it('defaults floorIndex to 0', () => {
    const sweep = createSweep(
      'sweep-2',
      { x: 0, y: 0, z: 0 },
      { x: 0, y: 0, z: 0, w: 1 },
    );
    expect(sweep.floorIndex).toBe(0);
  });

  it('assigns a valid ISO-8601 timestamp', () => {
    const sweep = createSweep(
      's',
      { x: 0, y: 0, z: 0 },
      { x: 0, y: 0, z: 0, w: 1 },
    );
    expect(sweep.timestamp).toBeDefined();
    const parsed = Date.parse(sweep.timestamp!);
    expect(Number.isNaN(parsed)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// sweepDistance
// ---------------------------------------------------------------------------

describe('sweepDistance', () => {
  it('returns 0 for identical sweeps', () => {
    const a = createSweep('a', { x: 5, y: 5, z: 5 }, { x: 0, y: 0, z: 0, w: 1 });
    expect(sweepDistance(a, a)).toBeCloseTo(0, 10);
  });

  it('returns correct Euclidean distance for axis-aligned offset', () => {
    const a = createSweep('a', { x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 0, w: 1 });
    const b = createSweep('b', { x: 3, y: 4, z: 0 }, { x: 0, y: 0, z: 0, w: 1 });
    expect(sweepDistance(a, b)).toBeCloseTo(5, 10);
  });

  it('returns correct distance for a 3D diagonal', () => {
    const a = createSweep('a', { x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 0, w: 1 });
    const b = createSweep('b', { x: 1, y: 1, z: 1 }, { x: 0, y: 0, z: 0, w: 1 });
    expect(sweepDistance(a, b)).toBeCloseTo(Math.sqrt(3), 10);
  });
});

// ---------------------------------------------------------------------------
// computeRoomArea — Shoelace formula
// ---------------------------------------------------------------------------

describe('computeRoomArea', () => {
  it('computes the area of a unit square', () => {
    const square: Vec2[] = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
      { x: 0, y: 1 },
    ];
    expect(computeRoomArea(square)).toBeCloseTo(1, 10);
  });

  it('computes the area of a right triangle (3-4-5)', () => {
    const triangle: Vec2[] = [
      { x: 0, y: 0 },
      { x: 3, y: 0 },
      { x: 0, y: 4 },
    ];
    expect(computeRoomArea(triangle)).toBeCloseTo(6, 10);
  });

  it('returns 0 for fewer than 3 vertices', () => {
    expect(computeRoomArea([{ x: 0, y: 0 }])).toBe(0);
    expect(computeRoomArea([])).toBe(0);
  });

  it('works for a 10x5 rectangle', () => {
    const rect: Vec2[] = [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 5 },
      { x: 0, y: 5 },
    ];
    expect(computeRoomArea(rect)).toBeCloseTo(50, 10);
  });

  it('gives the same area regardless of CW/CCW winding', () => {
    const ccw: Vec2[] = [
      { x: 0, y: 0 },
      { x: 4, y: 0 },
      { x: 4, y: 3 },
    ];
    const cw: Vec2[] = [...ccw].reverse();
    expect(computeRoomArea(ccw)).toBeCloseTo(computeRoomArea(cw), 10);
  });
});

// ---------------------------------------------------------------------------
// isPointInRoom
// ---------------------------------------------------------------------------

describe('isPointInRoom', () => {
  const squareRoom = makeRoom(
    'square',
    [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 10 },
      { x: 0, y: 10 },
    ],
    100,
  );

  it('returns true for a point inside the room', () => {
    expect(isPointInRoom({ x: 5, y: 5 }, squareRoom)).toBe(true);
  });

  it('returns false for a point outside the room', () => {
    expect(isPointInRoom({ x: 15, y: 5 }, squareRoom)).toBe(false);
  });

  it('returns false for a point clearly below the room', () => {
    expect(isPointInRoom({ x: 5, y: -5 }, squareRoom)).toBe(false);
  });

  it('returns true near the centre of a triangle room', () => {
    const triRoom = makeRoom(
      'tri',
      [
        { x: 0, y: 0 },
        { x: 10, y: 0 },
        { x: 5, y: 10 },
      ],
      50,
    );
    expect(isPointInRoom({ x: 5, y: 3 }, triRoom)).toBe(true);
  });

  it('returns false for a degenerate room (< 3 vertices)', () => {
    const degenerate = makeRoom('degen', [{ x: 0, y: 0 }, { x: 1, y: 1 }], 0);
    expect(isPointInRoom({ x: 0.5, y: 0.5 }, degenerate)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// computeFloorPlanBounds
// ---------------------------------------------------------------------------

describe('computeFloorPlanBounds', () => {
  it('returns zeros for empty rooms array', () => {
    const bounds = computeFloorPlanBounds([]);
    expect(bounds.min).toEqual({ x: 0, y: 0 });
    expect(bounds.max).toEqual({ x: 0, y: 0 });
  });

  it('returns correct bounding box for a single room', () => {
    const room = makeRoom(
      'r',
      [
        { x: 2, y: 3 },
        { x: 8, y: 3 },
        { x: 8, y: 7 },
        { x: 2, y: 7 },
      ],
      30,
    );
    const bounds = computeFloorPlanBounds([room]);
    expect(bounds.min.x).toBeCloseTo(2, 10);
    expect(bounds.min.y).toBeCloseTo(3, 10);
    expect(bounds.max.x).toBeCloseTo(8, 10);
    expect(bounds.max.y).toBeCloseTo(7, 10);
  });

  it('returns the union bounding box across multiple rooms', () => {
    const r1 = makeRoom('r1', [{ x: 0, y: 0 }, { x: 5, y: 0 }, { x: 5, y: 5 }], 12.5);
    const r2 = makeRoom('r2', [{ x: 10, y: 10 }, { x: 20, y: 10 }, { x: 20, y: 20 }], 50);
    const bounds = computeFloorPlanBounds([r1, r2]);
    expect(bounds.min.x).toBeCloseTo(0, 10);
    expect(bounds.min.y).toBeCloseTo(0, 10);
    expect(bounds.max.x).toBeCloseTo(20, 10);
    expect(bounds.max.y).toBeCloseTo(20, 10);
  });
});

// ---------------------------------------------------------------------------
// extractRoomBoundaries
// ---------------------------------------------------------------------------

describe('extractRoomBoundaries', () => {
  it('returns one boundary per room', () => {
    const rooms = [
      makeRoom('a', [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }], 0.5),
      makeRoom('b', [{ x: 2, y: 2 }, { x: 3, y: 2 }, { x: 3, y: 3 }], 0.5),
    ];
    const boundaries = extractRoomBoundaries(rooms);
    expect(boundaries).toHaveLength(2);
    expect(boundaries[0]!).toHaveLength(3);
    expect(boundaries[1]!).toHaveLength(3);
  });

  it('deep-copies vertices so mutations do not propagate', () => {
    const rooms = [makeRoom('x', [{ x: 5, y: 5 }], 0)];
    const boundaries = extractRoomBoundaries(rooms);
    boundaries[0]![0]!.x = 999;
    expect(rooms[0]!.polygon[0]!.x).toBe(5);
  });
});

// ---------------------------------------------------------------------------
// createFloorPlan
// ---------------------------------------------------------------------------

describe('createFloorPlan', () => {
  it('computes width and height from rooms', () => {
    const rooms = [
      makeRoom(
        'r',
        [
          { x: 0, y: 0 },
          { x: 10, y: 0 },
          { x: 10, y: 5 },
          { x: 0, y: 5 },
        ],
        50,
      ),
    ];
    const fp = createFloorPlan(rooms);
    expect(fp.width).toBeCloseTo(10, 10);
    expect(fp.height).toBeCloseTo(5, 10);
    expect(fp.floorIndex).toBe(0);
    expect(fp.rooms).toBe(rooms);
  });

  it('returns zero dimensions for empty rooms', () => {
    const fp = createFloorPlan([]);
    expect(fp.width).toBe(0);
    expect(fp.height).toBe(0);
    expect(fp.boundary).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// parseE57Header
// ---------------------------------------------------------------------------

describe('parseE57Header', () => {
  it('parses scan count, point count, and bounds from Float64Array', () => {
    const data = new Float64Array([3, 100000, -1, -2, -3, 4, 5, 6]);
    const header = parseE57Header(data);
    expect(header.scanCount).toBe(3);
    expect(header.totalPointCount).toBe(100000);
    expect(header.boundsMin).toEqual({ x: -1, y: -2, z: -3 });
    expect(header.boundsMax).toEqual({ x: 4, y: 5, z: 6 });
  });

  it('throws when data is shorter than 8 elements', () => {
    expect(() => parseE57Header(new Float64Array([1, 2]))).toThrow();
  });

  it('assigns placeholder guid and coordinateSystem', () => {
    const header = parseE57Header(new Float64Array(8));
    expect(header.guid).toContain('0000');
    expect(header.coordinateSystem).toBe('unknown');
  });

  it('handles fractional values for bounds', () => {
    const data = new Float64Array([1, 1, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]);
    const header = parseE57Header(data);
    expect(header.boundsMin.x).toBeCloseTo(0.5, 10);
    expect(header.boundsMax.z).toBeCloseTo(5.5, 10);
  });

  it('parses large arrays (extra elements beyond 8 are ignored)', () => {
    const data = new Float64Array(64);
    data[0] = 7;
    data[1] = 999;
    const header = parseE57Header(data);
    expect(header.scanCount).toBe(7);
    expect(header.totalPointCount).toBe(999);
  });
});

// ---------------------------------------------------------------------------
// transformPoint
// ---------------------------------------------------------------------------

describe('transformPoint', () => {
  it('identity transform returns the same point', () => {
    const matrix = mat4Identity();
    const result = transformPoint(
      { x: 3, y: 4, z: 5 },
      { matrix, sourceSystem: 'a', targetSystem: 'b', scale: 1 },
    );
    expect(result.x).toBeCloseTo(3, 10);
    expect(result.y).toBeCloseTo(4, 10);
    expect(result.z).toBeCloseTo(5, 10);
  });

  it('applies translation from column 3 of the matrix', () => {
    const matrix = mat4Identity();
    matrix[12] = 10;
    matrix[13] = 20;
    matrix[14] = 30;
    const result = transformPoint(
      { x: 0, y: 0, z: 0 },
      { matrix, sourceSystem: '', targetSystem: '', scale: 1 },
    );
    expect(result.x).toBeCloseTo(10, 10);
    expect(result.y).toBeCloseTo(20, 10);
    expect(result.z).toBeCloseTo(30, 10);
  });

  it('applies uniform scale before the matrix transform', () => {
    const matrix = mat4Identity();
    const result = transformPoint(
      { x: 1, y: 2, z: 3 },
      { matrix, sourceSystem: '', targetSystem: '', scale: 2 },
    );
    expect(result.x).toBeCloseTo(2, 10);
    expect(result.y).toBeCloseTo(4, 10);
    expect(result.z).toBeCloseTo(6, 10);
  });

  it('combined scale + translation', () => {
    const matrix = mat4Identity();
    matrix[12] = 5;
    matrix[13] = 5;
    matrix[14] = 5;
    const result = transformPoint(
      { x: 1, y: 1, z: 1 },
      { matrix, sourceSystem: '', targetSystem: '', scale: 3 },
    );
    expect(result.x).toBeCloseTo(8, 10);
    expect(result.y).toBeCloseTo(8, 10);
    expect(result.z).toBeCloseTo(8, 10);
  });
});

// ---------------------------------------------------------------------------
// computeTransformFromPoints
// ---------------------------------------------------------------------------

describe('computeTransformFromPoints', () => {
  it('throws for fewer than 3 correspondences', () => {
    const pts: Vector3[] = [
      { x: 0, y: 0, z: 0 },
      { x: 1, y: 0, z: 0 },
    ];
    expect(() => computeTransformFromPoints(pts, pts)).toThrow();
  });

  it('throws when src and dst have different lengths', () => {
    const src: Vector3[] = [
      { x: 0, y: 0, z: 0 },
      { x: 1, y: 0, z: 0 },
      { x: 0, y: 1, z: 0 },
    ];
    const dst: Vector3[] = [
      { x: 0, y: 0, z: 0 },
      { x: 1, y: 0, z: 0 },
    ];
    expect(() => computeTransformFromPoints(src, dst)).toThrow();
  });

  it('identity case: transform from points to themselves', () => {
    const pts: Vector3[] = [
      { x: 1, y: 0, z: 0 },
      { x: 0, y: 1, z: 0 },
      { x: 0, y: 0, z: 1 },
      { x: 1, y: 1, z: 1 },
    ];
    const transform = computeTransformFromPoints(pts, pts);

    // Apply the transform to one of the source points
    const result = transformPoint(pts[0]!, transform);
    expect(result.x).toBeCloseTo(1, 5);
    expect(result.y).toBeCloseTo(0, 5);
    expect(result.z).toBeCloseTo(0, 5);
  });

  it('recovers a pure translation', () => {
    const src: Vector3[] = [
      { x: 0, y: 0, z: 0 },
      { x: 1, y: 0, z: 0 },
      { x: 0, y: 1, z: 0 },
      { x: 0, y: 0, z: 1 },
    ];
    const offset = { x: 5, y: -3, z: 7 };
    const dst: Vector3[] = src.map((p) => ({
      x: p.x + offset.x,
      y: p.y + offset.y,
      z: p.z + offset.z,
    }));
    const transform = computeTransformFromPoints(src, dst);

    for (let i = 0; i < src.length; i++) {
      const result = transformPoint(src[i]!, transform);
      expect(result.x).toBeCloseTo(dst[i]!.x, 5);
      expect(result.y).toBeCloseTo(dst[i]!.y, 5);
      expect(result.z).toBeCloseTo(dst[i]!.z, 5);
    }
  });

  it('returns scale = 1', () => {
    const pts: Vector3[] = [
      { x: 1, y: 0, z: 0 },
      { x: 0, y: 1, z: 0 },
      { x: 0, y: 0, z: 1 },
    ];
    const transform = computeTransformFromPoints(pts, pts);
    expect(transform.scale).toBe(1);
  });
});
