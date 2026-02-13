import { describe, it, expect } from 'vitest';
import {
  createMask,
  maskUnion,
  maskIntersection,
  computeIoU,
  connectedComponents,
  nonMaxSuppression,
  softNMS,
  bboxIoU,
  createBBox2D,
  bboxArea,
  bboxContains,
  fitOrientedBBox,
  mergeBBoxes,
} from '../segmentation/index.js';

// ---------------------------------------------------------------------------
// Helper: fill a rectangular region of a mask with 1s
// ---------------------------------------------------------------------------
function fillRect(
  mask: { data: Uint32Array; width: number; height: number },
  x0: number,
  y0: number,
  w: number,
  h: number,
): void {
  for (let y = y0; y < y0 + h; y++) {
    for (let x = x0; x < x0 + w; x++) {
      mask.data[y * mask.width + x] = 1;
    }
  }
}

// ---------------------------------------------------------------------------
// Mask operations
// ---------------------------------------------------------------------------

describe('maskUnion', () => {
  it('should produce a superset of both input masks', () => {
    const a = createMask(10, 10);
    const b = createMask(10, 10);
    fillRect(a, 0, 0, 5, 5); // top-left quadrant
    fillRect(b, 5, 5, 5, 5); // bottom-right quadrant

    const union = maskUnion(a, b);
    // Every pixel set in a must be set in the union
    for (let i = 0; i < 100; i++) {
      if (a.data[i]! !== 0) {
        expect(union.data[i]!).toBe(1);
      }
      if (b.data[i]! !== 0) {
        expect(union.data[i]!).toBe(1);
      }
    }
    // Union should have exactly 25 + 25 = 50 foreground pixels (disjoint)
    let fgCount = 0;
    for (let i = 0; i < 100; i++) {
      if (union.data[i]! !== 0) fgCount++;
    }
    expect(fgCount).toBe(50);
  });
});

describe('computeIoU', () => {
  it('should return 1 for identical masks', () => {
    const a = createMask(8, 8);
    fillRect(a, 1, 1, 4, 4);
    const b = createMask(8, 8);
    fillRect(b, 1, 1, 4, 4);

    expect(computeIoU(a, b)).toBeCloseTo(1.0, 10);
  });

  it('should return 0 for completely disjoint masks', () => {
    const a = createMask(10, 10);
    const b = createMask(10, 10);
    fillRect(a, 0, 0, 3, 3);
    fillRect(b, 7, 7, 3, 3);

    expect(computeIoU(a, b)).toBeCloseTo(0.0, 10);
  });

  it('should return a value between 0 and 1 for partially overlapping masks', () => {
    const a = createMask(10, 10);
    const b = createMask(10, 10);
    fillRect(a, 0, 0, 6, 6); // 36 pixels
    fillRect(b, 3, 3, 6, 6); // 36 pixels, overlap = 3x3 = 9

    const iou = computeIoU(a, b);
    // intersection = 9, union = 36 + 36 - 9 = 63
    expect(iou).toBeCloseTo(9 / 63, 10);
  });
});

// ---------------------------------------------------------------------------
// Connected components
// ---------------------------------------------------------------------------

describe('connectedComponents', () => {
  it('should count two separate components in a disjoint mask', () => {
    const mask = createMask(10, 10);
    fillRect(mask, 0, 0, 3, 3); // top-left block
    fillRect(mask, 7, 7, 3, 3); // bottom-right block

    const result = connectedComponents(mask);
    expect(result.count).toBe(2);
  });

  it('should count one component when regions are connected', () => {
    const mask = createMask(10, 1);
    // Fill entire row
    for (let x = 0; x < 10; x++) {
      mask.data[x] = 1;
    }

    const result = connectedComponents(mask);
    expect(result.count).toBe(1);
  });

  it('should return 0 components for an empty mask', () => {
    const mask = createMask(5, 5);
    const result = connectedComponents(mask);
    expect(result.count).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// NMS
// ---------------------------------------------------------------------------

describe('nonMaxSuppression', () => {
  it('should suppress overlapping boxes above the IoU threshold', () => {
    // Two nearly identical boxes, one with higher score
    const boxes = [
      createBBox2D(0, 0, 100, 100),
      createBBox2D(5, 5, 100, 100),
      createBBox2D(200, 200, 50, 50), // far away, should survive
    ];
    const scores = new Float64Array([0.9, 0.8, 0.7]);

    const kept = nonMaxSuppression(boxes, scores, 0.5);
    // The first two overlap heavily so only one survives, plus the third
    expect(kept.length).toBe(2);
    expect(kept).toContain(0); // highest score
    expect(kept).toContain(2); // far away
  });

  it('should keep all boxes when none overlap', () => {
    const boxes = [
      createBBox2D(0, 0, 10, 10),
      createBBox2D(100, 100, 10, 10),
      createBBox2D(200, 200, 10, 10),
    ];
    const scores = new Float64Array([0.9, 0.8, 0.7]);

    const kept = nonMaxSuppression(boxes, scores, 0.5);
    expect(kept.length).toBe(3);
  });
});

describe('softNMS', () => {
  it('should preserve more boxes than hard NMS by decaying scores', () => {
    // Two overlapping boxes
    const boxes = [
      createBBox2D(0, 0, 100, 100),
      createBBox2D(10, 10, 100, 100),
    ];
    const scores = new Float64Array([0.9, 0.85]);

    // Hard NMS would suppress the second box
    const hardKept = nonMaxSuppression(boxes, scores, 0.5);

    // Soft NMS should keep both (with decayed score for second)
    const adjusted = softNMS(boxes, scores, 0.3, 0.5);

    // Hard NMS keeps only 1
    expect(hardKept.length).toBe(1);

    // Soft NMS: second box's score should be decayed but still > 0
    expect(adjusted[1]!).toBeGreaterThan(0);
    expect(adjusted[1]!).toBeLessThan(0.85);
    // First box keeps its score
    expect(adjusted[0]!).toBeCloseTo(0.9, 10);
  });
});

// ---------------------------------------------------------------------------
// BBox utilities
// ---------------------------------------------------------------------------

describe('bboxArea', () => {
  it('should return correct area for a known box', () => {
    const box = createBBox2D(10, 20, 30, 40);
    expect(bboxArea(box)).toBe(1200);
  });

  it('should return 0 for a zero-dimension box', () => {
    const box = createBBox2D(5, 5, 0, 10);
    expect(bboxArea(box)).toBe(0);
  });
});

describe('bboxContains', () => {
  it('should return true for a point inside the box', () => {
    const box = createBBox2D(10, 10, 50, 50);
    expect(bboxContains(box, { x: 30, y: 30 })).toBe(true);
  });

  it('should return false for a point outside the box', () => {
    const box = createBBox2D(10, 10, 50, 50);
    expect(bboxContains(box, { x: 100, y: 100 })).toBe(false);
  });

  it('should be inclusive on the min edge, exclusive on the max edge', () => {
    const box = createBBox2D(0, 0, 10, 10);
    expect(bboxContains(box, { x: 0, y: 0 })).toBe(true); // min edge: inclusive
    expect(bboxContains(box, { x: 10, y: 10 })).toBe(false); // max edge: exclusive
  });
});

describe('bboxIoU', () => {
  it('should return 1 for identical boxes', () => {
    const a = createBBox2D(0, 0, 10, 10);
    const b = createBBox2D(0, 0, 10, 10);
    expect(bboxIoU(a, b)).toBeCloseTo(1.0, 10);
  });

  it('should return 0 for non-overlapping boxes', () => {
    const a = createBBox2D(0, 0, 10, 10);
    const b = createBBox2D(20, 20, 10, 10);
    expect(bboxIoU(a, b)).toBeCloseTo(0.0, 10);
  });
});

// ---------------------------------------------------------------------------
// Oriented BBox
// ---------------------------------------------------------------------------

describe('fitOrientedBBox', () => {
  it('should produce an OBB that contains all input points', () => {
    const points = [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 5 },
      { x: 0, y: 5 },
      { x: 5, y: 2.5 },
    ];

    const obb = fitOrientedBBox(points);

    // The oriented bbox center should be approximately at the centroid of
    // a rectangle spanning [0,10] x [0,5]
    expect(obb.center.x).toBeCloseTo(5.0, 0);
    expect(obb.center.y).toBeCloseTo(2.5, 0);

    // Half-extents should be non-negative
    expect(obb.halfExtents.x).toBeGreaterThanOrEqual(0);
    expect(obb.halfExtents.y).toBeGreaterThanOrEqual(0);

    // Quaternion should be normalised
    const q = obb.orientation;
    const qLen = Math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    expect(qLen).toBeCloseTo(1.0, 10);
  });
});

// ---------------------------------------------------------------------------
// Merge bboxes
// ---------------------------------------------------------------------------

describe('mergeBBoxes', () => {
  it('should produce a box that encloses all input boxes', () => {
    const boxes = [
      createBBox2D(0, 0, 10, 10),
      createBBox2D(20, 20, 10, 10),
      createBBox2D(5, 5, 5, 5),
    ];

    const merged = mergeBBoxes(boxes);

    // Merged box must start at (0,0) and extend to (30,30)
    expect(merged.x).toBeCloseTo(0, 10);
    expect(merged.y).toBeCloseTo(0, 10);
    expect(merged.width).toBeCloseTo(30, 10);
    expect(merged.height).toBeCloseTo(30, 10);
  });

  it('should return a zero-size box for an empty array', () => {
    const merged = mergeBBoxes([]);
    expect(merged.width).toBe(0);
    expect(merged.height).toBe(0);
  });
});
