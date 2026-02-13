// ---------------------------------------------------------------------------
// CV-10: Non-Maximum Suppression — standard NMS, soft-NMS, and box IoU.
// ---------------------------------------------------------------------------

import type { BBox2D } from '../types.js';

// ---------------------------------------------------------------------------
// bboxIoU
// ---------------------------------------------------------------------------

/**
 * Compute the Intersection over Union (IoU) between two 2D axis-aligned
 * bounding boxes.
 *
 * Each box is defined by its top-left corner `(x, y)` and its
 * `width` / `height`.  If the boxes do not overlap the IoU is 0.
 *
 * @param a First bounding box.
 * @param b Second bounding box.
 * @returns IoU value in [0, 1].
 */
export function bboxIoU(a: BBox2D, b: BBox2D): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);

  const interW = Math.max(0, x2 - x1);
  const interH = Math.max(0, y2 - y1);
  const interArea = interW * interH;

  const areaA = a.width * a.height;
  const areaB = b.width * b.height;
  const unionArea = areaA + areaB - interArea;

  return unionArea > 0 ? interArea / unionArea : 0;
}

// ---------------------------------------------------------------------------
// nonMaxSuppression
// ---------------------------------------------------------------------------

/**
 * Standard greedy Non-Maximum Suppression (NMS).
 *
 * Boxes are sorted by descending score.  The highest-scoring box is kept,
 * and all remaining boxes with an IoU above `iouThreshold` against it are
 * suppressed.  The process repeats until no boxes remain.
 *
 * @param boxes        Array of bounding boxes.
 * @param scores       Confidence scores for each box.
 * @param iouThreshold IoU threshold above which a box is suppressed.
 * @returns Array of kept box indices (into the original `boxes` array).
 */
export function nonMaxSuppression(
  boxes: BBox2D[],
  scores: Float64Array,
  iouThreshold: number,
): number[] {
  const n = boxes.length;

  // Build an index array sorted by descending score
  const order: number[] = [];
  for (let i = 0; i < n; i++) order.push(i);
  order.sort((a, b) => scores[b]! - scores[a]!);

  const suppressed = new Uint8Array(n);
  const kept: number[] = [];

  for (let i = 0; i < n; i++) {
    const idx = order[i]!;
    if (suppressed[idx]!) continue;

    kept.push(idx);

    for (let j = i + 1; j < n; j++) {
      const jdx = order[j]!;
      if (suppressed[jdx]!) continue;

      if (bboxIoU(boxes[idx]!, boxes[jdx]!) >= iouThreshold) {
        suppressed[jdx] = 1;
      }
    }
  }

  return kept;
}

// ---------------------------------------------------------------------------
// softNMS
// ---------------------------------------------------------------------------

/**
 * Gaussian Soft Non-Maximum Suppression (soft-NMS).
 *
 * Instead of hard suppression, overlapping boxes have their scores
 * decayed by a Gaussian penalty:
 *
 *   score_i *= exp( -iou^2 / sigma )
 *
 * The function processes boxes in order of current highest score (greedy).
 *
 * @param boxes        Array of bounding boxes.
 * @param scores       Confidence scores for each box.
 * @param iouThreshold IoU threshold below which no penalty is applied.
 * @param sigma        Gaussian decay parameter.
 * @returns A new Float64Array of adjusted scores (same length as input).
 */
export function softNMS(
  boxes: BBox2D[],
  scores: Float64Array,
  iouThreshold: number,
  sigma: number,
): Float64Array {
  const n = boxes.length;
  const adjusted = new Float64Array(scores);

  // Track which boxes have already been selected as the "current best"
  const processed = new Uint8Array(n);

  for (let iter = 0; iter < n; iter++) {
    // Find the unprocessed box with the highest current score
    let bestIdx = -1;
    let bestScore = -1;
    for (let i = 0; i < n; i++) {
      if (processed[i]!) continue;
      if (adjusted[i]! > bestScore) {
        bestScore = adjusted[i]!;
        bestIdx = i;
      }
    }

    if (bestIdx === -1) break;
    processed[bestIdx] = 1;

    // Decay overlapping boxes
    for (let i = 0; i < n; i++) {
      if (processed[i]!) continue;

      const iou = bboxIoU(boxes[bestIdx]!, boxes[i]!);
      if (iou >= iouThreshold) {
        // Gaussian decay
        adjusted[i] = adjusted[i]! * Math.exp(-(iou * iou) / sigma);
      }
    }
  }

  return adjusted;
}
