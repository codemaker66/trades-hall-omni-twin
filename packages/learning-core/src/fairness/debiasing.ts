// ---------------------------------------------------------------------------
// Debiasing via Exponentiated Gradient (Agarwal et al. 2018)
// Post-processing approach to achieve fairness constraints.
// ---------------------------------------------------------------------------

/**
 * Compute accuracy: fraction of correct predictions at threshold 0.5.
 */
function accuracy(predictions: number[], labels: number[]): number {
  const n = Math.min(predictions.length, labels.length);
  if (n === 0) return 0;
  let correct = 0;
  for (let i = 0; i < n; i++) {
    const pred = (predictions[i] ?? 0) >= 0.5 ? 1 : 0;
    const label = (labels[i] ?? 0) >= 0.5 ? 1 : 0;
    if (pred === label) correct++;
  }
  return correct / n;
}

/**
 * Compute demographic parity violation for a set of adjusted predictions.
 */
function dpViolation(predictions: number[], sensitiveAttr: number[]): number {
  const n = Math.min(predictions.length, sensitiveAttr.length);
  let posGroup0 = 0, countGroup0 = 0;
  let posGroup1 = 0, countGroup1 = 0;

  for (let i = 0; i < n; i++) {
    const pred = (predictions[i] ?? 0) >= 0.5 ? 1 : 0;
    if ((sensitiveAttr[i] ?? 0) < 0.5) {
      countGroup0++;
      posGroup0 += pred;
    } else {
      countGroup1++;
      posGroup1 += pred;
    }
  }

  const rate0 = countGroup0 > 0 ? posGroup0 / countGroup0 : 0;
  const rate1 = countGroup1 > 0 ? posGroup1 / countGroup1 : 0;
  return Math.abs(rate0 - rate1);
}

/**
 * Compute equalized odds violation (max of TPR diff and FPR diff).
 */
function eoViolation(
  predictions: number[],
  labels: number[],
  sensitiveAttr: number[],
): number {
  const n = Math.min(predictions.length, labels.length, sensitiveAttr.length);
  let tp0 = 0, fn0 = 0, fp0 = 0, tn0 = 0;
  let tp1 = 0, fn1 = 0, fp1 = 0, tn1 = 0;

  for (let i = 0; i < n; i++) {
    const pred = (predictions[i] ?? 0) >= 0.5 ? 1 : 0;
    const label = (labels[i] ?? 0) >= 0.5 ? 1 : 0;

    if ((sensitiveAttr[i] ?? 0) < 0.5) {
      if (label === 1 && pred === 1) tp0++;
      else if (label === 1 && pred === 0) fn0++;
      else if (label === 0 && pred === 1) fp0++;
      else tn0++;
    } else {
      if (label === 1 && pred === 1) tp1++;
      else if (label === 1 && pred === 0) fn1++;
      else if (label === 0 && pred === 1) fp1++;
      else tn1++;
    }
  }

  const tpr0 = (tp0 + fn0) > 0 ? tp0 / (tp0 + fn0) : 0;
  const tpr1 = (tp1 + fn1) > 0 ? tp1 / (tp1 + fn1) : 0;
  const fpr0 = (fp0 + tn0) > 0 ? fp0 / (fp0 + tn0) : 0;
  const fpr1 = (fp1 + tn1) > 0 ? fp1 / (fp1 + tn1) : 0;

  return Math.max(Math.abs(tpr0 - tpr1), Math.abs(fpr0 - fpr1));
}

/**
 * Exponentiated gradient method for fair classification.
 *
 * Adjusts prediction scores to satisfy fairness constraints within epsilon
 * by learning per-group thresholds via exponentiated gradient updates.
 *
 * The algorithm maintains multipliers (weights) for constraint violations:
 * 1. Compute the fairness constraint violation for current adjusted predictions
 * 2. Update multipliers: λ *= exp(η * violation) (exponentiated gradient)
 * 3. Adjust predictions: shift group-specific predictions to reduce violations
 * 4. Repeat until convergence or maxIter
 *
 * @param predictions - Original predicted probabilities in [0, 1]
 * @param labels - Binary labels (0 or 1)
 * @param sensitiveAttr - Binary sensitive attribute (0 or 1)
 * @param constraint - Fairness constraint type: 'dp' (demographic parity) or 'eo' (equalized odds)
 * @param epsilon - Maximum allowed fairness violation
 * @param maxIter - Maximum iterations
 * @param lr - Learning rate for exponentiated gradient updates
 * @returns Adjusted prediction scores
 */
export function exponentiatedGradient(
  predictions: number[],
  labels: number[],
  sensitiveAttr: number[],
  constraint: 'dp' | 'eo',
  epsilon: number,
  maxIter: number,
  lr: number,
): number[] {
  const n = Math.min(predictions.length, labels.length, sensitiveAttr.length);
  if (n === 0) return [];

  // Work on a copy of predictions
  const adjusted: number[] = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    adjusted[i] = predictions[i] ?? 0;
  }

  // Compute group indices
  const group0Indices: number[] = [];
  const group1Indices: number[] = [];
  for (let i = 0; i < n; i++) {
    if ((sensitiveAttr[i] ?? 0) < 0.5) {
      group0Indices.push(i);
    } else {
      group1Indices.push(i);
    }
  }

  // Exponentiated gradient multipliers
  // For DP: one multiplier per direction (rate0 > rate1, rate0 < rate1)
  // For EO: multipliers for TPR and FPR differences
  let lambdaPos = 1.0; // penalizes group0 rate > group1 rate
  let lambdaNeg = 1.0; // penalizes group1 rate > group0 rate

  for (let iter = 0; iter < maxIter; iter++) {
    // Compute current violation
    let violation: number;
    if (constraint === 'dp') {
      violation = dpViolation(adjusted, sensitiveAttr);
    } else {
      violation = eoViolation(adjusted, labels, sensitiveAttr);
    }

    // Check convergence
    if (violation <= epsilon) break;

    // Compute signed violation (directional)
    // Positive means group0 has higher rate than group1
    let signedViolation: number;
    if (constraint === 'dp') {
      let posGroup0 = 0, posGroup1 = 0;
      for (const idx of group0Indices) {
        if ((adjusted[idx] ?? 0) >= 0.5) posGroup0++;
      }
      for (const idx of group1Indices) {
        if ((adjusted[idx] ?? 0) >= 0.5) posGroup1++;
      }
      const rate0 = group0Indices.length > 0 ? posGroup0 / group0Indices.length : 0;
      const rate1 = group1Indices.length > 0 ? posGroup1 / group1Indices.length : 0;
      signedViolation = rate0 - rate1;
    } else {
      // For EO, use the TPR difference as signed violation
      let tp0 = 0, fn0 = 0, tp1 = 0, fn1 = 0;
      for (const idx of group0Indices) {
        const pred = (adjusted[idx] ?? 0) >= 0.5 ? 1 : 0;
        const label = (labels[idx] ?? 0) >= 0.5 ? 1 : 0;
        if (label === 1) { if (pred === 1) tp0++; else fn0++; }
      }
      for (const idx of group1Indices) {
        const pred = (adjusted[idx] ?? 0) >= 0.5 ? 1 : 0;
        const label = (labels[idx] ?? 0) >= 0.5 ? 1 : 0;
        if (label === 1) { if (pred === 1) tp1++; else fn1++; }
      }
      const tpr0 = (tp0 + fn0) > 0 ? tp0 / (tp0 + fn0) : 0;
      const tpr1 = (tp1 + fn1) > 0 ? tp1 / (tp1 + fn1) : 0;
      signedViolation = tpr0 - tpr1;
    }

    // Exponentiated gradient update
    lambdaPos *= Math.exp(lr * Math.max(0, signedViolation - epsilon));
    lambdaNeg *= Math.exp(lr * Math.max(0, -signedViolation - epsilon));

    // Compute adjustment: shift predictions to reduce violation
    // The net multiplier determines which group gets shifted
    const netLambda = lambdaPos - lambdaNeg;

    // Adjust scores: if netLambda > 0, group0 rate is too high
    // → decrease group0 scores or increase group1 scores
    const shift = netLambda * lr * 0.1;

    if (signedViolation > epsilon) {
      // Group 0 rate too high: decrease group 0 predictions
      for (const idx of group0Indices) {
        adjusted[idx] = Math.max(0, Math.min(1, (adjusted[idx] ?? 0) - Math.abs(shift)));
      }
    } else if (signedViolation < -epsilon) {
      // Group 1 rate too high: decrease group 1 predictions
      for (const idx of group1Indices) {
        adjusted[idx] = Math.max(0, Math.min(1, (adjusted[idx] ?? 0) - Math.abs(shift)));
      }
    }
  }

  return adjusted;
}

/**
 * Compute the Pareto front of accuracy vs. fairness violation.
 *
 * Sweeps over threshold values for binarizing predictions and computes
 * the accuracy and fairness violation at each threshold. Returns the
 * Pareto-optimal points (no other point has both higher accuracy AND
 * lower fairness violation).
 *
 * @param predictions - Predicted probabilities in [0, 1]
 * @param labels - Binary labels (0 or 1)
 * @param sensitiveAttr - Binary sensitive attribute (0 or 1)
 * @param nPoints - Number of threshold points to evaluate
 * @returns Array of Pareto-optimal { accuracy, fairnessViolation, threshold }
 */
export function paretoFairnessAccuracy(
  predictions: number[],
  labels: number[],
  sensitiveAttr: number[],
  nPoints: number,
): Array<{ accuracy: number; fairnessViolation: number; threshold: number }> {
  const n = Math.min(predictions.length, labels.length, sensitiveAttr.length);
  if (n === 0 || nPoints <= 0) return [];

  // Evaluate accuracy and fairness at different thresholds
  const candidates: Array<{ accuracy: number; fairnessViolation: number; threshold: number }> = [];

  for (let t = 0; t < nPoints; t++) {
    const threshold = (t + 0.5) / nPoints;

    // Create binary predictions at this threshold
    const binaryPreds: number[] = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      binaryPreds[i] = (predictions[i] ?? 0) >= threshold ? 1 : 0;
    }

    const acc = accuracy(binaryPreds, labels);
    const violation = dpViolation(binaryPreds, sensitiveAttr);

    candidates.push({
      accuracy: acc,
      fairnessViolation: violation,
      threshold,
    });
  }

  // Extract Pareto front:
  // A point is Pareto-optimal if no other point dominates it
  // (has both higher accuracy AND lower fairness violation)
  // Sort by accuracy descending for efficient filtering
  candidates.sort((a, b) => b.accuracy - a.accuracy);

  const pareto: Array<{ accuracy: number; fairnessViolation: number; threshold: number }> = [];
  let minViolation = Infinity;

  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i]!;
    // A point is Pareto-optimal if its fairness violation is the lowest
    // seen so far (considering we're going in decreasing accuracy order)
    if (c.fairnessViolation < minViolation) {
      pareto.push(c);
      minViolation = c.fairnessViolation;
    }
  }

  // Sort Pareto front by threshold for readability
  pareto.sort((a, b) => a.threshold - b.threshold);

  return pareto;
}
