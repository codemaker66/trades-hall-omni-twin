// ---------------------------------------------------------------------------
// SP-4: Median Filter
// ---------------------------------------------------------------------------
// Robust nonlinear filter: 50% breakdown point.
// Removes spike outliers without affecting step edges.
// O(N·k) where k is kernel size; efficient for small kernels.

/**
 * Median filter with specified kernel size (must be odd).
 */
export function medianFilter(signal: Float64Array, kernelSize: number = 5): Float64Array {
  if (kernelSize % 2 === 0) throw new Error('kernelSize must be odd');

  const N = signal.length;
  const half = Math.floor(kernelSize / 2);
  const result = new Float64Array(N);
  const window = new Float64Array(kernelSize);

  for (let i = 0; i < N; i++) {
    // Fill window with values from signal (mirror boundary)
    for (let j = 0; j < kernelSize; j++) {
      let idx = i + j - half;
      if (idx < 0) idx = -idx;
      if (idx >= N) idx = 2 * (N - 1) - idx;
      idx = Math.max(0, Math.min(N - 1, idx));
      window[j] = signal[idx]!;
    }

    // Sort and take median
    const sorted = Float64Array.from(window).sort();
    result[i] = sorted[half]!;
  }

  return result;
}

/**
 * Weighted median filter: center-weighted for better peak preservation.
 * Center weight is tripled.
 */
export function weightedMedianFilter(signal: Float64Array, kernelSize: number = 5): Float64Array {
  if (kernelSize % 2 === 0) throw new Error('kernelSize must be odd');

  const N = signal.length;
  const half = Math.floor(kernelSize / 2);
  const result = new Float64Array(N);

  for (let i = 0; i < N; i++) {
    // Build weighted sample set (center value repeated 3×)
    const values: number[] = [];
    for (let j = 0; j < kernelSize; j++) {
      let idx = i + j - half;
      if (idx < 0) idx = -idx;
      if (idx >= N) idx = 2 * (N - 1) - idx;
      idx = Math.max(0, Math.min(N - 1, idx));
      const val = signal[idx]!;
      values.push(val);
      if (j === half) {
        values.push(val);
        values.push(val);
      }
    }

    values.sort((a, b) => a - b);
    result[i] = values[Math.floor(values.length / 2)]!;
  }

  return result;
}
