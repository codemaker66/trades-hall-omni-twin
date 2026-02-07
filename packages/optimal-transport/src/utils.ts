/**
 * Numerical utilities for the OT package.
 */

/**
 * Log-sum-exp with max-shift for numerical stability:
 * LSE(x) = max(x) + log(Σ exp(x_i - max(x)))
 */
export function logSumExp(values: Float64Array): number {
  if (values.length === 0) return -Infinity
  let maxVal = -Infinity
  for (let i = 0; i < values.length; i++) {
    if (values[i]! > maxVal) maxVal = values[i]!
  }
  if (maxVal === -Infinity) return -Infinity

  let sum = 0
  for (let i = 0; i < values.length; i++) {
    sum += Math.exp(values[i]! - maxVal)
  }
  return maxVal + Math.log(sum)
}

/**
 * Log-sum-exp over a slice of an array (avoids allocation).
 * Computes LSE of values[start..start+len-1].
 */
export function logSumExpSlice(
  values: Float64Array,
  start: number,
  len: number,
): number {
  if (len === 0) return -Infinity
  let maxVal = -Infinity
  for (let i = 0; i < len; i++) {
    const v = values[start + i]!
    if (v > maxVal) maxVal = v
  }
  if (maxVal === -Infinity) return -Infinity

  let sum = 0
  for (let i = 0; i < len; i++) {
    sum += Math.exp(values[start + i]! - maxVal)
  }
  return maxVal + Math.log(sum)
}

/**
 * Normalize an array in-place to [0, 1] using min-max scaling.
 * If all values are identical, sets all to 0.
 */
export function normalize01(arr: Float64Array): void {
  if (arr.length === 0) return
  let min = Infinity
  let max = -Infinity
  for (let i = 0; i < arr.length; i++) {
    if (arr[i]! < min) min = arr[i]!
    if (arr[i]! > max) max = arr[i]!
  }
  const range = max - min
  if (range < 1e-15) {
    arr.fill(0)
    return
  }
  for (let i = 0; i < arr.length; i++) {
    arr[i] = (arr[i]! - min) / range
  }
}

/**
 * Convert degrees to radians.
 */
export function toRad(deg: number): number {
  return deg * (Math.PI / 180)
}

/**
 * Compute row sums of an N×M row-major matrix.
 */
export function computeRowSums(
  plan: Float64Array,
  N: number,
  M: number,
): Float64Array {
  const sums = new Float64Array(N)
  for (let i = 0; i < N; i++) {
    let s = 0
    for (let j = 0; j < M; j++) {
      s += plan[i * M + j]!
    }
    sums[i] = s
  }
  return sums
}

/**
 * Compute column sums of an N×M row-major matrix.
 */
export function computeColSums(
  plan: Float64Array,
  N: number,
  M: number,
): Float64Array {
  const sums = new Float64Array(M)
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < M; j++) {
      sums[j]! += plan[i * M + j]!
    }
  }
  return sums
}

/**
 * Matrix-vector multiply: result = A · v (A is N×M row-major, v is M).
 * Writes into pre-allocated result array of length N.
 */
export function matVecMul(
  A: Float64Array,
  v: Float64Array,
  N: number,
  M: number,
  result: Float64Array,
): void {
  for (let i = 0; i < N; i++) {
    let s = 0
    for (let j = 0; j < M; j++) {
      s += A[i * M + j]! * v[j]!
    }
    result[i] = s
  }
}

/**
 * Matrix-transpose-vector multiply: result = Aᵀ · v (A is N×M row-major, v is N).
 * Writes into pre-allocated result array of length M.
 */
export function matTransVecMul(
  A: Float64Array,
  v: Float64Array,
  N: number,
  M: number,
  result: Float64Array,
): void {
  result.fill(0)
  for (let i = 0; i < N; i++) {
    const vi = v[i]!
    for (let j = 0; j < M; j++) {
      result[j]! += A[i * M + j]! * vi
    }
  }
}

/**
 * Normalize a distribution in-place to sum to 1.
 * Returns the original sum.
 */
export function normalizeDistribution(arr: Float64Array): number {
  let sum = 0
  for (let i = 0; i < arr.length; i++) {
    sum += arr[i]!
  }
  if (sum > 0) {
    for (let i = 0; i < arr.length; i++) {
      arr[i] = arr[i]! / sum
    }
  }
  return sum
}

/**
 * Create a uniform distribution of length n.
 */
export function uniformDistribution(n: number): Float64Array {
  return new Float64Array(n).fill(1 / n)
}

/**
 * L1 distance (total variation) between two arrays.
 */
export function l1Distance(a: Float64Array, b: Float64Array): number {
  let sum = 0
  for (let i = 0; i < a.length; i++) {
    sum += Math.abs(a[i]! - b[i]!)
  }
  return sum
}

/**
 * Dot product of two arrays.
 */
export function dot(a: Float64Array, b: Float64Array): number {
  let sum = 0
  for (let i = 0; i < a.length; i++) {
    sum += a[i]! * b[i]!
  }
  return sum
}

/**
 * Element-wise clamp: max(arr[i], floor).
 */
export function clampFloor(arr: Float64Array, floor: number): void {
  for (let i = 0; i < arr.length; i++) {
    if (arr[i]! < floor) arr[i] = floor
  }
}
