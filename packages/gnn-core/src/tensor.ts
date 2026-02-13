// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — Tensor / Linear Algebra Infrastructure
// Float64Array-based operations for all GNN layers.
// ---------------------------------------------------------------------------

import type { PRNG } from './types.js';

// ---- Matrix Operations ----

/** Dense matrix multiply: (M×K) × (K×N) → (M×N), row-major layout. */
export function matMul(
  a: Float64Array,
  b: Float64Array,
  M: number,
  K: number,
  N: number,
): Float64Array {
  const out = new Float64Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let k = 0; k < K; k++) {
      const aik = a[i * K + k]!;
      for (let j = 0; j < N; j++) {
        out[i * N + j]! += aik * b[k * N + j]!;
      }
    }
  }
  return out;
}

/** Matrix-vector multiply: (M×N) × (N) → (M), row-major. */
export function matVecMul(
  mat: Float64Array,
  vec: Float64Array,
  M: number,
  N: number,
): Float64Array {
  const out = new Float64Array(M);
  for (let i = 0; i < M; i++) {
    let s = 0;
    for (let j = 0; j < N; j++) {
      s += mat[i * N + j]! * vec[j]!;
    }
    out[i] = s;
  }
  return out;
}

/** Transpose M×N → N×M, row-major. */
export function transpose(
  mat: Float64Array,
  M: number,
  N: number,
): Float64Array {
  const out = new Float64Array(N * M);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      out[j * M + i] = mat[i * N + j]!;
    }
  }
  return out;
}

// ---- Element-wise Operations ----

/** Element-wise addition. */
export function add(a: Float64Array, b: Float64Array): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = a[i]! + b[i]!;
  }
  return out;
}

/** Element-wise subtraction. */
export function sub(a: Float64Array, b: Float64Array): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = a[i]! - b[i]!;
  }
  return out;
}

/** Element-wise multiplication (Hadamard product). */
export function mul(a: Float64Array, b: Float64Array): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = a[i]! * b[i]!;
  }
  return out;
}

/** Scalar multiplication. */
export function scale(a: Float64Array, s: number): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = a[i]! * s;
  }
  return out;
}

/** Dot product. */
export function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    s += a[i]! * b[i]!;
  }
  return s;
}

// ---- Activation Functions ----

/** ReLU activation. */
export function relu(x: Float64Array): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = x[i]! > 0 ? x[i]! : 0;
  }
  return out;
}

/** Leaky ReLU activation. */
export function leakyRelu(x: Float64Array, slope = 0.01): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const v = x[i]!;
    out[i] = v > 0 ? v : slope * v;
  }
  return out;
}

/** ELU activation: alpha * (exp(x) - 1) for x < 0. */
export function elu(x: Float64Array, alpha: number): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const v = x[i]!;
    out[i] = v >= 0 ? v : alpha * (Math.exp(v) - 1);
  }
  return out;
}

/** Sigmoid activation: 1 / (1 + exp(-x)). */
export function sigmoid(x: Float64Array): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const v = x[i]!;
    // Numerically stable sigmoid
    if (v >= 0) {
      const ez = Math.exp(-v);
      out[i] = 1 / (1 + ez);
    } else {
      const ez = Math.exp(v);
      out[i] = ez / (1 + ez);
    }
  }
  return out;
}

/** Tanh activation. */
export function tanhActivation(x: Float64Array): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = Math.tanh(x[i]!);
  }
  return out;
}

// ---- Normalization ----

/**
 * Numerically stable softmax over blocks of n elements.
 * If x has more than n elements, applies softmax independently to each
 * contiguous block of n elements (row-wise softmax for matrices).
 */
export function softmax(x: Float64Array, n: number): Float64Array {
  const out = new Float64Array(x.length);
  const numRows = x.length / n;
  for (let r = 0; r < numRows; r++) {
    const offset = r * n;
    // Find max for numerical stability
    let maxVal = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = x[offset + i]!;
      if (v > maxVal) maxVal = v;
    }
    // Compute exp(x - max) and sum
    let sumExp = 0;
    for (let i = 0; i < n; i++) {
      const e = Math.exp(x[offset + i]! - maxVal);
      out[offset + i] = e;
      sumExp += e;
    }
    // Normalize
    for (let i = 0; i < n; i++) {
      out[offset + i] = out[offset + i]! / sumExp;
    }
  }
  return out;
}

/**
 * Layer normalization. x is a vector of length n (or multiple rows of length n).
 * gamma and beta are scale/shift of length n.
 */
export function layerNorm(
  x: Float64Array,
  n: number,
  gamma: Float64Array,
  beta: Float64Array,
): Float64Array {
  const out = new Float64Array(x.length);
  const numRows = x.length / n;
  const eps = 1e-5;
  for (let r = 0; r < numRows; r++) {
    const offset = r * n;
    // Compute mean
    let mu = 0;
    for (let i = 0; i < n; i++) {
      mu += x[offset + i]!;
    }
    mu /= n;
    // Compute variance
    let variance = 0;
    for (let i = 0; i < n; i++) {
      const d = x[offset + i]! - mu;
      variance += d * d;
    }
    variance /= n;
    const invStd = 1 / Math.sqrt(variance + eps);
    // Normalize, scale, shift
    for (let i = 0; i < n; i++) {
      out[offset + i] = (x[offset + i]! - mu) * invStd * gamma[i]! + beta[i]!;
    }
  }
  return out;
}

/**
 * Batch normalization using provided running mean/variance.
 * x is (n × dim), row-major. gamma, beta, mean, variance are all length dim.
 */
export function batchNorm(
  x: Float64Array,
  n: number,
  dim: number,
  gamma: Float64Array,
  beta: Float64Array,
  runningMean: Float64Array,
  runningVariance: Float64Array,
  epsilon = 1e-5,
): Float64Array {
  const out = new Float64Array(x.length);
  // Precompute inverse std per dimension
  const invStd = new Float64Array(dim);
  for (let d = 0; d < dim; d++) {
    invStd[d] = 1 / Math.sqrt(runningVariance[d]! + epsilon);
  }
  for (let i = 0; i < n; i++) {
    const offset = i * dim;
    for (let d = 0; d < dim; d++) {
      out[offset + d] =
        (x[offset + d]! - runningMean[d]!) * invStd[d]! * gamma[d]! + beta[d]!;
    }
  }
  return out;
}

// ---- Initialization ----

/**
 * Xavier/Glorot uniform initialization.
 * U(-sqrt(6/(fanIn+fanOut)), sqrt(6/(fanIn+fanOut))).
 * Returns fanIn×fanOut array.
 */
export function xavierInit(
  fanIn: number,
  fanOut: number,
  rng: PRNG,
): Float64Array {
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  const size = fanIn * fanOut;
  const out = new Float64Array(size);
  for (let i = 0; i < size; i++) {
    out[i] = (rng() * 2 - 1) * limit;
  }
  return out;
}

/**
 * He/Kaiming normal initialization.
 * N(0, sqrt(2/fanIn)). Returns fanIn×fanOut array.
 * Uses Box-Muller transform for normal distribution.
 */
export function heInit(
  fanIn: number,
  fanOut: number,
  rng: PRNG,
): Float64Array {
  const std = Math.sqrt(2 / fanIn);
  const size = fanIn * fanOut;
  const out = new Float64Array(size);
  for (let i = 0; i < size; i += 2) {
    // Box-Muller transform
    const u1 = rng() || 1e-15; // avoid log(0)
    const u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    out[i] = r * Math.cos(theta) * std;
    if (i + 1 < size) {
      out[i + 1] = r * Math.sin(theta) * std;
    }
  }
  return out;
}

/** Return a zero-filled array of the same length as x. */
export function zerosLike(x: Float64Array): Float64Array {
  return new Float64Array(x.length);
}

/** Return a ones-filled array of the same length as x. */
export function onesLike(x: Float64Array): Float64Array {
  const out = new Float64Array(x.length);
  out.fill(1);
  return out;
}

// ---- Norms and Distances ----

/** L2 (Euclidean) norm. */
export function l2Norm(x: Float64Array): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) {
    s += x[i]! * x[i]!;
  }
  return Math.sqrt(s);
}

/** L2 normalize: return x / ||x||. Returns zero vector if norm is 0. */
export function normalize(x: Float64Array): Float64Array {
  const norm = l2Norm(x);
  if (norm === 0) return new Float64Array(x.length);
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = x[i]! / norm;
  }
  return out;
}

// ---- Loss Functions ----

/**
 * Cross-entropy loss for a single sample.
 * logits is length numClasses, targetIndex is the correct class.
 * Uses log-sum-exp for numerical stability.
 */
export function crossEntropyLoss(
  logits: Float64Array,
  targetIndex: number,
  numClasses: number,
): number {
  // Find max for numerical stability
  let maxVal = -Infinity;
  for (let i = 0; i < numClasses; i++) {
    if (logits[i]! > maxVal) maxVal = logits[i]!;
  }
  // log-sum-exp
  let sumExp = 0;
  for (let i = 0; i < numClasses; i++) {
    sumExp += Math.exp(logits[i]! - maxVal);
  }
  const logSumExp = maxVal + Math.log(sumExp);
  return logSumExp - logits[targetIndex]!;
}

/**
 * BPR (Bayesian Personalized Ranking) loss.
 * -mean(ln(sigmoid(pos - neg))).
 */
export function bprLoss(
  posScores: Float64Array,
  negScores: Float64Array,
): number {
  const n = posScores.length;
  let total = 0;
  for (let i = 0; i < n; i++) {
    const diff = posScores[i]! - negScores[i]!;
    // Numerically stable log-sigmoid: log(sigmoid(x)) = -log(1+exp(-x))
    // For large positive x: ≈ 0. For large negative x: ≈ x.
    const logSig = diff >= 0
      ? -Math.log(1 + Math.exp(-diff))
      : diff - Math.log(1 + Math.exp(diff));
    total += logSig;
  }
  return -total / n;
}

// ---- Scatter Operations ----

/**
 * Scatter operation. src is (numElements × dim), index maps each element
 * to a target bin [0..dimSize). Returns (dimSize × dim).
 * reduce: 'sum' accumulates, 'mean' divides by count, 'max' takes max per dim.
 */
export function scatter(
  src: Float64Array,
  index: Uint32Array,
  dimSize: number,
  dim: number,
  reduce: 'sum' | 'mean' | 'max',
): Float64Array {
  const numElements = index.length;
  const out = new Float64Array(dimSize * dim);

  if (reduce === 'max') {
    // Initialize to -Infinity for max reduce
    out.fill(-Infinity);
  }

  const counts = reduce === 'mean' ? new Uint32Array(dimSize) : null;

  for (let e = 0; e < numElements; e++) {
    const target = index[e]!;
    const srcOffset = e * dim;
    const dstOffset = target * dim;

    if (reduce === 'max') {
      for (let d = 0; d < dim; d++) {
        const v = src[srcOffset + d]!;
        if (v > out[dstOffset + d]!) {
          out[dstOffset + d] = v;
        }
      }
    } else {
      for (let d = 0; d < dim; d++) {
        out[dstOffset + d] = out[dstOffset + d]! + src[srcOffset + d]!;
      }
    }

    if (counts) counts[target]!++;
  }

  // For max reduce, replace -Infinity with 0 for bins with no elements
  if (reduce === 'max') {
    for (let i = 0; i < out.length; i++) {
      if (out[i] === -Infinity) out[i] = 0;
    }
  }

  // For mean reduce, divide by counts
  if (counts) {
    for (let t = 0; t < dimSize; t++) {
      const c = counts[t]!;
      if (c > 0) {
        const offset = t * dim;
        for (let d = 0; d < dim; d++) {
          out[offset + d] = out[offset + d]! / c;
        }
      }
    }
  }

  return out;
}

// ---- Optimizer ----

/**
 * In-place Adam optimizer update.
 * Updates params, m (first moment), and v (second moment) in place.
 */
export function adamUpdate(
  params: Float64Array,
  grads: Float64Array,
  m: Float64Array,
  v: Float64Array,
  lr: number,
  beta1: number,
  beta2: number,
  t: number,
  epsilon = 1e-8,
): void {
  const bc1 = 1 - Math.pow(beta1, t);
  const bc2 = 1 - Math.pow(beta2, t);
  for (let i = 0; i < params.length; i++) {
    const g = grads[i]!;
    // Update biased first and second moment estimates
    m[i] = beta1 * m[i]! + (1 - beta1) * g;
    v[i] = beta2 * v[i]! + (1 - beta2) * g * g;
    // Bias-corrected estimates
    const mHat = m[i]! / bc1;
    const vHat = v[i]! / bc2;
    // Update parameters
    params[i] = params[i]! - lr * mHat / (Math.sqrt(vHat) + epsilon);
  }
}

// ---- Utility ----

/** Concatenate multiple Float64Arrays into one. */
export function concatenate(arrays: Float64Array[]): Float64Array {
  let totalLen = 0;
  for (const a of arrays) totalLen += a.length;
  const out = new Float64Array(totalLen);
  let offset = 0;
  for (const a of arrays) {
    out.set(a, offset);
    offset += a.length;
  }
  return out;
}

/** Slice a Float64Array from start (inclusive) to end (exclusive). */
export function slice(x: Float64Array, start: number, end: number): Float64Array {
  return x.slice(start, end);
}

/** Index of the maximum value. */
export function argmax(x: Float64Array): number {
  let bestIdx = 0;
  let bestVal = x[0]!;
  for (let i = 1; i < x.length; i++) {
    if (x[i]! > bestVal) {
      bestVal = x[i]!;
      bestIdx = i;
    }
  }
  return bestIdx;
}

/** Sum of all elements. */
export function sum(x: Float64Array): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) {
    s += x[i]!;
  }
  return s;
}

/** Mean of all elements. */
export function mean(x: Float64Array): number {
  return sum(x) / x.length;
}

// ---- GRU Cell ----

/**
 * Single GRU cell step. Returns the new hidden state.
 *
 * z = sigmoid(W_z * x + U_z * hPrev + b_z)        (update gate)
 * r = sigmoid(W_r * x + U_r * hPrev + b_r)        (reset gate)
 * hCandidate = tanh(W_h * x + U_h * (r ⊙ hPrev) + b_h)
 * hNew = (1 - z) ⊙ hPrev + z ⊙ hCandidate
 *
 * W matrices are (hidden×input), U matrices are (hidden×hidden), biases are (hidden).
 */
export function gruCell(
  x: Float64Array,
  hPrev: Float64Array,
  W_z: Float64Array,
  U_z: Float64Array,
  b_z: Float64Array,
  W_r: Float64Array,
  U_r: Float64Array,
  b_r: Float64Array,
  W_h: Float64Array,
  U_h: Float64Array,
  b_h: Float64Array,
): Float64Array {
  const hiddenDim = hPrev.length;
  const inputDim = x.length;

  // Update gate: z = sigmoid(W_z * x + U_z * hPrev + b_z)
  const z = new Float64Array(hiddenDim);
  for (let i = 0; i < hiddenDim; i++) {
    let val = b_z[i]!;
    for (let j = 0; j < inputDim; j++) {
      val += W_z[i * inputDim + j]! * x[j]!;
    }
    for (let j = 0; j < hiddenDim; j++) {
      val += U_z[i * hiddenDim + j]! * hPrev[j]!;
    }
    // Numerically stable sigmoid
    if (val >= 0) {
      z[i] = 1 / (1 + Math.exp(-val));
    } else {
      const ev = Math.exp(val);
      z[i] = ev / (1 + ev);
    }
  }

  // Reset gate: r = sigmoid(W_r * x + U_r * hPrev + b_r)
  const r = new Float64Array(hiddenDim);
  for (let i = 0; i < hiddenDim; i++) {
    let val = b_r[i]!;
    for (let j = 0; j < inputDim; j++) {
      val += W_r[i * inputDim + j]! * x[j]!;
    }
    for (let j = 0; j < hiddenDim; j++) {
      val += U_r[i * hiddenDim + j]! * hPrev[j]!;
    }
    if (val >= 0) {
      r[i] = 1 / (1 + Math.exp(-val));
    } else {
      const ev = Math.exp(val);
      r[i] = ev / (1 + ev);
    }
  }

  // Candidate hidden state: hCandidate = tanh(W_h * x + U_h * (r ⊙ hPrev) + b_h)
  const hCandidate = new Float64Array(hiddenDim);
  for (let i = 0; i < hiddenDim; i++) {
    let val = b_h[i]!;
    for (let j = 0; j < inputDim; j++) {
      val += W_h[i * inputDim + j]! * x[j]!;
    }
    for (let j = 0; j < hiddenDim; j++) {
      val += U_h[i * hiddenDim + j]! * (r[j]! * hPrev[j]!);
    }
    hCandidate[i] = Math.tanh(val);
  }

  // New hidden state: hNew = (1 - z) ⊙ hPrev + z ⊙ hCandidate
  const hNew = new Float64Array(hiddenDim);
  for (let i = 0; i < hiddenDim; i++) {
    hNew[i] = (1 - z[i]!) * hPrev[i]! + z[i]! * hCandidate[i]!;
  }

  return hNew;
}
