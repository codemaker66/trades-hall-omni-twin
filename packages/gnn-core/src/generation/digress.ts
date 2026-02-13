// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-5: Graph Generation
// DiGress — Discrete Denoising Diffusion for Graph Generation
// (Vignac et al., 2023).
//
// Operates on graphs with categorical node types and edge types.
// Forward process: corrupt discrete state via Markov transition matrices.
// Reverse process: predict clean graph from noisy state using a learned
// denoising network (simplified here as a small MLP).
// ---------------------------------------------------------------------------

import type {
  PRNG,
  DiGressConfig,
  DiGressState,
  GeneratedGraph,
} from '../types.js';

/**
 * Numerically stable sigmoid for a single scalar value.
 */
function sigmoidScalar(x: number): number {
  if (x >= 0) {
    const ez = Math.exp(-x);
    return 1 / (1 + ez);
  }
  const ez = Math.exp(x);
  return ez / (1 + ez);
}

/**
 * Sample from a categorical distribution given probabilities.
 * probs should sum to ~1.0. Returns the chosen category index.
 */
function sampleCategorical(probs: Float64Array, numCategories: number, rng: PRNG): number {
  const u = rng();
  let cumulative = 0;
  for (let c = 0; c < numCategories; c++) {
    cumulative += probs[c]!;
    if (u < cumulative) return c;
  }
  return numCategories - 1; // fallback for rounding
}

/**
 * Build a uniform transition matrix for `k` categories at noise level `beta`.
 *
 * Q(t) = (1 - beta) * I + beta * (1/k) * 11^T
 *
 * This means with probability (1 - beta) the category stays the same,
 * and with probability beta it becomes uniformly random.
 *
 * Returns a flat k x k Float64Array (row-major).
 */
function buildTransitionMatrix(k: number, beta: number): Float64Array {
  const Q = new Float64Array(k * k);
  const uniform = beta / k;
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      Q[i * k + j] = i === j ? (1 - beta) + uniform : uniform;
    }
  }
  return Q;
}

/**
 * Apply the transition matrix to a one-hot encoded categorical variable.
 * Returns the resulting probability distribution (Float64Array of length k).
 */
function applyTransition(
  currentCategory: number,
  Q: Float64Array,
  k: number,
): Float64Array {
  // One-hot row: just read row `currentCategory` from Q
  const probs = new Float64Array(k);
  for (let j = 0; j < k; j++) {
    probs[j] = Q[currentCategory * k + j]!;
  }
  return probs;
}

/**
 * Compute a noise schedule beta(t) that linearly increases from a small
 * value to near 1 over diffusionSteps timesteps.
 */
function noiseSchedule(t: number, totalSteps: number): number {
  // Linear schedule: beta increases from 0.01 to 0.99
  return 0.01 + (0.98 * t) / totalSteps;
}

/**
 * Forward diffusion step: corrupt the graph state at timestep t.
 *
 * Algorithm:
 * 1. Compute noise level beta(t) from the schedule.
 * 2. Build transition matrices Q_X (for nodes) and Q_E (for edges).
 * 3. For each node, compute the noisy categorical distribution by
 *    multiplying the current one-hot type by Q_X^t and sampling.
 * 4. For each edge, do the same with Q_E^t.
 *
 * For simplicity, we apply the single-step transition (not the full
 * product Q_X^1 * Q_X^2 * ... * Q_X^t) but use the marginal transition
 * parameterised by the cumulative noise level.
 *
 * @param state  - Current graph state { nodeTypes, adjMatrix, numNodes }.
 * @param t      - Current timestep (1-indexed, where 1 = least noisy).
 * @param config - DiGress configuration.
 * @param rng    - Deterministic PRNG.
 * @returns      - Noisy DiGressState at timestep t.
 */
export function discreteDiffusionForward(
  state: DiGressState,
  t: number,
  config: DiGressConfig,
  rng: PRNG,
): DiGressState {
  const { numNodeTypes, numEdgeTypes, diffusionSteps } = config;
  const n = state.numNodes;

  // Cumulative noise level for timestep t
  const beta = noiseSchedule(t, diffusionSteps);

  // Build transition matrices
  const Q_X = buildTransitionMatrix(numNodeTypes, beta);
  const Q_E = buildTransitionMatrix(numEdgeTypes, beta);

  // Corrupt node types
  const noisyNodeTypes = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const currentType = state.nodeTypes[i]!;
    const probs = applyTransition(currentType, Q_X, numNodeTypes);
    noisyNodeTypes[i] = sampleCategorical(probs, numNodeTypes, rng);
  }

  // Corrupt edge types (upper triangle stored flat)
  // Upper triangle size = n*(n-1)/2
  const triSize = (n * (n - 1)) / 2;
  const noisyAdj = new Uint8Array(triSize);
  for (let idx = 0; idx < triSize; idx++) {
    const currentType = state.adjMatrix[idx]!;
    const probs = applyTransition(currentType, Q_E, numEdgeTypes);
    noisyAdj[idx] = sampleCategorical(probs, numEdgeTypes, rng);
  }

  return {
    nodeTypes: noisyNodeTypes,
    adjMatrix: noisyAdj,
    numNodes: n,
  };
}

/**
 * Single reverse (denoising) step: predict the clean graph from the noisy
 * state using a simple MLP denoiser.
 *
 * Algorithm:
 * 1. Flatten the noisy state into a feature vector.
 * 2. Pass through a 2-layer MLP (with ReLU) to produce logits for each
 *    node type and each edge type.
 * 3. Sample from the predicted categorical distributions.
 *
 * The MLP layout within denoiseWeights:
 *   Layer 1: W1 (hiddenDim × inputDim), b1 (hiddenDim)
 *   Layer 2: W2 (outputDim × hiddenDim), b2 (outputDim)
 * where inputDim = n + triSize, outputDim = n * numNodeTypes + triSize * numEdgeTypes.
 *
 * @param noisyState     - Current noisy graph state.
 * @param denoiseWeights - Flattened MLP weights [W1, b1, W2, b2].
 * @param config         - DiGress configuration.
 * @param rng            - Deterministic PRNG.
 * @returns              - Denoised DiGressState (single step).
 */
export function discreteDiffusionReverse(
  noisyState: DiGressState,
  denoiseWeights: Float64Array,
  config: DiGressConfig,
  rng: PRNG,
): DiGressState {
  const { numNodeTypes, numEdgeTypes, hiddenDim } = config;
  const n = noisyState.numNodes;
  const triSize = (n * (n - 1)) / 2;

  // Build input vector: concatenate node types and edge types (as floats)
  const inputDim = n + triSize;
  const input = new Float64Array(inputDim);
  for (let i = 0; i < n; i++) {
    input[i] = noisyState.nodeTypes[i]! / Math.max(numNodeTypes - 1, 1);
  }
  for (let i = 0; i < triSize; i++) {
    input[n + i] = noisyState.adjMatrix[i]! / Math.max(numEdgeTypes - 1, 1);
  }

  // Parse weight offsets
  const outputDim = n * numNodeTypes + triSize * numEdgeTypes;
  let offset = 0;
  const W1 = denoiseWeights.subarray(offset, offset + hiddenDim * inputDim);
  offset += hiddenDim * inputDim;
  const b1 = denoiseWeights.subarray(offset, offset + hiddenDim);
  offset += hiddenDim;
  const W2 = denoiseWeights.subarray(offset, offset + outputDim * hiddenDim);
  offset += outputDim * hiddenDim;
  const b2 = denoiseWeights.subarray(offset, offset + outputDim);

  // Layer 1: h = ReLU(W1 * input + b1)
  const h = new Float64Array(hiddenDim);
  for (let i = 0; i < hiddenDim; i++) {
    let val = b1[i]!;
    for (let j = 0; j < inputDim; j++) {
      val += W1[i * inputDim + j]! * input[j]!;
    }
    h[i] = val > 0 ? val : 0; // ReLU
  }

  // Layer 2: logits = W2 * h + b2
  const logits = new Float64Array(outputDim);
  for (let i = 0; i < outputDim; i++) {
    let val = b2[i]!;
    for (let j = 0; j < hiddenDim; j++) {
      val += W2[i * hiddenDim + j]! * h[j]!;
    }
    logits[i] = val;
  }

  // Decode node types: sample from softmax over numNodeTypes per node
  const denoisedNodeTypes = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const start = i * numNodeTypes;
    // Softmax over this node's logits
    const probs = new Float64Array(numNodeTypes);
    let maxLogit = -Infinity;
    for (let c = 0; c < numNodeTypes; c++) {
      const v = logits[start + c]!;
      if (v > maxLogit) maxLogit = v;
    }
    let sumExp = 0;
    for (let c = 0; c < numNodeTypes; c++) {
      probs[c] = Math.exp(logits[start + c]! - maxLogit);
      sumExp += probs[c]!;
    }
    for (let c = 0; c < numNodeTypes; c++) {
      probs[c] = probs[c]! / sumExp;
    }
    denoisedNodeTypes[i] = sampleCategorical(probs, numNodeTypes, rng);
  }

  // Decode edge types: sample from softmax over numEdgeTypes per edge
  const edgeLogitOffset = n * numNodeTypes;
  const denoisedAdj = new Uint8Array(triSize);
  for (let idx = 0; idx < triSize; idx++) {
    const start = edgeLogitOffset + idx * numEdgeTypes;
    const probs = new Float64Array(numEdgeTypes);
    let maxLogit = -Infinity;
    for (let c = 0; c < numEdgeTypes; c++) {
      const v = logits[start + c]!;
      if (v > maxLogit) maxLogit = v;
    }
    let sumExp = 0;
    for (let c = 0; c < numEdgeTypes; c++) {
      probs[c] = Math.exp(logits[start + c]! - maxLogit);
      sumExp += probs[c]!;
    }
    for (let c = 0; c < numEdgeTypes; c++) {
      probs[c] = probs[c]! / sumExp;
    }
    denoisedAdj[idx] = sampleCategorical(probs, numEdgeTypes, rng);
  }

  return {
    nodeTypes: denoisedNodeTypes,
    adjMatrix: denoisedAdj,
    numNodes: n,
  };
}

/**
 * Full graph generation using DiGress.
 *
 * Algorithm:
 * 1. Start from a fully noisy state (uniform random node types and edge types).
 * 2. Iteratively denoise for `diffusionSteps` reverse steps.
 * 3. At each step, call discreteDiffusionReverse to predict the clean graph.
 * 4. Convert the final DiGressState to a GeneratedGraph by expanding the
 *    upper-triangle adjacency to a full n x n matrix.
 *
 * @param config         - DiGress configuration.
 * @param denoiseWeights - Flattened MLP denoiser weights.
 * @param numNodes       - Number of nodes in the graph to generate.
 * @param rng            - Deterministic PRNG.
 * @returns              - GeneratedGraph { adjacency, numNodes, nodeTypes }.
 */
export function generateWithDiGress(
  config: DiGressConfig,
  denoiseWeights: Float64Array,
  numNodes: number,
  rng: PRNG,
): GeneratedGraph {
  const { numNodeTypes, numEdgeTypes, diffusionSteps } = config;
  const triSize = (numNodes * (numNodes - 1)) / 2;

  // Start from pure noise: uniformly random categories
  const noisyNodeTypes = new Uint8Array(numNodes);
  for (let i = 0; i < numNodes; i++) {
    noisyNodeTypes[i] = Math.floor(rng() * numNodeTypes);
  }
  const noisyAdj = new Uint8Array(triSize);
  for (let i = 0; i < triSize; i++) {
    noisyAdj[i] = Math.floor(rng() * numEdgeTypes);
  }

  let state: DiGressState = {
    nodeTypes: noisyNodeTypes,
    adjMatrix: noisyAdj,
    numNodes,
  };

  // Reverse diffusion: denoise from t = diffusionSteps down to t = 1
  for (let t = diffusionSteps; t >= 1; t--) {
    state = discreteDiffusionReverse(state, denoiseWeights, config, rng);
  }

  // Convert upper-triangle adjMatrix to full n x n adjacency.
  // Edge type 0 = no edge, any nonzero type = edge present.
  const adjacency = new Uint8Array(numNodes * numNodes);
  let idx = 0;
  for (let i = 0; i < numNodes; i++) {
    for (let j = i + 1; j < numNodes; j++) {
      const edgeType = state.adjMatrix[idx]!;
      if (edgeType > 0) {
        adjacency[i * numNodes + j] = 1;
        adjacency[j * numNodes + i] = 1;
      }
      idx++;
    }
  }

  return {
    adjacency,
    numNodes,
    nodeTypes: state.nodeTypes,
  };
}
