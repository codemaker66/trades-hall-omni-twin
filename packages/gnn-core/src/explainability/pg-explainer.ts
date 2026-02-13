// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — PGExplainer (Luo et al. 2020)
// Parameterized explainer: an MLP generates edge masks for ALL nodes at once,
// avoiding per-node optimization. Uses Binary Concrete relaxation for
// differentiable edge sampling.
// ---------------------------------------------------------------------------

import type {
  Graph,
  GNNForwardFn,
  ExplanationResult,
  MLPWeights,
  PRNG,
} from '../types.js';
import { sigmoid, relu, matVecMul } from '../tensor.js';
import { getEdgeIndex } from '../graph.js';

// ---- Helpers ----

/**
 * Forward pass through a small MLP.
 *
 * For each layer: output = activation(W * input + bias)
 * Last layer uses identity activation (logits).
 * Intermediate layers use ReLU.
 *
 * @param input - Input vector.
 * @param weights - MLP layer weights.
 * @returns Output vector from the final layer.
 */
function mlpForward(input: Float64Array, weights: MLPWeights): Float64Array {
  let h = input;
  const numLayers = weights.layers.length;

  for (let l = 0; l < numLayers; l++) {
    const layer = weights.layers[l]!;
    // h = W * h + bias
    const z = matVecMul(layer.W, h, layer.outDim, layer.inDim);
    for (let i = 0; i < layer.outDim; i++) {
      z[i] = z[i]! + layer.bias[i]!;
    }
    // ReLU for all but last layer
    if (l < numLayers - 1) {
      h = relu(z);
    } else {
      h = z;
    }
  }

  return h;
}

/**
 * Sample from Binary Concrete distribution (Maddison et al. 2017).
 *
 * mask = sigmoid((log(z/(1-z)) + logistic_noise) / temperature)
 *
 * where z is the MLP output (pre-sigmoid probability), and logistic_noise
 * is sampled by inverse CDF: log(u / (1-u)) for u ~ Uniform(0,1).
 *
 * @param logAlpha - Log-odds from the MLP (z_ij values).
 * @param temperature - Temperature for the concrete distribution.
 * @param rng - PRNG for sampling noise.
 * @returns Sampled mask values in (0, 1).
 */
function binaryConcreteSample(
  logAlpha: Float64Array,
  temperature: number,
  rng: PRNG,
): Float64Array {
  const n = logAlpha.length;
  const out = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    // Sample uniform noise and convert to logistic noise
    const u = Math.max(1e-8, Math.min(1 - 1e-8, rng()));
    const logisticNoise = Math.log(u / (1 - u));

    // Binary Concrete: sigmoid((logAlpha + noise) / temperature)
    const val = (logAlpha[i]! + logisticNoise) / temperature;

    // Numerically stable sigmoid
    if (val >= 0) {
      const ez = Math.exp(-val);
      out[i] = 1 / (1 + ez);
    } else {
      const ez = Math.exp(val);
      out[i] = ez / (1 + ez);
    }
  }

  return out;
}

/**
 * Extract node embeddings from a GNN model by running a forward pass
 * and treating the output as per-node embeddings.
 *
 * @param model - GNN forward function.
 * @param graph - Input graph.
 * @returns Per-node embedding matrix as Float64Array (numNodes * embDim).
 */
function getNodeEmbeddings(
  model: GNNForwardFn,
  graph: Graph,
): { embeddings: Float64Array; embDim: number } {
  const output = model(graph, graph.nodeFeatures);
  const embDim = output.length / graph.numNodes;
  return { embeddings: output, embDim };
}

// ---- Main Export ----

/**
 * PGExplainer — Luo et al. 2020.
 *
 * A parameterized explainer that uses an MLP to generate edge masks for
 * ALL nodes simultaneously, without per-node optimization.
 *
 * Algorithm:
 * 1. Run the GNN model to obtain node embeddings h_i for all nodes.
 * 2. For each edge (i, j):
 *    a. Concatenate embeddings: z_ij = MLP([h_i || h_j]).
 *    b. The MLP output is the log-odds for edge importance.
 * 3. Sample edge mask from Binary Concrete distribution:
 *    mask = sigmoid((log(z/(1-z)) + logistic_noise) / temperature)
 * 4. Apply mask to graph edges.
 * 5. Threshold to get important edges.
 *
 * @param model - GNN forward function.
 * @param graph - Input CSR graph with node features.
 * @param mlpWeights - Pre-trained MLP weights for edge mask generation.
 *                     Input dim = 2 * embDim (concatenated endpoint embeddings).
 *                     Output dim = 1 (scalar log-odds per edge).
 * @param temperature - Temperature for Binary Concrete sampling (lower = more discrete).
 * @param rng - Deterministic PRNG for Binary Concrete noise.
 * @param edgeMaskThreshold - Threshold for binarizing the mask (default 0.5).
 * @returns ExplanationResult with edge mask and important edges.
 */
export function pgExplainer(
  model: GNNForwardFn,
  graph: Graph,
  mlpWeights: MLPWeights,
  temperature: number,
  rng: PRNG,
  edgeMaskThreshold = 0.5,
): ExplanationResult {
  const numEdges = graph.numEdges;

  // Step 1: Get node embeddings from the GNN model
  const { embeddings, embDim } = getNodeEmbeddings(model, graph);

  // Step 2: For each edge (i, j), compute log-odds via MLP([h_i || h_j])
  const [srcArray, dstArray] = getEdgeIndex(graph);
  const logAlpha = new Float64Array(numEdges);

  // Concatenation buffer for [h_i || h_j]
  const concatBuf = new Float64Array(2 * embDim);

  for (let e = 0; e < numEdges; e++) {
    const src = srcArray[e]!;
    const dst = dstArray[e]!;

    // Concatenate h_src and h_dst
    const srcOffset = src * embDim;
    const dstOffset = dst * embDim;
    for (let d = 0; d < embDim; d++) {
      concatBuf[d] = embeddings[srcOffset + d]!;
      concatBuf[embDim + d] = embeddings[dstOffset + d]!;
    }

    // MLP forward: output is a scalar (1-dim)
    const mlpOut = mlpForward(concatBuf, mlpWeights);
    logAlpha[e] = mlpOut[0]!;
  }

  // Step 3: Sample edge mask from Binary Concrete distribution
  const edgeMask = binaryConcreteSample(logAlpha, temperature, rng);

  // Step 4: Threshold to get important edges
  const importantEdges: [number, number][] = [];
  for (let e = 0; e < numEdges; e++) {
    if (edgeMask[e]! >= edgeMaskThreshold) {
      importantEdges.push([srcArray[e]!, dstArray[e]!]);
    }
  }

  return {
    edgeMask,
    importantEdges,
  };
}
