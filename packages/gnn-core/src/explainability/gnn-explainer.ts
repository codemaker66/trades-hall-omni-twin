// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNNExplainer (Ying et al. 2019)
// Learn continuous edge/feature masks to maximize mutual information
// MI(Y, (G_S, X_S)) for single-node explanations.
// ---------------------------------------------------------------------------

import type {
  Graph,
  GNNForwardFn,
  ExplainerConfig,
  ExplanationResult,
  PRNG,
} from '../types.js';
import { sigmoid, argmax } from '../tensor.js';
import { getEdgeIndex } from '../graph.js';

// ---- Helpers ----

/**
 * Compute L1 norm (sum of absolute values) of an array.
 */
function l1Norm(x: Float64Array): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) {
    s += Math.abs(x[i]!);
  }
  return s;
}

/**
 * Compute binary entropy regularization: -sum(m * log(m) + (1-m) * log(1-m)).
 * Encourages masks towards 0 or 1 (discrete).
 */
function entropyReg(mask: Float64Array): number {
  const eps = 1e-8;
  let h = 0;
  for (let i = 0; i < mask.length; i++) {
    const m = mask[i]!;
    const clamped = Math.max(eps, Math.min(1 - eps, m));
    h -= clamped * Math.log(clamped) + (1 - clamped) * Math.log(1 - clamped);
  }
  return h;
}

/**
 * Build a masked graph: apply edge mask weights and feature mask to
 * a copy of the original graph.
 */
function buildMaskedGraph(
  graph: Graph,
  edgeMaskSigmoid: Float64Array,
  featureMaskSigmoid: Float64Array,
): { maskedGraph: Graph; maskedFeatures: Float64Array } {
  // Apply edge mask to edge weights
  const maskedEdgeWeights = new Float64Array(graph.numEdges);
  for (let e = 0; e < graph.numEdges; e++) {
    const origWeight = graph.edgeWeights ? graph.edgeWeights[e]! : 1.0;
    maskedEdgeWeights[e] = origWeight * edgeMaskSigmoid[e]!;
  }

  // Apply feature mask element-wise to node features
  const maskedFeatures = new Float64Array(graph.nodeFeatures.length);
  for (let i = 0; i < graph.numNodes; i++) {
    for (let f = 0; f < graph.featureDim; f++) {
      const idx = i * graph.featureDim + f;
      maskedFeatures[idx] = graph.nodeFeatures[idx]! * featureMaskSigmoid[f]!;
    }
  }

  const maskedGraph: Graph = {
    numNodes: graph.numNodes,
    numEdges: graph.numEdges,
    rowPtr: graph.rowPtr,
    colIdx: graph.colIdx,
    edgeWeights: maskedEdgeWeights,
    nodeFeatures: maskedFeatures,
    featureDim: graph.featureDim,
    edgeFeatures: graph.edgeFeatures,
    edgeFeatureDim: graph.edgeFeatureDim,
  };

  return { maskedGraph, maskedFeatures };
}

/**
 * Compute the GNNExplainer loss:
 *   loss = -log(pred[targetClass]) + sizeReg * ||mask||_1 + entropyReg * H(mask)
 *
 * @param prediction - Softmax/logit output from the model at nodeIdx
 * @param targetClass - The ground-truth class index
 * @param edgeMaskSigmoid - Sigmoid-activated edge mask
 * @param featureMaskSigmoid - Sigmoid-activated feature mask
 * @param config - Explainer configuration with regularization coefficients
 */
function computeLoss(
  prediction: Float64Array,
  targetClass: number,
  edgeMaskSigmoid: Float64Array,
  featureMaskSigmoid: Float64Array,
  config: ExplainerConfig,
): number {
  const eps = 1e-8;
  // Prediction loss: negative log probability of the correct class
  const predLoss = -Math.log(Math.max(eps, prediction[targetClass]!));

  // Size regularization on edge mask (encourages sparsity)
  const sizeL = config.sizeReg * l1Norm(edgeMaskSigmoid);

  // Entropy regularization on edge mask (encourages discreteness)
  const entL = config.entropyReg * entropyReg(edgeMaskSigmoid);

  // Additional size + entropy regularization on feature mask
  const featureSizeL = config.sizeReg * 0.1 * l1Norm(featureMaskSigmoid);
  const featureEntL = config.entropyReg * 0.1 * entropyReg(featureMaskSigmoid);

  return predLoss + sizeL + entL + featureSizeL + featureEntL;
}

// ---- Main Export ----

/**
 * GNNExplainer — Ying et al. 2019.
 *
 * Learns continuous edge mask M and feature mask F to maximize
 * MI(Y, (G_S, X_S)) for a target node's prediction.
 *
 * Uses gradient-free optimization: at each epoch, perturb the mask
 * parameters and keep the perturbation if loss decreases.
 *
 * Algorithm:
 * 1. Initialize edgeMask and featureMask with random values in (0, 1).
 * 2. For each epoch:
 *    a. Apply sigmoid to masks to keep values in (0, 1).
 *    b. Mask edges: weight each edge by sigmoid(edgeMask[e]).
 *    c. Mask features: element-wise multiply node features by sigmoid(featureMask).
 *    d. Forward pass through the model with the masked graph.
 *    e. Loss = -log(pred[targetClass]) + sizeReg * ||mask||_1 + entropyReg * H(mask).
 *    f. Gradient-free: perturb mask, keep if loss decreases.
 * 3. Threshold masks to get binary important edges/features.
 *
 * @param model - GNN forward function mapping (graph, features) => node logits.
 * @param graph - Input CSR graph with node features.
 * @param nodeIdx - Target node index to explain.
 * @param config - Explainer configuration (epochs, lr, regularization, thresholds).
 * @param rng - Deterministic PRNG for initialization and perturbation.
 * @returns ExplanationResult with edge mask, feature mask, and thresholded importance.
 */
export function gnnExplainer(
  model: GNNForwardFn,
  graph: Graph,
  nodeIdx: number,
  config: ExplainerConfig,
  rng: PRNG,
): ExplanationResult {
  const numEdges = graph.numEdges;
  const featureDim = graph.featureDim;

  // Get the original (unmasked) prediction to determine target class
  const origOutput = model(graph, graph.nodeFeatures);
  const numClasses = origOutput.length / graph.numNodes;
  const nodeLogits = origOutput.slice(
    nodeIdx * numClasses,
    (nodeIdx + 1) * numClasses,
  );
  const targetClass = argmax(nodeLogits);

  // Initialize mask parameters (pre-sigmoid space, centered around 0 => sigmoid ≈ 0.5)
  const edgeMaskParams = new Float64Array(numEdges);
  const featureMaskParams = new Float64Array(featureDim);

  for (let i = 0; i < numEdges; i++) {
    edgeMaskParams[i] = (rng() - 0.5) * 2; // range [-1, 1]
  }
  for (let i = 0; i < featureDim; i++) {
    featureMaskParams[i] = (rng() - 0.5) * 2;
  }

  // Compute initial loss
  let edgeMaskSig = sigmoid(edgeMaskParams);
  let featureMaskSig = sigmoid(featureMaskParams);

  const { maskedGraph: initMaskedGraph, maskedFeatures: initMaskedFeatures } =
    buildMaskedGraph(graph, edgeMaskSig, featureMaskSig);
  const initPred = model(initMaskedGraph, initMaskedFeatures);
  const initNodePred = initPred.slice(
    nodeIdx * numClasses,
    (nodeIdx + 1) * numClasses,
  );

  // Softmax the prediction for loss computation
  let maxVal = -Infinity;
  for (let c = 0; c < numClasses; c++) {
    if (initNodePred[c]! > maxVal) maxVal = initNodePred[c]!;
  }
  let sumExp = 0;
  const initSoftmax = new Float64Array(numClasses);
  for (let c = 0; c < numClasses; c++) {
    initSoftmax[c] = Math.exp(initNodePred[c]! - maxVal);
    sumExp += initSoftmax[c]!;
  }
  for (let c = 0; c < numClasses; c++) {
    initSoftmax[c] = initSoftmax[c]! / sumExp;
  }

  let bestLoss = computeLoss(
    initSoftmax,
    targetClass,
    edgeMaskSig,
    featureMaskSig,
    config,
  );

  // Gradient-free optimization loop
  const perturbScale = config.lr;

  for (let epoch = 0; epoch < config.epochs; epoch++) {
    // Decay perturbation scale over epochs
    const currentScale = perturbScale * (1 - epoch / (config.epochs + 1));

    // Create perturbation
    const edgePert = new Float64Array(numEdges);
    const featurePert = new Float64Array(featureDim);

    for (let i = 0; i < numEdges; i++) {
      edgePert[i] = (rng() - 0.5) * 2 * currentScale;
    }
    for (let i = 0; i < featureDim; i++) {
      featurePert[i] = (rng() - 0.5) * 2 * currentScale;
    }

    // Apply perturbation
    const candidateEdgeParams = new Float64Array(numEdges);
    const candidateFeatureParams = new Float64Array(featureDim);

    for (let i = 0; i < numEdges; i++) {
      candidateEdgeParams[i] = edgeMaskParams[i]! + edgePert[i]!;
    }
    for (let i = 0; i < featureDim; i++) {
      candidateFeatureParams[i] = featureMaskParams[i]! + featurePert[i]!;
    }

    // Sigmoid activation
    const candEdgeSig = sigmoid(candidateEdgeParams);
    const candFeatureSig = sigmoid(candidateFeatureParams);

    // Build masked graph and forward pass
    const { maskedGraph, maskedFeatures } = buildMaskedGraph(
      graph,
      candEdgeSig,
      candFeatureSig,
    );
    const pred = model(maskedGraph, maskedFeatures);
    const nodePred = pred.slice(
      nodeIdx * numClasses,
      (nodeIdx + 1) * numClasses,
    );

    // Softmax
    let mx = -Infinity;
    for (let c = 0; c < numClasses; c++) {
      if (nodePred[c]! > mx) mx = nodePred[c]!;
    }
    let se = 0;
    const sm = new Float64Array(numClasses);
    for (let c = 0; c < numClasses; c++) {
      sm[c] = Math.exp(nodePred[c]! - mx);
      se += sm[c]!;
    }
    for (let c = 0; c < numClasses; c++) {
      sm[c] = sm[c]! / se;
    }

    const loss = computeLoss(sm, targetClass, candEdgeSig, candFeatureSig, config);

    // Keep perturbation if loss improved
    if (loss < bestLoss) {
      bestLoss = loss;
      for (let i = 0; i < numEdges; i++) {
        edgeMaskParams[i] = candidateEdgeParams[i]!;
      }
      for (let i = 0; i < featureDim; i++) {
        featureMaskParams[i] = candidateFeatureParams[i]!;
      }
      edgeMaskSig = candEdgeSig;
      featureMaskSig = candFeatureSig;
    }
  }

  // Threshold to get important edges and features
  const [srcArray, dstArray] = getEdgeIndex(graph);
  const importantEdges: [number, number][] = [];

  for (let e = 0; e < numEdges; e++) {
    if (edgeMaskSig[e]! >= config.edgeMaskThreshold) {
      importantEdges.push([srcArray[e]!, dstArray[e]!]);
    }
  }

  const importantFeatures: number[] = [];
  for (let f = 0; f < featureDim; f++) {
    if (featureMaskSig[f]! >= config.featureMaskThreshold) {
      importantFeatures.push(f);
    }
  }

  return {
    edgeMask: edgeMaskSig,
    featureMask: featureMaskSig,
    importantEdges,
    importantFeatures,
  };
}
