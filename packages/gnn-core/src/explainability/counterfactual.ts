// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — CF-GNNExplainer (Lucic et al. 2022)
// Counterfactual explanation: find minimal edge deletions that change
// the model's prediction for a target node.
// ---------------------------------------------------------------------------

import type {
  Graph,
  GNNForwardFn,
  ExplainerConfig,
  CounterfactualResult,
  PRNG,
} from '../types.js';
import { argmax } from '../tensor.js';
import { getEdgeIndex, buildCSR } from '../graph.js';

// ---- Helpers ----

/**
 * Build a graph with specific edges removed.
 *
 * @param graph - Original CSR graph.
 * @param removedEdgeIndices - Set of edge indices to remove.
 * @returns A new Graph without the removed edges, preserving node features.
 */
function buildGraphWithoutEdges(
  graph: Graph,
  removedEdgeIndices: Set<number>,
): Graph {
  const edges: [number, number][] = [];
  const weights: number[] = [];

  for (let i = 0; i < graph.numNodes; i++) {
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;
    for (let e = start; e < end; e++) {
      if (!removedEdgeIndices.has(e)) {
        edges.push([i, graph.colIdx[e]!]);
        weights.push(graph.edgeWeights ? graph.edgeWeights[e]! : 1.0);
      }
    }
  }

  const csrGraph = buildCSR(edges, graph.numNodes, weights);

  return {
    ...csrGraph,
    nodeFeatures: graph.nodeFeatures,
    featureDim: graph.featureDim,
    edgeFeatures: graph.edgeFeatures,
    edgeFeatureDim: graph.edgeFeatureDim,
  };
}

/**
 * Get the predicted class for a specific node.
 *
 * @param model - GNN forward function.
 * @param graph - Input graph.
 * @param nodeIdx - Target node index.
 * @returns The predicted class index and the number of classes.
 */
function getPrediction(
  model: GNNForwardFn,
  graph: Graph,
  nodeIdx: number,
): { predictedClass: number; numClasses: number } {
  const output = model(graph, graph.nodeFeatures);
  const numClasses = output.length / graph.numNodes;
  const nodeLogits = output.slice(
    nodeIdx * numClasses,
    (nodeIdx + 1) * numClasses,
  );
  return { predictedClass: argmax(nodeLogits), numClasses };
}

/**
 * Compute the prediction confidence for a specific class at a node.
 * Uses softmax to convert logits to probabilities.
 *
 * @param model - GNN forward function.
 * @param graph - Input graph.
 * @param nodeIdx - Target node index.
 * @param numClasses - Number of output classes.
 * @returns Softmax probabilities for the target node.
 */
function getPredictionProbabilities(
  model: GNNForwardFn,
  graph: Graph,
  nodeIdx: number,
  numClasses: number,
): Float64Array {
  const output = model(graph, graph.nodeFeatures);
  const nodeLogits = output.slice(
    nodeIdx * numClasses,
    (nodeIdx + 1) * numClasses,
  );

  // Softmax
  let maxVal = -Infinity;
  for (let c = 0; c < numClasses; c++) {
    if (nodeLogits[c]! > maxVal) maxVal = nodeLogits[c]!;
  }
  let sumExp = 0;
  const probs = new Float64Array(numClasses);
  for (let c = 0; c < numClasses; c++) {
    probs[c] = Math.exp(nodeLogits[c]! - maxVal);
    sumExp += probs[c]!;
  }
  for (let c = 0; c < numClasses; c++) {
    probs[c] = probs[c]! / sumExp;
  }

  return probs;
}

// ---- Main Export ----

/**
 * CF-GNNExplainer — Lucic et al. 2022.
 *
 * Finds minimal edge deletions that change the model's prediction for
 * a target node. Uses a greedy search strategy.
 *
 * Algorithm:
 * 1. Get the original prediction for the target node.
 * 2. Build a candidate set of all edges adjacent to the target node
 *    (and optionally edges in the local neighborhood).
 * 3. Iteratively:
 *    a. For each candidate edge not yet removed:
 *       - Temporarily remove it and check the new prediction.
 *       - Score the removal by how much it decreases confidence in the
 *         original class.
 *    b. Select the edge removal that most decreases original-class confidence.
 *    c. If the prediction flips (argmax changes), stop.
 * 4. Return the set of removed edges, original/new predictions, and edit count.
 *
 * @param model - GNN forward function.
 * @param graph - Input CSR graph with node features.
 * @param nodeIdx - Target node index to generate counterfactual explanation for.
 * @param config - Explainer configuration (epochs used as max edits).
 * @param _rng - PRNG (reserved for tie-breaking; greedy search is deterministic).
 * @returns CounterfactualResult with removed edges and prediction change info.
 */
export function cfExplainer(
  model: GNNForwardFn,
  graph: Graph,
  nodeIdx: number,
  config: ExplainerConfig,
  _rng: PRNG,
): CounterfactualResult {
  const maxEdits = config.epochs; // reuse epochs as max edit budget

  // Step 1: Get original prediction
  const { predictedClass: originalPred, numClasses } = getPrediction(
    model,
    graph,
    nodeIdx,
  );

  // Build candidate edge set: all edges in the graph that are adjacent
  // to nodeIdx or within its local neighborhood (2-hop for better coverage)
  const [srcArray, dstArray] = getEdgeIndex(graph);

  // Collect 1-hop neighbors
  const neighbors = new Set<number>();
  neighbors.add(nodeIdx);
  const start = graph.rowPtr[nodeIdx]!;
  const end = graph.rowPtr[nodeIdx + 1]!;
  for (let e = start; e < end; e++) {
    neighbors.add(graph.colIdx[e]!);
  }

  // Collect 2-hop neighbors
  const twoHopNeighbors = new Set<number>(neighbors);
  for (const n of neighbors) {
    const nStart = graph.rowPtr[n]!;
    const nEnd = graph.rowPtr[n + 1]!;
    for (let e = nStart; e < nEnd; e++) {
      twoHopNeighbors.add(graph.colIdx[e]!);
    }
  }

  // Candidate edges: those where at least one endpoint is in the neighborhood
  const candidateEdgeIndices: number[] = [];
  for (let e = 0; e < graph.numEdges; e++) {
    if (twoHopNeighbors.has(srcArray[e]!) || twoHopNeighbors.has(dstArray[e]!)) {
      candidateEdgeIndices.push(e);
    }
  }

  // Step 2: Greedy iterative removal
  const removedEdgeIndices = new Set<number>();
  const removedEdges: [number, number][] = [];
  let currentPred = originalPred;
  let editsUsed = 0;

  for (let edit = 0; edit < maxEdits; edit++) {
    let bestEdgeIdx = -1;
    let bestConfidenceDrop = -Infinity;
    let bestNewClass = originalPred;

    // Get current probabilities
    const currentGraph = buildGraphWithoutEdges(graph, removedEdgeIndices);
    const currentProbs = getPredictionProbabilities(
      model,
      currentGraph,
      nodeIdx,
      numClasses,
    );
    const currentOrigClassConf = currentProbs[originalPred]!;

    // Try removing each candidate edge
    for (const edgeIdx of candidateEdgeIndices) {
      if (removedEdgeIndices.has(edgeIdx)) continue;

      // Temporarily add this edge to the removed set
      const trialRemoved = new Set(removedEdgeIndices);
      trialRemoved.add(edgeIdx);

      // Build trial graph and evaluate
      const trialGraph = buildGraphWithoutEdges(graph, trialRemoved);
      const trialProbs = getPredictionProbabilities(
        model,
        trialGraph,
        nodeIdx,
        numClasses,
      );

      // Score: how much does the original-class confidence drop?
      const confidenceDrop = currentOrigClassConf - trialProbs[originalPred]!;

      if (confidenceDrop > bestConfidenceDrop) {
        bestConfidenceDrop = confidenceDrop;
        bestEdgeIdx = edgeIdx;
        bestNewClass = argmax(trialProbs);
      }
    }

    // If no candidate edge found, stop
    if (bestEdgeIdx < 0) break;

    // Commit the best removal
    removedEdgeIndices.add(bestEdgeIdx);
    removedEdges.push([srcArray[bestEdgeIdx]!, dstArray[bestEdgeIdx]!]);
    currentPred = bestNewClass;
    editsUsed++;

    // Check if prediction has flipped
    if (currentPred !== originalPred) {
      break;
    }
  }

  return {
    removedEdges,
    originalPrediction: originalPred,
    newPrediction: currentPred,
    numEditsRequired: editsUsed,
  };
}
