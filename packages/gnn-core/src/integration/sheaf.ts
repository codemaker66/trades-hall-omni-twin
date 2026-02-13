// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — Sheaf Neural Networks (GNN-11)
// Cellular sheaf diffusion on graphs: learn restriction maps per edge,
// build the sheaf Laplacian, and run heat diffusion in the stalk space.
// ---------------------------------------------------------------------------

import type { Graph, SheafConfig, SheafWeights } from '../types.js';
import { matMul, matVecMul, relu, add } from '../tensor.js';

// ---------------------------------------------------------------------------
// 1. buildRestrictionMaps — MLP-based restriction maps per edge
// ---------------------------------------------------------------------------

/**
 * Compute restriction maps for every edge in the graph using a small MLP.
 *
 * For each edge (i, j), the restriction maps F_{i<-e} and F_{j<-e} are
 * stalkDim x stalkDim matrices that define how the stalk at each endpoint
 * relates to the stalk over the edge.
 *
 * Algorithm:
 * 1. For each edge (i, j), concatenate features x_i and x_j to form the
 *    MLP input of dimension 2 * featureDim.
 * 2. Pass through a 2-layer MLP:
 *    h = ReLU(W1 * concat(x_i, x_j) + b1)
 *    out = W2 * h + b2
 * 3. The output has dimension 2 * stalkDim^2.
 *    First stalkDim^2 values form F_{i<-e}, second stalkDim^2 form F_{j<-e}.
 * 4. Pack all maps into a single array: numEdges * 2 * stalkDim^2.
 *
 * @param graph - CSR graph.
 * @param X - Node features, flat row-major (numNodes x featureDim).
 * @param numNodes - Number of nodes.
 * @param featureDim - Feature dimension per node.
 * @param stalkDim - Stalk dimension (restriction maps are stalkDim x stalkDim).
 * @param mlpWeights - MLP weights { W1, b1, W2, b2 }.
 * @returns Packed restriction maps: Float64Array of length numEdges * 2 * stalkDim^2.
 */
export function buildRestrictionMaps(
  graph: Graph,
  X: Float64Array,
  numNodes: number,
  featureDim: number,
  stalkDim: number,
  mlpWeights: { W1: Float64Array; b1: Float64Array; W2: Float64Array; b2: Float64Array },
): Float64Array {
  const mapSize = stalkDim * stalkDim;
  const outputPerEdge = 2 * mapSize;
  const numEdges = graph.numEdges;
  const result = new Float64Array(numEdges * outputPerEdge);

  const inputDim = 2 * featureDim;
  const hiddenDim = mlpWeights.b1.length;

  // Process each edge
  let edgeIdx = 0;
  for (let i = 0; i < numNodes; i++) {
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;

    for (let e = start; e < end; e++) {
      const j = graph.colIdx[e]!;

      // Concatenate features of i and j
      const input = new Float64Array(inputDim);
      for (let f = 0; f < featureDim; f++) {
        input[f] = X[i * featureDim + f]!;
        input[featureDim + f] = X[j * featureDim + f]!;
      }

      // Layer 1: h = ReLU(W1 * input + b1)
      // W1 is hiddenDim x inputDim
      const z1 = matVecMul(mlpWeights.W1, input, hiddenDim, inputDim);
      const h = relu(add(z1, mlpWeights.b1));

      // Layer 2: out = W2 * h + b2
      // W2 is outputPerEdge x hiddenDim
      const z2 = matVecMul(mlpWeights.W2, h, outputPerEdge, hiddenDim);
      const out = add(z2, mlpWeights.b2);

      // Store in result
      const outOffset = edgeIdx * outputPerEdge;
      for (let k = 0; k < outputPerEdge; k++) {
        result[outOffset + k] = out[k]!;
      }

      edgeIdx++;
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// 2. sheafLaplacian — Compute the sheaf Laplacian L_F
// ---------------------------------------------------------------------------

/**
 * Compute the sheaf Laplacian L_F = delta^T delta where delta is the
 * sheaf coboundary operator.
 *
 * The sheaf Laplacian is a block matrix of size (N*d) x (N*d) where
 * N = numNodes and d = stalkDim. For each edge e = (i, j):
 *
 *   L_F[i*d:(i+1)*d, i*d:(i+1)*d] += F_{i<-e}^T F_{i<-e}
 *   L_F[j*d:(j+1)*d, j*d:(j+1)*d] += F_{j<-e}^T F_{j<-e}
 *   L_F[i*d:(i+1)*d, j*d:(j+1)*d] -= F_{i<-e}^T F_{j<-e}
 *   L_F[j*d:(j+1)*d, i*d:(i+1)*d] -= F_{j<-e}^T F_{i<-e}
 *
 * @param graph - CSR graph.
 * @param restrictionMaps - Packed maps from buildRestrictionMaps.
 * @param numNodes - Number of nodes.
 * @param stalkDim - Stalk dimension.
 * @returns Dense sheaf Laplacian of size (numNodes*stalkDim)^2.
 */
export function sheafLaplacian(
  graph: Graph,
  restrictionMaps: Float64Array,
  numNodes: number,
  stalkDim: number,
): Float64Array {
  const totalDim = numNodes * stalkDim;
  const L = new Float64Array(totalDim * totalDim);
  const d = stalkDim;
  const mapSize = d * d;

  // Iterate over all edges in CSR order
  let edgeIdx = 0;
  for (let i = 0; i < numNodes; i++) {
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;

    for (let e = start; e < end; e++) {
      const j = graph.colIdx[e]!;

      // Extract restriction maps for this edge
      const baseOffset = edgeIdx * 2 * mapSize;

      // F_ie: stalkDim x stalkDim (restriction from edge to node i)
      // F_je: stalkDim x stalkDim (restriction from edge to node j)

      // Compute F_ie^T F_ie and add to L[i,i] block
      // Compute F_ie^T F_je and subtract from L[i,j] block
      // Compute F_je^T F_je and add to L[j,j] block
      // Compute F_je^T F_ie and subtract from L[j,i] block

      const iBlock = i * d;
      const jBlock = j * d;

      for (let r = 0; r < d; r++) {
        for (let c = 0; c < d; c++) {
          // F_ie^T F_ie: sum_k F_ie[k,r] * F_ie[k,c]
          let ftieFie = 0;
          // F_ie^T F_je: sum_k F_ie[k,r] * F_je[k,c]
          let ftieFje = 0;
          // F_je^T F_je: sum_k F_je[k,r] * F_je[k,c]
          let ftjeFje = 0;
          // F_je^T F_ie: sum_k F_je[k,r] * F_ie[k,c]
          let ftjeFie = 0;

          for (let k = 0; k < d; k++) {
            const fie_kr = restrictionMaps[baseOffset + k * d + r]!;
            const fie_kc = restrictionMaps[baseOffset + k * d + c]!;
            const fje_kr = restrictionMaps[baseOffset + mapSize + k * d + r]!;
            const fje_kc = restrictionMaps[baseOffset + mapSize + k * d + c]!;

            ftieFie += fie_kr * fie_kc;
            ftieFje += fie_kr * fje_kc;
            ftjeFje += fje_kr * fje_kc;
            ftjeFie += fje_kr * fie_kc;
          }

          // L[i,i] block += F_ie^T F_ie
          L[(iBlock + r) * totalDim + (iBlock + c)] =
            L[(iBlock + r) * totalDim + (iBlock + c)]! + ftieFie;

          // L[j,j] block += F_je^T F_je
          L[(jBlock + r) * totalDim + (jBlock + c)] =
            L[(jBlock + r) * totalDim + (jBlock + c)]! + ftjeFje;

          // L[i,j] block -= F_ie^T F_je
          L[(iBlock + r) * totalDim + (jBlock + c)] =
            L[(iBlock + r) * totalDim + (jBlock + c)]! - ftieFje;

          // L[j,i] block -= F_je^T F_ie
          L[(jBlock + r) * totalDim + (iBlock + c)] =
            L[(jBlock + r) * totalDim + (iBlock + c)]! - ftjeFie;
        }
      }

      edgeIdx++;
    }
  }

  return L;
}

// ---------------------------------------------------------------------------
// 3. neuralSheafDiffusion — Heat diffusion on sheaf Laplacian
// ---------------------------------------------------------------------------

/**
 * Neural Sheaf Diffusion: project features to stalk space, build restriction
 * maps via MLP, then perform T steps of heat diffusion on the sheaf Laplacian.
 *
 * Algorithm:
 * 1. Project node features to stalk space: x_stalk_v = W_proj * x_v
 *    for each node v, where W_proj is (stalkDim x featureDim).
 * 2. Build restriction maps from current features via the MLP.
 * 3. Compute the sheaf Laplacian L_F.
 * 4. For T diffusion steps:
 *    x_stalk = x_stalk - dt * L_F * x_stalk
 *    (Explicit Euler discretisation of heat equation dx/dt = -L_F x)
 * 5. Return final stalk features (numNodes x stalkDim).
 *
 * @param graph - CSR graph.
 * @param X - Node features, flat row-major (numNodes x featureDim).
 * @param numNodes - Number of nodes.
 * @param featureDim - Feature dimension per node.
 * @param config - SheafConfig with stalkDim, diffusionSteps, learningRate (used as dt).
 * @param weights - SheafWeights with restrictionMLP and stalkDim.
 * @returns Final stalk features, flat row-major (numNodes x stalkDim).
 */
export function neuralSheafDiffusion(
  graph: Graph,
  X: Float64Array,
  numNodes: number,
  featureDim: number,
  config: SheafConfig,
  weights: SheafWeights,
): Float64Array {
  const { stalkDim, diffusionSteps, learningRate: dt } = config;

  // Step 1: Project features to stalk space
  // We use the first layer of the restriction MLP's W1 to derive a projection.
  // The projection matrix is stalkDim x featureDim.
  // Extract from the MLP: use first stalkDim rows of W1 (which is hiddenDim x (2*featureDim)).
  // For a clean design, we build W_proj from the first stalkDim*featureDim entries
  // of the MLP's W1 (the part corresponding to the first node's features).
  const mlpLayers = weights.restrictionMLP.layers;
  const W1 = mlpLayers[0]!.W;
  const mlpHiddenDim = mlpLayers[0]!.outDim;

  // Build a simple projection: take the stalkDim x featureDim sub-block of W1
  // W1 is mlpHiddenDim x (2*featureDim), we take the top-left stalkDim x featureDim block
  const W_proj = new Float64Array(stalkDim * featureDim);
  const inputDimFull = 2 * featureDim;
  for (let r = 0; r < stalkDim && r < mlpHiddenDim; r++) {
    for (let c = 0; c < featureDim; c++) {
      W_proj[r * featureDim + c] = W1[r * inputDimFull + c]!;
    }
  }

  // Project each node: x_stalk = W_proj * x
  const totalStalkDim = numNodes * stalkDim;
  let xStalk = new Float64Array(totalStalkDim);

  for (let v = 0; v < numNodes; v++) {
    const nodeFeats = X.subarray(v * featureDim, (v + 1) * featureDim);
    const projected = matVecMul(W_proj, nodeFeats, stalkDim, featureDim);
    for (let d = 0; d < stalkDim; d++) {
      xStalk[v * stalkDim + d] = projected[d]!;
    }
  }

  // Step 2: Build restriction maps from node features
  // Construct MLP weights in the format expected by buildRestrictionMaps
  const b1 = mlpLayers[0]!.bias;
  const W2 = mlpLayers.length > 1 ? mlpLayers[1]!.W : new Float64Array(2 * stalkDim * stalkDim * mlpHiddenDim);
  const b2 = mlpLayers.length > 1 ? mlpLayers[1]!.bias : new Float64Array(2 * stalkDim * stalkDim);

  const maps = buildRestrictionMaps(
    graph,
    X,
    numNodes,
    featureDim,
    stalkDim,
    { W1, b1, W2, b2 },
  );

  // Step 3: Compute sheaf Laplacian
  const L = sheafLaplacian(graph, maps, numNodes, stalkDim);

  // Step 4: Heat diffusion for T steps
  // x_stalk = x_stalk - dt * L_F * x_stalk
  for (let t = 0; t < diffusionSteps; t++) {
    // L_F * x_stalk (matrix-vector multiply, L is totalStalkDim x totalStalkDim)
    const Lx = matVecMul(L, xStalk, totalStalkDim, totalStalkDim);

    // x_stalk = x_stalk - dt * Lx
    const newStalk = new Float64Array(totalStalkDim);
    for (let i = 0; i < totalStalkDim; i++) {
      newStalk[i] = xStalk[i]! - dt * Lx[i]!;
    }
    xStalk = newStalk;
  }

  return xStalk;
}
