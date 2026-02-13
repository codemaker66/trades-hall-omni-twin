// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — Demand Prediction with GNN (GNN-11)
// GCN-based demand prediction for venue nodes with time features,
// plus stochastic pricing optimisation driven by predicted demand.
// ---------------------------------------------------------------------------

import type { Graph, PRNG, DemandForecast, PricingResult } from '../types.js';
import { matVecMul, relu, add, dot } from '../tensor.js';
import { addSelfLoops, normalizeAdjacency, getNeighbors } from '../graph.js';

// ---------------------------------------------------------------------------
// 1. demandPredictorGNN — 2-layer GCN + MLP head for demand forecasting
// ---------------------------------------------------------------------------

/**
 * Predict per-node demand using a 2-layer Graph Convolutional Network with
 * time-feature concatenation and an MLP head that outputs both mean and
 * variance (aleatoric uncertainty).
 *
 * Architecture:
 * 1. Concatenate node features (featureDim) with time features (timeDim)
 *    to form input of dimension (featureDim + timeDim) per node.
 * 2. GCN Layer 1: H1 = ReLU(D^{-1/2} A D^{-1/2} * X_concat * W1 + b1)
 * 3. GCN Layer 2: H2 = ReLU(D^{-1/2} A D^{-1/2} * H1 * W2 + b2)
 * 4. MLP Head Layer 1: Z = ReLU(W_mlp1 * H2 + b_mlp1)
 * 5. MLP Head Layer 2: [mean, log_var] = W_mlp2 * Z + b_mlp2
 *    variance = exp(log_var) for numerical stability.
 *
 * @param graph - Venue graph in CSR format.
 * @param nodeFeatures - Node features, flat row-major (numNodes x featureDim).
 * @param numNodes - Number of venue nodes.
 * @param featureDim - Feature dimension per node.
 * @param timeFeatures - Time features per node, flat row-major (numNodes x timeDim).
 * @param timeDim - Time feature dimension.
 * @param model - Pre-trained weights for GCN layers and MLP head.
 * @returns DemandForecast with mean, variance, and timestamps per node.
 */
export function demandPredictorGNN(
  graph: Graph,
  nodeFeatures: Float64Array,
  numNodes: number,
  featureDim: number,
  timeFeatures: Float64Array,
  timeDim: number,
  model: {
    gnnWeights: { W: Float64Array; b: Float64Array }[];
    mlpWeights: { W: Float64Array; b: Float64Array }[];
  },
): DemandForecast {
  // --- Prepare normalised adjacency ---
  const graphSL = addSelfLoops(graph);
  const normGraph = normalizeAdjacency(graphSL, 'symmetric');

  // --- Step 1: Concatenate node features with time features ---
  const inputDim = featureDim + timeDim;
  let X = new Float64Array(numNodes * inputDim);

  for (let v = 0; v < numNodes; v++) {
    for (let f = 0; f < featureDim; f++) {
      X[v * inputDim + f] = nodeFeatures[v * featureDim + f]!;
    }
    for (let t = 0; t < timeDim; t++) {
      X[v * inputDim + featureDim + t] = timeFeatures[v * timeDim + t]!;
    }
  }

  // --- Step 2 & 3: Two GCN layers ---
  let H = X;
  let currentDim = inputDim;

  for (let layer = 0; layer < 2 && layer < model.gnnWeights.length; layer++) {
    const layerW = model.gnnWeights[layer]!;
    const W = layerW.W;
    const b = layerW.b;
    const outDim = b.length;

    // Transform: Z = H * W (per node)
    // W is currentDim x outDim
    const Z = new Float64Array(numNodes * outDim);
    for (let v = 0; v < numNodes; v++) {
      const nodeH = H.subarray(v * currentDim, (v + 1) * currentDim);
      const transformed = matVecMul(
        // Need to treat W as outDim x currentDim for matVecMul (it computes M*v)
        // But our W is currentDim x outDim (row-major). We need W^T * nodeH
        // which equals the rows of W^T dotted with nodeH.
        // Alternatively, compute each output dimension manually.
        W, nodeH, outDim, currentDim,
      );
      const biased = add(transformed, b);
      for (let d = 0; d < outDim; d++) {
        Z[v * outDim + d] = biased[d]!;
      }
    }

    // Aggregate: AZ = A_norm * Z (message passing)
    const AZ = new Float64Array(numNodes * outDim);
    for (let v = 0; v < numNodes; v++) {
      const start = normGraph.rowPtr[v]!;
      const end = normGraph.rowPtr[v + 1]!;

      for (let e = start; e < end; e++) {
        const neighbor = normGraph.colIdx[e]!;
        const w = normGraph.edgeWeights ? normGraph.edgeWeights[e]! : 1.0;

        for (let d = 0; d < outDim; d++) {
          AZ[v * outDim + d] = AZ[v * outDim + d]! + w * Z[neighbor * outDim + d]!;
        }
      }
    }

    // Apply ReLU activation
    H = relu(AZ) as Float64Array<ArrayBuffer>;
    currentDim = outDim;
  }

  // --- Step 4 & 5: MLP head ---
  // MLP Layer 1: Z_mlp = ReLU(W_mlp1 * H_node + b_mlp1)
  const mlp1W = model.mlpWeights[0]!;
  const mlp1OutDim = mlp1W.b.length;

  let mlpH = new Float64Array(numNodes * mlp1OutDim);
  for (let v = 0; v < numNodes; v++) {
    const nodeH = H.subarray(v * currentDim, (v + 1) * currentDim);
    const z = matVecMul(mlp1W.W, nodeH, mlp1OutDim, currentDim);
    const activated = relu(add(z, mlp1W.b));
    for (let d = 0; d < mlp1OutDim; d++) {
      mlpH[v * mlp1OutDim + d] = activated[d]!;
    }
  }

  // MLP Layer 2: [mean, log_var] = W_mlp2 * Z_mlp + b_mlp2
  // Output dim = 2 (mean + log_variance)
  const mlp2W = model.mlpWeights.length > 1 ? model.mlpWeights[1]! : model.mlpWeights[0]!;
  const finalOutDim = mlp2W.b.length; // Should be 2

  const means = new Float64Array(numNodes);
  const variances = new Float64Array(numNodes);
  const timestamps = new Float64Array(numNodes);

  for (let v = 0; v < numNodes; v++) {
    const nodeH2 = mlpH.subarray(v * mlp1OutDim, (v + 1) * mlp1OutDim);
    const out = matVecMul(mlp2W.W, nodeH2, finalOutDim, mlp1OutDim);
    const biasedOut = add(out, mlp2W.b);

    // First output is mean demand, second is log-variance
    means[v] = biasedOut[0]!;
    const logVar = finalOutDim > 1 ? biasedOut[1]! : 0;
    // Clamp log_var to avoid numerical issues
    variances[v] = Math.exp(Math.min(Math.max(logVar, -10), 10));
    timestamps[v] = v; // Placeholder timestamp (node index)
  }

  return { mean: means, variance: variances, timestamps };
}

// ---------------------------------------------------------------------------
// 2. stochasticPricingOptimizer — Revenue-maximising price per node
// ---------------------------------------------------------------------------

/**
 * For each node, find the price that maximises expected revenue under a
 * simple linear demand model with uncertainty.
 *
 * Demand model:
 *   demand(price) = mean_demand * (1 - (price - minPrice) / (maxPrice - minPrice) * elasticity)
 * where elasticity is derived from the demand variance (higher variance => more elastic).
 *
 * Revenue = price * demand(price).
 *
 * Algorithm:
 * 1. For each node, sample `numSamples` candidate prices uniformly from
 *    [minPrice, maxPrice].
 * 2. For each candidate, compute demand and expected revenue, accounting
 *    for demand uncertainty via Monte Carlo sampling from N(mean, variance).
 * 3. Select the price that maximises expected revenue.
 *
 * @param demand - DemandForecast from the GNN predictor.
 * @param priceRange - [minPrice, maxPrice] range.
 * @param numSamples - Number of candidate prices to evaluate per node.
 * @param rng - Deterministic PRNG.
 * @returns PricingResult with optimal price, expected revenue, and demand at that price.
 */
export function stochasticPricingOptimizer(
  demand: DemandForecast,
  priceRange: [number, number],
  numSamples: number,
  rng: PRNG,
): PricingResult {
  const [minPrice, maxPrice] = priceRange;
  const priceSpan = maxPrice - minPrice;
  const numNodes = demand.mean.length;

  // We return the best aggregate result across all nodes
  let totalBestRevenue = 0;
  let totalDemandAtBest = 0;

  // Per-node optimal prices and revenues
  const optimalPrices = new Float64Array(numNodes);
  const expectedRevenues = new Float64Array(numNodes);

  for (let v = 0; v < numNodes; v++) {
    const baseDemand = Math.max(demand.mean[v]!, 0);
    const demandVar = demand.variance[v]!;

    // Elasticity: higher variance implies more uncertain/elastic demand
    // Clamp elasticity between 0.1 and 0.95
    const elasticity = Math.min(0.95, Math.max(0.1, 0.5 + 0.1 * Math.sqrt(demandVar)));

    let bestPrice = minPrice;
    let bestRevenue = -Infinity;
    let bestDemand = 0;

    for (let s = 0; s < numSamples; s++) {
      const price = minPrice + rng() * priceSpan;

      // Linear demand model: demand decreases with price
      const priceFraction = priceSpan > 0 ? (price - minPrice) / priceSpan : 0;
      const expectedDemand = baseDemand * (1 - priceFraction * elasticity);

      // Monte Carlo: sample a few demand realisations and average revenue
      const mcSamples = 5;
      let revenueSum = 0;

      for (let mc = 0; mc < mcSamples; mc++) {
        // Box-Muller for normal sample
        const u1 = rng() || 1e-15;
        const u2 = rng();
        const normal = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);

        const sampledDemand = Math.max(0, expectedDemand + Math.sqrt(demandVar) * normal);
        revenueSum += price * sampledDemand;
      }

      const avgRevenue = revenueSum / mcSamples;

      if (avgRevenue > bestRevenue) {
        bestRevenue = avgRevenue;
        bestPrice = price;
        bestDemand = Math.max(0, expectedDemand);
      }
    }

    optimalPrices[v] = bestPrice;
    expectedRevenues[v] = bestRevenue;
    totalBestRevenue += bestRevenue;
    totalDemandAtBest += bestDemand;
  }

  // Return aggregate result (sum across all nodes)
  // Per the type definition: single optimal price, revenue, demand
  // Use the mean optimal price weighted by expected revenue
  let weightedPriceSum = 0;
  let revenueTotal = 0;
  for (let v = 0; v < numNodes; v++) {
    weightedPriceSum += optimalPrices[v]! * expectedRevenues[v]!;
    revenueTotal += expectedRevenues[v]!;
  }

  const avgOptimalPrice = revenueTotal > 0 ? weightedPriceSum / revenueTotal : (minPrice + maxPrice) / 2;

  return {
    optimalPrice: avgOptimalPrice,
    expectedRevenue: totalBestRevenue,
    demandAtPrice: totalDemandAtBest / Math.max(numNodes, 1),
  };
}
