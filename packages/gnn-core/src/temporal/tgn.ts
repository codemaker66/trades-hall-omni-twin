// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-6: Temporal Graph Networks (Rossi et al. 2020)
// Streaming temporal graph model with per-node memory, message passing,
// and GRU-based memory updates.
// ---------------------------------------------------------------------------

import type { TemporalGraph, TemporalEdge, TGNConfig, TGNWeights } from '../types.js';
import { gruCell, matVecMul } from '../tensor.js';
import { bochnerTimeEncoding } from './time-encoding.js';

/**
 * Create a temporal graph with zero-initialized memory and timestamps.
 *
 * @param numNodes - total number of nodes in the graph
 * @param memoryDim - dimensionality of per-node memory vectors
 * @returns initialized TemporalGraph
 */
export function createTemporalGraph(
  numNodes: number,
  memoryDim: number,
): TemporalGraph {
  return {
    numNodes,
    temporalEdges: [],
    nodeMemory: new Float64Array(numNodes * memoryDim),
    memoryDim,
    lastUpdate: new Float64Array(numNodes),
  };
}

/**
 * Process a batch of temporal edge events through the TGN pipeline:
 *   1. Compute messages from interaction events
 *   2. Aggregate messages per node
 *   3. Update node memory via GRU cell
 *
 * Mutates temporal.nodeMemory and temporal.lastUpdate in place.
 *
 * Message computation:
 *   msg = msgW * [memory_src || memory_dst || time_enc(dt) || edge_features]
 *
 * @param temporal - the temporal graph (mutated in place)
 * @param events - batch of temporal edge events (should be time-sorted)
 * @param weights - TGN weight matrices
 * @param config - TGN configuration
 */
export function tgnUpdate(
  temporal: TemporalGraph,
  events: TemporalEdge[],
  weights: TGNWeights,
  config: TGNConfig,
): void {
  const { memoryDim } = temporal;
  const timeDim = config.timeDim;
  const numFreqs = timeDim / 2; // Bochner encoding produces 2*numFreqs dims

  // Build frequency vector for time encoding (linearly spaced)
  const frequencies = new Float64Array(numFreqs);
  for (let i = 0; i < numFreqs; i++) {
    frequencies[i] = (i + 1) * 0.1;
  }

  // ---- Step 1: Compute raw messages for each event ----
  // Each event produces two messages: one for src, one for dst
  // Message input = [memory_src || memory_dst || time_enc(dt) || edge_features]

  // Collect messages per node: Map<nodeIdx, message[]>
  const nodeMessages = new Map<number, { msg: Float64Array; timestamp: number }[]>();

  for (const event of events) {
    const { src, dst, timestamp, features } = event;

    // Time deltas for src and dst
    const dtSrc = timestamp - temporal.lastUpdate[src]!;
    const dtDst = timestamp - temporal.lastUpdate[dst]!;

    // Time encodings (single timestamp -> 1 x timeDim)
    const timeEncSrc = bochnerTimeEncoding(
      new Float64Array([dtSrc]),
      frequencies,
    );
    const timeEncDst = bochnerTimeEncoding(
      new Float64Array([dtDst]),
      frequencies,
    );

    // Extract source and destination memory
    const memorySrc = temporal.nodeMemory.slice(
      src * memoryDim,
      (src + 1) * memoryDim,
    );
    const memoryDst = temporal.nodeMemory.slice(
      dst * memoryDim,
      (dst + 1) * memoryDim,
    );

    // Build concatenated input for src message:
    // [memory_src || memory_dst || time_enc_src || features]
    const inputDimTotal = memoryDim + memoryDim + timeDim + features.length;
    const srcInput = new Float64Array(inputDimTotal);
    let offset = 0;
    srcInput.set(memorySrc, offset); offset += memoryDim;
    srcInput.set(memoryDst, offset); offset += memoryDim;
    srcInput.set(timeEncSrc, offset); offset += timeDim;
    srcInput.set(features, offset);

    // Build concatenated input for dst message:
    // [memory_dst || memory_src || time_enc_dst || features]
    const dstInput = new Float64Array(inputDimTotal);
    offset = 0;
    dstInput.set(memoryDst, offset); offset += memoryDim;
    dstInput.set(memorySrc, offset); offset += memoryDim;
    dstInput.set(timeEncDst, offset); offset += timeDim;
    dstInput.set(features, offset);

    // Apply message weight matrix: msg = msgW * input
    // msgW is (messageDim x inputDimTotal)
    const msgSrc = matVecMul(weights.msgW, srcInput, config.messageDim, inputDimTotal);
    const msgDst = matVecMul(weights.msgW, dstInput, config.messageDim, inputDimTotal);

    // Store messages for aggregation
    if (!nodeMessages.has(src)) nodeMessages.set(src, []);
    nodeMessages.get(src)!.push({ msg: msgSrc, timestamp });

    if (!nodeMessages.has(dst)) nodeMessages.set(dst, []);
    nodeMessages.get(dst)!.push({ msg: msgDst, timestamp });
  }

  // ---- Step 2: Aggregate messages per node ----
  // ---- Step 3: Update memory via GRU ----

  for (const [nodeIdx, messages] of nodeMessages) {
    let aggregated: Float64Array;

    if (config.aggregator === 'last') {
      // Take the message with the most recent timestamp
      let bestIdx = 0;
      let bestTime = messages[0]!.timestamp;
      for (let i = 1; i < messages.length; i++) {
        if (messages[i]!.timestamp >= bestTime) {
          bestTime = messages[i]!.timestamp;
          bestIdx = i;
        }
      }
      aggregated = messages[bestIdx]!.msg;
    } else {
      // Mean aggregation
      const dim = messages[0]!.msg.length;
      aggregated = new Float64Array(dim);
      for (const m of messages) {
        for (let d = 0; d < dim; d++) {
          aggregated[d] = aggregated[d]! + m.msg[d]!;
        }
      }
      const count = messages.length;
      for (let d = 0; d < dim; d++) {
        aggregated[d] = aggregated[d]! / count;
      }
    }

    // GRU update: new_memory = GRU(aggregated_msg, current_memory)
    const hPrev = temporal.nodeMemory.slice(
      nodeIdx * memoryDim,
      (nodeIdx + 1) * memoryDim,
    );

    const hNew = gruCell(
      aggregated,
      hPrev,
      weights.gru.W_z,
      weights.gru.U_z,
      weights.gru.b_z,
      weights.gru.W_r,
      weights.gru.U_r,
      weights.gru.b_r,
      weights.gru.W_h,
      weights.gru.U_h,
      weights.gru.b_h,
    );

    // Write updated memory back
    temporal.nodeMemory.set(hNew, nodeIdx * memoryDim);
  }

  // ---- Update lastUpdate timestamps ----
  for (const event of events) {
    const { src, dst, timestamp } = event;
    if (timestamp > temporal.lastUpdate[src]!) {
      temporal.lastUpdate[src] = timestamp;
    }
    if (timestamp > temporal.lastUpdate[dst]!) {
      temporal.lastUpdate[dst] = timestamp;
    }
  }
}

/**
 * Simple identity embedding: return memory state for target nodes.
 *
 * @param temporal - temporal graph with node memory
 * @param targetNodes - node indices to embed
 * @param _featureDim - unused (memory dim is used from temporal graph)
 * @returns (targetNodes.length x memoryDim) matrix, row-major
 */
export function tgnEmbed(
  temporal: TemporalGraph,
  targetNodes: Uint32Array,
  _featureDim: number,
): Float64Array {
  const { memoryDim } = temporal;
  const numTargets = targetNodes.length;
  const out = new Float64Array(numTargets * memoryDim);

  for (let i = 0; i < numTargets; i++) {
    const nodeIdx = targetNodes[i]!;
    const memStart = nodeIdx * memoryDim;
    for (let d = 0; d < memoryDim; d++) {
      out[i * memoryDim + d] = temporal.nodeMemory[memStart + d]!;
    }
  }

  return out;
}
