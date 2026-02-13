// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — Venue Heterogeneous Graph Builder
// Constructs a HeteroGraph for venue/planner/event domain modeling.
//
// Node types: 'venue', 'planner', 'event'
// Edge types: ['planner','books','venue'], ['event','held_at','venue']
// Reverse:    ['venue','booked_by','planner'], ['venue','hosts','event']
// ---------------------------------------------------------------------------

import type { HeteroGraph, HeteroNodeStore, HeteroEdgeStore } from '../types.js';
import { buildCSR } from '../graph.js';

/**
 * Build a heterogeneous venue graph from domain data.
 *
 * Algorithm:
 * 1. Register three node types with their feature stores.
 * 2. Build CSR for each forward edge type from the provided edge lists.
 * 3. Build CSR for reverse edge types by flipping source/destination.
 * 4. Package into a HeteroGraph.
 *
 * Edge type keys follow "srcType/relation/dstType" format.
 *
 * @param venues - Venue node data: features (count × featureDim), count, featureDim.
 * @param planners - Planner node data.
 * @param events - Event node data.
 * @param bookingEdges - Array of [plannerIdx, venueIdx] edges.
 * @param eventVenueEdges - Array of [eventIdx, venueIdx] edges.
 * @returns A HeteroGraph with forward and reverse edges.
 */
export function buildVenueHeteroGraph(
  venues: { features: Float64Array; count: number; featureDim: number },
  planners: { features: Float64Array; count: number; featureDim: number },
  events: { features: Float64Array; count: number; featureDim: number },
  bookingEdges: [number, number][],
  eventVenueEdges: [number, number][],
): HeteroGraph {
  // --- Node stores ---
  const nodes = new Map<string, HeteroNodeStore>();
  nodes.set('venue', {
    features: venues.features,
    count: venues.count,
    featureDim: venues.featureDim,
  });
  nodes.set('planner', {
    features: planners.features,
    count: planners.count,
    featureDim: planners.featureDim,
  });
  nodes.set('event', {
    features: events.features,
    count: events.count,
    featureDim: events.featureDim,
  });

  // --- Edge stores ---
  const edges = new Map<string, HeteroEdgeStore>();

  // Forward edge: planner --books--> venue
  // CSR indexed by destination (venue), colIdx contains source (planner)
  // For message passing: rowPtr[venue] → list of planners that booked this venue
  const booksCSR = buildEdgeTypeCSR(bookingEdges, venues.count);
  edges.set('planner/books/venue', booksCSR);

  // Forward edge: event --held_at--> venue
  const heldAtCSR = buildEdgeTypeCSR(eventVenueEdges, venues.count);
  edges.set('event/held_at/venue', heldAtCSR);

  // Reverse edge: venue --booked_by--> planner
  // Flip: [planner, venue] → [venue, planner]
  const reverseBookingEdges: [number, number][] = bookingEdges.map(
    ([p, v]) => [v, p] as [number, number],
  );
  const bookedByCSR = buildEdgeTypeCSR(reverseBookingEdges, planners.count);
  edges.set('venue/booked_by/planner', bookedByCSR);

  // Reverse edge: venue --hosts--> event
  // Flip: [event, venue] → [venue, event]
  const reverseEventEdges: [number, number][] = eventVenueEdges.map(
    ([e, v]) => [v, e] as [number, number],
  );
  const hostsCSR = buildEdgeTypeCSR(reverseEventEdges, events.count);
  edges.set('venue/hosts/event', hostsCSR);

  // --- Edge type triplets ---
  const edgeTypes: [string, string, string][] = [
    ['planner', 'books', 'venue'],
    ['event', 'held_at', 'venue'],
    ['venue', 'booked_by', 'planner'],
    ['venue', 'hosts', 'event'],
  ];

  return {
    nodeTypes: ['venue', 'planner', 'event'],
    edgeTypes,
    nodes,
    edges,
  };
}

/**
 * Build a CSR-like HeteroEdgeStore from an edge list.
 *
 * The edge list is [srcNode, dstNode][]. The CSR is indexed by dstNode
 * for message-passing (rowPtr[dst] gives the range of source neighbors).
 *
 * @param edgeList - Array of [src, dst] pairs.
 * @param numDstNodes - Number of destination nodes (determines rowPtr length).
 * @returns HeteroEdgeStore with CSR arrays.
 */
function buildEdgeTypeCSR(
  edgeList: [number, number][],
  numDstNodes: number,
): HeteroEdgeStore {
  if (edgeList.length === 0) {
    return {
      rowPtr: new Uint32Array(numDstNodes + 1),
      colIdx: new Uint32Array(0),
      numEdges: 0,
    };
  }

  // Sort by destination node, then by source node
  const sorted = edgeList
    .map((e, i) => ({ src: e[0]!, dst: e[1]!, idx: i }))
    .sort((a, b) => a.dst - b.dst || a.src - b.src);

  const numEdges = sorted.length;
  const rowPtr = new Uint32Array(numDstNodes + 1);
  const colIdx = new Uint32Array(numEdges);

  // Fill colIdx (source nodes) from sorted edges
  for (let i = 0; i < sorted.length; i++) {
    colIdx[i] = sorted[i]!.src;
  }

  // Count edges per destination node
  const counts = new Uint32Array(numDstNodes);
  for (let i = 0; i < sorted.length; i++) {
    counts[sorted[i]!.dst]!++;
  }

  // Prefix sum for rowPtr
  rowPtr[0] = 0;
  for (let i = 0; i < numDstNodes; i++) {
    rowPtr[i + 1] = rowPtr[i]! + counts[i]!;
  }

  return {
    rowPtr,
    colIdx,
    numEdges,
  };
}

/**
 * Add reverse edges for every edge type in a HeteroGraph.
 *
 * For each edge type [srcType, relation, dstType], adds a reverse edge type
 * [dstType, rev_{relation}, srcType] with flipped source/destination edges.
 * If the reverse edge type already exists, it is skipped.
 *
 * @param heteroGraph - Input heterogeneous graph.
 * @returns A new HeteroGraph with reverse edges added.
 */
export function addReverseEdges(heteroGraph: HeteroGraph): HeteroGraph {
  const newEdgeTypes: [string, string, string][] = [...heteroGraph.edgeTypes];
  const newEdges = new Map<string, HeteroEdgeStore>(heteroGraph.edges);

  for (const [srcType, relation, dstType] of heteroGraph.edgeTypes) {
    const revRelation = `rev_${relation}`;
    const revKey = `${dstType}/${revRelation}/${srcType}`;

    // Skip if reverse already exists
    if (newEdges.has(revKey)) continue;

    const fwdKey = `${srcType}/${relation}/${dstType}`;
    const fwdStore = heteroGraph.edges.get(fwdKey);
    if (!fwdStore) continue;

    // The forward CSR is indexed by dstNode with colIdx = srcNode.
    // The reverse CSR should be indexed by srcNode with colIdx = dstNode.
    // So we need to "transpose" the CSR.

    const srcStore = heteroGraph.nodes.get(srcType);
    if (!srcStore) continue;
    const numNewDstNodes = srcStore.count; // reverse dst is original src

    // Extract all edges from forward CSR: (srcNode in colIdx, dstNode from rowPtr)
    const dstStore = heteroGraph.nodes.get(dstType);
    if (!dstStore) continue;

    const reverseEdgeList: [number, number][] = [];
    for (let dst = 0; dst < dstStore.count; dst++) {
      const start = fwdStore.rowPtr[dst]!;
      const end = fwdStore.rowPtr[dst + 1]!;
      for (let e = start; e < end; e++) {
        const src = fwdStore.colIdx[e]!;
        // Reverse: original dst → original src
        // In reverse CSR, rowPtr is indexed by original src (new dst)
        // colIdx contains original dst (new src)
        reverseEdgeList.push([dst, src]);
      }
    }

    // Build reverse CSR indexed by the new destination (= original srcType)
    const revStore = buildEdgeTypeCSRFromPairs(reverseEdgeList, numNewDstNodes);

    newEdgeTypes.push([dstType, revRelation, srcType]);
    newEdges.set(revKey, revStore);
  }

  return {
    nodeTypes: heteroGraph.nodeTypes,
    edgeTypes: newEdgeTypes,
    nodes: heteroGraph.nodes,
    edges: newEdges,
  };
}

/**
 * Build HeteroEdgeStore from [src, dst] pairs where CSR is indexed by dst.
 */
function buildEdgeTypeCSRFromPairs(
  edgeList: [number, number][],
  numDstNodes: number,
): HeteroEdgeStore {
  if (edgeList.length === 0) {
    return {
      rowPtr: new Uint32Array(numDstNodes + 1),
      colIdx: new Uint32Array(0),
      numEdges: 0,
    };
  }

  // Sort by destination (second element), then source
  const sorted = edgeList
    .map((e) => ({ src: e[0]!, dst: e[1]! }))
    .sort((a, b) => a.dst - b.dst || a.src - b.src);

  const numEdges = sorted.length;
  const rowPtr = new Uint32Array(numDstNodes + 1);
  const colIdx = new Uint32Array(numEdges);

  for (let i = 0; i < sorted.length; i++) {
    colIdx[i] = sorted[i]!.src;
  }

  const counts = new Uint32Array(numDstNodes);
  for (let i = 0; i < sorted.length; i++) {
    counts[sorted[i]!.dst]!++;
  }

  rowPtr[0] = 0;
  for (let i = 0; i < numDstNodes; i++) {
    rowPtr[i + 1] = rowPtr[i]! + counts[i]!;
  }

  return {
    rowPtr,
    colIdx,
    numEdges,
  };
}
