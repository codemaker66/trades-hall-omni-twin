// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-12 Production Architecture: Event Processor
// Streaming event processor for incremental graph updates.
// ---------------------------------------------------------------------------

import type { GraphStore } from './graph-store.js';
import type { EmbeddingCache } from './cache-manager.js';

// ---- Types ----

/** A single graph mutation event. */
export interface EventUpdate {
  readonly type:
    | 'node_add'
    | 'node_remove'
    | 'node_update'
    | 'edge_add'
    | 'edge_remove';
  /** Unique ID for the node or edge being mutated. */
  readonly entityId: string;
  /** For node events: node type string. */
  readonly nodeType?: string;
  /** For node events: feature vector. */
  readonly features?: Float64Array;
  /** For edge events: source node ID. */
  readonly src?: string;
  /** For edge events: destination node ID. */
  readonly dst?: string;
  /** For edge events: edge type string. */
  readonly edgeType?: string;
  /** For edge events: optional edge features. */
  readonly edgeFeatures?: Float64Array;
}

/** Result of processing one or more events. */
export interface ProcessResult {
  /** Node IDs whose embeddings may be stale and need recomputation. */
  readonly affectedNodes: string[];
  /** Whether any affected nodes exist (i.e. recomputation is warranted). */
  readonly recomputeNeeded: boolean;
}

// ---------------------------------------------------------------------------
// EventProcessor — Applies streaming events to a GraphStore
// ---------------------------------------------------------------------------

/**
 * Processes streaming graph mutation events, applies them to a `GraphStore`,
 * invalidates stale cache entries in an `EmbeddingCache`, and determines which
 * nodes require embedding recomputation.
 *
 * The `recomputeRadius` parameter controls how many hops around a changed node
 * are considered affected. A value of 2 means that 2-hop neighbors of any
 * mutated node will have their cached embeddings invalidated.
 */
export class EventProcessor {
  private readonly graphStore: GraphStore;
  private readonly cache: EmbeddingCache;
  private readonly recomputeRadius: number;

  constructor(
    graphStore: GraphStore,
    cache: EmbeddingCache,
    recomputeRadius: number,
  ) {
    this.graphStore = graphStore;
    this.cache = cache;
    this.recomputeRadius = recomputeRadius;
  }

  // ---- Single event processing ----

  /**
   * Apply a single event to the graph store, invalidate affected cache entries,
   * and return the set of nodes that need embedding recomputation.
   */
  processEvent(event: EventUpdate): ProcessResult {
    const directlyAffected = this.applyEvent(event);

    // Expand affected set to the recompute radius
    const allAffected = this.expandAffectedSet(directlyAffected);

    // Invalidate cache entries for all affected nodes
    for (const nodeId of allAffected) {
      this.cache.invalidate(nodeId);
    }

    return {
      affectedNodes: allAffected,
      recomputeNeeded: allAffected.length > 0,
    };
  }

  // ---- Batch event processing ----

  /**
   * Process multiple events, deduplicating the affected node set.
   *
   * Events are applied in order. The union of all affected neighborhoods
   * is computed once at the end to avoid redundant BFS expansions.
   */
  processBatch(events: EventUpdate[]): ProcessResult {
    const directlyAffectedSet = new Set<string>();

    for (const event of events) {
      const affected = this.applyEvent(event);
      for (const nodeId of affected) {
        directlyAffectedSet.add(nodeId);
      }
    }

    // Expand all directly affected nodes to the recompute radius
    const allAffected = this.expandAffectedSet(Array.from(directlyAffectedSet));

    // Invalidate cache entries
    for (const nodeId of allAffected) {
      this.cache.invalidate(nodeId);
    }

    return {
      affectedNodes: allAffected,
      recomputeNeeded: allAffected.length > 0,
    };
  }

  // ---- Subgraph extraction ----

  /**
   * Get the subgraph around a set of affected nodes for incremental
   * GNN recomputation.
   *
   * The returned subgraph contains all nodes within `recomputeRadius` hops
   * of any affected node, plus all edges between those nodes.
   */
  getRecomputeSubgraph(
    affectedNodes: string[],
  ): { nodeIds: string[]; edgeIds: string[] } {
    const allNodeIds = new Set<string>();
    const allEdgeIds = new Set<string>();

    for (const nodeId of affectedNodes) {
      const sub = this.graphStore.getKHopSubgraph(nodeId, this.recomputeRadius);
      for (const nid of sub.nodeIds) {
        allNodeIds.add(nid);
      }
      for (const eid of sub.edgeIds) {
        allEdgeIds.add(eid);
      }
    }

    return {
      nodeIds: Array.from(allNodeIds),
      edgeIds: Array.from(allEdgeIds),
    };
  }

  // ---- Private helpers ----

  /**
   * Apply a single event to the graph store and return the list of
   * directly affected node IDs (the node itself plus its immediate
   * neighbors *before* the mutation).
   */
  private applyEvent(event: EventUpdate): string[] {
    const affected: string[] = [];

    switch (event.type) {
      case 'node_add': {
        const features = event.features ?? new Float64Array(0);
        const nodeType = event.nodeType ?? 'unknown';
        this.graphStore.addNode(event.entityId, nodeType, features);
        affected.push(event.entityId);
        break;
      }

      case 'node_remove': {
        // Collect neighbors before removal so we know what is affected
        const neighbors = this.graphStore.getNeighborIds(event.entityId);
        affected.push(event.entityId, ...neighbors);
        this.graphStore.removeNode(event.entityId);
        break;
      }

      case 'node_update': {
        if (event.features) {
          this.graphStore.updateNodeFeatures(event.entityId, event.features);
        }
        affected.push(event.entityId);
        // Neighbors are affected too because their aggregated messages change
        const neighbors = this.graphStore.getNeighborIds(event.entityId);
        affected.push(...neighbors);
        break;
      }

      case 'edge_add': {
        if (event.src && event.dst) {
          const edgeType = event.edgeType ?? 'default';
          this.graphStore.addEdge(
            event.entityId,
            event.src,
            event.dst,
            edgeType,
            event.edgeFeatures,
          );
          affected.push(event.src, event.dst);
        }
        break;
      }

      case 'edge_remove': {
        // Look up the edge endpoints before removal
        // We need to iterate edges to find this edge's src/dst
        // The entityId is the edge ID in the store
        const node = this.graphStore.getNode(event.entityId);
        if (!node) {
          // entityId is an edge ID — just remove it; the GraphStore
          // handles adjacency cleanup internally
          // We need the edge endpoints for affected set, but we cannot
          // query the edge directly. We pass the event's src/dst if provided.
          if (event.src) affected.push(event.src);
          if (event.dst) affected.push(event.dst);
        }
        this.graphStore.removeEdge(event.entityId);
        break;
      }
    }

    // Deduplicate
    return Array.from(new Set(affected));
  }

  /**
   * Expand a set of directly affected node IDs to include all nodes
   * within `recomputeRadius` hops. Uses iterative BFS.
   */
  private expandAffectedSet(directlyAffected: string[]): string[] {
    const visited = new Set<string>(directlyAffected);
    let frontier = new Set<string>(directlyAffected);

    for (let hop = 0; hop < this.recomputeRadius; hop++) {
      const nextFrontier = new Set<string>();
      for (const nodeId of frontier) {
        const neighbors = this.graphStore.getNeighborIds(nodeId);
        for (const neighbor of neighbors) {
          if (!visited.has(neighbor)) {
            visited.add(neighbor);
            nextFrontier.add(neighbor);
          }
        }
      }
      frontier = nextFrontier;
      if (frontier.size === 0) break;
    }

    return Array.from(visited);
  }
}
