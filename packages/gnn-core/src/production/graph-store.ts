// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-12 Production Architecture: Graph Store
// In-memory graph storage with adjacency tracking and CSR conversion.
// ---------------------------------------------------------------------------

import type { Graph } from '../types.js';
import { buildCSR } from '../graph.js';

// ---- Types ----

interface StoredNode {
  type: string;
  features: Float64Array;
}

interface StoredEdge {
  src: string;
  dst: string;
  type: string;
  features?: Float64Array;
}

interface SerializedNode {
  type: string;
  features: number[];
}

interface SerializedEdge {
  src: string;
  dst: string;
  type: string;
  features?: number[];
}

interface SerializedGraphStore {
  nodes: [string, SerializedNode][];
  edges: [string, SerializedEdge][];
}

// ---------------------------------------------------------------------------
// GraphStore — In-memory graph storage with adjacency index
// ---------------------------------------------------------------------------

/**
 * In-memory graph store supporting typed nodes and edges with feature vectors.
 *
 * Maintains an adjacency index (node -> neighbor set) for fast k-hop
 * neighborhood queries. Provides serialization and conversion to CSR
 * `Graph` format for GNN inference.
 */
export class GraphStore {
  private nodes: Map<string, StoredNode> = new Map();
  private edges: Map<string, StoredEdge> = new Map();
  private adjacency: Map<string, Set<string>> = new Map();

  // ---- Node operations ----

  /**
   * Add a node with the given ID, type, and feature vector.
   * If a node with this ID already exists it is overwritten.
   */
  addNode(id: string, type: string, features: Float64Array): void {
    this.nodes.set(id, { type, features });
    if (!this.adjacency.has(id)) {
      this.adjacency.set(id, new Set());
    }
  }

  /**
   * Remove a node and all of its incident edges.
   */
  removeNode(id: string): void {
    // Find and remove all incident edges
    const edgesToRemove: string[] = [];
    for (const [edgeId, edge] of this.edges) {
      if (edge.src === id || edge.dst === id) {
        edgesToRemove.push(edgeId);
      }
    }
    for (const edgeId of edgesToRemove) {
      this.removeEdge(edgeId);
    }

    // Remove node from adjacency lists of neighbors
    const neighbors = this.adjacency.get(id);
    if (neighbors) {
      for (const neighbor of neighbors) {
        const neighborSet = this.adjacency.get(neighbor);
        if (neighborSet) {
          neighborSet.delete(id);
        }
      }
    }

    this.adjacency.delete(id);
    this.nodes.delete(id);
  }

  /**
   * Replace the feature vector for an existing node.
   * Throws if the node does not exist.
   */
  updateNodeFeatures(id: string, features: Float64Array): void {
    const node = this.nodes.get(id);
    if (!node) {
      throw new Error(`GraphStore.updateNodeFeatures: node '${id}' not found`);
    }
    node.features = features;
  }

  // ---- Edge operations ----

  /**
   * Add a directed edge between two existing nodes.
   * Also updates the adjacency index in both directions (undirected view).
   */
  addEdge(
    id: string,
    src: string,
    dst: string,
    type: string,
    features?: Float64Array,
  ): void {
    this.edges.set(id, { src, dst, type, features });

    // Update adjacency (treat as undirected for neighbor queries)
    if (!this.adjacency.has(src)) {
      this.adjacency.set(src, new Set());
    }
    if (!this.adjacency.has(dst)) {
      this.adjacency.set(dst, new Set());
    }
    this.adjacency.get(src)!.add(dst);
    this.adjacency.get(dst)!.add(src);
  }

  /**
   * Remove an edge by its ID and update the adjacency index.
   */
  removeEdge(id: string): void {
    const edge = this.edges.get(id);
    if (!edge) return;

    this.edges.delete(id);

    // Check if there are any remaining edges between src and dst
    let stillConnected = false;
    for (const [, e] of this.edges) {
      if (
        (e.src === edge.src && e.dst === edge.dst) ||
        (e.src === edge.dst && e.dst === edge.src)
      ) {
        stillConnected = true;
        break;
      }
    }

    if (!stillConnected) {
      const srcSet = this.adjacency.get(edge.src);
      if (srcSet) srcSet.delete(edge.dst);
      const dstSet = this.adjacency.get(edge.dst);
      if (dstSet) dstSet.delete(edge.src);
    }
  }

  // ---- Query operations ----

  /**
   * Retrieve a node by ID, or undefined if it does not exist.
   */
  getNode(id: string): { type: string; features: Float64Array } | undefined {
    return this.nodes.get(id);
  }

  /**
   * Get the IDs of all neighbors of the given node.
   */
  getNeighborIds(nodeId: string): string[] {
    const neighbors = this.adjacency.get(nodeId);
    if (!neighbors) return [];
    return Array.from(neighbors);
  }

  /**
   * Extract the k-hop subgraph around a node.
   *
   * Algorithm — BFS from `nodeId` up to depth `k`, collecting all visited
   * node IDs and every edge whose both endpoints are in the visited set.
   */
  getKHopSubgraph(
    nodeId: string,
    k: number,
  ): { nodeIds: string[]; edgeIds: string[] } {
    const visited = new Set<string>();
    let frontier = new Set<string>();

    if (!this.nodes.has(nodeId)) {
      return { nodeIds: [], edgeIds: [] };
    }

    frontier.add(nodeId);
    visited.add(nodeId);

    for (let hop = 0; hop < k; hop++) {
      const nextFrontier = new Set<string>();
      for (const nid of frontier) {
        const neighbors = this.adjacency.get(nid);
        if (!neighbors) continue;
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

    // Collect edges with both endpoints in the visited set
    const edgeIds: string[] = [];
    for (const [edgeId, edge] of this.edges) {
      if (visited.has(edge.src) && visited.has(edge.dst)) {
        edgeIds.push(edgeId);
      }
    }

    return { nodeIds: Array.from(visited), edgeIds };
  }

  /**
   * Return all node IDs with the given type.
   */
  queryByType(nodeType: string): string[] {
    const result: string[] = [];
    for (const [id, node] of this.nodes) {
      if (node.type === nodeType) {
        result.push(id);
      }
    }
    return result;
  }

  /** Total number of nodes. */
  nodeCount(): number {
    return this.nodes.size;
  }

  /** Total number of edges. */
  edgeCount(): number {
    return this.edges.size;
  }

  // ---- Serialization ----

  /**
   * Serialize the entire graph store to a JSON string.
   * Float64Arrays are converted to plain number arrays for portability.
   */
  serialize(): string {
    const serializedNodes: [string, SerializedNode][] = [];
    for (const [id, node] of this.nodes) {
      serializedNodes.push([id, { type: node.type, features: Array.from(node.features) }]);
    }

    const serializedEdges: [string, SerializedEdge][] = [];
    for (const [id, edge] of this.edges) {
      const se: SerializedEdge = { src: edge.src, dst: edge.dst, type: edge.type };
      if (edge.features) {
        se.features = Array.from(edge.features);
      }
      serializedEdges.push([id, se]);
    }

    const data: SerializedGraphStore = {
      nodes: serializedNodes,
      edges: serializedEdges,
    };

    return JSON.stringify(data);
  }

  /**
   * Reconstruct a GraphStore from a JSON string produced by `serialize()`.
   */
  static deserialize(json: string): GraphStore {
    const data = JSON.parse(json) as SerializedGraphStore;
    const store = new GraphStore();

    for (const [id, sn] of data.nodes) {
      store.addNode(id, sn.type, new Float64Array(sn.features));
    }

    for (const [id, se] of data.edges) {
      const features = se.features ? new Float64Array(se.features) : undefined;
      store.addEdge(id, se.src, se.dst, se.type, features);
    }

    return store;
  }

  // ---- CSR conversion ----

  /**
   * Convert the graph store to a CSR `Graph` suitable for GNN inference.
   *
   * Algorithm:
   * 1. Assign each node a contiguous integer index (0..N-1).
   *    If `nodeIdOrder` is provided, use that order; otherwise use insertion order.
   * 2. Build an edge list from stored edges with remapped indices.
   * 3. Concatenate node features into a single row-major matrix.
   * 4. Delegate to `buildCSR` and attach features.
   *
   * @param nodeIdOrder - Optional array specifying the mapping from index to node ID.
   * @returns The CSR Graph and the node ID ordering used.
   */
  toGraph(nodeIdOrder?: string[]): { graph: Graph; nodeIds: string[] } {
    const nodeIds = nodeIdOrder ?? Array.from(this.nodes.keys());
    const numNodes = nodeIds.length;

    // Build ID -> index mapping
    const idToIndex = new Map<string, number>();
    for (let i = 0; i < numNodes; i++) {
      idToIndex.set(nodeIds[i]!, i);
    }

    // Determine feature dimension from the first node (or 0 if empty)
    let featureDim = 0;
    if (numNodes > 0) {
      const firstNode = this.nodes.get(nodeIds[0]!);
      if (firstNode) {
        featureDim = firstNode.features.length;
      }
    }

    // Build edge list
    const edges: [number, number][] = [];
    for (const [, edge] of this.edges) {
      const srcIdx = idToIndex.get(edge.src);
      const dstIdx = idToIndex.get(edge.dst);
      if (srcIdx !== undefined && dstIdx !== undefined) {
        edges.push([srcIdx, dstIdx]);
      }
    }

    // Build CSR
    const csrGraph = buildCSR(edges, numNodes);

    // Build node feature matrix
    const nodeFeatures = new Float64Array(numNodes * featureDim);
    for (let i = 0; i < numNodes; i++) {
      const node = this.nodes.get(nodeIds[i]!);
      if (node) {
        const offset = i * featureDim;
        for (let f = 0; f < featureDim; f++) {
          nodeFeatures[offset + f] = node.features[f]!;
        }
      }
    }

    const graph: Graph = {
      numNodes: csrGraph.numNodes,
      numEdges: csrGraph.numEdges,
      rowPtr: csrGraph.rowPtr,
      colIdx: csrGraph.colIdx,
      edgeWeights: csrGraph.edgeWeights,
      nodeFeatures,
      featureDim,
    };

    return { graph, nodeIds };
  }
}
