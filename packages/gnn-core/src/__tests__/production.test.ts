// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-12: Production Architecture Tests
// Tests for GraphStore, EmbeddingCache, EventProcessor, and ServingPipeline.
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';

// Production modules under test
import { GraphStore } from '../production/graph-store.js';
import { EmbeddingCache } from '../production/cache-manager.js';
import { EventProcessor, type EventUpdate } from '../production/event-processor.js';
import { ServingPipeline, type ServingPipelineConfig } from '../production/serving-pipeline.js';

// ---------------------------------------------------------------------------
// GraphStore Tests
// ---------------------------------------------------------------------------

describe('GraphStore', () => {
  it('addNode/getNode roundtrip', () => {
    const store = new GraphStore();
    const features = new Float64Array([1.0, 2.0, 3.0]);

    store.addNode('venue-1', 'venue', features);

    const node = store.getNode('venue-1');
    expect(node).toBeDefined();
    expect(node!.type).toBe('venue');
    expect(node!.features.length).toBe(3);
    expect(node!.features[0]).toBe(1.0);
    expect(node!.features[1]).toBe(2.0);
    expect(node!.features[2]).toBe(3.0);
  });

  it('removeNode also removes incident edges', () => {
    const store = new GraphStore();
    store.addNode('A', 'room', new Float64Array([1, 0]));
    store.addNode('B', 'room', new Float64Array([0, 1]));
    store.addNode('C', 'room', new Float64Array([1, 1]));

    store.addEdge('e1', 'A', 'B', 'adjacent');
    store.addEdge('e2', 'B', 'C', 'adjacent');

    expect(store.nodeCount()).toBe(3);
    expect(store.edgeCount()).toBe(2);

    // Remove B — should also remove edges e1 and e2
    store.removeNode('B');

    expect(store.nodeCount()).toBe(2);
    expect(store.getNode('B')).toBeUndefined();
    expect(store.edgeCount()).toBe(0);

    // A and C should no longer have B as a neighbor
    expect(store.getNeighborIds('A')).toEqual([]);
    expect(store.getNeighborIds('C')).toEqual([]);
  });

  it('getKHopSubgraph returns correct neighborhood', () => {
    const store = new GraphStore();

    // Build a chain: A -- B -- C -- D -- E
    store.addNode('A', 't', new Float64Array([1]));
    store.addNode('B', 't', new Float64Array([2]));
    store.addNode('C', 't', new Float64Array([3]));
    store.addNode('D', 't', new Float64Array([4]));
    store.addNode('E', 't', new Float64Array([5]));

    store.addEdge('e1', 'A', 'B', 'link');
    store.addEdge('e2', 'B', 'C', 'link');
    store.addEdge('e3', 'C', 'D', 'link');
    store.addEdge('e4', 'D', 'E', 'link');

    // 1-hop from C should include B, C, D
    const hop1 = store.getKHopSubgraph('C', 1);
    expect(hop1.nodeIds.sort()).toEqual(['B', 'C', 'D']);

    // 2-hop from C should include A, B, C, D, E
    const hop2 = store.getKHopSubgraph('C', 2);
    expect(hop2.nodeIds.sort()).toEqual(['A', 'B', 'C', 'D', 'E']);
    // All 4 edges should be included since all endpoints are in the visited set
    expect(hop2.edgeIds.sort()).toEqual(['e1', 'e2', 'e3', 'e4']);
  });

  it('serialize/deserialize roundtrip', () => {
    const store = new GraphStore();
    store.addNode('n1', 'venue', new Float64Array([1.5, 2.5]));
    store.addNode('n2', 'room', new Float64Array([3.5, 4.5]));
    store.addEdge('e1', 'n1', 'n2', 'contains');

    const json = store.serialize();
    const restored = GraphStore.deserialize(json);

    expect(restored.nodeCount()).toBe(2);
    expect(restored.edgeCount()).toBe(1);

    const n1 = restored.getNode('n1');
    expect(n1).toBeDefined();
    expect(n1!.type).toBe('venue');
    expect(n1!.features[0]).toBeCloseTo(1.5);
    expect(n1!.features[1]).toBeCloseTo(2.5);

    const n2 = restored.getNode('n2');
    expect(n2).toBeDefined();
    expect(n2!.type).toBe('room');

    // Adjacency should be restored
    expect(restored.getNeighborIds('n1')).toContain('n2');
    expect(restored.getNeighborIds('n2')).toContain('n1');
  });

  it('toGraph produces valid CSR graph', () => {
    const store = new GraphStore();
    store.addNode('x', 'type_a', new Float64Array([1, 2, 3]));
    store.addNode('y', 'type_a', new Float64Array([4, 5, 6]));
    store.addNode('z', 'type_a', new Float64Array([7, 8, 9]));

    store.addEdge('e1', 'x', 'y', 'link');
    store.addEdge('e2', 'y', 'z', 'link');

    const { graph, nodeIds } = store.toGraph();

    expect(graph.numNodes).toBe(3);
    expect(graph.numEdges).toBe(2);
    expect(graph.featureDim).toBe(3);
    expect(graph.nodeFeatures.length).toBe(9); // 3 nodes * 3 features
    expect(graph.rowPtr.length).toBe(4); // numNodes + 1
    expect(nodeIds.length).toBe(3);

    // Verify the CSR structure has valid pointers
    expect(graph.rowPtr[0]).toBe(0);
    expect(graph.rowPtr[graph.numNodes]).toBe(graph.numEdges);
  });
});

// ---------------------------------------------------------------------------
// EmbeddingCache Tests
// ---------------------------------------------------------------------------

describe('EmbeddingCache', () => {
  it('get/set basic functionality', () => {
    const cache = new EmbeddingCache({
      maxSize: 10,
      ttlMs: 60_000,
      embeddingDim: 3,
    });

    const embedding = new Float64Array([1.0, 2.0, 3.0]);
    const now = 1000;

    cache.set('node-1', embedding, now);

    const retrieved = cache.get('node-1', now + 100);
    expect(retrieved).toBeDefined();
    expect(retrieved!.length).toBe(3);
    expect(retrieved![0]).toBe(1.0);
    expect(retrieved![1]).toBe(2.0);
    expect(retrieved![2]).toBe(3.0);

    // Non-existent key returns undefined
    const missing = cache.get('node-999', now);
    expect(missing).toBeUndefined();
  });

  it('LRU eviction when over maxSize', () => {
    const cache = new EmbeddingCache({
      maxSize: 3,
      ttlMs: 60_000,
      embeddingDim: 2,
    });

    const now = 1000;
    cache.set('a', new Float64Array([1, 1]), now);
    cache.set('b', new Float64Array([2, 2]), now + 1);
    cache.set('c', new Float64Array([3, 3]), now + 2);

    // Cache is full at 3 entries
    expect(cache.size()).toBe(3);

    // Adding a 4th entry should evict the LRU entry ('a')
    cache.set('d', new Float64Array([4, 4]), now + 3);
    expect(cache.size()).toBe(3);

    // 'a' should have been evicted
    expect(cache.get('a', now + 4)).toBeUndefined();
    // 'b', 'c', 'd' should still be present
    expect(cache.get('b', now + 4)).toBeDefined();
    expect(cache.get('c', now + 4)).toBeDefined();
    expect(cache.get('d', now + 4)).toBeDefined();
  });

  it('TTL expiration', () => {
    const ttlMs = 5000;
    const cache = new EmbeddingCache({
      maxSize: 10,
      ttlMs,
      embeddingDim: 2,
    });

    const now = 1000;
    cache.set('x', new Float64Array([1, 2]), now);

    // Before expiration: should be found
    expect(cache.get('x', now + ttlMs - 1)).toBeDefined();

    // After expiration: should be gone
    expect(cache.get('x', now + ttlMs + 1)).toBeUndefined();
  });

  it('stats tracking (hits/misses/evictions)', () => {
    const cache = new EmbeddingCache({
      maxSize: 2,
      ttlMs: 60_000,
      embeddingDim: 2,
    });

    const now = 1000;

    // Miss: key not found
    cache.get('missing', now);

    // Set and hit
    cache.set('a', new Float64Array([1, 1]), now);
    cache.get('a', now + 1);  // hit

    // Set two more to trigger eviction
    cache.set('b', new Float64Array([2, 2]), now + 2);
    cache.set('c', new Float64Array([3, 3]), now + 3); // evicts 'a'

    const stats = cache.stats();
    expect(stats.hits).toBe(1);
    expect(stats.misses).toBe(1);
    expect(stats.evictions).toBe(1);
    expect(stats.size).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// EventProcessor Tests
// ---------------------------------------------------------------------------

describe('EventProcessor', () => {
  it('processEvent adds node correctly', () => {
    const store = new GraphStore();
    const cache = new EmbeddingCache({
      maxSize: 100,
      ttlMs: 60_000,
      embeddingDim: 3,
    });
    const processor = new EventProcessor(store, cache, 1);

    const event: EventUpdate = {
      type: 'node_add',
      entityId: 'venue-1',
      nodeType: 'venue',
      features: new Float64Array([1.0, 2.0, 3.0]),
    };

    const result = processor.processEvent(event);

    // Node should have been added to the store
    const node = store.getNode('venue-1');
    expect(node).toBeDefined();
    expect(node!.type).toBe('venue');
    expect(node!.features[0]).toBe(1.0);

    // The node itself should be in the affected set
    expect(result.affectedNodes).toContain('venue-1');
    expect(result.recomputeNeeded).toBe(true);
  });

  it('invalidates cache on update', () => {
    const store = new GraphStore();
    const cache = new EmbeddingCache({
      maxSize: 100,
      ttlMs: 60_000,
      embeddingDim: 3,
    });
    const processor = new EventProcessor(store, cache, 1);

    // Set up a graph with two connected nodes
    store.addNode('A', 'room', new Float64Array([1, 0, 0]));
    store.addNode('B', 'room', new Float64Array([0, 1, 0]));
    store.addEdge('e1', 'A', 'B', 'adjacent');

    // Pre-populate cache for both nodes
    const now = 1000;
    cache.set('A', new Float64Array([0.1, 0.2, 0.3]), now);
    cache.set('B', new Float64Array([0.4, 0.5, 0.6]), now);

    expect(cache.get('A', now + 1)).toBeDefined();
    expect(cache.get('B', now + 1)).toBeDefined();

    // Update node A's features
    const updateEvent: EventUpdate = {
      type: 'node_update',
      entityId: 'A',
      features: new Float64Array([9, 9, 9]),
    };

    const result = processor.processEvent(updateEvent);

    // Both A and its neighbor B should be affected (recomputeRadius=1)
    expect(result.affectedNodes).toContain('A');
    expect(result.affectedNodes).toContain('B');

    // Cache entries for affected nodes should be invalidated
    expect(cache.get('A', now + 2)).toBeUndefined();
    expect(cache.get('B', now + 2)).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// ServingPipeline Tests
// ---------------------------------------------------------------------------

describe('ServingPipeline', () => {
  it('serve returns response with results', () => {
    const store = new GraphStore();
    const embeddingDim = 4;

    // Add some nodes
    store.addNode('n1', 'venue', new Float64Array([1, 2, 3, 4]));
    store.addNode('n2', 'venue', new Float64Array([5, 6, 7, 8]));
    store.addNode('n3', 'venue', new Float64Array([2, 3, 4, 5]));

    store.addEdge('e1', 'n1', 'n2', 'link');
    store.addEdge('e2', 'n2', 'n3', 'link');

    const cache = new EmbeddingCache({
      maxSize: 100,
      ttlMs: 60_000,
      embeddingDim,
    });

    const pipeline = new ServingPipeline({
      mode: 'realtime',
      graphStore: store,
      cache,
      embeddingDim,
    });

    const response = pipeline.serve({
      queryNodeId: 'n1',
      taskType: 'predict',
    });

    // Should return a valid response
    expect(response.results).toBeInstanceOf(Float64Array);
    expect(response.results.length).toBe(embeddingDim);
    expect(response.nodeIds).toContain('n1');
    expect(typeof response.latencyMs).toBe('number');
    expect(response.latencyMs).toBeGreaterThanOrEqual(0);
    expect(typeof response.cached).toBe('boolean');
  });

  it('healthCheck returns status', () => {
    const store = new GraphStore();
    const embeddingDim = 3;

    store.addNode('a', 'venue', new Float64Array([1, 2, 3]));

    const cache = new EmbeddingCache({
      maxSize: 100,
      ttlMs: 60_000,
      embeddingDim,
    });

    const pipeline = new ServingPipeline({
      mode: 'realtime',
      graphStore: store,
      cache,
      embeddingDim,
    });

    const health = pipeline.healthCheck();

    expect(health.status).toBe('ok');
    expect(health.graphStats.nodeCount).toBe(1);
    expect(health.graphStats.edgeCount).toBe(0);
    expect(typeof health.cacheStats.hits).toBe('number');
    expect(typeof health.cacheStats.misses).toBe('number');
    expect(typeof health.cacheStats.evictions).toBe('number');
    expect(typeof health.cacheStats.size).toBe('number');
  });
});
