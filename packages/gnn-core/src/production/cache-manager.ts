// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-12 Production Architecture: Cache Manager
// LRU embedding cache with TTL expiration and statistics tracking.
// ---------------------------------------------------------------------------

import type { CacheConfig } from '../types.js';

// ---- Types ----

interface CacheEntry {
  embedding: Float64Array;
  accessTime: number;
  insertTime: number;
}

interface CacheStats {
  hits: number;
  misses: number;
  evictions: number;
  size: number;
}

// ---------------------------------------------------------------------------
// EmbeddingCache — LRU cache with TTL for GNN embedding vectors
// ---------------------------------------------------------------------------

/**
 * LRU embedding cache with time-to-live expiration.
 *
 * Stores Float64Array embeddings keyed by string identifiers. When the cache
 * exceeds `maxSize`, the least-recently-used entry is evicted. Entries older
 * than `ttlMs` are treated as expired and removed on access.
 *
 * Statistics (hits, misses, evictions) are tracked for monitoring.
 */
export class EmbeddingCache {
  private cache: Map<string, CacheEntry> = new Map();
  private accessOrder: string[] = [];

  private readonly maxSize: number;
  private readonly ttlMs: number;
  private readonly embeddingDim: number;

  // Stats counters
  private hitCount = 0;
  private missCount = 0;
  private evictionCount = 0;

  constructor(config: CacheConfig) {
    this.maxSize = config.maxSize;
    this.ttlMs = config.ttlMs;
    this.embeddingDim = config.embeddingDim;
  }

  // ---- Single-entry operations ----

  /**
   * Retrieve an embedding by key.
   *
   * Returns `undefined` if the key is not found or the entry has expired.
   * A successful get promotes the key to most-recently-used.
   *
   * @param key - Cache key.
   * @param now - Current timestamp in milliseconds (defaults to Date.now()).
   */
  get(key: string, now?: number): Float64Array | undefined {
    const ts = now ?? Date.now();
    const entry = this.cache.get(key);

    if (!entry) {
      this.missCount++;
      return undefined;
    }

    // Check TTL expiration
    if (ts - entry.insertTime > this.ttlMs) {
      this.invalidate(key);
      this.missCount++;
      return undefined;
    }

    // Update access time and promote in LRU order
    entry.accessTime = ts;
    this.promoteKey(key);
    this.hitCount++;
    return entry.embedding;
  }

  /**
   * Store an embedding in the cache.
   *
   * If the cache is at capacity, the least-recently-used entry is evicted
   * before insertion.
   *
   * @param key - Cache key.
   * @param embedding - The embedding vector to store.
   * @param now - Current timestamp in milliseconds (defaults to Date.now()).
   */
  set(key: string, embedding: Float64Array, now?: number): void {
    const ts = now ?? Date.now();

    // If key already exists, update in place
    if (this.cache.has(key)) {
      const entry = this.cache.get(key)!;
      entry.embedding = embedding;
      entry.accessTime = ts;
      entry.insertTime = ts;
      this.promoteKey(key);
      return;
    }

    // Evict LRU if at capacity
    while (this.cache.size >= this.maxSize && this.accessOrder.length > 0) {
      this.evictLRU();
    }

    // Insert new entry
    this.cache.set(key, { embedding, accessTime: ts, insertTime: ts });
    this.accessOrder.push(key);
  }

  // ---- Batch operations ----

  /**
   * Retrieve embeddings for multiple keys at once.
   *
   * @param keys - Array of cache keys.
   * @param now - Current timestamp in milliseconds (defaults to Date.now()).
   * @returns Map from key to embedding for all found (non-expired) entries.
   */
  batchGet(keys: string[], now?: number): Map<string, Float64Array> {
    const result = new Map<string, Float64Array>();
    const ts = now ?? Date.now();
    for (const key of keys) {
      const emb = this.get(key, ts);
      if (emb) {
        result.set(key, emb);
      }
    }
    return result;
  }

  /**
   * Store multiple embeddings at once.
   *
   * @param entries - Map from key to embedding vector.
   * @param now - Current timestamp in milliseconds (defaults to Date.now()).
   */
  batchSet(entries: Map<string, Float64Array>, now?: number): void {
    const ts = now ?? Date.now();
    for (const [key, embedding] of entries) {
      this.set(key, embedding, ts);
    }
  }

  // ---- Invalidation ----

  /**
   * Remove a specific entry from the cache.
   */
  invalidate(key: string): void {
    if (this.cache.has(key)) {
      this.cache.delete(key);
      this.removeFromAccessOrder(key);
    }
  }

  /**
   * Remove all entries whose key starts with the given prefix.
   */
  invalidateByPrefix(prefix: string): void {
    const keysToRemove: string[] = [];
    for (const key of this.cache.keys()) {
      if (key.startsWith(prefix)) {
        keysToRemove.push(key);
      }
    }
    for (const key of keysToRemove) {
      this.invalidate(key);
    }
  }

  // ---- Info ----

  /** Current number of entries in the cache. */
  size(): number {
    return this.cache.size;
  }

  /** Remove all entries and reset statistics. */
  clear(): void {
    this.cache.clear();
    this.accessOrder = [];
    this.hitCount = 0;
    this.missCount = 0;
    this.evictionCount = 0;
  }

  /** Return cache performance statistics. */
  stats(): CacheStats {
    return {
      hits: this.hitCount,
      misses: this.missCount,
      evictions: this.evictionCount,
      size: this.cache.size,
    };
  }

  // ---- Private helpers ----

  /**
   * Evict the least-recently-used entry (front of accessOrder).
   */
  private evictLRU(): void {
    // Walk from the front of accessOrder to find a key still in the cache
    while (this.accessOrder.length > 0) {
      const lruKey = this.accessOrder.shift()!;
      if (this.cache.has(lruKey)) {
        this.cache.delete(lruKey);
        this.evictionCount++;
        return;
      }
      // Key was already removed — skip and try the next one
    }
  }

  /**
   * Promote a key to most-recently-used by moving it to the end
   * of the access order list.
   */
  private promoteKey(key: string): void {
    this.removeFromAccessOrder(key);
    this.accessOrder.push(key);
  }

  /**
   * Remove a key from the access order array.
   */
  private removeFromAccessOrder(key: string): void {
    const idx = this.accessOrder.indexOf(key);
    if (idx !== -1) {
      this.accessOrder.splice(idx, 1);
    }
  }
}
