// ---------------------------------------------------------------------------
// HPC-7: CRDT — Observed-Remove Set (add-wins semantics)
// ---------------------------------------------------------------------------
// An OR-Set tracks elements via unique "add tags". Each add generates a
// globally unique tag; remove marks observed tags as tombstones.
// An element is present if it has at least one tag that is not tombstoned.
//
// Merge is commutative, associative, and idempotent:
//   tags = union of tags
//   tombstones = union of tombstones
//   element present iff (tags \ tombstones) is non-empty
//
// Because add generates a fresh tag unknown to concurrent removes, this
// gives add-wins semantics: concurrent add + remove keeps the element.
// ---------------------------------------------------------------------------

/**
 * Internal state of an OR-Set.
 * - `elements`: maps a serialized value key to `{ value, tags }`.
 * - `tombstones`: set of tags that have been removed.
 */
export type ORSetState<T> = {
  elements: Map<string, { value: T; tags: Set<string> }>;
  tombstones: Set<string>;
};

/** Global counter used to generate unique tags within a peer session. */
let globalTagCounter = 0;

/**
 * Serialize a value to a string key for map lookup.
 */
function serializeKey<T>(value: T): string {
  if (typeof value === 'string') return `s:${value}`;
  if (typeof value === 'number') return `n:${value}`;
  return `j:${JSON.stringify(value)}`;
}

/**
 * Create an empty OR-Set.
 */
export function createORSet<T>(): ORSetState<T> {
  return {
    elements: new Map(),
    tombstones: new Set(),
  };
}

/**
 * Add a value to the OR-Set.
 * Returns the unique tag generated for this add operation.
 */
export function orSetAdd<T>(
  set: ORSetState<T>,
  value: T,
  peerId: string,
): string {
  const tag = `${peerId}:${++globalTagCounter}`;
  const key = serializeKey(value);

  const existing = set.elements.get(key);
  if (existing) {
    existing.tags.add(tag);
  } else {
    const tags = new Set<string>();
    tags.add(tag);
    set.elements.set(key, { value, tags });
  }

  return tag;
}

/**
 * Remove a value from the OR-Set by tombstoning all its currently observed tags.
 * Concurrent adds (with tags unknown at remove time) will survive.
 */
export function orSetRemove<T>(set: ORSetState<T>, value: T): void {
  const key = serializeKey(value);
  const entry = set.elements.get(key);
  if (!entry) return;

  // Tombstone all currently observed tags
  for (const tag of entry.tags) {
    set.tombstones.add(tag);
  }
}

/**
 * Check whether a value is currently present in the OR-Set.
 * A value is present if it has at least one tag not in the tombstone set.
 */
export function orSetContains<T>(set: ORSetState<T>, value: T): boolean {
  const key = serializeKey(value);
  const entry = set.elements.get(key);
  if (!entry) return false;

  for (const tag of entry.tags) {
    if (!set.tombstones.has(tag)) {
      return true;
    }
  }
  return false;
}

/**
 * Return all elements currently present in the OR-Set.
 */
export function orSetElements<T>(set: ORSetState<T>): T[] {
  const result: T[] = [];
  for (const [, entry] of set.elements) {
    let alive = false;
    for (const tag of entry.tags) {
      if (!set.tombstones.has(tag)) {
        alive = true;
        break;
      }
    }
    if (alive) {
      result.push(entry.value);
    }
  }
  return result;
}

/**
 * Merge two OR-Sets. The result contains:
 *   - Union of all tags for each element
 *   - Union of all tombstones
 * An element is present if it has at least one non-tombstoned tag.
 */
export function orSetMerge<T>(
  a: ORSetState<T>,
  b: ORSetState<T>,
): ORSetState<T> {
  const merged: ORSetState<T> = {
    elements: new Map(),
    tombstones: new Set<string>(),
  };

  // Union of tombstones
  for (const t of a.tombstones) merged.tombstones.add(t);
  for (const t of b.tombstones) merged.tombstones.add(t);

  // Merge element entries from a
  for (const [key, entryA] of a.elements) {
    const tags = new Set<string>();
    for (const tag of entryA.tags) tags.add(tag);
    merged.elements.set(key, { value: entryA.value, tags });
  }

  // Merge element entries from b
  for (const [key, entryB] of b.elements) {
    const existing = merged.elements.get(key);
    if (existing) {
      for (const tag of entryB.tags) {
        existing.tags.add(tag);
      }
    } else {
      const tags = new Set<string>();
      for (const tag of entryB.tags) tags.add(tag);
      merged.elements.set(key, { value: entryB.value, tags });
    }
  }

  return merged;
}
