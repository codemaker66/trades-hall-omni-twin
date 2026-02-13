// ---------------------------------------------------------------------------
// HPC-8: Scheduling — Binary Min-Heap Priority Queue
// ---------------------------------------------------------------------------
// A standard binary min-heap where lower priority numbers are dequeued first.
// Stability is maintained by insertion order: items with equal priority are
// returned in FIFO order using a monotonically increasing sequence number.
// ---------------------------------------------------------------------------

/**
 * A single entry in the heap, carrying the item, its priority, and an
 * insertion sequence number for stable ordering.
 */
type HeapEntry<T> = {
  item: T;
  priority: number;
  seq: number;
};

/**
 * Binary min-heap priority queue.
 * `heap` is the backing array; `nextSeq` provides stable ordering.
 */
export type PriorityQueue<T> = {
  heap: Array<{ item: T; priority: number }>;
  /** @internal monotonic counter for insertion-order tiebreaking */
  _nextSeq: number;
  /** @internal actual entries with sequence numbers */
  _entries: Array<HeapEntry<T>>;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function lessThan<T>(a: HeapEntry<T>, b: HeapEntry<T>): boolean {
  if (a.priority !== b.priority) return a.priority < b.priority;
  return a.seq < b.seq;
}

function siftUp<T>(entries: Array<HeapEntry<T>>, idx: number): void {
  while (idx > 0) {
    const parentIdx = (idx - 1) >> 1;
    const parent = entries[parentIdx]!;
    const current = entries[idx]!;
    if (lessThan(current, parent)) {
      entries[parentIdx] = current;
      entries[idx] = parent;
      idx = parentIdx;
    } else {
      break;
    }
  }
}

function siftDown<T>(entries: Array<HeapEntry<T>>, idx: number): void {
  const len = entries.length;
  while (true) {
    let smallest = idx;
    const left = 2 * idx + 1;
    const right = 2 * idx + 2;

    if (left < len && lessThan(entries[left]!, entries[smallest]!)) {
      smallest = left;
    }
    if (right < len && lessThan(entries[right]!, entries[smallest]!)) {
      smallest = right;
    }

    if (smallest !== idx) {
      const tmp = entries[idx]!;
      entries[idx] = entries[smallest]!;
      entries[smallest] = tmp;
      idx = smallest;
    } else {
      break;
    }
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Create an empty priority queue.
 */
export function createPriorityQueue<T>(): PriorityQueue<T> {
  return {
    heap: [],
    _nextSeq: 0,
    _entries: [],
  };
}

/**
 * Push an item onto the priority queue with the given priority.
 * Lower priority numbers are dequeued first.
 */
export function pqPush<T>(pq: PriorityQueue<T>, item: T, priority: number): void {
  const entry: HeapEntry<T> = { item, priority, seq: pq._nextSeq++ };
  pq._entries.push(entry);
  siftUp(pq._entries, pq._entries.length - 1);

  // Keep the public `heap` array in sync
  syncPublicHeap(pq);
}

/**
 * Remove and return the item with the highest priority (lowest number).
 * Returns null if the queue is empty.
 */
export function pqPop<T>(pq: PriorityQueue<T>): T | null {
  if (pq._entries.length === 0) return null;

  const top = pq._entries[0]!;
  const last = pq._entries.pop()!;

  if (pq._entries.length > 0) {
    pq._entries[0] = last;
    siftDown(pq._entries, 0);
  }

  syncPublicHeap(pq);
  return top.item;
}

/**
 * Peek at the highest-priority item without removing it.
 * Returns null if the queue is empty.
 */
export function pqPeek<T>(pq: PriorityQueue<T>): T | null {
  if (pq._entries.length === 0) return null;
  return pq._entries[0]!.item;
}

/**
 * Return the number of items in the queue.
 */
export function pqSize<T>(pq: PriorityQueue<T>): number {
  return pq._entries.length;
}

/**
 * Return true if the queue is empty.
 */
export function pqIsEmpty<T>(pq: PriorityQueue<T>): boolean {
  return pq._entries.length === 0;
}

/**
 * Update the priority of an existing item in the queue.
 * Uses reference equality to find the item.
 * Returns true if the item was found and updated; false otherwise.
 */
export function pqUpdatePriority<T>(
  pq: PriorityQueue<T>,
  item: T,
  newPriority: number,
): boolean {
  const idx = pq._entries.findIndex((e) => e.item === item);
  if (idx === -1) return false;

  const oldPriority = pq._entries[idx]!.priority;
  pq._entries[idx]!.priority = newPriority;

  if (newPriority < oldPriority) {
    siftUp(pq._entries, idx);
  } else if (newPriority > oldPriority) {
    siftDown(pq._entries, idx);
  }

  syncPublicHeap(pq);
  return true;
}

// ---------------------------------------------------------------------------
// Internal — keep the public heap array consistent
// ---------------------------------------------------------------------------

function syncPublicHeap<T>(pq: PriorityQueue<T>): void {
  pq.heap.length = pq._entries.length;
  for (let i = 0; i < pq._entries.length; i++) {
    const e = pq._entries[i]!;
    pq.heap[i] = { item: e.item, priority: e.priority };
  }
}
