// ---------------------------------------------------------------------------
// HPC-3: Workers — Lock-free SPSC Ring Buffer
// ---------------------------------------------------------------------------
// Single-producer single-consumer ring buffer using modular arithmetic.
// Backed by Float64Array for numeric payloads (inter-worker data exchange).
//
// Full condition:  (tail + 1) % capacity === head
// Empty condition: tail === head
//
// One slot is always sacrificed to distinguish full from empty, so a buffer
// with capacity N can hold at most N-1 elements.
// ---------------------------------------------------------------------------

/**
 * Mutable state for the ring buffer.
 * `head` is the read pointer; `tail` is the write pointer.
 */
export type RingBufState = {
  buffer: Float64Array;
  head: number;
  tail: number;
  capacity: number;
};

/**
 * Create a new ring buffer with the given capacity.
 * The usable capacity is `capacity - 1` due to the full/empty disambiguation slot.
 */
export function createRingBuffer(capacity: number): RingBufState {
  if (capacity < 2) {
    throw new RangeError('Ring buffer capacity must be at least 2');
  }
  return {
    buffer: new Float64Array(capacity),
    head: 0,
    tail: 0,
    capacity,
  };
}

/**
 * Push a value onto the ring buffer.
 * Returns false if the buffer is full.
 */
export function ringBufferPush(rb: RingBufState, value: number): boolean {
  const nextTail = (rb.tail + 1) % rb.capacity;
  if (nextTail === rb.head) {
    return false; // full
  }
  rb.buffer[rb.tail] = value;
  rb.tail = nextTail;
  return true;
}

/**
 * Pop a value from the ring buffer.
 * Returns null if the buffer is empty.
 */
export function ringBufferPop(rb: RingBufState): number | null {
  if (rb.head === rb.tail) {
    return null; // empty
  }
  const value = rb.buffer[rb.head]!;
  rb.head = (rb.head + 1) % rb.capacity;
  return value;
}

/**
 * Peek at the front element without consuming it.
 * Returns null if the buffer is empty.
 */
export function ringBufferPeek(rb: RingBufState): number | null {
  if (rb.head === rb.tail) {
    return null;
  }
  return rb.buffer[rb.head]!;
}

/**
 * Returns the current number of elements in the buffer.
 */
export function ringBufferSize(rb: RingBufState): number {
  if (rb.tail >= rb.head) {
    return rb.tail - rb.head;
  }
  return rb.capacity - rb.head + rb.tail;
}

/**
 * Returns true if the buffer is full (no space for another push).
 */
export function ringBufferIsFull(rb: RingBufState): boolean {
  return (rb.tail + 1) % rb.capacity === rb.head;
}

/**
 * Returns true if the buffer is empty.
 */
export function ringBufferIsEmpty(rb: RingBufState): boolean {
  return rb.head === rb.tail;
}

/**
 * Clear the buffer by resetting head and tail pointers.
 * Does not zero the underlying Float64Array.
 */
export function ringBufferClear(rb: RingBufState): void {
  rb.head = 0;
  rb.tail = 0;
}

/**
 * Snapshot the current buffer contents into a new Float64Array.
 * Elements are returned in FIFO order (head -> tail).
 */
export function ringBufferToArray(rb: RingBufState): Float64Array {
  const size = ringBufferSize(rb);
  const result = new Float64Array(size);
  for (let i = 0; i < size; i++) {
    const idx = (rb.head + i) % rb.capacity;
    result[i] = rb.buffer[idx]!;
  }
  return result;
}
