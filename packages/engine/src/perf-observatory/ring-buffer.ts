/**
 * Generic ring buffer (circular buffer) for fixed-size rolling windows.
 *
 * Supports O(1) push, O(1) indexed access, and O(N) iteration.
 * When full, oldest samples are overwritten.
 */

export class RingBuffer<T> {
  private readonly buffer: (T | undefined)[]
  private head = 0
  private _size = 0

  constructor(readonly capacity: number) {
    this.buffer = new Array<T | undefined>(capacity)
  }

  /** Push a new item. Overwrites oldest if at capacity. */
  push(item: T): void {
    this.buffer[this.head] = item
    this.head = (this.head + 1) % this.capacity
    if (this._size < this.capacity) this._size++
  }

  /** Number of items currently stored. */
  get size(): number {
    return this._size
  }

  /** Whether the buffer is full. */
  get full(): boolean {
    return this._size === this.capacity
  }

  /** Get the item at index (0 = oldest). */
  at(index: number): T | undefined {
    if (index < 0 || index >= this._size) return undefined
    const actualIndex = (this.head - this._size + index + this.capacity) % this.capacity
    return this.buffer[actualIndex]
  }

  /** Get the most recent item. */
  latest(): T | undefined {
    if (this._size === 0) return undefined
    return this.buffer[(this.head - 1 + this.capacity) % this.capacity]
  }

  /** Iterate all items from oldest to newest. */
  *[Symbol.iterator](): Iterator<T> {
    for (let i = 0; i < this._size; i++) {
      const idx = (this.head - this._size + i + this.capacity) % this.capacity
      yield this.buffer[idx] as T
    }
  }

  /** Convert to array (oldest to newest). */
  toArray(): T[] {
    return Array.from(this)
  }

  /** Clear all items. */
  clear(): void {
    this.head = 0
    this._size = 0
    this.buffer.fill(undefined)
  }
}
