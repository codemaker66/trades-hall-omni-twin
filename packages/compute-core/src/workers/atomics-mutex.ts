// ---------------------------------------------------------------------------
// HPC-3: Workers — Atomics-based Mutex
// ---------------------------------------------------------------------------
// Pure logic implementation of an Atomics-style mutex using Int32Array.
// Uses plain reads/writes that mirror the Atomics.compareExchange pattern
// so the logic is testable in any JS environment.
// ---------------------------------------------------------------------------

/**
 * Creates initial mutex state (unlocked).
 * Value 0 = unlocked, 1 = locked.
 */
export function createMutexState(): { locked: number } {
  return { locked: 0 };
}

/** Value stored when the mutex is unlocked. */
const UNLOCKED = 0;

/** Value stored when the mutex is locked. */
const LOCKED = 1;

/**
 * Non-blocking lock attempt using compareExchange logic.
 * Returns true if the lock was successfully acquired.
 *
 * Mirrors `Atomics.compareExchange(view, offset, UNLOCKED, LOCKED)`:
 * if current value equals UNLOCKED, atomically swap to LOCKED.
 */
export function tryLock(view: Int32Array, offset: number): boolean {
  const current = view[offset];
  if (current === UNLOCKED) {
    view[offset] = LOCKED;
    return true;
  }
  return false;
}

/**
 * Release the mutex by storing the UNLOCKED value.
 * Mirrors `Atomics.store(view, offset, UNLOCKED)`.
 */
export function unlock(view: Int32Array, offset: number): void {
  view[offset] = UNLOCKED;
}

/**
 * Check whether the mutex is currently locked.
 * Mirrors `Atomics.load(view, offset)`.
 */
export function isLocked(view: Int32Array, offset: number): boolean {
  return view[offset] === LOCKED;
}

/**
 * Spin-wait lock with a maximum number of retries.
 * Attempts to acquire the lock up to `maxSpins` times.
 * Returns true if the lock was acquired within the spin budget.
 *
 * In a real SharedArrayBuffer environment this would use
 * `Atomics.wait` / `Atomics.notify` for parking, but here
 * we spin on tryLock to keep the logic pure.
 */
export function spinLock(
  view: Int32Array,
  offset: number,
  maxSpins: number,
): boolean {
  for (let i = 0; i < maxSpins; i++) {
    if (tryLock(view, offset)) {
      return true;
    }
    // In a real implementation: Atomics.wait or a brief pause
  }
  return false;
}
