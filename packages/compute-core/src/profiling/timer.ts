// ---------------------------------------------------------------------------
// HPC-10: Profiling — High-resolution timing utilities
// ---------------------------------------------------------------------------
// Provides a simple named timer with lap support. Uses Date.now() as the
// time source so it works identically in browsers, workers, Node, and Deno
// without relying on performance.now().
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// TimerState
// ---------------------------------------------------------------------------

/** Mutable timer state carrying name, lap records, and running flag. */
export type TimerState = {
  name: string;
  startMs: number;
  laps: Array<{ name: string; ms: number }>;
  running: boolean;
};

// ---------------------------------------------------------------------------
// createTimer
// ---------------------------------------------------------------------------

/**
 * Create a new named timer in the stopped state.
 */
export function createTimer(name: string): TimerState {
  return {
    name,
    startMs: 0,
    laps: [],
    running: false,
  };
}

// ---------------------------------------------------------------------------
// timerStart
// ---------------------------------------------------------------------------

/**
 * Start (or restart) the timer. Clears any existing laps.
 */
export function timerStart(timer: TimerState): void {
  timer.startMs = Date.now();
  timer.laps = [];
  timer.running = true;
}

// ---------------------------------------------------------------------------
// timerStop
// ---------------------------------------------------------------------------

/**
 * Stop the timer and return the total elapsed time in milliseconds.
 * If the timer was not running, returns 0.
 */
export function timerStop(timer: TimerState): number {
  if (!timer.running) return 0;
  const elapsed = Date.now() - timer.startMs;
  timer.running = false;
  return elapsed;
}

// ---------------------------------------------------------------------------
// timerLap
// ---------------------------------------------------------------------------

/**
 * Record a named lap. Returns the elapsed time (in ms) from the timer's
 * start to now. The lap is appended to the internal laps array.
 *
 * If the timer is not running, returns 0 without recording.
 */
export function timerLap(timer: TimerState, lapName: string): number {
  if (!timer.running) return 0;
  const elapsed = Date.now() - timer.startMs;
  timer.laps.push({ name: lapName, ms: elapsed });
  return elapsed;
}

// ---------------------------------------------------------------------------
// timerReset
// ---------------------------------------------------------------------------

/**
 * Reset the timer to its initial state (stopped, no laps).
 */
export function timerReset(timer: TimerState): void {
  timer.startMs = 0;
  timer.laps = [];
  timer.running = false;
}

// ---------------------------------------------------------------------------
// timerGetLaps
// ---------------------------------------------------------------------------

/**
 * Return a copy of all recorded laps.
 */
export function timerGetLaps(
  timer: TimerState,
): Array<{ name: string; ms: number }> {
  return timer.laps.slice();
}
