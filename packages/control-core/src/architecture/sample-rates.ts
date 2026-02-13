// ---------------------------------------------------------------------------
// OC-11  Real-Time Control Architecture -- Multi-Rate Scheduling
// ---------------------------------------------------------------------------

import type { SampleRateConfig } from '../types.js';

// ---------------------------------------------------------------------------
// Subsystem names (used as schedule keys)
// ---------------------------------------------------------------------------

const SUBSYSTEM_NAMES = ['pricing', 'staffing', 'marketing', 'crowdOps'] as const;

// ---------------------------------------------------------------------------
// getActiveSubsystems
// ---------------------------------------------------------------------------

/**
 * Return the list of subsystem names whose update period aligns with time `t`.
 *
 * A subsystem fires when `t` is an integer multiple of its configured period
 * (within a small floating-point tolerance of 1e-9 seconds).
 *
 * @param config  Per-subsystem sample periods in seconds.
 * @param t       Current simulation / wall-clock time in seconds.
 * @returns       Array of subsystem name strings that should execute at `t`.
 */
export function getActiveSubsystems(
  config: SampleRateConfig,
  t: number,
): string[] {
  const active: string[] = [];

  for (const name of SUBSYSTEM_NAMES) {
    const period = config[name]!;
    if (period <= 0) continue;

    const remainder = t % period;
    const tolerance = 1e-9;
    if (remainder < tolerance || period - remainder < tolerance) {
      active.push(name);
    }
  }

  return active;
}

// ---------------------------------------------------------------------------
// createMultiRateScheduler
// ---------------------------------------------------------------------------

/**
 * Create a stateful multi-rate scheduler that wraps {@link getActiveSubsystems}
 * and tracks the last scheduled time for each subsystem.
 *
 * The returned `schedule(t)` function deduplicates: if the same time `t` is
 * queried twice, previously-reported subsystems will not fire again until
 * their next period boundary.
 *
 * @param config  Per-subsystem sample periods in seconds.
 * @returns       Object with a `schedule` method.
 */
export function createMultiRateScheduler(config: SampleRateConfig): {
  schedule: (t: number) => string[];
} {
  // Track last fire time per subsystem to avoid double-firing.
  const lastFired = new Map<string, number>();

  return {
    schedule(t: number): string[] {
      const candidates = getActiveSubsystems(config, t);
      const fired: string[] = [];

      for (let i = 0; i < candidates.length; i++) {
        const name = candidates[i]!;
        const prev = lastFired.get(name);

        // Only fire if this is the first invocation or time has advanced
        // past the previous fire time.
        if (prev === undefined || t > prev + 1e-12) {
          fired.push(name);
          lastFired.set(name, t);
        }
      }

      return fired;
    },
  };
}
