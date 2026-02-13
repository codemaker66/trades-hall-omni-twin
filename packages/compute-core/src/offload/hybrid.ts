// ---------------------------------------------------------------------------
// HPC-9: Offload — Hybrid execution patterns
// ---------------------------------------------------------------------------
// Strategies for splitting work between browser and server, running them in
// parallel, progressively refining results, and falling back through a chain
// of targets when the preferred one is unavailable.
// ---------------------------------------------------------------------------

import type { OffloadDecision } from '../types.js';

// ---------------------------------------------------------------------------
// splitWorkload
// ---------------------------------------------------------------------------

/**
 * Split a total number of elements between browser and server execution.
 *
 * The browser receives up to `browserCapacity` elements; the remainder is
 * offloaded. If the browser can handle the entire workload, the server
 * share is zero.
 */
export function splitWorkload(
  totalElements: number,
  browserCapacity: number,
): { browser: number; server: number } {
  if (totalElements <= 0) return { browser: 0, server: 0 };
  const cap = Math.max(0, browserCapacity);
  const browser = Math.min(totalElements, cap);
  const server = totalElements - browser;
  return { browser, server };
}

// ---------------------------------------------------------------------------
// parallelExecutionTimeMs
// ---------------------------------------------------------------------------

/**
 * Estimate the wall-clock time when browser and server execute in parallel.
 *
 * The browser starts immediately while the server must also pay for the
 * round-trip transfer. The total time is the maximum of both legs:
 *
 *   total = max(browserMs, serverMs + transferMs)
 */
export function parallelExecutionTimeMs(
  browserMs: number,
  serverMs: number,
  transferMs: number,
): number {
  return Math.max(browserMs, serverMs + transferMs);
}

// ---------------------------------------------------------------------------
// isWorthParallelizing
// ---------------------------------------------------------------------------

/**
 * Determine whether parallel browser+server execution is worthwhile.
 *
 * Parallel is beneficial only when the combined wall-clock time is strictly
 * less than **both** the browser-only time and the server-only time.
 */
export function isWorthParallelizing(
  browserMs: number,
  serverMs: number,
  transferMs: number,
): boolean {
  const parallel = parallelExecutionTimeMs(browserMs, serverMs, transferMs);
  const browserOnly = browserMs;
  const serverOnly = serverMs + transferMs;
  return parallel < browserOnly && parallel < serverOnly;
}

// ---------------------------------------------------------------------------
// progressiveRefinement
// ---------------------------------------------------------------------------

/**
 * Compute the strategy for progressive refinement:
 *   1. Browser computes a coarse result immediately (fast, local).
 *   2. Server computes a fine result (slow, remote).
 *   3. The fine result replaces the coarse one when it arrives.
 *
 * Returns whether to show the coarse result first and the total time until
 * the fine result is available.
 *
 * Show the coarse result first if the fine result (server + transfer) would
 * take longer than the coarse result alone, otherwise skip the coarse pass.
 */
export function progressiveRefinement(
  coarseMs: number,
  fineMs: number,
  transferMs: number,
): { showCoarseFirst: boolean; totalMs: number } {
  const fineTotalMs = fineMs + transferMs;
  const showCoarseFirst = fineTotalMs > coarseMs;

  // Total time is always dominated by the fine result arriving.
  // If we show coarse first, the total is the max of coarse and fine (they
  // can run in parallel).
  const totalMs = showCoarseFirst
    ? Math.max(coarseMs, fineTotalMs)
    : fineTotalMs;

  return { showCoarseFirst, totalMs };
}

// ---------------------------------------------------------------------------
// fallbackChain
// ---------------------------------------------------------------------------

/** Ordered set of browser-side targets for fallback resolution. */
const AVAILABLE_BROWSER_TARGETS: ReadonlySet<OffloadDecision['target']> =
  new Set<OffloadDecision['target']>([
    'browser-js',
    'browser-wasm',
    'browser-gpu',
  ]);

/**
 * Walk a prioritised list of targets and return the first one that is
 * considered "available".
 *
 * All browser targets are assumed available. Server-gpu and edge are assumed
 * available (the caller should pre-filter if connectivity is down).
 *
 * If no targets are provided or the list is empty, returns `'browser-js'`
 * as the universal fallback.
 */
export function fallbackChain(
  targets: OffloadDecision['target'][],
): OffloadDecision['target'] {
  for (let i = 0; i < targets.length; i++) {
    const t = targets[i]!;
    // All listed targets are considered reachable — return the first one.
    if (
      AVAILABLE_BROWSER_TARGETS.has(t) ||
      t === 'server-gpu' ||
      t === 'edge'
    ) {
      return t;
    }
  }
  // Universal fallback: plain JS in the main thread
  return 'browser-js';
}
