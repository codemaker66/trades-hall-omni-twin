/**
 * @omni-twin/tda-wasm — Browser-side TDA via WASM + pure TypeScript (TDA-7).
 *
 * Provides persistent homology computation in the browser for small
 * interactive datasets (< 200 furniture points).
 *
 * Performance expectations:
 *   100 points, H₀:  < 50ms  (pure TS)
 *   100 points, H₁:  < 1 sec (WASM, when available)
 *   200 points, H₁:  1-5 sec (WASM, when available)
 *   500 points, H₁:  5-30 sec (show loading indicator)
 *   H₂:              limit to ≤100 points in browser
 *
 * When to use browser vs server:
 *   Browser: layout analysis of current floor plan (< 200 points),
 *            quick topology previews, educational/demo
 *   Server:  full venue database analysis, time series, Mapper,
 *            simplicial complexes, > 500 points or H₂
 */

// ─── Types ────────────────────────────────────────────────────────────────

export type {
  PersistenceResult,
  BrowserPersistenceConfig,
  DimensionStats,
  LayoutPoint,
  DeadSpaceResult,
  WorkerRequest,
  WorkerResponse,
} from './types'

// ─── Core Persistence (pure TS) ───────────────────────────────────────────

export {
  computeH0Persistence,
  buildDistanceMatrix2D,
  computeStats,
} from './persistence'

// ─── Layout Analysis ──────────────────────────────────────────────────────

export {
  analyzeLayoutBrowser,
  compareLayoutsBrowser,
} from './layout-analysis'

// ─── High-Level API ───────────────────────────────────────────────────────

import type { PersistenceResult, BrowserPersistenceConfig } from './types'
import { computeH0Persistence, computeStats } from './persistence'

/**
 * Compute persistence in the browser.
 *
 * Uses a Web Worker for non-blocking computation when configured.
 * Falls back to main-thread pure TS for H₀-only computations.
 *
 * @param distanceMatrix - N×N distance matrix (row-major Float64Array)
 * @param config - Configuration options
 */
export async function computePersistenceBrowser(
  distanceMatrix: Float64Array,
  config: BrowserPersistenceConfig = {},
): Promise<PersistenceResult> {
  const {
    maxDim = 1,
    threshold,
    useWorker = true,
    onProgress,
  } = config

  const n = Math.round(Math.sqrt(distanceMatrix.length))

  if (n * n !== distanceMatrix.length) {
    throw new Error(
      `Distance matrix must be square. Got ${distanceMatrix.length} elements (not a perfect square).`,
    )
  }

  onProgress?.(0)

  // For H₀-only, compute on main thread (fast enough)
  if (maxDim === 0 || !useWorker) {
    const start = performance.now()
    const h0 = computeH0Persistence(distanceMatrix, n, threshold)
    const computeTimeMs = performance.now() - start

    onProgress?.(1)

    return {
      diagrams: { H0: h0 },
      computeTimeMs,
      numPoints: n,
      maxDim: 0,
    }
  }

  // Use Web Worker for H₁+
  return new Promise((resolve, reject) => {
    try {
      const worker = new Worker(new URL('./worker.ts', import.meta.url), {
        type: 'module',
      })

      worker.postMessage({
        type: 'compute',
        distanceMatrix,
        maxDim,
        threshold,
      })

      worker.onmessage = (event) => {
        const response = event.data
        if (response.type === 'result') {
          onProgress?.(1)
          resolve(response.result)
          worker.terminate()
        } else if (response.type === 'progress') {
          onProgress?.(response.progress)
        } else if (response.type === 'error') {
          reject(new Error(response.error))
          worker.terminate()
        }
      }

      worker.onerror = (error) => {
        reject(error)
        worker.terminate()
      }
    } catch {
      // Worker creation failed (e.g., no Worker support)
      // Fall back to main-thread H₀ only
      const start = performance.now()
      const h0 = computeH0Persistence(distanceMatrix, n, threshold)
      const computeTimeMs = performance.now() - start

      onProgress?.(1)

      resolve({
        diagrams: { H0: h0, H1: [], ...(maxDim >= 2 ? { H2: [] } : {}) },
        computeTimeMs,
        numPoints: n,
        maxDim,
      })
    }
  })
}
