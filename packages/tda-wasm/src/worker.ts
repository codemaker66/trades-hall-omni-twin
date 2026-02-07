/**
 * Web Worker for non-blocking TDA computation (TDA-7).
 *
 * Runs Ripser WASM (when available) or falls back to the pure TS
 * H₀ implementation in a separate thread.
 */

import type { WorkerRequest, WorkerResponse } from './types'
import { computeH0Persistence } from './persistence'

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const { distanceMatrix, maxDim, threshold } = event.data

  try {
    const startTime = performance.now()
    const n = Math.round(Math.sqrt(distanceMatrix.length))

    const diagrams: Record<string, [number, number][]> = {}

    // H₀ via pure TypeScript
    diagrams['H0'] = computeH0Persistence(distanceMatrix, n, threshold)

    // H₁ and H₂ require WASM Ripser
    // If WASM module is loaded, use it; otherwise, return empty
    if (maxDim >= 1) {
      // TODO: When WASM Ripser is compiled, call it here:
      // const wasmResult = ripserWasm(distanceMatrix, n, maxDim, threshold)
      // diagrams['H1'] = wasmResult.dgms[1]
      // if (maxDim >= 2) diagrams['H2'] = wasmResult.dgms[2]
      diagrams['H1'] = []
      if (maxDim >= 2) diagrams['H2'] = []
    }

    const computeTimeMs = performance.now() - startTime

    const response: WorkerResponse = {
      type: 'result',
      result: {
        diagrams,
        computeTimeMs,
        numPoints: n,
        maxDim,
      },
    }

    self.postMessage(response)
  } catch (error) {
    const response: WorkerResponse = {
      type: 'error',
      error: error instanceof Error ? error.message : String(error),
    }
    self.postMessage(response)
  }
}
