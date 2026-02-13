/**
 * Hook for physics-based auto-layout optimization.
 *
 * Spawns a Web Worker to run simulated annealing on the current floor plan items,
 * then applies the optimized positions back to the store (with undo support via loadTemplate).
 */
'use client'

import { useRef, useState, useCallback } from 'react'
import { useFloorPlanStore } from './store'
import {
  editorToSolverItems,
  solverToEditorItems,
  createRoomBoundary,
  DEFAULT_WEIGHTS,
} from './solverAdapter'
import type { WorkerInput, WorkerOutput, WorkerError } from './autoLayoutWorker'

interface AutoLayoutResult {
  energy: number
  iterations: number
}

export function useAutoLayout() {
  const [isRunning, setIsRunning] = useState(false)
  const [result, setResult] = useState<AutoLayoutResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const workerRef = useRef<Worker | null>(null)

  const optimize = useCallback((targetCapacity: number) => {
    // Terminate any existing worker
    workerRef.current?.terminate()
    setIsRunning(true)
    setResult(null)
    setError(null)

    const state = useFloorPlanStore.getState()
    const { solverItems, participatingIndices } = editorToSolverItems(state.items)

    if (solverItems.length === 0) {
      setIsRunning(false)
      setError('No unlocked items to optimize. Unlock some items first.')
      return
    }

    const room = createRoomBoundary(state.planWidthFt, state.planHeightFt)

    const input: WorkerInput = {
      items: solverItems,
      room,
      weights: DEFAULT_WEIGHTS,
      targetCapacity,
      seed: Date.now(),
      maxIterations: 5000,
    }

    try {
      const worker = new Worker(
        new URL('./autoLayoutWorker.ts', import.meta.url),
      )
      workerRef.current = worker

      worker.onmessage = (e: MessageEvent<WorkerOutput | WorkerError>) => {
        const msg = e.data

        if (msg.type === 'error') {
          setError(msg.message)
          setIsRunning(false)
          worker.terminate()
          workerRef.current = null
          return
        }

        // Apply results back to the store
        const currentState = useFloorPlanStore.getState()
        const updatedItems = solverToEditorItems(
          msg.items,
          currentState.items,
          participatingIndices,
        )

        // Use loadTemplate to record undo history — strips id/locked, we re-map them
        const templateItems = updatedItems.map((item) => ({
          name: item.name,
          category: item.category,
          x: item.x,
          y: item.y,
          widthFt: item.widthFt,
          depthFt: item.depthFt,
          rotation: item.rotation,
        }))
        currentState.loadTemplate(templateItems)

        setResult({ energy: msg.bestEnergy, iterations: msg.iterations })
        setIsRunning(false)
        worker.terminate()
        workerRef.current = null
      }

      worker.onerror = (ev) => {
        setError(ev.message || 'Worker error')
        setIsRunning(false)
        worker.terminate()
        workerRef.current = null
      }

      worker.postMessage(input)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start auto-layout')
      setIsRunning(false)
    }
  }, [])

  const cancel = useCallback(() => {
    workerRef.current?.terminate()
    workerRef.current = null
    setIsRunning(false)
  }, [])

  return { optimize, cancel, isRunning, result, error }
}
