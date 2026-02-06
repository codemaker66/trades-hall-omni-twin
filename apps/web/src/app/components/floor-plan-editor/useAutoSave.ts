/**
 * Auto-save hook for floor plan editor.
 *
 * Debounces store changes to localStorage (500ms) and exposes save status.
 * On mount, loads any previously saved plan.
 */
import { useEffect, useRef, useState, useCallback } from 'react'
import { useFloorPlanStore, type FloorPlanItem } from './store'

export type SaveStatus = 'idle' | 'saving' | 'saved' | 'error'

const STORAGE_KEY = 'omnitwin-floorplan-autosave'
const DEBOUNCE_MS = 500

interface SavedData {
  items: FloorPlanItem[]
  planWidthFt: number
  planHeightFt: number
  gridSizeFt: number
  snapEnabled: boolean
  savedAt: number
}

export function useAutoSave() {
  const [status, setStatus] = useState<SaveStatus>('idle')
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const suppressRef = useRef(false)

  // Load on mount
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (!raw) return
      const data: SavedData = JSON.parse(raw)
      if (!data.items || !Array.isArray(data.items)) return

      suppressRef.current = true
      const state = useFloorPlanStore.getState()
      // Only load if current editor is empty (don't overwrite user work)
      if (state.items.length === 0 && data.items.length > 0) {
        useFloorPlanStore.setState({
          items: data.items,
          planWidthFt: data.planWidthFt ?? 80,
          planHeightFt: data.planHeightFt ?? 50,
          gridSizeFt: data.gridSizeFt ?? 1,
          snapEnabled: data.snapEnabled ?? true,
        })
      }
      suppressRef.current = false
    } catch {
      // Silently ignore corrupt data
    }
  }, [])

  // Subscribe to changes and debounce saves
  useEffect(() => {
    const unsubscribe = useFloorPlanStore.subscribe((state) => {
      if (suppressRef.current) return

      if (timeoutRef.current) clearTimeout(timeoutRef.current)
      setStatus('saving')

      timeoutRef.current = setTimeout(() => {
        try {
          const data: SavedData = {
            items: state.items,
            planWidthFt: state.planWidthFt,
            planHeightFt: state.planHeightFt,
            gridSizeFt: state.gridSizeFt,
            snapEnabled: state.snapEnabled,
            savedAt: Date.now(),
          }
          localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
          setStatus('saved')
        } catch {
          setStatus('error')
        }
      }, DEBOUNCE_MS)
    })

    return () => {
      unsubscribe()
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
    }
  }, [])

  const clearSaved = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY)
    setStatus('idle')
  }, [])

  return { status, clearSaved }
}
