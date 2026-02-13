/**
 * Server sync hook for the 2D floor plan editor.
 *
 * When venueId + planId are provided, loads the floor plan from the server on mount
 * and auto-saves store changes back to the server with debounce.
 * When no params are provided, this is a no-op (localStorage-only mode).
 */
'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import { useFloorPlanStore, type FloorPlanItem } from './store'
import { apiFetch } from '../../../lib/api-client'
import type { SaveStatus } from './useAutoSave'

const SERVER_DEBOUNCE_MS = 2000

interface FloorPlanResponse {
  floorPlan: {
    id: string
    venueId: string
    name: string
    widthFt: number
    heightFt: number
    objects: unknown[]
    isTemplate: boolean
  }
}

/** Validate and coerce a raw objects array into FloorPlanItem[]. */
function sanitizeObjects(raw: unknown[]): FloorPlanItem[] {
  const items: FloorPlanItem[] = []

  for (const obj of raw) {
    if (!obj || typeof obj !== 'object') continue
    const candidate = obj as Record<string, unknown>

    if (typeof candidate['id'] !== 'string') continue
    if (typeof candidate['name'] !== 'string') continue
    if (typeof candidate['x'] !== 'number') continue
    if (typeof candidate['y'] !== 'number') continue
    if (typeof candidate['widthFt'] !== 'number') continue
    if (typeof candidate['depthFt'] !== 'number') continue

    const category = candidate['category']
    if (
      category !== 'table' && category !== 'chair' &&
      category !== 'stage' && category !== 'decor' && category !== 'equipment'
    ) continue

    items.push({
      id: candidate['id'] as string,
      name: candidate['name'] as string,
      category,
      x: candidate['x'] as number,
      y: candidate['y'] as number,
      widthFt: candidate['widthFt'] as number,
      depthFt: candidate['depthFt'] as number,
      rotation: typeof candidate['rotation'] === 'number' ? (candidate['rotation'] as number) : 0,
      locked: typeof candidate['locked'] === 'boolean' ? (candidate['locked'] as boolean) : false,
    })
  }

  return items
}

export function useServerSync(venueId?: string, planId?: string) {
  const [serverStatus, setServerStatus] = useState<SaveStatus>('idle')
  const [loadedFromServer, setLoadedFromServer] = useState(false)
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const suppressRef = useRef(false)
  const abortRef = useRef<AbortController | null>(null)

  const isEnabled = Boolean(venueId && planId)

  // ── Load from server on mount ────────────────────────────────────────────
  useEffect(() => {
    if (!venueId || !planId) return

    let cancelled = false
    const controller = new AbortController()

    async function load() {
      const res = await apiFetch<FloorPlanResponse>(
        `/venues/${venueId}/floor-plans/${planId}`,
        { signal: controller.signal },
      )

      if (cancelled) return

      if (res.ok && res.data) {
        const plan = res.data.floorPlan
        const items = Array.isArray(plan.objects) ? sanitizeObjects(plan.objects) : []

        suppressRef.current = true
        useFloorPlanStore.setState({
          items,
          planWidthFt: plan.widthFt,
          planHeightFt: plan.heightFt,
          selectedIds: [],
        })
        suppressRef.current = false
        setLoadedFromServer(true)
        setServerStatus('saved')
      } else {
        setServerStatus('error')
      }
    }

    load()

    return () => {
      cancelled = true
      controller.abort()
    }
  }, [venueId, planId])

  // ── Auto-save to server on store changes ─────────────────────────────────
  const saveToServer = useCallback(async () => {
    if (!venueId || !planId) return

    // Cancel any in-flight request
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setServerStatus('saving')

    const state = useFloorPlanStore.getState()
    const res = await apiFetch(`/venues/${venueId}/floor-plans/${planId}`, {
      method: 'PATCH',
      body: JSON.stringify({
        objects: state.items,
        widthFt: state.planWidthFt,
        heightFt: state.planHeightFt,
      }),
      signal: controller.signal,
    })

    if (controller.signal.aborted) return

    setServerStatus(res.ok ? 'saved' : 'error')
  }, [venueId, planId])

  useEffect(() => {
    if (!isEnabled) return

    const unsubscribe = useFloorPlanStore.subscribe(() => {
      if (suppressRef.current) return

      if (timeoutRef.current) clearTimeout(timeoutRef.current)
      setServerStatus('saving')

      timeoutRef.current = setTimeout(() => {
        saveToServer()
      }, SERVER_DEBOUNCE_MS)
    })

    return () => {
      unsubscribe()
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
      abortRef.current?.abort()
    }
  }, [isEnabled, saveToServer])

  return { serverStatus: isEnabled ? serverStatus : ('idle' as SaveStatus), loadedFromServer }
}
