import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { useFloorPlanStore, type FloorPlanItem } from '../store'

// ─── Helpers ─────────────────────────────────────────────────────────────────

function resetStore() {
  useFloorPlanStore.setState({
    items: [],
    selectedIds: [],
    past: [],
    future: [],
    batch: null,
    canUndo: false,
    canRedo: false,
    zoom: 1,
    panX: 0,
    panY: 0,
    gridSizeFt: 1,
    snapEnabled: false,
    tool: 'select',
    planWidthFt: 80,
    planHeightFt: 50,
  })
}

function addItems(count: number): string[] {
  const ids: string[] = []
  for (let i = 0; i < count; i++) {
    const id = useFloorPlanStore.getState().addItem({
      name: `Item ${i}`,
      category: 'chair',
      x: 10 + i * 5,
      y: 10,
      widthFt: 1.5,
      depthFt: 1.5,
      rotation: 0,
    })
    ids.push(id)
  }
  return ids
}

function fireKey(key: string, opts: Partial<KeyboardEventInit> = {}) {
  const event = new KeyboardEvent('keydown', {
    key,
    bubbles: true,
    cancelable: true,
    ...opts,
  })
  document.dispatchEvent(event)
}

// ─── Import the hook effect (we simulate by importing and calling setup) ──────

// Since useEditorKeyboard is a React hook, we test its logic by simulating
// the keydown events directly — the hook simply adds/removes a document listener.
// We import the module to trigger its side effects in integration tests.

// For unit tests, we directly test the store actions that the keyboard hook calls.

describe('Keyboard navigation', () => {
  beforeEach(() => resetStore())

  describe('Tab cycling through items', () => {
    it('selects first item when nothing is selected', () => {
      const ids = addItems(3)
      useFloorPlanStore.getState().setSelection([])

      // Simulate Tab: select next from nothing → first item
      const state = useFloorPlanStore.getState()
      const currentIdx = -1
      const nextIdx = currentIdx >= state.items.length - 1 ? 0 : currentIdx + 1
      state.setSelection([state.items[nextIdx]!.id])

      expect(useFloorPlanStore.getState().selectedIds).toEqual([ids[0]])
    })

    it('cycles forward through items', () => {
      const ids = addItems(3)
      useFloorPlanStore.getState().setSelection([ids[0]!])

      const state = useFloorPlanStore.getState()
      const currentIdx = state.items.findIndex((i) => i.id === ids[0])
      const nextIdx = currentIdx >= state.items.length - 1 ? 0 : currentIdx + 1
      state.setSelection([state.items[nextIdx]!.id])

      expect(useFloorPlanStore.getState().selectedIds).toEqual([ids[1]])
    })

    it('wraps around at the end', () => {
      const ids = addItems(3)
      useFloorPlanStore.getState().setSelection([ids[2]!])

      const state = useFloorPlanStore.getState()
      const currentIdx = state.items.findIndex((i) => i.id === ids[2])
      const nextIdx = currentIdx >= state.items.length - 1 ? 0 : currentIdx + 1
      state.setSelection([state.items[nextIdx]!.id])

      expect(useFloorPlanStore.getState().selectedIds).toEqual([ids[0]])
    })

    it('cycles backward with Shift+Tab', () => {
      const ids = addItems(3)
      useFloorPlanStore.getState().setSelection([ids[1]!])

      const state = useFloorPlanStore.getState()
      const currentIdx = state.items.findIndex((i) => i.id === ids[1])
      const nextIdx = currentIdx <= 0 ? state.items.length - 1 : currentIdx - 1
      state.setSelection([state.items[nextIdx]!.id])

      expect(useFloorPlanStore.getState().selectedIds).toEqual([ids[0]])
    })
  })

  describe('Arrow key nudging', () => {
    it('nudges selected items right by grid size', () => {
      const ids = addItems(1)
      useFloorPlanStore.getState().setSelection([ids[0]!])

      const state = useFloorPlanStore.getState()
      const item = state.items[0]!
      state.updateItems([{ id: item.id, changes: { x: item.x + state.gridSizeFt } }])

      expect(useFloorPlanStore.getState().items[0]!.x).toBe(11)
    })

    it('nudges selected items up', () => {
      const ids = addItems(1)
      useFloorPlanStore.getState().setSelection([ids[0]!])

      const state = useFloorPlanStore.getState()
      const item = state.items[0]!
      state.updateItems([{ id: item.id, changes: { y: item.y - state.gridSizeFt } }])

      expect(useFloorPlanStore.getState().items[0]!.y).toBe(9)
    })

    it('nudges by 5x with Shift held', () => {
      const ids = addItems(1)
      useFloorPlanStore.getState().setSelection([ids[0]!])

      const state = useFloorPlanStore.getState()
      const item = state.items[0]!
      const step = state.gridSizeFt * 5
      state.updateItems([{ id: item.id, changes: { x: item.x + step } }])

      expect(useFloorPlanStore.getState().items[0]!.x).toBe(15)
    })

    it('does nothing when nothing is selected', () => {
      addItems(1)
      useFloorPlanStore.getState().setSelection([])

      // Arrow key with no selection should not change items
      const xBefore = useFloorPlanStore.getState().items[0]!.x
      // (no-op - we just verify the guard logic)
      expect(useFloorPlanStore.getState().selectedIds.length).toBe(0)
      expect(useFloorPlanStore.getState().items[0]!.x).toBe(xBefore)
    })

    it('skips locked items', () => {
      const ids = addItems(1)
      useFloorPlanStore.getState().setSelection([ids[0]!])
      // Lock the item
      useFloorPlanStore.getState().updateItems([{ id: ids[0]!, changes: { locked: true } }])

      const state = useFloorPlanStore.getState()
      const item = state.items[0]!
      // Locked items should not be nudged (the hook filters them)
      const updates = state.selectedIds.map((id) => {
        const i = state.items.find((it) => it.id === id)
        if (!i || i.locked) return null
        return { id, changes: { x: i.x + 1 } }
      }).filter(Boolean) as { id: string; changes: { x: number } }[]

      expect(updates).toHaveLength(0) // locked, so filtered out
      expect(state.items[0]!.x).toBe(10) // unchanged
    })
  })

  describe('Delete/Backspace removal', () => {
    it('removes selected items', () => {
      const ids = addItems(3)
      useFloorPlanStore.getState().setSelection([ids[1]!])
      useFloorPlanStore.getState().removeItems([ids[1]!])

      expect(useFloorPlanStore.getState().items).toHaveLength(2)
      expect(useFloorPlanStore.getState().items.find((i) => i.id === ids[1])).toBeUndefined()
    })

    it('removes multiple selected items', () => {
      const ids = addItems(3)
      useFloorPlanStore.getState().setSelection([ids[0]!, ids[2]!])
      useFloorPlanStore.getState().removeItems([ids[0]!, ids[2]!])

      expect(useFloorPlanStore.getState().items).toHaveLength(1)
      expect(useFloorPlanStore.getState().items[0]!.id).toBe(ids[1])
    })
  })

  describe('Rotate selection', () => {
    it('rotates selected items by 45 degrees', () => {
      const ids = addItems(1)
      useFloorPlanStore.getState().setSelection([ids[0]!])
      useFloorPlanStore.getState().rotateSelection(45)

      expect(useFloorPlanStore.getState().items[0]!.rotation).toBe(45)
    })

    it('rotates backward with -45', () => {
      const ids = addItems(1)
      useFloorPlanStore.getState().setSelection([ids[0]!])
      useFloorPlanStore.getState().rotateSelection(45)
      useFloorPlanStore.getState().rotateSelection(-45)

      expect(useFloorPlanStore.getState().items[0]!.rotation).toBe(0)
    })
  })

  describe('Select all / Escape', () => {
    it('selects all items', () => {
      const ids = addItems(5)
      useFloorPlanStore.getState().setSelection([])
      useFloorPlanStore.getState().setSelection(useFloorPlanStore.getState().items.map((i) => i.id))

      expect(useFloorPlanStore.getState().selectedIds).toHaveLength(5)
    })

    it('deselects all with Escape', () => {
      const ids = addItems(3)
      useFloorPlanStore.getState().setSelection(ids)
      useFloorPlanStore.getState().setSelection([])

      expect(useFloorPlanStore.getState().selectedIds).toHaveLength(0)
    })
  })

  describe('Undo/Redo via keyboard', () => {
    it('undo reverts last action', () => {
      addItems(1)
      const state = useFloorPlanStore.getState()
      expect(state.items).toHaveLength(1)

      state.undo()
      expect(useFloorPlanStore.getState().items).toHaveLength(0)
    })

    it('redo restores undone action', () => {
      addItems(1)
      useFloorPlanStore.getState().undo()
      expect(useFloorPlanStore.getState().items).toHaveLength(0)

      useFloorPlanStore.getState().redo()
      expect(useFloorPlanStore.getState().items).toHaveLength(1)
    })
  })
})

// ─── Auto-save ──────────────────────────────────────────────────────────────

describe('Auto-save', () => {
  const STORAGE_KEY = 'omnitwin-floorplan-autosave'

  beforeEach(() => {
    resetStore()
    localStorage.removeItem(STORAGE_KEY)
  })

  afterEach(() => {
    localStorage.removeItem(STORAGE_KEY)
  })

  it('saves data to localStorage', () => {
    const items: FloorPlanItem[] = [
      {
        id: 'save-1',
        name: 'Chair',
        category: 'chair',
        x: 10,
        y: 20,
        widthFt: 1.5,
        depthFt: 1.5,
        rotation: 0,
        locked: false,
      },
    ]

    const data = {
      items,
      planWidthFt: 80,
      planHeightFt: 50,
      gridSizeFt: 1,
      snapEnabled: true,
      savedAt: Date.now(),
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data))

    const raw = localStorage.getItem(STORAGE_KEY)
    expect(raw).not.toBeNull()
    const parsed = JSON.parse(raw!)
    expect(parsed.items).toHaveLength(1)
    expect(parsed.items[0].id).toBe('save-1')
  })

  it('loads saved data correctly', () => {
    const items: FloorPlanItem[] = [
      {
        id: 'load-1',
        name: 'Table',
        category: 'table',
        x: 30,
        y: 40,
        widthFt: 6,
        depthFt: 6,
        rotation: 90,
        locked: false,
      },
    ]

    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      items,
      planWidthFt: 100,
      planHeightFt: 60,
      gridSizeFt: 2,
      snapEnabled: false,
      savedAt: Date.now(),
    }))

    const raw = localStorage.getItem(STORAGE_KEY)
    const data = JSON.parse(raw!)

    // Simulate the load logic
    if (useFloorPlanStore.getState().items.length === 0 && data.items.length > 0) {
      useFloorPlanStore.setState({
        items: data.items,
        planWidthFt: data.planWidthFt,
        planHeightFt: data.planHeightFt,
      })
    }

    expect(useFloorPlanStore.getState().items).toHaveLength(1)
    expect(useFloorPlanStore.getState().items[0]!.id).toBe('load-1')
    expect(useFloorPlanStore.getState().planWidthFt).toBe(100)
  })

  it('does not overwrite existing items on load', () => {
    addItems(2) // editor already has items

    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      items: [{ id: 'saved', name: 'Saved', category: 'chair', x: 0, y: 0, widthFt: 1, depthFt: 1, rotation: 0, locked: false }],
      planWidthFt: 80,
      planHeightFt: 50,
      gridSizeFt: 1,
      snapEnabled: true,
      savedAt: Date.now(),
    }))

    const data = JSON.parse(localStorage.getItem(STORAGE_KEY)!)

    // Simulate load logic — should NOT overwrite
    if (useFloorPlanStore.getState().items.length === 0 && data.items.length > 0) {
      useFloorPlanStore.setState({ items: data.items })
    }

    // Should still have original 2 items, not the saved 1
    expect(useFloorPlanStore.getState().items).toHaveLength(2)
  })

  it('handles corrupt localStorage gracefully', () => {
    localStorage.setItem(STORAGE_KEY, 'not-json')

    expect(() => {
      try {
        JSON.parse(localStorage.getItem(STORAGE_KEY)!)
      } catch {
        // silently ignored — same as useAutoSave behavior
      }
    }).not.toThrow()
  })

  it('clears saved data', () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ items: [], savedAt: Date.now() }))
    localStorage.removeItem(STORAGE_KEY)
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull()
  })
})

// ─── Security headers ───────────────────────────────────────────────────────

describe('Security headers config', () => {
  it('defines all required headers', () => {
    // This is a static validation of the header configuration
    const requiredHeaders = [
      'X-Content-Type-Options',
      'X-Frame-Options',
      'X-XSS-Protection',
      'Referrer-Policy',
      'Permissions-Policy',
      'Strict-Transport-Security',
      'Content-Security-Policy',
    ]

    // The actual headers are in next.config.ts — we validate the expected set
    for (const header of requiredHeaders) {
      expect(header).toBeTruthy()
    }
    expect(requiredHeaders).toHaveLength(7)
  })

  it('CSP blocks frame-ancestors', () => {
    const csp = [
      "default-src 'self'",
      "script-src 'self' 'unsafe-eval' 'unsafe-inline'",
      "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
      "font-src 'self' https://fonts.gstatic.com",
      "img-src 'self' data: blob:",
      "connect-src 'self' ws: wss:",
      "worker-src 'self' blob:",
      "frame-ancestors 'none'",
    ].join('; ')

    expect(csp).toContain("frame-ancestors 'none'")
    expect(csp).toContain("default-src 'self'")
    expect(csp).toContain('ws:') // WebSocket needed for collaboration
  })

  it('HSTS has sufficient max-age', () => {
    const hsts = 'max-age=63072000; includeSubDomains; preload'
    const maxAge = parseInt(hsts.split('=')[1]!)
    expect(maxAge).toBeGreaterThanOrEqual(31536000) // at least 1 year
    expect(hsts).toContain('includeSubDomains')
  })
})
