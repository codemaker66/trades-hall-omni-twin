/**
 * 2D Floor Plan Editor Store
 *
 * Self-contained Zustand store for the 2D editor.
 * Works in feet. Positions are { x, y } where x = horizontal, y = vertical (top-down view).
 * Reuses undo/redo pattern from the venue store.
 */
import { create } from 'zustand'

// ─── Types ───────────────────────────────────────────────────────────────────

export type FurnitureCategory = 'table' | 'chair' | 'stage' | 'decor' | 'equipment'

export interface FloorPlanItem {
  id: string
  name: string
  category: FurnitureCategory
  x: number       // feet from left
  y: number       // feet from top
  widthFt: number
  depthFt: number
  rotation: number // degrees
  locked: boolean
}

export interface FloorPlanSnapshot {
  items: FloorPlanItem[]
  selectedIds: string[]
}

interface FloorPlanState {
  // Floor plan dimensions
  planWidthFt: number
  planHeightFt: number

  // Items
  items: FloorPlanItem[]
  selectedIds: string[]

  // Viewport
  zoom: number
  panX: number
  panY: number

  // Grid & Snap
  gridSizeFt: number  // 1 = 1ft, 0.5 = 6in
  snapEnabled: boolean

  // Tool mode
  tool: 'select' | 'pan'

  // Undo/Redo
  canUndo: boolean
  canRedo: boolean
  past: FloorPlanSnapshot[]
  future: FloorPlanSnapshot[]
  batch: FloorPlanSnapshot | null

  // Actions
  setPlanDimensions: (w: number, h: number) => void
  setZoom: (zoom: number) => void
  setPan: (x: number, y: number) => void
  setTool: (tool: 'select' | 'pan') => void
  setGridSize: (ft: number) => void
  toggleSnap: () => void
  setSelection: (ids: string[]) => void

  addItem: (item: Omit<FloorPlanItem, 'id' | 'locked'>) => string
  updateItems: (updates: { id: string; changes: Partial<FloorPlanItem> }[]) => void
  removeItems: (ids: string[]) => void
  rotateSelection: (degrees: number) => void

  beginBatch: () => void
  endBatch: () => void
  undo: () => void
  redo: () => void

  getMetrics: () => { chairs: number; tables: number; stages: number; totalSeats: number }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const HISTORY_LIMIT = 50

let idCounter = 0
function genId(): string {
  return `fp-${Date.now().toString(36)}-${(idCounter++).toString(36)}`
}

function snapshot(state: { items: FloorPlanItem[]; selectedIds: string[] }): FloorPlanSnapshot {
  return {
    items: state.items.map((i) => ({ ...i })),
    selectedIds: [...state.selectedIds],
  }
}

function snapshotEq(a: FloorPlanSnapshot, b: FloorPlanSnapshot): boolean {
  if (a.items.length !== b.items.length) return false
  for (let i = 0; i < a.items.length; i++) {
    const ai = a.items[i]!
    const bi = b.items[i]!
    if (ai.id !== bi.id || ai.x !== bi.x || ai.y !== bi.y || ai.rotation !== bi.rotation) return false
  }
  return true
}

function pushHistory(past: FloorPlanSnapshot[], snap: FloorPlanSnapshot): FloorPlanSnapshot[] {
  return [...past, snap].slice(-HISTORY_LIMIT)
}

// ─── Store ───────────────────────────────────────────────────────────────────

export const useFloorPlanStore = create<FloorPlanState>()((set, get) => ({
  planWidthFt: 80,
  planHeightFt: 50,
  items: [],
  selectedIds: [],
  zoom: 1,
  panX: 0,
  panY: 0,
  gridSizeFt: 1,
  snapEnabled: true,
  tool: 'select',
  canUndo: false,
  canRedo: false,
  past: [],
  future: [],
  batch: null,

  setPlanDimensions: (w, h) => set({ planWidthFt: w, planHeightFt: h }),
  setZoom: (zoom) => set({ zoom: Math.max(0.1, Math.min(5, zoom)) }),
  setPan: (x, y) => set({ panX: x, panY: y }),
  setTool: (tool) => set({ tool }),
  setGridSize: (ft) => set({ gridSizeFt: ft }),
  toggleSnap: () => set((s) => ({ snapEnabled: !s.snapEnabled })),

  setSelection: (ids) => set({ selectedIds: ids }),

  addItem: (item) => {
    const id = genId()
    const state = get()
    const snap = snapshot(state)
    const newItem: FloorPlanItem = { ...item, id, locked: false }

    // Snap to grid if enabled
    if (state.snapEnabled) {
      newItem.x = Math.round(newItem.x / state.gridSizeFt) * state.gridSizeFt
      newItem.y = Math.round(newItem.y / state.gridSizeFt) * state.gridSizeFt
    }

    set({
      items: [...state.items, newItem],
      selectedIds: [id],
      past: pushHistory(state.past, snap),
      future: [],
      canUndo: true,
      canRedo: false,
    })
    return id
  },

  updateItems: (updates) => {
    const state = get()
    if (state.batch) {
      // Inside batch — don't push history
      const items = state.items.map((item) => {
        const u = updates.find((up) => up.id === item.id)
        return u ? { ...item, ...u.changes } : item
      })
      set({ items })
      return
    }

    const snap = snapshot(state)
    const items = state.items.map((item) => {
      const u = updates.find((up) => up.id === item.id)
      return u ? { ...item, ...u.changes } : item
    })
    set({
      items,
      past: pushHistory(state.past, snap),
      future: [],
      canUndo: true,
      canRedo: false,
    })
  },

  removeItems: (ids) => {
    const state = get()
    const snap = snapshot(state)
    set({
      items: state.items.filter((i) => !ids.includes(i.id)),
      selectedIds: state.selectedIds.filter((id) => !ids.includes(id)),
      past: pushHistory(state.past, snap),
      future: [],
      canUndo: true,
      canRedo: false,
    })
  },

  rotateSelection: (degrees) => {
    const state = get()
    if (state.selectedIds.length === 0) return
    const snap = snapshot(state)
    const items = state.items.map((item) => {
      if (!state.selectedIds.includes(item.id)) return item
      return { ...item, rotation: (item.rotation + degrees) % 360 }
    })
    set({
      items,
      past: pushHistory(state.past, snap),
      future: [],
      canUndo: true,
      canRedo: false,
    })
  },

  beginBatch: () => {
    const state = get()
    if (state.batch) return
    set({ batch: snapshot(state) })
  },

  endBatch: () => {
    const state = get()
    if (!state.batch) return
    const current = snapshot(state)
    if (snapshotEq(state.batch, current)) {
      set({ batch: null })
      return
    }
    set({
      past: pushHistory(state.past, state.batch),
      future: [],
      canUndo: true,
      canRedo: false,
      batch: null,
    })
  },

  undo: () => {
    const state = get()
    if (state.past.length === 0) return
    const prev = state.past[state.past.length - 1]!
    const current = snapshot(state)
    set({
      items: prev.items,
      selectedIds: prev.selectedIds,
      past: state.past.slice(0, -1),
      future: [current, ...state.future].slice(0, HISTORY_LIMIT),
      canUndo: state.past.length > 1,
      canRedo: true,
    })
  },

  redo: () => {
    const state = get()
    if (state.future.length === 0) return
    const next = state.future[0]!
    const current = snapshot(state)
    set({
      items: next.items,
      selectedIds: next.selectedIds,
      past: pushHistory(state.past, current),
      future: state.future.slice(1),
      canUndo: true,
      canRedo: state.future.length > 1,
    })
  },

  getMetrics: () => {
    const { items } = get()
    let chairs = 0, tables = 0, stages = 0
    for (const item of items) {
      if (item.category === 'chair') chairs++
      else if (item.category === 'table') tables++
      else if (item.category === 'stage') stages++
    }
    // Estimate seats: chairs + table capacity (round tables ~8, rect ~6)
    const tableSeats = items
      .filter((i) => i.category === 'table')
      .reduce((sum, t) => sum + (t.widthFt >= 5 ? 8 : 6), 0)
    return { chairs, tables, stages, totalSeats: chairs + tableSeats }
  },
}))

// Snap helper for external use
export function snapToGrid2D(val: number, grid: number): number {
  return Math.round(val / grid) * grid
}
