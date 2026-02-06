import { describe, it, expect, beforeEach } from 'vitest'
import { useFloorPlanStore, snapToGrid2D } from '../store'
import { templates } from '../templates'
import { generateLegend } from '../exportFloorPlan'

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
    snapEnabled: true,
    tool: 'select',
    planWidthFt: 80,
    planHeightFt: 50,
  })
}

function addChair(x = 10, y = 10) {
  return useFloorPlanStore.getState().addItem({
    name: 'Chair',
    category: 'chair',
    x,
    y,
    widthFt: 1.5,
    depthFt: 1.5,
    rotation: 0,
  })
}

function addTable(x = 20, y = 20) {
  return useFloorPlanStore.getState().addItem({
    name: 'Round Table 6ft',
    category: 'table',
    x,
    y,
    widthFt: 6,
    depthFt: 6,
    rotation: 0,
  })
}

// ─── snapToGrid2D ────────────────────────────────────────────────────────────

describe('snapToGrid2D', () => {
  it('snaps to 1ft grid', () => {
    expect(snapToGrid2D(3.3, 1)).toBe(3)
    expect(snapToGrid2D(3.6, 1)).toBe(4)
    expect(snapToGrid2D(3.5, 1)).toBe(4)
  })

  it('snaps to 0.5ft (6in) grid', () => {
    expect(snapToGrid2D(3.3, 0.5)).toBe(3.5)
    expect(snapToGrid2D(3.1, 0.5)).toBe(3)
    expect(snapToGrid2D(3.74, 0.5)).toBe(3.5)
    expect(snapToGrid2D(3.76, 0.5)).toBe(4)
  })

  it('snaps to 2ft grid', () => {
    expect(snapToGrid2D(5, 2)).toBe(6)
    expect(snapToGrid2D(4.9, 2)).toBe(4)
  })

  it('handles zero and negative values', () => {
    expect(snapToGrid2D(0, 1)).toBe(0)
    expect(snapToGrid2D(-1.3, 1)).toBe(-1)
    expect(snapToGrid2D(-1.7, 1)).toBe(-2)
  })
})

// ─── Store: add / remove / update ────────────────────────────────────────────

describe('floor plan store', () => {
  beforeEach(resetStore)

  describe('addItem', () => {
    it('adds an item and selects it', () => {
      const id = addChair()
      const state = useFloorPlanStore.getState()
      expect(state.items).toHaveLength(1)
      expect(state.items[0]!.id).toBe(id)
      expect(state.selectedIds).toEqual([id])
    })

    it('snaps new item to grid when snap is enabled', () => {
      useFloorPlanStore.setState({ snapEnabled: true, gridSizeFt: 1 })
      addChair(10.3, 15.7)
      const item = useFloorPlanStore.getState().items[0]!
      expect(item.x).toBe(10)
      expect(item.y).toBe(16)
    })

    it('preserves exact position when snap is disabled', () => {
      useFloorPlanStore.setState({ snapEnabled: false })
      addChair(10.3, 15.7)
      const item = useFloorPlanStore.getState().items[0]!
      expect(item.x).toBe(10.3)
      expect(item.y).toBe(15.7)
    })

    it('records undo history', () => {
      expect(useFloorPlanStore.getState().canUndo).toBe(false)
      addChair()
      expect(useFloorPlanStore.getState().canUndo).toBe(true)
    })
  })

  describe('removeItems', () => {
    it('removes items by id', () => {
      const id1 = addChair(5, 5)
      const id2 = addChair(10, 10)
      expect(useFloorPlanStore.getState().items).toHaveLength(2)

      useFloorPlanStore.getState().removeItems([id1])
      const state = useFloorPlanStore.getState()
      expect(state.items).toHaveLength(1)
      expect(state.items[0]!.id).toBe(id2)
    })

    it('clears removed items from selection', () => {
      const id = addChair()
      expect(useFloorPlanStore.getState().selectedIds).toContain(id)

      useFloorPlanStore.getState().removeItems([id])
      expect(useFloorPlanStore.getState().selectedIds).not.toContain(id)
    })
  })

  describe('updateItems', () => {
    it('updates item properties', () => {
      const id = addChair(10, 10)
      useFloorPlanStore.getState().updateItems([{ id, changes: { x: 20, y: 30 } }])
      const item = useFloorPlanStore.getState().items[0]!
      expect(item.x).toBe(20)
      expect(item.y).toBe(30)
    })

    it('records undo history for non-batch updates', () => {
      const id = addChair()
      const pastBefore = useFloorPlanStore.getState().past.length
      useFloorPlanStore.getState().updateItems([{ id, changes: { x: 99 } }])
      expect(useFloorPlanStore.getState().past.length).toBe(pastBefore + 1)
    })
  })

  describe('rotateSelection', () => {
    it('rotates selected items by given degrees', () => {
      const id = addChair()
      useFloorPlanStore.getState().setSelection([id])
      useFloorPlanStore.getState().rotateSelection(45)
      expect(useFloorPlanStore.getState().items[0]!.rotation).toBe(45)
    })

    it('wraps rotation at 360', () => {
      const id = addChair()
      useFloorPlanStore.getState().setSelection([id])
      useFloorPlanStore.getState().rotateSelection(350)
      useFloorPlanStore.getState().rotateSelection(20)
      expect(useFloorPlanStore.getState().items[0]!.rotation).toBe(10)
    })

    it('does nothing when nothing is selected', () => {
      addChair()
      useFloorPlanStore.getState().setSelection([])
      useFloorPlanStore.getState().rotateSelection(45)
      expect(useFloorPlanStore.getState().items[0]!.rotation).toBe(0)
    })
  })

  // ─── Undo / Redo ──────────────────────────────────────────────────────────

  describe('undo / redo', () => {
    it('undoes the last action', () => {
      addChair()
      expect(useFloorPlanStore.getState().items).toHaveLength(1)

      useFloorPlanStore.getState().undo()
      expect(useFloorPlanStore.getState().items).toHaveLength(0)
    })

    it('redoes after undo', () => {
      addChair()
      useFloorPlanStore.getState().undo()
      expect(useFloorPlanStore.getState().items).toHaveLength(0)

      useFloorPlanStore.getState().redo()
      expect(useFloorPlanStore.getState().items).toHaveLength(1)
    })

    it('clears redo stack on new action', () => {
      addChair()
      useFloorPlanStore.getState().undo()
      expect(useFloorPlanStore.getState().canRedo).toBe(true)

      addTable()
      expect(useFloorPlanStore.getState().canRedo).toBe(false)
    })

    it('does nothing when no undo/redo available', () => {
      useFloorPlanStore.getState().undo() // no-op
      expect(useFloorPlanStore.getState().items).toHaveLength(0)

      useFloorPlanStore.getState().redo() // no-op
      expect(useFloorPlanStore.getState().items).toHaveLength(0)
    })
  })

  // ─── Batching ──────────────────────────────────────────────────────────────

  describe('batch operations', () => {
    it('groups multiple updates into one undo entry', () => {
      const id = addChair()
      const pastBefore = useFloorPlanStore.getState().past.length

      useFloorPlanStore.getState().beginBatch()
      useFloorPlanStore.getState().updateItems([{ id, changes: { x: 20 } }])
      useFloorPlanStore.getState().updateItems([{ id, changes: { x: 30 } }])
      useFloorPlanStore.getState().updateItems([{ id, changes: { x: 40 } }])
      useFloorPlanStore.getState().endBatch()

      // Only one history entry for the entire batch
      expect(useFloorPlanStore.getState().past.length).toBe(pastBefore + 1)
      expect(useFloorPlanStore.getState().items[0]!.x).toBe(40)
    })

    it('undo restores state from before the batch', () => {
      const id = addChair(10, 10)

      useFloorPlanStore.getState().beginBatch()
      useFloorPlanStore.getState().updateItems([{ id, changes: { x: 50 } }])
      useFloorPlanStore.getState().endBatch()

      useFloorPlanStore.getState().undo()
      expect(useFloorPlanStore.getState().items[0]!.x).toBe(10)
    })

    it('skips history if batch has no changes', () => {
      addChair()
      const pastBefore = useFloorPlanStore.getState().past.length

      useFloorPlanStore.getState().beginBatch()
      useFloorPlanStore.getState().endBatch()

      expect(useFloorPlanStore.getState().past.length).toBe(pastBefore)
    })
  })

  // ─── Templates ─────────────────────────────────────────────────────────────

  describe('loadTemplate', () => {
    it('replaces all items with template items', () => {
      addChair()
      addTable()
      expect(useFloorPlanStore.getState().items).toHaveLength(2)

      useFloorPlanStore.getState().loadTemplate(templates[0]!.items)
      expect(useFloorPlanStore.getState().items.length).toBe(templates[0]!.items.length)
    })

    it('assigns new ids to template items', () => {
      useFloorPlanStore.getState().loadTemplate(templates[0]!.items)
      const ids = useFloorPlanStore.getState().items.map((i) => i.id)
      const uniqueIds = new Set(ids)
      expect(uniqueIds.size).toBe(ids.length)
    })

    it('can be undone to restore previous items', () => {
      addChair()
      addTable()

      useFloorPlanStore.getState().loadTemplate(templates[0]!.items)
      useFloorPlanStore.getState().undo()

      expect(useFloorPlanStore.getState().items).toHaveLength(2)
    })
  })

  // ─── Metrics ───────────────────────────────────────────────────────────────

  describe('getMetrics', () => {
    it('counts chairs and tables', () => {
      addChair()
      addChair(12, 12)
      addTable()
      const m = useFloorPlanStore.getState().getMetrics()
      expect(m.chairs).toBe(2)
      expect(m.tables).toBe(1)
    })

    it('calculates total seats (chairs + table capacity)', () => {
      addChair()
      addChair(12, 12)
      addTable() // 6ft round → 8 seats
      const m = useFloorPlanStore.getState().getMetrics()
      expect(m.totalSeats).toBe(2 + 8) // 2 chairs + 8 at table
    })

    it('counts small tables as 6 seats', () => {
      useFloorPlanStore.getState().addItem({
        name: 'Cocktail Table',
        category: 'table',
        x: 10,
        y: 10,
        widthFt: 2,
        depthFt: 2,
        rotation: 0,
      })
      const m = useFloorPlanStore.getState().getMetrics()
      expect(m.totalSeats).toBe(6)
    })
  })

  // ─── Viewport ──────────────────────────────────────────────────────────────

  describe('viewport controls', () => {
    it('clamps zoom between 0.1 and 5', () => {
      useFloorPlanStore.getState().setZoom(0.01)
      expect(useFloorPlanStore.getState().zoom).toBe(0.1)

      useFloorPlanStore.getState().setZoom(10)
      expect(useFloorPlanStore.getState().zoom).toBe(5)
    })

    it('sets pan coordinates', () => {
      useFloorPlanStore.getState().setPan(100, 200)
      expect(useFloorPlanStore.getState().panX).toBe(100)
      expect(useFloorPlanStore.getState().panY).toBe(200)
    })

    it('toggles snap', () => {
      expect(useFloorPlanStore.getState().snapEnabled).toBe(true)
      useFloorPlanStore.getState().toggleSnap()
      expect(useFloorPlanStore.getState().snapEnabled).toBe(false)
      useFloorPlanStore.getState().toggleSnap()
      expect(useFloorPlanStore.getState().snapEnabled).toBe(true)
    })
  })
})

// ─── Templates ───────────────────────────────────────────────────────────────

describe('floor plan templates', () => {
  it('all templates have unique ids', () => {
    const ids = templates.map((t) => t.id)
    expect(new Set(ids).size).toBe(ids.length)
  })

  it('all template items have valid categories', () => {
    const validCategories = new Set(['table', 'chair', 'stage', 'decor', 'equipment'])
    for (const t of templates) {
      for (const item of t.items) {
        expect(validCategories.has(item.category)).toBe(true)
      }
    }
  })

  it('all template items have positive dimensions', () => {
    for (const t of templates) {
      for (const item of t.items) {
        expect(item.widthFt).toBeGreaterThan(0)
        expect(item.depthFt).toBeGreaterThan(0)
      }
    }
  })

  it('theater template has a stage and 120 chairs', () => {
    const theater = templates.find((t) => t.id === 'theater')!
    const chairs = theater.items.filter((i) => i.category === 'chair')
    const stages = theater.items.filter((i) => i.category === 'stage')
    expect(chairs.length).toBe(120)
    expect(stages.length).toBeGreaterThanOrEqual(1)
  })

  it('boardroom template has 16 chairs', () => {
    const boardroom = templates.find((t) => t.id === 'boardroom')!
    const chairs = boardroom.items.filter((i) => i.category === 'chair')
    expect(chairs.length).toBe(16)
  })
})

// ─── Export Legend ────────────────────────────────────────────────────────────

describe('generateLegend', () => {
  it('includes floor plan dimensions', () => {
    const legend = generateLegend([], 80, 50)
    expect(legend).toContain('80ft x 50ft')
  })

  it('counts items correctly', () => {
    const items = [
      { id: '1', name: 'Chair', category: 'chair' as const, x: 0, y: 0, widthFt: 1.5, depthFt: 1.5, rotation: 0, locked: false },
      { id: '2', name: 'Chair', category: 'chair' as const, x: 2, y: 0, widthFt: 1.5, depthFt: 1.5, rotation: 0, locked: false },
      { id: '3', name: 'Round Table 6ft', category: 'table' as const, x: 10, y: 10, widthFt: 6, depthFt: 6, rotation: 0, locked: false },
    ]
    const legend = generateLegend(items, 80, 50)
    expect(legend).toContain('Total items: 3')
    expect(legend).toContain('Tables: 1')
    expect(legend).toContain('Chair: 2')
    expect(legend).toContain('Round Table 6ft: 1')
  })

  it('calculates seat count with table capacity', () => {
    const items = [
      { id: '1', name: 'Chair', category: 'chair' as const, x: 0, y: 0, widthFt: 1.5, depthFt: 1.5, rotation: 0, locked: false },
      { id: '2', name: 'Round Table 6ft', category: 'table' as const, x: 10, y: 10, widthFt: 6, depthFt: 6, rotation: 0, locked: false },
    ]
    const legend = generateLegend(items, 80, 50)
    // 1 chair + 8 at table = 9 seats
    expect(legend).toContain('Seats: 9')
  })
})
