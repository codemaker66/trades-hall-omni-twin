/**
 * Yjs ↔ Zustand Bidirectional Bridge
 *
 * Observes changes in both the Yjs Y.Doc and the Zustand floor plan store,
 * keeping them in sync. Uses echo suppression to prevent infinite loops.
 *
 * Usage:
 *   const bridge = createYjsBridge(doc)
 *   bridge.connect()
 *   // ... collaboration active ...
 *   bridge.disconnect()
 */
import * as Y from 'yjs'
import { useFloorPlanStore, type FloorPlanItem } from '../store'
import {
  getItemsArray,
  readItemsFromDoc,
  yMapToFloorPlanItem,
  floorPlanItemToYMap,
} from './yjsModel'

export interface YjsBridge {
  /** Start observing both Yjs and Zustand for changes. */
  connect: () => void
  /** Stop observing. */
  disconnect: () => void
  /** True if bridge is active. */
  isConnected: () => boolean
  /** Push current store state into Y.Doc (initial sync). */
  pushToDoc: () => void
  /** Pull Y.Doc state into the store (initial sync). */
  pullFromDoc: () => void
}

export function createYjsBridge(doc: Y.Doc): YjsBridge {
  let connected = false
  let suppressYjsUpdate = false
  let suppressStoreUpdate = false
  let storeUnsubscribe: (() => void) | null = null

  const yItems = getItemsArray(doc)

  // ── Yjs → Store ─────────────────────────────────────────────────────────

  function onYjsChange() {
    if (suppressYjsUpdate) return
    suppressStoreUpdate = true
    try {
      const items = readItemsFromDoc(doc)
      useFloorPlanStore.setState({ items })
    } finally {
      suppressStoreUpdate = false
    }
  }

  // ── Store → Yjs ─────────────────────────────────────────────────────────

  function onStoreChange(items: FloorPlanItem[], prevItems: FloorPlanItem[]) {
    if (suppressStoreUpdate) return
    suppressYjsUpdate = true
    try {
      reconcileToDoc(doc, prevItems, items)
    } finally {
      suppressYjsUpdate = false
    }
  }

  // ── Reconciliation (incremental diff) ──────────────────────────────────

  function reconcileToDoc(doc: Y.Doc, prev: FloorPlanItem[], next: FloorPlanItem[]) {
    const prevById = new Map(prev.map((i) => [i.id, i]))
    const nextById = new Map(next.map((i) => [i.id, i]))

    doc.transact(() => {
      // Remove items that no longer exist
      const toRemove: number[] = []
      for (let i = yItems.length - 1; i >= 0; i--) {
        const id = yItems.get(i).get('id') as string
        if (!nextById.has(id)) {
          toRemove.push(i)
        }
      }
      for (const idx of toRemove) {
        yItems.delete(idx, 1)
      }

      // Update existing items (only changed properties)
      for (let i = 0; i < yItems.length; i++) {
        const yMap = yItems.get(i)
        const id = yMap.get('id') as string
        const nextItem = nextById.get(id)
        if (!nextItem) continue

        const prevItem = prevById.get(id)
        if (prevItem) {
          if (prevItem.x !== nextItem.x) yMap.set('x', nextItem.x)
          if (prevItem.y !== nextItem.y) yMap.set('y', nextItem.y)
          if (prevItem.rotation !== nextItem.rotation) yMap.set('rotation', nextItem.rotation)
          if (prevItem.name !== nextItem.name) yMap.set('name', nextItem.name)
          if (prevItem.category !== nextItem.category) yMap.set('category', nextItem.category)
          if (prevItem.widthFt !== nextItem.widthFt) yMap.set('widthFt', nextItem.widthFt)
          if (prevItem.depthFt !== nextItem.depthFt) yMap.set('depthFt', nextItem.depthFt)
          if (prevItem.locked !== nextItem.locked) yMap.set('locked', nextItem.locked)
        }
      }

      // Add new items
      const existingIds = new Set<string>()
      for (let i = 0; i < yItems.length; i++) {
        existingIds.add(yItems.get(i).get('id') as string)
      }
      for (const item of next) {
        if (!existingIds.has(item.id)) {
          yItems.push([floorPlanItemToYMap(item)])
        }
      }
    })
  }

  // ── Public API ──────────────────────────────────────────────────────────

  return {
    connect() {
      if (connected) return
      connected = true

      // Observe Yjs deep changes (item property changes)
      yItems.observeDeep(onYjsChange)

      // Observe Zustand store items
      let prevItems = useFloorPlanStore.getState().items
      storeUnsubscribe = useFloorPlanStore.subscribe((state) => {
        if (state.items !== prevItems) {
          onStoreChange(state.items, prevItems)
          prevItems = state.items
        }
      })
    },

    disconnect() {
      if (!connected) return
      connected = false
      yItems.unobserveDeep(onYjsChange)
      storeUnsubscribe?.()
      storeUnsubscribe = null
    },

    isConnected() {
      return connected
    },

    pushToDoc() {
      suppressYjsUpdate = true
      try {
        const { items } = useFloorPlanStore.getState()
        doc.transact(() => {
          yItems.delete(0, yItems.length)
          for (const item of items) {
            yItems.push([floorPlanItemToYMap(item)])
          }
        })
      } finally {
        suppressYjsUpdate = false
      }
    },

    pullFromDoc() {
      suppressStoreUpdate = true
      try {
        const items = readItemsFromDoc(doc)
        useFloorPlanStore.setState({ items })
      } finally {
        suppressStoreUpdate = false
      }
    },
  }
}
