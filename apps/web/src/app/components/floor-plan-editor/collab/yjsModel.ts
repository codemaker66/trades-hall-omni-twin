/**
 * Yjs Data Model for Floor Plan Collaboration
 *
 * Each floor plan has a Y.Doc containing:
 * - Y.Array<Y.Map> for furniture items
 * - Y.Map for plan settings (dimensions, etc.)
 *
 * Item Y.Map keys: id, name, category, x, y, widthFt, depthFt, rotation, locked
 */
import * as Y from 'yjs'
import type { FloorPlanItem, FurnitureCategory } from '../store'

// ─── Y.Doc shape ─────────────────────────────────────────────────────────────

export function getItemsArray(doc: Y.Doc): Y.Array<Y.Map<string | number | boolean>> {
  return doc.getArray('items')
}

export function getSettingsMap(doc: Y.Doc): Y.Map<number> {
  return doc.getMap('settings')
}

// ─── Convert between FloorPlanItem and Y.Map ─────────────────────────────────

export function floorPlanItemToYMap(item: FloorPlanItem): Y.Map<string | number | boolean> {
  const map = new Y.Map<string | number | boolean>()
  map.set('id', item.id)
  map.set('name', item.name)
  map.set('category', item.category)
  map.set('x', item.x)
  map.set('y', item.y)
  map.set('widthFt', item.widthFt)
  map.set('depthFt', item.depthFt)
  map.set('rotation', item.rotation)
  map.set('locked', item.locked)
  return map
}

export function yMapToFloorPlanItem(map: Y.Map<string | number | boolean>): FloorPlanItem {
  return {
    id: map.get('id') as string,
    name: map.get('name') as string,
    category: map.get('category') as FurnitureCategory,
    x: map.get('x') as number,
    y: map.get('y') as number,
    widthFt: map.get('widthFt') as number,
    depthFt: map.get('depthFt') as number,
    rotation: map.get('rotation') as number,
    locked: map.get('locked') as boolean,
  }
}

/**
 * Sync all floor plan items into the Y.Doc.
 * Clears existing items and replaces with the provided array.
 */
export function syncItemsToDoc(doc: Y.Doc, items: FloorPlanItem[]): void {
  const yItems = getItemsArray(doc)
  doc.transact(() => {
    yItems.delete(0, yItems.length)
    for (const item of items) {
      yItems.push([floorPlanItemToYMap(item)])
    }
  })
}

/**
 * Read all floor plan items from the Y.Doc.
 */
export function readItemsFromDoc(doc: Y.Doc): FloorPlanItem[] {
  const yItems = getItemsArray(doc)
  const result: FloorPlanItem[] = []
  for (let i = 0; i < yItems.length; i++) {
    result.push(yMapToFloorPlanItem(yItems.get(i)))
  }
  return result
}

/**
 * Find the index of an item in the Y.Array by its ID.
 */
export function findItemIndex(doc: Y.Doc, id: string): number {
  const yItems = getItemsArray(doc)
  for (let i = 0; i < yItems.length; i++) {
    if (yItems.get(i).get('id') === id) return i
  }
  return -1
}

/**
 * Add a single item to the Y.Doc.
 */
export function addItemToDoc(doc: Y.Doc, item: FloorPlanItem): void {
  const yItems = getItemsArray(doc)
  yItems.push([floorPlanItemToYMap(item)])
}

/**
 * Update properties of an item in the Y.Doc.
 */
export function updateItemInDoc(doc: Y.Doc, id: string, changes: Partial<FloorPlanItem>): void {
  const yItems = getItemsArray(doc)
  for (let i = 0; i < yItems.length; i++) {
    const map = yItems.get(i)
    if (map.get('id') === id) {
      doc.transact(() => {
        for (const [key, value] of Object.entries(changes)) {
          if (key !== 'id') {
            map.set(key, value as string | number | boolean)
          }
        }
      })
      return
    }
  }
}

/**
 * Remove items from the Y.Doc by their IDs.
 */
export function removeItemsFromDoc(doc: Y.Doc, ids: string[]): void {
  const yItems = getItemsArray(doc)
  const idSet = new Set(ids)
  doc.transact(() => {
    // Remove from end to start to keep indices stable
    for (let i = yItems.length - 1; i >= 0; i--) {
      if (idSet.has(yItems.get(i).get('id') as string)) {
        yItems.delete(i, 1)
      }
    }
  })
}

/**
 * Save plan settings (dimensions) to the Y.Doc.
 */
export function syncSettingsToDoc(doc: Y.Doc, widthFt: number, heightFt: number): void {
  const settings = getSettingsMap(doc)
  doc.transact(() => {
    settings.set('planWidthFt', widthFt)
    settings.set('planHeightFt', heightFt)
  })
}

/**
 * Read plan settings from the Y.Doc.
 */
export function readSettingsFromDoc(doc: Y.Doc): { planWidthFt: number; planHeightFt: number } | null {
  const settings = getSettingsMap(doc)
  const w = settings.get('planWidthFt')
  const h = settings.get('planHeightFt')
  if (w === undefined || h === undefined) return null
  return { planWidthFt: w, planHeightFt: h }
}
