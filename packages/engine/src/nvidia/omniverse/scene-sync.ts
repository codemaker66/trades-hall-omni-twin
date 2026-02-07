/**
 * Omniverse scene sync bridge: converts venue state to Omniverse scene format
 * and computes incremental diffs for efficient network updates.
 */

import type { ProjectedVenueState, ProjectedItem } from '../../projector'
import type { OmniverseSceneItem, OmniverseSceneState } from '../types'

// ─── Material Mapping ────────────────────────────────────────────────────────

const MATERIAL_MAP: Record<string, string> = {
  chair: 'fabric_burgundy',
  'round-table': 'wood_oak',
  'rect-table': 'wood_walnut',
  'trestle-table': 'wood_pine',
  podium: 'wood_mahogany',
  stage: 'wood_maple',
  bar: 'metal_brushed',
}

// ─── Convert State ───────────────────────────────────────────────────────────

/** Convert a projected item to Omniverse scene format. */
function toOmniverseItem(item: ProjectedItem): OmniverseSceneItem {
  return {
    id: item.id,
    type: item.furnitureType,
    position: [...item.position],
    rotation: [...item.rotation],
    scale: [...item.scale],
    material: MATERIAL_MAP[item.furnitureType],
  }
}

/** Convert full venue state to Omniverse scene state. */
export function toOmniverseScene(
  state: ProjectedVenueState,
  dimensions: { width: number; depth: number; height: number },
  lighting: 'warm' | 'cool' | 'natural' | 'dramatic' = 'warm',
): OmniverseSceneState {
  const items: OmniverseSceneItem[] = []
  for (const [, item] of state.items) {
    items.push(toOmniverseItem(item))
  }

  return {
    items,
    venueDimensions: dimensions,
    lighting,
  }
}

// ─── Incremental Diff ────────────────────────────────────────────────────────

export type SceneDiffOp =
  | { type: 'add'; item: OmniverseSceneItem }
  | { type: 'remove'; id: string }
  | { type: 'update'; item: OmniverseSceneItem }

/** Compute incremental diff between old and new scene states. */
export function computeSceneDiff(
  oldScene: OmniverseSceneState,
  newScene: OmniverseSceneState,
): SceneDiffOp[] {
  const ops: SceneDiffOp[] = []

  const oldMap = new Map<string, OmniverseSceneItem>()
  for (const item of oldScene.items) {
    oldMap.set(item.id, item)
  }

  const newMap = new Map<string, OmniverseSceneItem>()
  for (const item of newScene.items) {
    newMap.set(item.id, item)
  }

  // Check for added and updated items
  for (const [id, newItem] of newMap) {
    const oldItem = oldMap.get(id)
    if (!oldItem) {
      ops.push({ type: 'add', item: newItem })
    } else if (!sceneItemsEqual(oldItem, newItem)) {
      ops.push({ type: 'update', item: newItem })
    }
  }

  // Check for removed items
  for (const id of oldMap.keys()) {
    if (!newMap.has(id)) {
      ops.push({ type: 'remove', id })
    }
  }

  return ops
}

function sceneItemsEqual(a: OmniverseSceneItem, b: OmniverseSceneItem): boolean {
  return (
    a.type === b.type &&
    a.position[0] === b.position[0] && a.position[1] === b.position[1] && a.position[2] === b.position[2] &&
    a.rotation[0] === b.rotation[0] && a.rotation[1] === b.rotation[1] && a.rotation[2] === b.rotation[2] &&
    a.scale[0] === b.scale[0] && a.scale[1] === b.scale[1] && a.scale[2] === b.scale[2] &&
    a.material === b.material
  )
}

export { toOmniverseItem }
