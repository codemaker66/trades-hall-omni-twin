/**
 * ECS ↔ Zustand bridge.
 *
 * Synchronizes between domain-level item arrays (Zustand store) and the
 * bitECS world (entity-component system). The bridge:
 *   - Converts domain FurnitureItems to ECS entities on full sync
 *   - Applies incremental updates (add, remove, move)
 *   - Reads ECS state back into domain-friendly structures
 *   - Maintains a bidirectional ID map (string ↔ eid)
 */

import type { FurnitureType } from '@omni-twin/types'
import {
  Position,
  Rotation,
  Scale,
  FurnitureTag,
  GroupMember,
  BoundingBox,
  furnitureTypeToIndex,
  indexToFurnitureType,
} from './components'
import {
  createEcsWorld,
  createFurnitureEntity,
  removeFurnitureEntity,
  getAllEntityIds,
  EntityIdMap,
  type EcsWorld,
} from './world'
import { SpatialHash } from './systems/spatial-index'

// ─── Domain Types ───────────────────────────────────────────────────────────

/**
 * Minimal item shape matching the Zustand store's FurnitureItem.
 * The bridge accepts any object with these fields.
 */
export interface BridgeItem {
  id: string
  type: FurnitureType
  position: [number, number, number]
  rotation: [number, number, number]
  scale?: [number, number, number]
  groupId?: string
}

// ─── ECS Bridge ─────────────────────────────────────────────────────────────

/**
 * Bridges between a domain item array and the ECS world.
 * Manages the world, ID map, and spatial hash together.
 */
export class EcsBridge {
  readonly world: EcsWorld
  readonly idMap: EntityIdMap
  readonly spatialHash: SpatialHash

  /** Map from domain groupId (string) → ECS group entity eid. */
  private groupIdMap = new Map<string, number>()
  private nextGroupEid = 1_000_000 // synthetic group eids start high

  constructor(cellSize = 2) {
    this.world = createEcsWorld()
    this.idMap = new EntityIdMap()
    this.spatialHash = new SpatialHash(cellSize)
  }

  // ─── Full Sync ──────────────────────────────────────────────────────────

  /**
   * Full sync: replace the entire ECS world with the given items.
   * Clears existing entities and rebuilds from scratch.
   */
  syncAll(items: readonly BridgeItem[]): void {
    // Clear everything
    for (const eid of getAllEntityIds(this.world)) {
      removeFurnitureEntity(this.world, eid)
    }
    this.idMap.clear()
    this.spatialHash.clear()
    this.groupIdMap.clear()

    // Rebuild
    for (const item of items) {
      this.addItem(item)
    }
  }

  // ─── Incremental Updates ────────────────────────────────────────────────

  /** Add a single item to the ECS world. */
  addItem(item: BridgeItem): number {
    const groupEid = item.groupId ? this.resolveGroupEid(item.groupId) : 0

    const eid = createFurnitureEntity(this.world, {
      type: item.type,
      position: item.position,
      rotation: item.rotation,
      scale: item.scale,
      groupId: groupEid,
    })

    this.idMap.set(item.id, eid)
    this.spatialHash.insert(eid, item.position[0], item.position[2])

    return eid
  }

  /** Remove an item by domain ID. */
  removeItem(domainId: string): void {
    const eid = this.idMap.getEcs(domainId)
    if (eid === undefined) return

    this.spatialHash.remove(eid)
    removeFurnitureEntity(this.world, eid)
    this.idMap.delete(domainId)
  }

  /** Update an item's position. */
  moveItem(domainId: string, position: [number, number, number]): void {
    const eid = this.idMap.getEcs(domainId)
    if (eid === undefined) return

    Position.x[eid] = position[0]
    Position.y[eid] = position[1]
    Position.z[eid] = position[2]

    this.spatialHash.update(eid)
  }

  /** Update an item's rotation. */
  rotateItem(domainId: string, rotation: [number, number, number]): void {
    const eid = this.idMap.getEcs(domainId)
    if (eid === undefined) return

    Rotation.x[eid] = rotation[0]
    Rotation.y[eid] = rotation[1]
    Rotation.z[eid] = rotation[2]
  }

  /** Batch move multiple items. */
  moveItems(updates: Array<{ id: string; position: [number, number, number] }>): void {
    for (const { id, position } of updates) {
      this.moveItem(id, position)
    }
  }

  // ─── Read Back ──────────────────────────────────────────────────────────

  /** Read a single entity's state back as a domain item. */
  readItem(domainId: string): BridgeItem | undefined {
    const eid = this.idMap.getEcs(domainId)
    if (eid === undefined) return undefined

    const typeIndex = FurnitureTag.type[eid]!
    const furnitureType = indexToFurnitureType(typeIndex)
    if (!furnitureType) return undefined

    return {
      id: domainId,
      type: furnitureType,
      position: [Position.x[eid]!, Position.y[eid]!, Position.z[eid]!],
      rotation: [Rotation.x[eid]!, Rotation.y[eid]!, Rotation.z[eid]!],
      scale: [Scale.x[eid]!, Scale.y[eid]!, Scale.z[eid]!],
    }
  }

  /** Read all entities back as domain items. */
  readAll(): BridgeItem[] {
    const items: BridgeItem[] = []
    const entities = getAllEntityIds(this.world)

    for (const eid of entities) {
      const domainId = this.idMap.getDomain(eid)
      if (!domainId) continue

      const item = this.readItem(domainId)
      if (item) items.push(item)
    }

    return items
  }

  /** Get the entity ID for a domain item. */
  getEid(domainId: string): number | undefined {
    return this.idMap.getEcs(domainId)
  }

  /** Get the domain ID for an entity. */
  getDomainId(eid: number): string | undefined {
    return this.idMap.getDomain(eid)
  }

  /** Get the number of tracked entities. */
  get size(): number {
    return this.idMap.size
  }

  // ─── Group Management ───────────────────────────────────────────────────

  private resolveGroupEid(groupId: string): number {
    let eid = this.groupIdMap.get(groupId)
    if (eid === undefined) {
      eid = this.nextGroupEid++
      this.groupIdMap.set(groupId, eid)
    }
    return eid
  }
}
