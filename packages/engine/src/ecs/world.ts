/**
 * ECS world creation and entity management.
 * Uses bitECS for Structure-of-Arrays performance.
 */

import {
  createWorld,
  addEntity,
  removeEntity,
  addComponent,
  hasComponent,
  getAllEntities,
  resetWorld,
} from 'bitecs'
import type { FurnitureType } from '@omni-twin/types'
import {
  Position,
  Rotation,
  Scale,
  BoundingBox,
  FurnitureTag,
  GroupMember,
  Selectable,
  Draggable,
  furnitureTypeToIndex,
} from './components'

// ─── Types ──────────────────────────────────────────────────────────────────

/** Input data for creating a furniture entity. */
export interface FurnitureEntityInput {
  type: FurnitureType
  position: [number, number, number]
  rotation?: [number, number, number]
  scale?: [number, number, number]
  groupId?: number
}

/** Maps domain item IDs (string) ↔ ECS entity IDs (number). */
export class EntityIdMap {
  private domainToEcs = new Map<string, number>()
  private ecsToDomain = new Map<number, string>()

  set(domainId: string, eid: number): void {
    this.domainToEcs.set(domainId, eid)
    this.ecsToDomain.set(eid, domainId)
  }

  getEcs(domainId: string): number | undefined {
    return this.domainToEcs.get(domainId)
  }

  getDomain(eid: number): string | undefined {
    return this.ecsToDomain.get(eid)
  }

  delete(domainId: string): void {
    const eid = this.domainToEcs.get(domainId)
    if (eid !== undefined) {
      this.ecsToDomain.delete(eid)
    }
    this.domainToEcs.delete(domainId)
  }

  deleteByEid(eid: number): void {
    const domainId = this.ecsToDomain.get(eid)
    if (domainId !== undefined) {
      this.domainToEcs.delete(domainId)
    }
    this.ecsToDomain.delete(eid)
  }

  get size(): number {
    return this.domainToEcs.size
  }

  clear(): void {
    this.domainToEcs.clear()
    this.ecsToDomain.clear()
  }
}

// ─── Default bounding boxes per furniture type ──────────────────────────────

const DEFAULT_BOUNDS: Record<FurnitureType, [number, number, number]> = {
  'chair': [0.22, 0.45, 0.22],
  'round-table': [0.6, 0.38, 0.6],
  'rect-table': [0.6, 0.38, 0.3],
  'trestle-table': [0.9, 0.38, 0.38],
  'podium': [0.3, 0.6, 0.3],
  'stage': [2.0, 0.3, 1.5],
  'bar': [1.0, 0.55, 0.3],
}

// ─── World Factory ──────────────────────────────────────────────────────────

export type EcsWorld = ReturnType<typeof createWorld>

/** Create a fresh ECS world. */
export function createEcsWorld(): EcsWorld {
  return createWorld()
}

/** Reset a world, removing all entities and component data. */
export function resetEcsWorld(world: EcsWorld): void {
  resetWorld(world)
}

// ─── Entity Factory ─────────────────────────────────────────────────────────

/**
 * Create a furniture entity in the ECS world with all standard components.
 * Returns the entity ID (number).
 */
export function createFurnitureEntity(world: EcsWorld, input: FurnitureEntityInput): number {
  const eid = addEntity(world)

  // Position
  addComponent(world, Position, eid)
  Position.x[eid] = input.position[0]
  Position.y[eid] = input.position[1]
  Position.z[eid] = input.position[2]

  // Rotation
  addComponent(world, Rotation, eid)
  if (input.rotation) {
    Rotation.x[eid] = input.rotation[0]
    Rotation.y[eid] = input.rotation[1]
    Rotation.z[eid] = input.rotation[2]
  }

  // Scale
  addComponent(world, Scale, eid)
  if (input.scale) {
    Scale.x[eid] = input.scale[0]
    Scale.y[eid] = input.scale[1]
    Scale.z[eid] = input.scale[2]
  } else {
    Scale.x[eid] = 1
    Scale.y[eid] = 1
    Scale.z[eid] = 1
  }

  // BoundingBox
  addComponent(world, BoundingBox, eid)
  const bounds = DEFAULT_BOUNDS[input.type]
  BoundingBox.halfX[eid] = bounds[0]
  BoundingBox.halfY[eid] = bounds[1]
  BoundingBox.halfZ[eid] = bounds[2]

  // FurnitureTag
  addComponent(world, FurnitureTag, eid)
  FurnitureTag.type[eid] = furnitureTypeToIndex(input.type)

  // GroupMember
  addComponent(world, GroupMember, eid)
  GroupMember.groupId[eid] = input.groupId ?? 0

  // Interaction tags
  addComponent(world, Selectable, eid)
  addComponent(world, Draggable, eid)

  return eid
}

/**
 * Remove a furniture entity from the world.
 */
export function removeFurnitureEntity(world: EcsWorld, eid: number): void {
  removeEntity(world, eid)
}

/**
 * Check if an entity exists and has Position (proxy for "is a furniture entity").
 */
export function isFurnitureEntity(world: EcsWorld, eid: number): boolean {
  return hasComponent(world, Position, eid) && hasComponent(world, FurnitureTag, eid)
}

/**
 * Get all entity IDs currently in the world.
 */
export function getAllEntityIds(world: EcsWorld): number[] {
  return getAllEntities(world)
}
