/**
 * ECS component definitions using bitECS Structure-of-Arrays layout.
 * Each component is a typed array store keyed by entity ID.
 */

import { defineComponent, Types } from 'bitecs'

const { f32, ui8, ui32 } = Types

// ─── Spatial Components ─────────────────────────────────────────────────────

/** 3D position (world-space). */
export const Position = defineComponent({ x: f32, y: f32, z: f32 })

/** 3D rotation (Euler angles in radians). */
export const Rotation = defineComponent({ x: f32, y: f32, z: f32 })

/** 3D scale (uniform or per-axis). */
export const Scale = defineComponent({ x: f32, y: f32, z: f32 })

/** Axis-aligned bounding box half-extents for broadphase collision. */
export const BoundingBox = defineComponent({ halfX: f32, halfY: f32, halfZ: f32 })

// ─── Identity Components ────────────────────────────────────────────────────

/**
 * Furniture type tag.
 * Stored as a uint8 enum index for SoA efficiency.
 *
 * Mapping:
 *   0 = chair, 1 = round-table, 2 = rect-table, 3 = trestle-table,
 *   4 = podium, 5 = stage, 6 = bar
 */
export const FurnitureTag = defineComponent({ type: ui8 })

/** Group membership. groupId is the entity ID of the group entity (0 = none). */
export const GroupMember = defineComponent({ groupId: ui32 })

// ─── Flag Components (zero-size tags) ───────────────────────────────────────

/** Entity can be selected by the user. */
export const Selectable = defineComponent()

/** Entity can be dragged / moved by the user. */
export const Draggable = defineComponent()

// ─── Furniture type enum mapping ────────────────────────────────────────────

import type { FurnitureType } from '@omni-twin/types'

const FURNITURE_TYPE_TO_INDEX: Record<FurnitureType, number> = {
  'chair': 0,
  'round-table': 1,
  'rect-table': 2,
  'trestle-table': 3,
  'podium': 4,
  'stage': 5,
  'bar': 6,
}

const INDEX_TO_FURNITURE_TYPE: FurnitureType[] = [
  'chair',
  'round-table',
  'rect-table',
  'trestle-table',
  'podium',
  'stage',
  'bar',
]

/** Convert a FurnitureType string to its uint8 index. */
export function furnitureTypeToIndex(type: FurnitureType): number {
  return FURNITURE_TYPE_TO_INDEX[type]
}

/** Convert a uint8 index back to a FurnitureType string. */
export function indexToFurnitureType(index: number): FurnitureType | undefined {
  return INDEX_TO_FURNITURE_TYPE[index]
}
