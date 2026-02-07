/**
 * Wire protocol types for compact binary encoding of spatial operations.
 *
 * Wire format: [1B opType] [8B hlcTimestamp] [4B objectId] [variable payload]
 * Total: 13 + payload bytes (vs ~200+ bytes JSON)
 */

// ─── Op Type Tags ──────────────────────────────────────────────────────────

export const OP_MOVE         = 0x01 as const
export const OP_ROTATE       = 0x02 as const
export const OP_PLACE        = 0x03 as const
export const OP_REMOVE       = 0x04 as const
export const OP_SCALE        = 0x05 as const
export const OP_BATCH_MOVE   = 0x06 as const
export const OP_BATCH_ROTATE = 0x07 as const

export type OpType =
  | typeof OP_MOVE
  | typeof OP_ROTATE
  | typeof OP_PLACE
  | typeof OP_REMOVE
  | typeof OP_SCALE
  | typeof OP_BATCH_MOVE
  | typeof OP_BATCH_ROTATE

// ─── Furniture Type Enum (matches ECS FurnitureTag.type) ────────────────────

export const FTYPE_CHAIR         = 0 as const
export const FTYPE_ROUND_TABLE   = 1 as const
export const FTYPE_RECT_TABLE    = 2 as const
export const FTYPE_TRESTLE_TABLE = 3 as const
export const FTYPE_PODIUM        = 4 as const
export const FTYPE_STAGE         = 5 as const
export const FTYPE_BAR           = 6 as const

export type FurnitureTypeIndex = 0 | 1 | 2 | 3 | 4 | 5 | 6

const FURNITURE_TYPE_NAMES = [
  'chair', 'round-table', 'rect-table', 'trestle-table', 'podium', 'stage', 'bar',
] as const

export type FurnitureTypeName = (typeof FURNITURE_TYPE_NAMES)[number]

export function furnitureIndexToName(index: FurnitureTypeIndex): FurnitureTypeName {
  return FURNITURE_TYPE_NAMES[index]
}

export function furnitureNameToIndex(name: FurnitureTypeName): FurnitureTypeIndex {
  const idx = FURNITURE_TYPE_NAMES.indexOf(name)
  if (idx === -1) throw new Error(`Unknown furniture type: ${name}`)
  return idx as FurnitureTypeIndex
}

// ─── HLC Timestamp ──────────────────────────────────────────────────────────

/**
 * Hybrid Logical Clock timestamp.
 * - wallMs: physical wall clock in milliseconds (48-bit range)
 * - counter: logical counter for ordering within same millisecond (16-bit range)
 *
 * Wire encoding: 8 bytes = (wallMs << 16) | counter as BigInt written as Uint64.
 */
export interface HlcTimestamp {
  wallMs: number
  counter: number
}

// ─── Operation Types ────────────────────────────────────────────────────────

export interface MoveOp {
  op: typeof OP_MOVE
  hlc: HlcTimestamp
  objectId: number
  dx: number
  dy: number
  dz: number
}

export interface RotateOp {
  op: typeof OP_ROTATE
  hlc: HlcTimestamp
  objectId: number
  rx: number
  ry: number
  rz: number
}

export interface PlaceOp {
  op: typeof OP_PLACE
  hlc: HlcTimestamp
  objectId: number
  furnitureType: FurnitureTypeIndex
  x: number
  y: number
  z: number
  rx: number
  ry: number
  rz: number
}

export interface RemoveOp {
  op: typeof OP_REMOVE
  hlc: HlcTimestamp
  objectId: number
}

export interface ScaleOp {
  op: typeof OP_SCALE
  hlc: HlcTimestamp
  objectId: number
  sx: number
  sy: number
  sz: number
}

export interface BatchMoveOp {
  op: typeof OP_BATCH_MOVE
  hlc: HlcTimestamp
  objectId: 0 // unused, always 0 for batch ops
  moves: Array<{ objectId: number; dx: number; dy: number; dz: number }>
}

export interface BatchRotateOp {
  op: typeof OP_BATCH_ROTATE
  hlc: HlcTimestamp
  objectId: 0
  rotations: Array<{ objectId: number; rx: number; ry: number; rz: number }>
}

/** Discriminated union of all wire operations. */
export type WireOp =
  | MoveOp
  | RotateOp
  | PlaceOp
  | RemoveOp
  | ScaleOp
  | BatchMoveOp
  | BatchRotateOp

// ─── Header constants ───────────────────────────────────────────────────────

/** Fixed header: 1 byte opType + 8 bytes HLC + 4 bytes objectId = 13 bytes */
export const HEADER_SIZE = 13

// Payload sizes for fixed-length ops
export const MOVE_PAYLOAD   = 12  // 3 × float32
export const ROTATE_PAYLOAD = 12  // 3 × float32
export const PLACE_PAYLOAD  = 25  // 1B type + 6 × float32
export const REMOVE_PAYLOAD = 0
export const SCALE_PAYLOAD  = 12  // 3 × float32

// Batch item sizes
export const BATCH_ITEM_SIZE = 16  // 4B objectId + 3 × float32

// ─── Batch frame ────────────────────────────────────────────────────────────

/** Batch frame header: [4B total_length] [2B op_count] */
export const BATCH_FRAME_HEADER = 6
