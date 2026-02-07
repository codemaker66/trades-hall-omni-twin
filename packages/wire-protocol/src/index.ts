// Types
export type {
  OpType, FurnitureTypeIndex, FurnitureTypeName, HlcTimestamp,
  WireOp, MoveOp, RotateOp, PlaceOp, RemoveOp, ScaleOp,
  BatchMoveOp, BatchRotateOp,
} from './types'

export {
  OP_MOVE, OP_ROTATE, OP_PLACE, OP_REMOVE, OP_SCALE,
  OP_BATCH_MOVE, OP_BATCH_ROTATE,
  FTYPE_CHAIR, FTYPE_ROUND_TABLE, FTYPE_RECT_TABLE, FTYPE_TRESTLE_TABLE,
  FTYPE_PODIUM, FTYPE_STAGE, FTYPE_BAR,
  furnitureIndexToName, furnitureNameToIndex,
  HEADER_SIZE, MOVE_PAYLOAD, ROTATE_PAYLOAD, PLACE_PAYLOAD,
  REMOVE_PAYLOAD, SCALE_PAYLOAD, BATCH_ITEM_SIZE, BATCH_FRAME_HEADER,
} from './types'

// Clock
export { HybridLogicalClock, hlcCompare, hlcToUint64, uint64ToHlc } from './clock'

// Encoder
export { encode, encodeInto, encodedSize } from './encoder'

// Decoder
export { decode, decodeAt, type DecodeResult } from './decoder'

// Batch framing
export { encodeBatchFrame, decodeBatchFrame, peekBatchCount, peekFrameLength } from './batch'

// Delta compression
export {
  DeltaCompressor,
  encodeCompressedMove, decodeCompressedMove, compressedMoveSize,
  SCALE_FACTOR, MAX_DELTA, DEADZONE,
  FLAG_DELTA, FLAG_RELATIVE,
  type CompressedMove, type FullMove, type DeltaMove,
} from './compress'
