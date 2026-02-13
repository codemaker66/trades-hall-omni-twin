// ---------------------------------------------------------------------------
// HPC-7: CRDT — Barrel Export
// ---------------------------------------------------------------------------

export {
  createVectorClock,
  vcIncrement,
  vcMerge,
  vcCompare,
  vcHappensBefore,
  vcGet,
  vcClone,
} from './vector-clock.js';

export {
  createLWWRegister,
  lwwSet,
  lwwGet,
  lwwMerge,
} from './lww-register.js';

export {
  createORSet,
  orSetAdd,
  orSetRemove,
  orSetContains,
  orSetElements,
  orSetMerge,
} from './or-set.js';
export type { ORSetState } from './or-set.js';

export {
  createLayoutDoc,
  layoutDocAddItem,
  layoutDocMoveItem,
  layoutDocRemoveItem,
  layoutDocGetItem,
  layoutDocGetAll,
  layoutDocMerge,
  layoutDocApplyOp,
} from './layout-doc.js';
export type { LayoutItem, LayoutDocState } from './layout-doc.js';
