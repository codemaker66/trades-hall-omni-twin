// ---------------------------------------------------------------------------
// HPC-5: Spatial Indexing — Barrel export
// ---------------------------------------------------------------------------

export {
  buildKDTree,
  kdTreeNearestN,
  kdTreeRadiusSearch,
  kdTreeSize,
} from './kdtree.js';

export {
  createRTree,
  rtreeInsert,
  rtreeBulkLoad,
  rtreeSearch,
  rtreeRemove,
  rtreeSize,
  rtreeAll,
  bboxIntersects,
  bboxContainsBBox,
  bboxArea,
  bboxEnlarged,
  bboxEnlargement,
} from './rtree.js';
export type { RTreeInternalNode, RTreeState } from './rtree.js';

export {
  buildBVH,
  bvhQuery,
  bvhRaycast,
  bvhFindOverlaps,
  bvhDepth,
} from './bvh.js';

export {
  createSpatialHashGrid,
  gridInsert,
  gridRemove,
  gridQuery,
  gridNearby,
  gridClear,
  gridSize,
} from './grid-hash.js';
export type { SpatialHashGrid } from './grid-hash.js';
