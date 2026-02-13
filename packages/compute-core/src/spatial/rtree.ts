// ---------------------------------------------------------------------------
// HPC-5: Spatial Indexing — R-tree
// ---------------------------------------------------------------------------
// R-tree for 2D bounding box queries with Sort-Tile-Recursive bulk loading.
// Pure TypeScript, zero external dependencies.
// ---------------------------------------------------------------------------

import type { BoundingBox2D, SpatialItem } from '../types.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Internal node of the R-tree. */
export interface RTreeInternalNode {
  bbox: BoundingBox2D;
  children: RTreeInternalNode[];
  items: SpatialItem[];
  leaf: boolean;
  height: number;
}

/** Root state of the R-tree. */
export type RTreeState = {
  root: RTreeInternalNode;
  size: number;
  maxEntries: number;
  minEntries: number;
};

// ---------------------------------------------------------------------------
// BBox helpers (exported for testing / reuse)
// ---------------------------------------------------------------------------

export function bboxIntersects(a: BoundingBox2D, b: BoundingBox2D): boolean {
  return a.minX <= b.maxX && a.maxX >= b.minX && a.minY <= b.maxY && a.maxY >= b.minY;
}

export function bboxContainsBBox(outer: BoundingBox2D, inner: BoundingBox2D): boolean {
  return (
    outer.minX <= inner.minX &&
    outer.minY <= inner.minY &&
    outer.maxX >= inner.maxX &&
    outer.maxY >= inner.maxY
  );
}

export function bboxArea(bb: BoundingBox2D): number {
  return (bb.maxX - bb.minX) * (bb.maxY - bb.minY);
}

export function bboxEnlarged(bb: BoundingBox2D, item: BoundingBox2D): BoundingBox2D {
  return {
    minX: Math.min(bb.minX, item.minX),
    minY: Math.min(bb.minY, item.minY),
    maxX: Math.max(bb.maxX, item.maxX),
    maxY: Math.max(bb.maxY, item.maxY),
  };
}

export function bboxEnlargement(bb: BoundingBox2D, item: BoundingBox2D): number {
  return bboxArea(bboxEnlarged(bb, item)) - bboxArea(bb);
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

const EMPTY_BBOX: BoundingBox2D = {
  minX: Infinity,
  minY: Infinity,
  maxX: -Infinity,
  maxY: -Infinity,
};

function createLeafNode(): RTreeInternalNode {
  return { bbox: { ...EMPTY_BBOX }, children: [], items: [], leaf: true, height: 1 };
}

function createInternalNode(height: number): RTreeInternalNode {
  return { bbox: { ...EMPTY_BBOX }, children: [], items: [], leaf: false, height };
}

function itemBBox(item: SpatialItem): BoundingBox2D {
  return item.bbox;
}

function nodeBBox(node: RTreeInternalNode): BoundingBox2D {
  return node.bbox;
}

function extendBBox(target: RTreeInternalNode, source: BoundingBox2D): void {
  target.bbox = bboxEnlarged(target.bbox, source);
}

function calcBBox(node: RTreeInternalNode): void {
  node.bbox = { ...EMPTY_BBOX };
  if (node.leaf) {
    for (let i = 0; i < node.items.length; i++) {
      extendBBox(node, itemBBox(node.items[i]!));
    }
  } else {
    for (let i = 0; i < node.children.length; i++) {
      extendBBox(node, nodeBBox(node.children[i]!));
    }
  }
}

// ---------------------------------------------------------------------------
// Create
// ---------------------------------------------------------------------------

/**
 * Create an empty R-tree.
 * @param maxEntries Maximum items per leaf node (default 9).
 */
export function createRTree(maxEntries?: number): RTreeState {
  const max = Math.max(4, maxEntries ?? 9);
  return {
    root: createLeafNode(),
    size: 0,
    maxEntries: max,
    minEntries: Math.max(2, Math.ceil(max * 0.4)),
  };
}

// ---------------------------------------------------------------------------
// Insert
// ---------------------------------------------------------------------------

/**
 * Insert a single spatial item into the R-tree. Mutates the tree in-place.
 */
export function rtreeInsert(tree: RTreeState, item: SpatialItem): void {
  insertItem(tree, item, tree.root.height - 1);
  tree.size++;
}

function insertItem(tree: RTreeState, item: SpatialItem, level: number): void {
  const bbox = itemBBox(item);
  const insertPath: RTreeInternalNode[] = [];

  // Choose the best leaf (or internal node at `level`) for this item
  const node = chooseSubtree(tree.root, bbox, level, insertPath);

  // Insert the item into the chosen leaf
  node.items.push(item);
  extendBBox(node, bbox);

  // Walk back up, extending bboxes and splitting overflow nodes
  let splitNode: RTreeInternalNode | null = null;
  let currentLevel = insertPath.length - 1;

  while (currentLevel >= 0) {
    const parent = insertPath[currentLevel]!;
    extendBBox(parent, splitNode ? nodeBBox(splitNode) : bbox);

    if (splitNode) {
      parent.children.push(splitNode);
      extendBBox(parent, nodeBBox(splitNode));
    }

    if (
      (parent.leaf && parent.items.length > tree.maxEntries) ||
      (!parent.leaf && parent.children.length > tree.maxEntries)
    ) {
      splitNode = splitNode ? null : splitOverflow(parent, tree);
    } else {
      splitNode = null;
    }

    currentLevel--;
  }

  // If root was split, create a new root
  if (splitNode) {
    const newRoot = createInternalNode(tree.root.height + 1);
    newRoot.children.push(tree.root, splitNode);
    calcBBox(newRoot);
    tree.root = newRoot;
  }
}

function chooseSubtree(
  node: RTreeInternalNode,
  bbox: BoundingBox2D,
  level: number,
  path: RTreeInternalNode[],
): RTreeInternalNode {
  let current = node;
  for (;;) {
    path.push(current);

    if (current.leaf || path.length - 1 === level) {
      return current;
    }

    let bestChild: RTreeInternalNode | null = null;
    let minEnlargement = Infinity;
    let minArea = Infinity;

    for (let i = 0; i < current.children.length; i++) {
      const child = current.children[i]!;
      const enlargement = bboxEnlargement(child.bbox, bbox);
      const area = bboxArea(child.bbox);

      if (enlargement < minEnlargement || (enlargement === minEnlargement && area < minArea)) {
        minEnlargement = enlargement;
        minArea = area;
        bestChild = child;
      }
    }

    current = bestChild!;
  }
}

/**
 * Split an overflowing node using the linear split algorithm.
 * Returns the newly created sibling node.
 */
function splitOverflow(node: RTreeInternalNode, tree: RTreeState): RTreeInternalNode {
  const isLeaf = node.leaf;

  if (isLeaf) {
    return splitLeaf(node, tree);
  } else {
    return splitInternal(node, tree);
  }
}

function splitLeaf(node: RTreeInternalNode, tree: RTreeState): RTreeInternalNode {
  const items = node.items;
  const total = items.length;

  // Choose split axis and sort
  const bestAxis = chooseSplitAxisItems(items);
  sortItemsByAxis(items, bestAxis);

  // Find the best split index (minimum overlap + area)
  const splitIndex = chooseSplitIndexItems(items, tree.minEntries, total);

  const newNode = createLeafNode();
  newNode.items = items.splice(splitIndex);
  node.items = items;

  calcBBox(node);
  calcBBox(newNode);

  return newNode;
}

function splitInternal(node: RTreeInternalNode, tree: RTreeState): RTreeInternalNode {
  const children = node.children;
  const total = children.length;

  const bestAxis = chooseSplitAxisChildren(children);
  sortChildrenByAxis(children, bestAxis);

  const splitIndex = chooseSplitIndexChildren(children, tree.minEntries, total);

  const newNode = createInternalNode(node.height);
  newNode.children = children.splice(splitIndex);
  node.children = children;

  calcBBox(node);
  calcBBox(newNode);

  return newNode;
}

function chooseSplitAxisItems(items: SpatialItem[]): 'x' | 'y' {
  // Compare margin sums along X vs Y
  const sortedX = items.slice().sort((a, b) => a.bbox.minX - b.bbox.minX);
  const sortedY = items.slice().sort((a, b) => a.bbox.minY - b.bbox.minY);

  let marginX = 0;
  let marginY = 0;

  for (let i = 1; i < items.length; i++) {
    let bb: BoundingBox2D = { ...EMPTY_BBOX };
    for (let j = 0; j < i; j++) bb = bboxEnlarged(bb, sortedX[j]!.bbox);
    marginX += bboxMargin(bb);

    bb = { ...EMPTY_BBOX };
    for (let j = 0; j < i; j++) bb = bboxEnlarged(bb, sortedY[j]!.bbox);
    marginY += bboxMargin(bb);
  }

  return marginX <= marginY ? 'x' : 'y';
}

function chooseSplitAxisChildren(children: RTreeInternalNode[]): 'x' | 'y' {
  const sortedX = children.slice().sort((a, b) => a.bbox.minX - b.bbox.minX);
  const sortedY = children.slice().sort((a, b) => a.bbox.minY - b.bbox.minY);

  let marginX = 0;
  let marginY = 0;

  for (let i = 1; i < children.length; i++) {
    let bb: BoundingBox2D = { ...EMPTY_BBOX };
    for (let j = 0; j < i; j++) bb = bboxEnlarged(bb, sortedX[j]!.bbox);
    marginX += bboxMargin(bb);

    bb = { ...EMPTY_BBOX };
    for (let j = 0; j < i; j++) bb = bboxEnlarged(bb, sortedY[j]!.bbox);
    marginY += bboxMargin(bb);
  }

  return marginX <= marginY ? 'x' : 'y';
}

function bboxMargin(bb: BoundingBox2D): number {
  return (bb.maxX - bb.minX) + (bb.maxY - bb.minY);
}

function sortItemsByAxis(items: SpatialItem[], axis: 'x' | 'y'): void {
  if (axis === 'x') {
    items.sort((a, b) => a.bbox.minX - b.bbox.minX || a.bbox.maxX - b.bbox.maxX);
  } else {
    items.sort((a, b) => a.bbox.minY - b.bbox.minY || a.bbox.maxY - b.bbox.maxY);
  }
}

function sortChildrenByAxis(children: RTreeInternalNode[], axis: 'x' | 'y'): void {
  if (axis === 'x') {
    children.sort((a, b) => a.bbox.minX - b.bbox.minX || a.bbox.maxX - b.bbox.maxX);
  } else {
    children.sort((a, b) => a.bbox.minY - b.bbox.minY || a.bbox.maxY - b.bbox.maxY);
  }
}

function chooseSplitIndexItems(
  items: SpatialItem[],
  minEntries: number,
  total: number,
): number {
  let bestOverlap = Infinity;
  let bestArea = Infinity;
  let bestIndex = minEntries;

  for (let i = minEntries; i <= total - minEntries; i++) {
    let bb1: BoundingBox2D = { ...EMPTY_BBOX };
    let bb2: BoundingBox2D = { ...EMPTY_BBOX };

    for (let j = 0; j < i; j++) bb1 = bboxEnlarged(bb1, items[j]!.bbox);
    for (let j = i; j < total; j++) bb2 = bboxEnlarged(bb2, items[j]!.bbox);

    const overlap = overlapArea(bb1, bb2);
    const area = bboxArea(bb1) + bboxArea(bb2);

    if (overlap < bestOverlap || (overlap === bestOverlap && area < bestArea)) {
      bestOverlap = overlap;
      bestArea = area;
      bestIndex = i;
    }
  }

  return bestIndex;
}

function chooseSplitIndexChildren(
  children: RTreeInternalNode[],
  minEntries: number,
  total: number,
): number {
  let bestOverlap = Infinity;
  let bestArea = Infinity;
  let bestIndex = minEntries;

  for (let i = minEntries; i <= total - minEntries; i++) {
    let bb1: BoundingBox2D = { ...EMPTY_BBOX };
    let bb2: BoundingBox2D = { ...EMPTY_BBOX };

    for (let j = 0; j < i; j++) bb1 = bboxEnlarged(bb1, children[j]!.bbox);
    for (let j = i; j < total; j++) bb2 = bboxEnlarged(bb2, children[j]!.bbox);

    const overlap = overlapArea(bb1, bb2);
    const area = bboxArea(bb1) + bboxArea(bb2);

    if (overlap < bestOverlap || (overlap === bestOverlap && area < bestArea)) {
      bestOverlap = overlap;
      bestArea = area;
      bestIndex = i;
    }
  }

  return bestIndex;
}

function overlapArea(a: BoundingBox2D, b: BoundingBox2D): number {
  const overlapX = Math.max(0, Math.min(a.maxX, b.maxX) - Math.max(a.minX, b.minX));
  const overlapY = Math.max(0, Math.min(a.maxY, b.maxY) - Math.max(a.minY, b.minY));
  return overlapX * overlapY;
}

// ---------------------------------------------------------------------------
// Bulk Load (Sort-Tile-Recursive)
// ---------------------------------------------------------------------------

/**
 * Bulk load items into the R-tree using the Sort-Tile-Recursive (STR) algorithm.
 * Much faster than individual inserts for large datasets. Replaces current contents.
 */
export function rtreeBulkLoad(tree: RTreeState, items: SpatialItem[]): void {
  if (items.length === 0) return;

  // If tree already has data, insert one-by-one (STR is for initial loading)
  if (tree.size > 0) {
    for (let i = 0; i < items.length; i++) {
      rtreeInsert(tree, items[i]!);
    }
    return;
  }

  tree.root = strBuild(items.slice(), tree.maxEntries);
  tree.size = items.length;
}

function strBuild(items: SpatialItem[], maxEntries: number): RTreeInternalNode {
  if (items.length <= maxEntries) {
    const leaf = createLeafNode();
    leaf.items = items;
    calcBBox(leaf);
    return leaf;
  }

  // Number of leaf nodes needed
  const numLeaves = Math.ceil(items.length / maxEntries);
  // Number of slices along X
  const numSlicesX = Math.ceil(Math.sqrt(numLeaves));

  // Sort by X center
  items.sort((a, b) => (a.bbox.minX + a.bbox.maxX) - (b.bbox.minX + b.bbox.maxX));

  const sliceSize = Math.ceil(items.length / numSlicesX);
  const childNodes: RTreeInternalNode[] = [];

  for (let i = 0; i < items.length; i += sliceSize) {
    const slice = items.slice(i, i + sliceSize);

    // Sort each slice by Y center
    slice.sort((a, b) => (a.bbox.minY + a.bbox.maxY) - (b.bbox.minY + b.bbox.maxY));

    // Create leaf nodes from this slice
    for (let j = 0; j < slice.length; j += maxEntries) {
      const leaf = createLeafNode();
      leaf.items = slice.slice(j, j + maxEntries);
      calcBBox(leaf);
      childNodes.push(leaf);
    }
  }

  // Recursively build internal levels
  return strBuildInternal(childNodes, maxEntries);
}

function strBuildInternal(
  nodes: RTreeInternalNode[],
  maxEntries: number,
): RTreeInternalNode {
  if (nodes.length === 1) return nodes[0]!;

  if (nodes.length <= maxEntries) {
    const parent = createInternalNode(nodes[0]!.height + 1);
    parent.children = nodes;
    calcBBox(parent);
    return parent;
  }

  const numGroups = Math.ceil(nodes.length / maxEntries);
  const numSlicesX = Math.ceil(Math.sqrt(numGroups));

  // Sort nodes by X center of their bbox
  nodes.sort(
    (a, b) => (a.bbox.minX + a.bbox.maxX) - (b.bbox.minX + b.bbox.maxX),
  );

  const sliceSize = Math.ceil(nodes.length / numSlicesX);
  const parentNodes: RTreeInternalNode[] = [];

  for (let i = 0; i < nodes.length; i += sliceSize) {
    const slice = nodes.slice(i, i + sliceSize);
    slice.sort(
      (a, b) => (a.bbox.minY + a.bbox.maxY) - (b.bbox.minY + b.bbox.maxY),
    );

    for (let j = 0; j < slice.length; j += maxEntries) {
      const parent = createInternalNode(nodes[0]!.height + 1);
      parent.children = slice.slice(j, j + maxEntries);
      calcBBox(parent);
      parentNodes.push(parent);
    }
  }

  return strBuildInternal(parentNodes, maxEntries);
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/**
 * Find all spatial items whose bounding boxes intersect the query box.
 */
export function rtreeSearch(tree: RTreeState, query: BoundingBox2D): SpatialItem[] {
  const results: SpatialItem[] = [];
  if (!bboxIntersects(tree.root.bbox, query)) return results;
  searchNode(tree.root, query, results);
  return results;
}

function searchNode(
  node: RTreeInternalNode,
  query: BoundingBox2D,
  results: SpatialItem[],
): void {
  if (node.leaf) {
    for (let i = 0; i < node.items.length; i++) {
      if (bboxIntersects(node.items[i]!.bbox, query)) {
        results.push(node.items[i]!);
      }
    }
  } else {
    for (let i = 0; i < node.children.length; i++) {
      const child = node.children[i]!;
      if (bboxIntersects(child.bbox, query)) {
        searchNode(child, query, results);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Remove
// ---------------------------------------------------------------------------

/**
 * Remove an item by its ID. Returns true if found and removed, false otherwise.
 */
export function rtreeRemove(tree: RTreeState, id: string): boolean {
  const path: RTreeInternalNode[] = [];
  const leaf = findLeafWithItem(tree.root, id, path);
  if (!leaf) return false;

  // Remove from leaf
  const idx = leaf.items.findIndex((it) => it.id === id);
  if (idx === -1) return false;
  leaf.items.splice(idx, 1);
  tree.size--;

  // Recalculate bboxes up the path
  calcBBox(leaf);
  for (let i = path.length - 1; i >= 0; i--) {
    calcBBox(path[i]!);
  }

  // Condense tree: remove underfull nodes and re-insert their contents
  condenseTree(tree, path, leaf);

  // If root has only one child and is not a leaf, shrink tree height
  if (!tree.root.leaf && tree.root.children.length === 1) {
    tree.root = tree.root.children[0]!;
  }

  // If tree is empty, reset to empty leaf
  if (tree.size === 0) {
    tree.root = createLeafNode();
  }

  return true;
}

function findLeafWithItem(
  node: RTreeInternalNode,
  id: string,
  path: RTreeInternalNode[],
): RTreeInternalNode | null {
  if (node.leaf) {
    for (let i = 0; i < node.items.length; i++) {
      if (node.items[i]!.id === id) return node;
    }
    return null;
  }

  for (let i = 0; i < node.children.length; i++) {
    path.push(node);
    const result = findLeafWithItem(node.children[i]!, id, path);
    if (result) return result;
    path.pop();
  }

  return null;
}

function condenseTree(
  tree: RTreeState,
  path: RTreeInternalNode[],
  leaf: RTreeInternalNode,
): void {
  const orphanedItems: SpatialItem[] = [];
  const orphanedNodes: RTreeInternalNode[] = [];

  // Check if the leaf is now underfull
  if (leaf.items.length < tree.minEntries && path.length > 0) {
    const parent = path[path.length - 1]!;
    const idx = parent.children.indexOf(leaf);
    if (idx !== -1) {
      parent.children.splice(idx, 1);
      collectAllItems(leaf, orphanedItems);
    }
  }

  // Walk up the path, removing underfull internal nodes
  for (let level = path.length - 1; level > 0; level--) {
    const node = path[level]!;
    const parent = path[level - 1]!;

    if (node.children.length < tree.minEntries) {
      const idx = parent.children.indexOf(node);
      if (idx !== -1) {
        parent.children.splice(idx, 1);
        if (node.leaf) {
          collectAllItems(node, orphanedItems);
        } else {
          for (let i = 0; i < node.children.length; i++) {
            orphanedNodes.push(node.children[i]!);
          }
        }
      }
    }

    calcBBox(parent);
  }

  // Re-insert orphaned items
  for (let i = 0; i < orphanedItems.length; i++) {
    rtreeInsert(tree, orphanedItems[i]!);
  }

  // Re-insert orphaned internal nodes at their appropriate level
  for (let i = 0; i < orphanedNodes.length; i++) {
    reinsertNode(tree, orphanedNodes[i]!);
  }
}

function reinsertNode(tree: RTreeState, node: RTreeInternalNode): void {
  // Find the level at which to insert this node
  const targetLevel = tree.root.height - node.height;
  const insertPath: RTreeInternalNode[] = [];
  const parent = chooseSubtree(tree.root, node.bbox, targetLevel - 1, insertPath);
  parent.children.push(node);
  extendBBox(parent, node.bbox);

  // Recalculate bboxes up the path
  for (let i = insertPath.length - 1; i >= 0; i--) {
    calcBBox(insertPath[i]!);
  }
}

function collectAllItems(node: RTreeInternalNode, out: SpatialItem[]): void {
  if (node.leaf) {
    for (let i = 0; i < node.items.length; i++) {
      out.push(node.items[i]!);
    }
  } else {
    for (let i = 0; i < node.children.length; i++) {
      collectAllItems(node.children[i]!, out);
    }
  }
}

// ---------------------------------------------------------------------------
// Size / All
// ---------------------------------------------------------------------------

/** Return the total number of items in the R-tree. */
export function rtreeSize(tree: RTreeState): number {
  return tree.size;
}

/** Return all items stored in the R-tree. */
export function rtreeAll(tree: RTreeState): SpatialItem[] {
  const results: SpatialItem[] = [];
  collectAllItems(tree.root, results);
  return results;
}
