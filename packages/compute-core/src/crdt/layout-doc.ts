// ---------------------------------------------------------------------------
// HPC-7: CRDT — Collaborative Layout Document
// ---------------------------------------------------------------------------
// A collaborative document for spatial layouts (venue floor plans, etc.).
// Uses an OR-Set to track which items exist (add-wins for concurrent
// add/remove) and LWW registers for each item's mutable properties
// (x, y, width, depth, rotation, type). A vector clock tracks causality.
// ---------------------------------------------------------------------------

import type { LWWRegister, MutableVectorClock, CRDTOperation } from '../types.js';
import { createVectorClock, vcIncrement, vcMerge } from './vector-clock.js';
import { createLWWRegister, lwwMerge } from './lww-register.js';
import {
  createORSet,
  orSetAdd,
  orSetRemove,
  orSetContains,
  orSetElements,
  orSetMerge,
} from './or-set.js';
import type { ORSetState } from './or-set.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** A layout item with all resolved properties. */
export type LayoutItem = {
  id: string;
  type: string;
  x: number;
  y: number;
  width: number;
  depth: number;
  rotation: number;
};

/** Mutable state of the collaborative layout document. */
export type LayoutDocState = {
  peerId: string;
  items: ORSetState<string>;
  /** Outer key = item ID, inner key = property name. */
  properties: Map<string, Map<string, LWWRegister<number | string>>>;
  clock: MutableVectorClock;
  counter: number;
};

// ---------------------------------------------------------------------------
// Property keys used for each layout item
// ---------------------------------------------------------------------------

const PROP_KEYS = ['type', 'x', 'y', 'width', 'depth', 'rotation'] as const;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function nextTimestamp(doc: LayoutDocState): number {
  doc.counter++;
  vcIncrement(doc.clock, doc.peerId);
  return doc.counter;
}

function setProperty(
  doc: LayoutDocState,
  itemId: string,
  prop: string,
  value: number | string,
): void {
  const ts = nextTimestamp(doc);
  let propMap = doc.properties.get(itemId);
  if (!propMap) {
    propMap = new Map();
    doc.properties.set(itemId, propMap);
  }
  propMap.set(prop, createLWWRegister(value, doc.peerId, ts));
}

function getPropertyValue(
  doc: LayoutDocState,
  itemId: string,
  prop: string,
): number | string | undefined {
  const propMap = doc.properties.get(itemId);
  if (!propMap) return undefined;
  const reg = propMap.get(prop);
  if (!reg) return undefined;
  return reg.value;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Create an empty collaborative layout document for the given peer.
 */
export function createLayoutDoc(peerId: string): LayoutDocState {
  return {
    peerId,
    items: createORSet<string>(),
    properties: new Map(),
    clock: createVectorClock(),
    counter: 0,
  };
}

/**
 * Add a new item to the layout document.
 * Returns the generated item ID.
 */
export function layoutDocAddItem(
  doc: LayoutDocState,
  item: {
    type: string;
    x: number;
    y: number;
    width: number;
    depth: number;
    rotation: number;
  },
): string {
  const itemId = `${doc.peerId}:item:${++doc.counter}`;
  vcIncrement(doc.clock, doc.peerId);

  // Add the item ID to the OR-Set
  orSetAdd(doc.items, itemId, doc.peerId);

  // Store each property as an LWW register
  setProperty(doc, itemId, 'type', item.type);
  setProperty(doc, itemId, 'x', item.x);
  setProperty(doc, itemId, 'y', item.y);
  setProperty(doc, itemId, 'width', item.width);
  setProperty(doc, itemId, 'depth', item.depth);
  setProperty(doc, itemId, 'rotation', item.rotation);

  return itemId;
}

/**
 * Move an existing item to new coordinates.
 * Returns false if the item is not found in the OR-Set.
 */
export function layoutDocMoveItem(
  doc: LayoutDocState,
  itemId: string,
  x: number,
  y: number,
): boolean {
  if (!orSetContains(doc.items, itemId)) return false;

  setProperty(doc, itemId, 'x', x);
  setProperty(doc, itemId, 'y', y);

  return true;
}

/**
 * Remove an item from the layout document.
 * Returns false if the item is not found.
 */
export function layoutDocRemoveItem(
  doc: LayoutDocState,
  itemId: string,
): boolean {
  if (!orSetContains(doc.items, itemId)) return false;

  orSetRemove(doc.items, itemId);
  vcIncrement(doc.clock, doc.peerId);
  doc.counter++;

  return true;
}

/**
 * Get a single item by ID, or null if not present.
 */
export function layoutDocGetItem(
  doc: LayoutDocState,
  itemId: string,
): LayoutItem | null {
  if (!orSetContains(doc.items, itemId)) return null;

  const type = getPropertyValue(doc, itemId, 'type');
  const x = getPropertyValue(doc, itemId, 'x');
  const y = getPropertyValue(doc, itemId, 'y');
  const width = getPropertyValue(doc, itemId, 'width');
  const depth = getPropertyValue(doc, itemId, 'depth');
  const rotation = getPropertyValue(doc, itemId, 'rotation');

  return {
    id: itemId,
    type: (type as string) ?? 'unknown',
    x: (x as number) ?? 0,
    y: (y as number) ?? 0,
    width: (width as number) ?? 0,
    depth: (depth as number) ?? 0,
    rotation: (rotation as number) ?? 0,
  };
}

/**
 * Get all items currently present in the layout document.
 */
export function layoutDocGetAll(doc: LayoutDocState): LayoutItem[] {
  const ids = orSetElements(doc.items);
  const result: LayoutItem[] = [];
  for (const id of ids) {
    const item = layoutDocGetItem(doc, id);
    if (item) result.push(item);
  }
  return result;
}

/**
 * Merge two layout documents from different peers.
 * - Items: add-wins via OR-Set merge
 * - Properties: LWW merge for each property register
 * - Clock: element-wise max merge
 */
export function layoutDocMerge(
  a: LayoutDocState,
  b: LayoutDocState,
): LayoutDocState {
  const merged: LayoutDocState = {
    peerId: a.peerId,
    items: orSetMerge(a.items, b.items),
    properties: new Map(),
    clock: vcMerge(a.clock, b.clock),
    counter: Math.max(a.counter, b.counter),
  };

  // Merge properties: for each item, for each property, LWW merge
  const allItemIds = new Set<string>();
  for (const key of a.properties.keys()) allItemIds.add(key);
  for (const key of b.properties.keys()) allItemIds.add(key);

  for (const itemId of allItemIds) {
    const propsA = a.properties.get(itemId);
    const propsB = b.properties.get(itemId);
    const mergedProps = new Map<string, LWWRegister<number | string>>();

    // Gather all property names
    const allPropNames = new Set<string>();
    if (propsA) for (const k of propsA.keys()) allPropNames.add(k);
    if (propsB) for (const k of propsB.keys()) allPropNames.add(k);

    for (const prop of allPropNames) {
      const regA = propsA?.get(prop);
      const regB = propsB?.get(prop);
      if (regA && regB) {
        mergedProps.set(prop, lwwMerge(regA, regB));
      } else if (regA) {
        mergedProps.set(prop, regA);
      } else if (regB) {
        mergedProps.set(prop, regB);
      }
    }

    merged.properties.set(itemId, mergedProps);
  }

  return merged;
}

/**
 * Apply a single CRDT operation to the layout document.
 * Operations are idempotent: replaying the same operation has no additional
 * effect once the timestamp is dominated.
 */
export function layoutDocApplyOp(
  doc: LayoutDocState,
  op: CRDTOperation,
): void {
  vcIncrement(doc.clock, op.peerId);

  switch (op.type) {
    case 'add': {
      const val = op.value as {
        type: string;
        x: number;
        y: number;
        width: number;
        depth: number;
        rotation: number;
      } | undefined;
      if (!val) break;

      // Use the op key as item ID
      orSetAdd(doc.items, op.key, op.peerId);

      const ts = op.timestamp;
      let propMap = doc.properties.get(op.key);
      if (!propMap) {
        propMap = new Map();
        doc.properties.set(op.key, propMap);
      }

      for (const prop of PROP_KEYS) {
        const existing = propMap.get(prop);
        const newReg = createLWWRegister(
          val[prop] as number | string,
          op.peerId,
          ts,
        );
        if (existing) {
          propMap.set(prop, lwwMerge(existing, newReg));
        } else {
          propMap.set(prop, newReg);
        }
      }
      break;
    }

    case 'remove': {
      orSetRemove(doc.items, op.key);
      break;
    }

    case 'move': {
      const coords = op.value as { x: number; y: number } | undefined;
      if (!coords) break;
      if (!orSetContains(doc.items, op.key)) break;

      const ts = op.timestamp;
      let propMap = doc.properties.get(op.key);
      if (!propMap) {
        propMap = new Map();
        doc.properties.set(op.key, propMap);
      }

      for (const prop of ['x', 'y'] as const) {
        const existing = propMap.get(prop);
        const newReg = createLWWRegister(coords[prop], op.peerId, ts);
        if (existing) {
          propMap.set(prop, lwwMerge(existing, newReg));
        } else {
          propMap.set(prop, newReg);
        }
      }
      break;
    }

    case 'set': {
      if (op.value === undefined) break;
      if (!orSetContains(doc.items, op.key)) break;

      // 'set' op uses key format "itemId:propName"
      const colonIdx = op.key.lastIndexOf(':');
      if (colonIdx === -1) break;

      const itemId = op.key.slice(0, colonIdx);
      const prop = op.key.slice(colonIdx + 1);

      let propMap = doc.properties.get(itemId);
      if (!propMap) {
        propMap = new Map();
        doc.properties.set(itemId, propMap);
      }

      const existing = propMap.get(prop);
      const newReg = createLWWRegister(
        op.value as number | string,
        op.peerId,
        op.timestamp,
      );
      if (existing) {
        propMap.set(prop, lwwMerge(existing, newReg));
      } else {
        propMap.set(prop, newReg);
      }
      break;
    }
  }

  // Advance counter to at least the op timestamp
  if (op.timestamp > doc.counter) {
    doc.counter = op.timestamp;
  }
}
