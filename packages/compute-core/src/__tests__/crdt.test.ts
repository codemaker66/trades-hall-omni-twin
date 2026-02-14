import { describe, it, expect } from 'vitest';
import {
  createVectorClock,
  vcIncrement,
  vcMerge,
  vcCompare,
  vcHappensBefore,
  vcGet,
  vcClone,
  createLWWRegister,
  lwwSet,
  lwwGet,
  lwwMerge,
  createORSet,
  orSetAdd,
  orSetRemove,
  orSetContains,
  orSetElements,
  orSetMerge,
  createLayoutDoc,
  layoutDocAddItem,
  layoutDocMoveItem,
  layoutDocRemoveItem,
  layoutDocGetItem,
  layoutDocGetAll,
  layoutDocMerge,
  layoutDocApplyOp,
} from '../crdt/index.js';

// ---------------------------------------------------------------------------
// Vector Clock
// ---------------------------------------------------------------------------

describe('VectorClock', () => {
  it('increment increases counter for the specified peer', () => {
    const vc = createVectorClock();
    vcIncrement(vc, 'peerA');
    expect(vcGet(vc, 'peerA')).toBe(1);
    vcIncrement(vc, 'peerA');
    expect(vcGet(vc, 'peerA')).toBe(2);
  });

  it('vcGet returns 0 for an unknown peer', () => {
    const vc = createVectorClock();
    expect(vcGet(vc, 'unknown')).toBe(0);
  });

  it('merge takes element-wise maximum of two clocks', () => {
    const a = createVectorClock();
    vcIncrement(a, 'peerA');
    vcIncrement(a, 'peerA');
    vcIncrement(a, 'peerB');

    const b = createVectorClock();
    vcIncrement(b, 'peerA');
    vcIncrement(b, 'peerB');
    vcIncrement(b, 'peerB');
    vcIncrement(b, 'peerB');

    const merged = vcMerge(a, b);
    expect(vcGet(merged, 'peerA')).toBe(2); // max(2, 1)
    expect(vcGet(merged, 'peerB')).toBe(3); // max(1, 3)
  });

  it('happensBefore detects causal ordering', () => {
    const a = createVectorClock();
    vcIncrement(a, 'peerA');

    const b = createVectorClock();
    vcIncrement(b, 'peerA');
    vcIncrement(b, 'peerB');

    // a = {peerA:1}, b = {peerA:1, peerB:1} => a happens before b
    expect(vcHappensBefore(a, b)).toBe(true);
    expect(vcHappensBefore(b, a)).toBe(false);
  });

  it('concurrent events are not causally ordered', () => {
    const a = createVectorClock();
    vcIncrement(a, 'peerA');

    const b = createVectorClock();
    vcIncrement(b, 'peerB');

    expect(vcCompare(a, b)).toBe('concurrent');
    expect(vcHappensBefore(a, b)).toBe(false);
    expect(vcHappensBefore(b, a)).toBe(false);
  });

  it('identical clocks compare as equal', () => {
    const a = createVectorClock();
    vcIncrement(a, 'peerA');
    vcIncrement(a, 'peerB');

    const b = createVectorClock();
    vcIncrement(b, 'peerA');
    vcIncrement(b, 'peerB');

    expect(vcCompare(a, b)).toBe('equal');
  });

  it('clone produces an independent copy', () => {
    const original = createVectorClock();
    vcIncrement(original, 'peerA');

    const cloned = vcClone(original);
    vcIncrement(cloned, 'peerA');

    expect(vcGet(original, 'peerA')).toBe(1);
    expect(vcGet(cloned, 'peerA')).toBe(2);
  });

  it('compare returns after when a dominates b', () => {
    const a = createVectorClock();
    vcIncrement(a, 'peerA');
    vcIncrement(a, 'peerA');
    vcIncrement(a, 'peerB');

    const b = createVectorClock();
    vcIncrement(b, 'peerA');

    expect(vcCompare(a, b)).toBe('after');
  });
});

// ---------------------------------------------------------------------------
// LWW Register
// ---------------------------------------------------------------------------

describe('LWWRegister', () => {
  it('stores and retrieves a value', () => {
    const reg = createLWWRegister('hello', 'peerA', 100);
    expect(lwwGet(reg)).toBe('hello');
  });

  it('later timestamp wins in lwwSet', () => {
    const reg = createLWWRegister('old', 'peerA', 100);
    const updated = lwwSet(reg, 'new', 'peerB', 200);
    expect(lwwGet(updated)).toBe('new');
  });

  it('earlier timestamp is rejected in lwwSet', () => {
    const reg = createLWWRegister('current', 'peerA', 200);
    const result = lwwSet(reg, 'outdated', 'peerB', 100);
    expect(lwwGet(result)).toBe('current');
    // Should return the same reference
    expect(result).toBe(reg);
  });

  it('same timestamp breaks tie by higher peerId', () => {
    const reg = createLWWRegister('fromA', 'peerA', 100);
    // peerB > peerA lexicographically, so peerB wins the tie
    const updated = lwwSet(reg, 'fromB', 'peerB', 100);
    expect(lwwGet(updated)).toBe('fromB');
  });

  it('same timestamp with lower peerId is rejected', () => {
    const reg = createLWWRegister('fromZ', 'peerZ', 100);
    // peerA < peerZ lexicographically, so peerA loses
    const result = lwwSet(reg, 'fromA', 'peerA', 100);
    expect(lwwGet(result)).toBe('fromZ');
  });

  it('merge picks the register with higher timestamp', () => {
    const a = createLWWRegister('early', 'peerA', 100);
    const b = createLWWRegister('late', 'peerB', 200);
    const merged = lwwMerge(a, b);
    expect(lwwGet(merged)).toBe('late');
  });

  it('merge with equal timestamps picks higher peerId', () => {
    const a = createLWWRegister('fromA', 'peerA', 100);
    const b = createLWWRegister('fromB', 'peerB', 100);
    const merged = lwwMerge(a, b);
    expect(lwwGet(merged)).toBe('fromB');
  });

  it('merge is commutative', () => {
    const a = createLWWRegister(10, 'peerA', 50);
    const b = createLWWRegister(20, 'peerB', 100);
    expect(lwwGet(lwwMerge(a, b))).toBe(lwwGet(lwwMerge(b, a)));
  });
});

// ---------------------------------------------------------------------------
// OR-Set
// ---------------------------------------------------------------------------

describe('ORSet', () => {
  it('add then contains returns true', () => {
    const set = createORSet<string>();
    orSetAdd(set, 'apple', 'peerA');
    expect(orSetContains(set, 'apple')).toBe(true);
  });

  it('remove then contains returns false', () => {
    const set = createORSet<string>();
    orSetAdd(set, 'apple', 'peerA');
    orSetRemove(set, 'apple');
    expect(orSetContains(set, 'apple')).toBe(false);
  });

  it('remove only tombstones existing tags; concurrent add wins', () => {
    const set = createORSet<string>();
    orSetAdd(set, 'apple', 'peerA');
    orSetRemove(set, 'apple');
    // Concurrent add after remove: the new tag survives
    orSetAdd(set, 'apple', 'peerB');
    expect(orSetContains(set, 'apple')).toBe(true);
  });

  it('elements returns all currently alive values', () => {
    const set = createORSet<number>();
    orSetAdd(set, 1, 'p1');
    orSetAdd(set, 2, 'p1');
    orSetAdd(set, 3, 'p1');
    orSetRemove(set, 2);

    const elems = orSetElements(set);
    expect(elems).toContain(1);
    expect(elems).toContain(3);
    expect(elems).not.toContain(2);
    expect(elems.length).toBe(2);
  });

  it('merge combines elements from both sets', () => {
    const a = createORSet<string>();
    orSetAdd(a, 'apple', 'peerA');

    const b = createORSet<string>();
    orSetAdd(b, 'banana', 'peerB');

    const merged = orSetMerge(a, b);
    expect(orSetContains(merged, 'apple')).toBe(true);
    expect(orSetContains(merged, 'banana')).toBe(true);
  });

  it('merge preserves tombstones from both sides', () => {
    const a = createORSet<string>();
    const tag = orSetAdd(a, 'apple', 'peerA');
    orSetRemove(a, 'apple');

    const b = createORSet<string>();
    // b has same element added with same tag via direct mutation for testing
    orSetAdd(b, 'banana', 'peerB');

    const merged = orSetMerge(a, b);
    // apple was removed in a, so it should not be present in merged
    expect(orSetContains(merged, 'apple')).toBe(false);
    expect(orSetContains(merged, 'banana')).toBe(true);
  });

  it('contains returns false for an element never added', () => {
    const set = createORSet<string>();
    expect(orSetContains(set, 'nonexistent')).toBe(false);
  });

  it('orSetAdd returns a unique tag string', () => {
    const set = createORSet<string>();
    const tag1 = orSetAdd(set, 'x', 'peerA');
    const tag2 = orSetAdd(set, 'y', 'peerA');
    expect(typeof tag1).toBe('string');
    expect(tag1).not.toBe(tag2);
  });
});

// ---------------------------------------------------------------------------
// Layout Document
// ---------------------------------------------------------------------------

describe('LayoutDoc', () => {
  const sampleItem = {
    type: 'chair',
    x: 10,
    y: 20,
    width: 1,
    depth: 1,
    rotation: 0,
  };

  it('add then get returns the item with correct properties', () => {
    const doc = createLayoutDoc('peerA');
    const id = layoutDocAddItem(doc, sampleItem);
    const item = layoutDocGetItem(doc, id);

    expect(item).not.toBeNull();
    expect(item!.type).toBe('chair');
    expect(item!.x).toBe(10);
    expect(item!.y).toBe(20);
    expect(item!.width).toBe(1);
    expect(item!.depth).toBe(1);
    expect(item!.rotation).toBe(0);
  });

  it('move updates x and y coordinates', () => {
    const doc = createLayoutDoc('peerA');
    const id = layoutDocAddItem(doc, sampleItem);

    const moved = layoutDocMoveItem(doc, id, 50, 60);
    expect(moved).toBe(true);

    const item = layoutDocGetItem(doc, id);
    expect(item!.x).toBe(50);
    expect(item!.y).toBe(60);
  });

  it('move returns false for non-existent item', () => {
    const doc = createLayoutDoc('peerA');
    expect(layoutDocMoveItem(doc, 'nonexistent', 0, 0)).toBe(false);
  });

  it('remove deletes the item', () => {
    const doc = createLayoutDoc('peerA');
    const id = layoutDocAddItem(doc, sampleItem);
    expect(layoutDocRemoveItem(doc, id)).toBe(true);
    expect(layoutDocGetItem(doc, id)).toBeNull();
  });

  it('remove returns false for non-existent item', () => {
    const doc = createLayoutDoc('peerA');
    expect(layoutDocRemoveItem(doc, 'nonexistent')).toBe(false);
  });

  it('getAll returns all items currently in the document', () => {
    const doc = createLayoutDoc('peerA');
    layoutDocAddItem(doc, { ...sampleItem, type: 'chair' });
    layoutDocAddItem(doc, { ...sampleItem, type: 'table' });
    layoutDocAddItem(doc, { ...sampleItem, type: 'podium' });

    const all = layoutDocGetAll(doc);
    expect(all.length).toBe(3);

    const types = all.map((i) => i.type);
    expect(types).toContain('chair');
    expect(types).toContain('table');
    expect(types).toContain('podium');
  });

  it('merge combines docs from different peers', () => {
    const docA = createLayoutDoc('peerA');
    const idA = layoutDocAddItem(docA, { ...sampleItem, type: 'chair' });

    const docB = createLayoutDoc('peerB');
    const idB = layoutDocAddItem(docB, { ...sampleItem, type: 'table' });

    const merged = layoutDocMerge(docA, docB);
    const all = layoutDocGetAll(merged);

    expect(all.length).toBe(2);
    const types = all.map((i) => i.type);
    expect(types).toContain('chair');
    expect(types).toContain('table');
  });

  it('applyOp with add operation inserts an item', () => {
    const doc = createLayoutDoc('peerA');
    layoutDocApplyOp(doc, {
      type: 'add',
      peerId: 'peerB',
      timestamp: 1,
      key: 'item-ext-1',
      value: {
        type: 'round-table',
        x: 100,
        y: 200,
        width: 2,
        depth: 2,
        rotation: 45,
      },
    });

    const item = layoutDocGetItem(doc, 'item-ext-1');
    expect(item).not.toBeNull();
    expect(item!.type).toBe('round-table');
    expect(item!.x).toBe(100);
    expect(item!.y).toBe(200);
    expect(item!.rotation).toBe(45);
  });

  it('applyOp with move operation updates coordinates', () => {
    const doc = createLayoutDoc('peerA');
    const id = layoutDocAddItem(doc, sampleItem);

    layoutDocApplyOp(doc, {
      type: 'move',
      peerId: 'peerB',
      timestamp: 9999,
      key: id,
      value: { x: 300, y: 400 },
    });

    const item = layoutDocGetItem(doc, id);
    expect(item!.x).toBe(300);
    expect(item!.y).toBe(400);
  });
});
