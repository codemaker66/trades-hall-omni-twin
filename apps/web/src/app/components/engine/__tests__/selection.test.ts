import { describe, it, expect } from 'vitest'
import {
    applySelectionModifier,
    resolveClickTargetIds,
    findDragAnchor,
} from '../selection'

describe('applySelectionModifier', () => {
    it('replaces selection when not multi-select', () => {
        const result = applySelectionModifier(['a', 'b'], ['c'], false)
        expect(result).toEqual(['c'])
    })

    it('adds to selection with multi-select', () => {
        const result = applySelectionModifier(['a'], ['b'], true)
        expect(result).toEqual(['a', 'b'])
    })

    it('removes from selection when all targets already selected', () => {
        const result = applySelectionModifier(['a', 'b', 'c'], ['b'], true)
        expect(result).toEqual(['a', 'c'])
    })

    it('adds multiple targets with multi-select', () => {
        const result = applySelectionModifier(['a'], ['b', 'c'], true)
        expect(result).toEqual(['a', 'b', 'c'])
    })

    it('deduplicates when adding existing items', () => {
        const result = applySelectionModifier(['a', 'b'], ['b', 'c'], true)
        expect(result).toEqual(['a', 'b', 'c'])
    })

    it('returns empty array replacing with nothing', () => {
        const result = applySelectionModifier(['a'], [], false)
        expect(result).toEqual([])
    })
})

describe('resolveClickTargetIds', () => {
    const items = [
        { id: '1', groupId: 'g1' },
        { id: '2', groupId: 'g1' },
        { id: '3', groupId: 'g2' },
        { id: '4' },
    ]

    it('returns entire group when not ctrl-click', () => {
        const result = resolveClickTargetIds('1', 'g1', false, items)
        expect(result).toEqual(['1', '2'])
    })

    it('returns single item on ctrl-click', () => {
        const result = resolveClickTargetIds('1', 'g1', true, items)
        expect(result).toEqual(['1'])
    })

    it('returns single item when no group', () => {
        const result = resolveClickTargetIds('4', undefined, false, items)
        expect(result).toEqual(['4'])
    })
})

describe('findDragAnchor', () => {
    it('returns lowest item', () => {
        const items = [
            { id: 'a', type: 'chair', position: [0, 2, 0] as [number, number, number] },
            { id: 'b', type: 'chair', position: [0, 0, 0] as [number, number, number] },
        ]
        const offsets = new Map([['a', 0], ['b', 0]])
        expect(findDragAnchor(items, offsets).id).toBe('b')
    })

    it('prefers platform when tied', () => {
        const items = [
            { id: 'a', type: 'chair', position: [0, 0, 0] as [number, number, number] },
            { id: 'b', type: 'platform', position: [0, 0, 0] as [number, number, number] },
        ]
        const offsets = new Map([['a', 0], ['b', 0]])
        expect(findDragAnchor(items, offsets).id).toBe('b')
    })

    it('accounts for bottom offsets', () => {
        const items = [
            { id: 'a', type: 'chair', position: [0, 1, 0] as [number, number, number] },
            { id: 'b', type: 'chair', position: [0, 2, 0] as [number, number, number] },
        ]
        // a: bottom = 1 - 0 = 1, b: bottom = 2 - 1.5 = 0.5
        const offsets = new Map([['a', 0], ['b', 1.5]])
        expect(findDragAnchor(items, offsets).id).toBe('b')
    })
})
