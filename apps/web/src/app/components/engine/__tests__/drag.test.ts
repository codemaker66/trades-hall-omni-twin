import { describe, it, expect } from 'vitest'
import {
    snapToGrid,
    snapChairPosition,
    getChairSpacing,
    computeGridPositions,
    boxesOverlap,
} from '../drag'
import * as THREE from 'three'

describe('snapToGrid', () => {
    it('snaps to nearest grid point', () => {
        expect(snapToGrid(1.3, 2.7, 0.5)).toEqual({ x: 1.5, z: 2.5 })
    })

    it('snaps exact values unchanged', () => {
        expect(snapToGrid(2.0, 3.0, 0.5)).toEqual({ x: 2, z: 3 })
    })

    it('snaps negative values', () => {
        expect(snapToGrid(-1.3, -2.7, 0.5)).toEqual({ x: -1.5, z: -2.5 })
    })

    it('handles grid size 1', () => {
        expect(snapToGrid(1.6, 2.4, 1)).toEqual({ x: 2, z: 2 })
    })
})

describe('snapChairPosition', () => {
    it('uses minimum 0.6 spacing when snapping disabled', () => {
        const result = snapChairPosition(0.4, 0.4, 0.5, false)
        expect(result.x).toBeCloseTo(0.6, 5)
        expect(result.z).toBeCloseTo(0.6, 5)
    })

    it('uses grid when grid is larger than 0.6', () => {
        const result = snapChairPosition(0.8, 0.8, 1.0, true)
        expect(result).toEqual({ x: 1, z: 1 })
    })

    it('uses 0.6 minimum when grid is smaller', () => {
        const result = snapChairPosition(0.4, 0.4, 0.1, true)
        expect(result.x).toBeCloseTo(0.6, 5)
        expect(result.z).toBeCloseTo(0.6, 5)
    })
})

describe('getChairSpacing', () => {
    it('returns 0.6 when snapping disabled', () => {
        expect(getChairSpacing(0.5, false)).toBe(0.6)
    })

    it('returns grid when grid >= 0.6', () => {
        expect(getChairSpacing(1.0, true)).toBe(1.0)
    })

    it('returns 0.6 when grid < 0.6', () => {
        expect(getChairSpacing(0.1, true)).toBe(0.6)
    })
})

describe('computeGridPositions', () => {
    it('returns single point for same start and end', () => {
        const result = computeGridPositions(1, 1, 1, 1, 0.6)
        expect(result.length).toBe(1)
    })

    it('fills a rectangular area', () => {
        const result = computeGridPositions(0, 0, 1.2, 0.6, 0.6)
        // 0, 0.6, 1.2 along X (3 points) x 0, 0.6 along Z (2 points) = 6
        expect(result.length).toBe(6)
    })

    it('handles reversed start/end', () => {
        const a = computeGridPositions(0, 0, 1.2, 0.6, 0.6)
        const b = computeGridPositions(1.2, 0.6, 0, 0, 0.6)
        expect(a.length).toBe(b.length)
    })
})

describe('boxesOverlap', () => {
    it('detects overlapping boxes', () => {
        const a = new THREE.Box3(new THREE.Vector3(0, 0, 0), new THREE.Vector3(2, 2, 2))
        const b = new THREE.Box3(new THREE.Vector3(1, 1, 1), new THREE.Vector3(3, 3, 3))
        expect(boxesOverlap(a, b)).toBe(true)
    })

    it('detects non-overlapping boxes', () => {
        const a = new THREE.Box3(new THREE.Vector3(0, 0, 0), new THREE.Vector3(1, 1, 1))
        const b = new THREE.Box3(new THREE.Vector3(2, 2, 2), new THREE.Vector3(3, 3, 3))
        expect(boxesOverlap(a, b)).toBe(false)
    })

    it('respects epsilon for near-touching boxes', () => {
        const a = new THREE.Box3(new THREE.Vector3(0, 0, 0), new THREE.Vector3(1, 1, 1))
        const b = new THREE.Box3(new THREE.Vector3(1.0005, 0, 0), new THREE.Vector3(2, 1, 1))
        // With default eps=0.001, these should NOT overlap
        expect(boxesOverlap(a, b)).toBe(false)
    })
})
