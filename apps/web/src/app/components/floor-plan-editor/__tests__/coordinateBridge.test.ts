import { describe, it, expect } from 'vitest'
import {
  FT_TO_M,
  M_TO_FT,
  categoryTo3DType,
  floorPlanItemTo3D,
  floorPlanTo3D,
  floorDimensions3D,
} from '../coordinateBridge'
import type { FloorPlanItem } from '../store'
import { getCameraPresets } from '../Scene3DPreview'

// ─── Constants ───────────────────────────────────────────────────────────────

describe('coordinate constants', () => {
  it('FT_TO_M is approximately 0.3048', () => {
    expect(FT_TO_M).toBeCloseTo(0.3048, 4)
  })

  it('M_TO_FT is the inverse of FT_TO_M', () => {
    expect(FT_TO_M * M_TO_FT).toBeCloseTo(1, 10)
  })

  it('10 feet equals approximately 3.048 meters', () => {
    expect(10 * FT_TO_M).toBeCloseTo(3.048, 3)
  })
})

// ─── Category mapping ────────────────────────────────────────────────────────

describe('categoryTo3DType', () => {
  const makeItem = (overrides: Partial<FloorPlanItem>): FloorPlanItem => ({
    id: 'test',
    name: 'Test',
    category: 'chair',
    x: 0,
    y: 0,
    widthFt: 1.5,
    depthFt: 1.5,
    rotation: 0,
    locked: false,
    ...overrides,
  })

  it('maps chair to "chair"', () => {
    expect(categoryTo3DType(makeItem({ category: 'chair' }))).toBe('chair')
  })

  it('maps round table (equal width/depth) to "round-table"', () => {
    expect(categoryTo3DType(makeItem({ category: 'table', widthFt: 6, depthFt: 6 }))).toBe('round-table')
  })

  it('maps rectangular table to "trestle-table"', () => {
    expect(categoryTo3DType(makeItem({ category: 'table', widthFt: 8, depthFt: 3 }))).toBe('trestle-table')
  })

  it('maps nearly-square table as round (within 0.5ft tolerance)', () => {
    expect(categoryTo3DType(makeItem({ category: 'table', widthFt: 5, depthFt: 5.4 }))).toBe('round-table')
  })

  it('maps stage to "platform"', () => {
    expect(categoryTo3DType(makeItem({ category: 'stage' }))).toBe('platform')
  })

  it('returns null for decor', () => {
    expect(categoryTo3DType(makeItem({ category: 'decor' }))).toBeNull()
  })

  it('returns null for equipment', () => {
    expect(categoryTo3DType(makeItem({ category: 'equipment' }))).toBeNull()
  })
})

// ─── Item conversion ─────────────────────────────────────────────────────────

describe('floorPlanItemTo3D', () => {
  const item: FloorPlanItem = {
    id: 'chair-1',
    name: 'Chair',
    category: 'chair',
    x: 40,
    y: 25,
    widthFt: 1.5,
    depthFt: 1.5,
    rotation: 0,
    locked: false,
  }

  it('centers origin: item at center of plan maps to (0, 0, 0)', () => {
    const result = floorPlanItemTo3D(item, 80, 50)
    expect(result.position[0]).toBeCloseTo(0, 5)
    expect(result.position[1]).toBe(0)
    expect(result.position[2]).toBeCloseTo(0, 5)
  })

  it('converts feet to meters for position', () => {
    const offCenter: FloorPlanItem = { ...item, x: 50, y: 30 }
    const result = floorPlanItemTo3D(offCenter, 80, 50)
    // x offset: (50 - 40) = 10ft = 3.048m
    expect(result.position[0]).toBeCloseTo(10 * FT_TO_M, 3)
    // z offset: (30 - 25) = 5ft = 1.524m
    expect(result.position[2]).toBeCloseTo(5 * FT_TO_M, 3)
  })

  it('converts 2D rotation (degrees CW) to 3D y-rotation (radians CCW)', () => {
    const rotated = { ...item, rotation: 90 }
    const result = floorPlanItemTo3D(rotated, 80, 50)
    expect(result.rotation[0]).toBe(0) // no x rotation
    expect(result.rotation[1]).toBeCloseTo(-Math.PI / 2, 5) // -90° in radians
    expect(result.rotation[2]).toBe(0) // no z rotation
  })

  it('converts 45° rotation correctly', () => {
    const rotated = { ...item, rotation: 45 }
    const result = floorPlanItemTo3D(rotated, 80, 50)
    expect(result.rotation[1]).toBeCloseTo(-Math.PI / 4, 5)
  })

  it('preserves item id and name', () => {
    const result = floorPlanItemTo3D(item, 80, 50)
    expect(result.id).toBe('chair-1')
    expect(result.name).toBe('Chair')
  })

  it('computes width and depth in meters', () => {
    const result = floorPlanItemTo3D(item, 80, 50)
    expect(result.widthM).toBeCloseTo(1.5 * FT_TO_M, 5)
    expect(result.depthM).toBeCloseTo(1.5 * FT_TO_M, 5)
  })

  it('maps item at top-left corner (0,0) to negative coordinates', () => {
    const topLeft = { ...item, x: 0, y: 0 }
    const result = floorPlanItemTo3D(topLeft, 80, 50)
    expect(result.position[0]).toBeCloseTo(-40 * FT_TO_M, 3)
    expect(result.position[2]).toBeCloseTo(-25 * FT_TO_M, 3)
  })
})

// ─── Batch conversion ────────────────────────────────────────────────────────

describe('floorPlanTo3D', () => {
  it('converts all items', () => {
    const items: FloorPlanItem[] = [
      { id: '1', name: 'Chair', category: 'chair', x: 10, y: 10, widthFt: 1.5, depthFt: 1.5, rotation: 0, locked: false },
      { id: '2', name: 'Table', category: 'table', x: 20, y: 20, widthFt: 6, depthFt: 6, rotation: 0, locked: false },
    ]
    const result = floorPlanTo3D(items, 80, 50)
    expect(result).toHaveLength(2)
    expect(result[0]!.id).toBe('1')
    expect(result[1]!.id).toBe('2')
  })

  it('returns empty array for empty input', () => {
    expect(floorPlanTo3D([], 80, 50)).toHaveLength(0)
  })
})

// ─── Floor dimensions ────────────────────────────────────────────────────────

describe('floorDimensions3D', () => {
  it('converts plan dimensions from feet to meters', () => {
    const result = floorDimensions3D(80, 50)
    expect(result.widthM).toBeCloseTo(80 * FT_TO_M, 3)
    expect(result.depthM).toBeCloseTo(50 * FT_TO_M, 3)
  })
})

// ─── Camera presets ──────────────────────────────────────────────────────────

describe('getCameraPresets', () => {
  const presets = getCameraPresets(24.384, 15.24) // ~80ft x 50ft

  it('returns 4 presets', () => {
    expect(presets).toHaveLength(4)
  })

  it('includes top-down, entrance, stage, and perspective', () => {
    const ids = presets.map((p) => p.id)
    expect(ids).toContain('top-down')
    expect(ids).toContain('entrance')
    expect(ids).toContain('stage')
    expect(ids).toContain('perspective')
  })

  it('top-down camera is high above, looking straight down', () => {
    const topDown = presets.find((p) => p.id === 'top-down')!
    expect(topDown.position[1]).toBeGreaterThan(20) // high up
    expect(topDown.target).toEqual([0, 0, 0])
  })

  it('entrance camera is at ground level, looking inward', () => {
    const entrance = presets.find((p) => p.id === 'entrance')!
    expect(entrance.position[1]).toBeCloseTo(2) // eye height
    expect(entrance.position[2]).toBeGreaterThan(0) // in front
  })

  it('stage camera faces opposite direction from entrance', () => {
    const entrance = presets.find((p) => p.id === 'entrance')!
    const stage = presets.find((p) => p.id === 'stage')!
    expect(Math.sign(stage.position[2])).toBe(-Math.sign(entrance.position[2]))
  })

  it('all presets have valid position and target arrays', () => {
    for (const preset of presets) {
      expect(preset.position).toHaveLength(3)
      expect(preset.target).toHaveLength(3)
      for (const v of [...preset.position, ...preset.target]) {
        expect(typeof v).toBe('number')
        expect(Number.isFinite(v)).toBe(true)
      }
    }
  })
})
