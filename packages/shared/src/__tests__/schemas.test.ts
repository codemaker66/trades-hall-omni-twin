import { describe, it, expect } from 'vitest'
import {
  createVenueSchema,
  updateVenueSchema,
  venueIdParam,
  createFloorPlanSchema,
  updateFloorPlanSchema,
  createOccasionSchema,
  updateOccasionSchema,
  updateOccasionStatusSchema,
  createCatalogItemSchema,
  updateCatalogItemSchema,
} from '../schemas/index'

// ─── Venue Schemas ──────────────────────────────────────────────────────────

describe('createVenueSchema', () => {
  it('accepts valid venue data', () => {
    const result = createVenueSchema.safeParse({
      name: 'Trades Hall',
      slug: 'trades-hall',
      description: 'A beautiful venue',
      venueType: 'ballroom',
    })
    expect(result.success).toBe(true)
  })

  it('requires name', () => {
    const result = createVenueSchema.safeParse({ slug: 'test' })
    expect(result.success).toBe(false)
  })

  it('requires slug', () => {
    const result = createVenueSchema.safeParse({ name: 'Test' })
    expect(result.success).toBe(false)
  })

  it('rejects invalid slug format', () => {
    const result = createVenueSchema.safeParse({
      name: 'Test',
      slug: 'Invalid Slug!',
    })
    expect(result.success).toBe(false)
  })

  it('accepts valid lat/lng', () => {
    const result = createVenueSchema.safeParse({
      name: 'Test',
      slug: 'test',
      latitude: -37.8136,
      longitude: 144.9631,
    })
    expect(result.success).toBe(true)
  })

  it('rejects out-of-range latitude', () => {
    const result = createVenueSchema.safeParse({
      name: 'Test',
      slug: 'test',
      latitude: 91,
    })
    expect(result.success).toBe(false)
  })

  it('rejects invalid venue type', () => {
    const result = createVenueSchema.safeParse({
      name: 'Test',
      slug: 'test',
      venueType: 'spaceship',
    })
    expect(result.success).toBe(false)
  })
})

describe('updateVenueSchema', () => {
  it('accepts partial updates', () => {
    const result = updateVenueSchema.safeParse({ name: 'New Name' })
    expect(result.success).toBe(true)
  })

  it('accepts empty update', () => {
    const result = updateVenueSchema.safeParse({})
    expect(result.success).toBe(true)
  })
})

describe('venueIdParam', () => {
  it('accepts valid UUID', () => {
    const result = venueIdParam.safeParse({
      venueId: '550e8400-e29b-41d4-a716-446655440000',
    })
    expect(result.success).toBe(true)
  })

  it('rejects non-UUID', () => {
    const result = venueIdParam.safeParse({ venueId: 'not-a-uuid' })
    expect(result.success).toBe(false)
  })
})

// ─── Floor Plan Schemas ─────────────────────────────────────────────────────

describe('createFloorPlanSchema', () => {
  it('accepts valid floor plan', () => {
    const result = createFloorPlanSchema.safeParse({
      name: 'Main Hall',
      widthFt: 100,
      heightFt: 60,
    })
    expect(result.success).toBe(true)
  })

  it('requires dimensions', () => {
    const result = createFloorPlanSchema.safeParse({ name: 'Test' })
    expect(result.success).toBe(false)
  })

  it('rejects zero dimensions', () => {
    const result = createFloorPlanSchema.safeParse({
      name: 'Test',
      widthFt: 0,
      heightFt: 50,
    })
    expect(result.success).toBe(false)
  })

  it('accepts objects array', () => {
    const result = createFloorPlanSchema.safeParse({
      name: 'Test',
      widthFt: 50,
      heightFt: 50,
      objects: [{ type: 'chair', x: 10, y: 20 }],
    })
    expect(result.success).toBe(true)
  })
})

describe('updateFloorPlanSchema', () => {
  it('accepts partial update', () => {
    const result = updateFloorPlanSchema.safeParse({ name: 'Renamed' })
    expect(result.success).toBe(true)
  })
})

// ─── Occasion Schemas ───────────────────────────────────────────────────────

describe('createOccasionSchema', () => {
  const validOccasion = {
    name: 'Annual Gala',
    type: 'gala' as const,
    dateStart: '2026-06-15T18:00:00Z',
    dateEnd: '2026-06-15T23:00:00Z',
    guestCount: 200,
  }

  it('accepts valid occasion', () => {
    const result = createOccasionSchema.safeParse(validOccasion)
    expect(result.success).toBe(true)
  })

  it('requires name', () => {
    const { name: _, ...rest } = validOccasion
    const result = createOccasionSchema.safeParse(rest)
    expect(result.success).toBe(false)
  })

  it('requires valid event type', () => {
    const result = createOccasionSchema.safeParse({
      ...validOccasion,
      type: 'rave',
    })
    expect(result.success).toBe(false)
  })

  it('requires positive guest count', () => {
    const result = createOccasionSchema.safeParse({
      ...validOccasion,
      guestCount: 0,
    })
    expect(result.success).toBe(false)
  })

  it('accepts optional fields', () => {
    const result = createOccasionSchema.safeParse({
      ...validOccasion,
      budget: 50000,
      notes: 'VIP event',
      setupTime: '2026-06-15T14:00:00Z',
    })
    expect(result.success).toBe(true)
  })
})

describe('updateOccasionStatusSchema', () => {
  it('accepts valid status', () => {
    const result = updateOccasionStatusSchema.safeParse({ status: 'confirmed' })
    expect(result.success).toBe(true)
  })

  it('rejects invalid status', () => {
    const result = updateOccasionStatusSchema.safeParse({ status: 'maybe' })
    expect(result.success).toBe(false)
  })
})

// ─── Catalog Schemas ────────────────────────────────────────────────────────

describe('createCatalogItemSchema', () => {
  const validItem = {
    name: 'Round Table 6ft',
    category: 'table' as const,
    widthFt: 6,
    depthFt: 6,
    heightFt: 2.5,
  }

  it('accepts valid catalog item', () => {
    const result = createCatalogItemSchema.safeParse(validItem)
    expect(result.success).toBe(true)
  })

  it('requires name', () => {
    const { name: _, ...rest } = validItem
    const result = createCatalogItemSchema.safeParse(rest)
    expect(result.success).toBe(false)
  })

  it('requires valid category', () => {
    const result = createCatalogItemSchema.safeParse({
      ...validItem,
      category: 'spaceship',
    })
    expect(result.success).toBe(false)
  })

  it('rejects zero dimensions', () => {
    const result = createCatalogItemSchema.safeParse({
      ...validItem,
      widthFt: 0,
    })
    expect(result.success).toBe(false)
  })

  it('accepts optional fields', () => {
    const result = createCatalogItemSchema.safeParse({
      ...validItem,
      capacity: 8,
      stackable: true,
      modelUrl: 'https://cdn.example.com/table.glb',
    })
    expect(result.success).toBe(true)
  })
})

describe('updateCatalogItemSchema', () => {
  it('accepts partial update', () => {
    const result = updateCatalogItemSchema.safeParse({ name: 'Renamed Table' })
    expect(result.success).toBe(true)
  })
})
