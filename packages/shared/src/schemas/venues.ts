import { z } from 'zod'

export const createVenueSchema = z.object({
  name: z.string().min(1, 'Name is required').max(200),
  slug: z.string().min(1).max(100).regex(/^[a-z0-9-]+$/, 'Slug must be lowercase alphanumeric with hyphens'),
  description: z.string().max(2000).optional(),
  address: z.string().max(500).optional(),
  latitude: z.number().min(-90).max(90).optional(),
  longitude: z.number().min(-180).max(180).optional(),
  capacity: z.number().int().min(0).optional(),
  squareFootage: z.number().min(0).optional(),
  venueType: z.enum(['ballroom', 'conference', 'outdoor', 'theater', 'restaurant', 'warehouse', 'other']).optional(),
  pricingModel: z.enum(['hourly', 'daily', 'flat', 'custom']).optional(),
  basePrice: z.number().min(0).optional(),
  amenities: z.array(z.string()).optional(),
  images: z.array(z.string().url()).optional(),
})

export const updateVenueSchema = createVenueSchema.partial()

export const venueIdParam = z.object({
  venueId: z.string().uuid('Invalid venue ID'),
})

export type CreateVenueInput = z.infer<typeof createVenueSchema>
export type UpdateVenueInput = z.infer<typeof updateVenueSchema>
