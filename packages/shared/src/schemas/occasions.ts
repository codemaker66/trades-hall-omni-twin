import { z } from 'zod'

export const createOccasionSchema = z.object({
  name: z.string().min(1, 'Name is required').max(200),
  type: z.enum(['wedding', 'corporate', 'social', 'conference', 'gala', 'other']),
  dateStart: z.string().datetime(),
  dateEnd: z.string().datetime(),
  setupTime: z.string().datetime().optional(),
  teardownTime: z.string().datetime().optional(),
  guestCount: z.number().int().min(1),
  floorPlanId: z.string().uuid().optional(),
  budget: z.number().min(0).optional(),
  notes: z.string().max(5000).optional(),
})

export const updateOccasionSchema = createOccasionSchema.partial()

export const updateOccasionStatusSchema = z.object({
  status: z.enum(['inquiry', 'confirmed', 'completed', 'cancelled']),
})

export type CreateOccasionInput = z.infer<typeof createOccasionSchema>
export type UpdateOccasionInput = z.infer<typeof updateOccasionSchema>
