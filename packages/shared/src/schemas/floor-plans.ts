import { z } from 'zod'

export const createFloorPlanSchema = z.object({
  name: z.string().min(1, 'Name is required').max(200),
  widthFt: z.number().min(1).max(10000),
  heightFt: z.number().min(1).max(10000),
  backgroundImageUrl: z.string().url().optional(),
  objects: z.array(z.record(z.unknown())).optional(),
  isTemplate: z.boolean().optional(),
})

export const updateFloorPlanSchema = createFloorPlanSchema.partial()

export type CreateFloorPlanInput = z.infer<typeof createFloorPlanSchema>
export type UpdateFloorPlanInput = z.infer<typeof updateFloorPlanSchema>
