import { z } from 'zod'

export const createCatalogItemSchema = z.object({
  name: z.string().min(1, 'Name is required').max(200),
  category: z.enum(['table', 'chair', 'stage', 'decor', 'equipment']),
  modelUrl: z.string().url().optional(),
  thumbnailUrl: z.string().url().optional(),
  widthFt: z.number().min(0.1).max(100),
  depthFt: z.number().min(0.1).max(100),
  heightFt: z.number().min(0.1).max(100),
  capacity: z.number().int().min(0).optional(),
  stackable: z.boolean().optional(),
})

export const updateCatalogItemSchema = createCatalogItemSchema.partial()

export type CreateCatalogItemInput = z.infer<typeof createCatalogItemSchema>
export type UpdateCatalogItemInput = z.infer<typeof updateCatalogItemSchema>
