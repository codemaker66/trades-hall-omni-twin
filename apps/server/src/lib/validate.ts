import type { Context } from 'hono'
import { z } from 'zod'

/** Parse and validate request body with a Zod schema. Returns 400 on failure. */
export async function parseBody<T extends z.ZodTypeAny>(
  c: Context,
  schema: T,
): Promise<z.infer<T> | Response> {
  let body: unknown
  try {
    body = await c.req.json()
  } catch {
    return c.json({ error: 'Invalid JSON body.' }, 400)
  }

  const result = schema.safeParse(body)
  if (!result.success) {
    const errors = result.error.flatten().fieldErrors
    return c.json({ error: 'Validation failed.', fields: errors }, 400)
  }

  return result.data as z.infer<T>
}

/** Check if a parseBody result is a Response (validation error). */
export function isResponse(value: unknown): value is Response {
  return value instanceof Response
}
