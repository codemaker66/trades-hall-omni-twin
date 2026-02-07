import Redis from 'ioredis'

const REDIS_URL = process.env['REDIS_URL'] ?? 'redis://localhost:6379'

/** Shared Redis client instance. */
export const redis = new Redis(REDIS_URL, {
  maxRetriesPerRequest: 3,
  lazyConnect: true,
})

/** Check if Redis is connected and responding. */
export async function redisHealthCheck(): Promise<boolean> {
  try {
    const pong = await redis.ping()
    return pong === 'PONG'
  } catch {
    return false
  }
}
