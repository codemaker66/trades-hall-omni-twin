import { redis } from './client'

/** Type-safe cache abstraction over Redis. */
export const cache = {
  /** Get a JSON value by key. Returns null if not found. */
  async get<T>(key: string): Promise<T | null> {
    const raw = await redis.get(key)
    if (raw === null) return null
    return JSON.parse(raw) as T
  },

  /** Set a JSON value with optional TTL in seconds. */
  async set<T>(key: string, value: T, ttlSeconds?: number): Promise<void> {
    const json = JSON.stringify(value)
    if (ttlSeconds !== undefined) {
      await redis.set(key, json, 'EX', ttlSeconds)
    } else {
      await redis.set(key, json)
    }
  },

  /** Set only if key does not exist. Returns true if set, false if key already exists. */
  async setIfAbsent<T>(key: string, value: T, ttlSeconds?: number): Promise<boolean> {
    const json = JSON.stringify(value)
    if (ttlSeconds !== undefined) {
      const result = await redis.set(key, json, 'EX', ttlSeconds, 'NX')
      return result === 'OK'
    }
    const result = await redis.set(key, json, 'NX')
    return result === 'OK'
  },

  /** Delete a key. */
  async del(key: string): Promise<void> {
    await redis.del(key)
  },

  /** Check if a key exists. */
  async exists(key: string): Promise<boolean> {
    const result = await redis.exists(key)
    return result === 1
  },

  /** Publish a message to a channel. */
  async publish(channel: string, message: unknown): Promise<void> {
    await redis.publish(channel, JSON.stringify(message))
  },
}
