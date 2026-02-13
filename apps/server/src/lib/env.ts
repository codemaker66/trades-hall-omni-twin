/**
 * Environment variable validation — fail-fast on startup.
 *
 * Import this module early in the server entry point.
 * If a required variable is missing, the process throws immediately
 * with a clear message rather than failing silently at runtime.
 */

function required(key: string): string {
  const val = process.env[key]
  if (!val) {
    throw new Error(
      `Missing required environment variable: ${key}. ` +
      `Set it in .env or your deployment configuration.`,
    )
  }
  return val
}

function optional(key: string, fallback: string): string {
  return process.env[key] ?? fallback
}

export const env = {
  DATABASE_URL: required('DATABASE_URL'),
  REDIS_URL: optional('REDIS_URL', 'redis://localhost:6379'),
  PORT: Number(optional('PORT', '4000')),
  NODE_ENV: optional('NODE_ENV', 'development'),
  CORS_ORIGINS: optional('CORS_ORIGINS', 'http://localhost:3000').split(','),
  SESSION_TTL_DAYS: Number(optional('SESSION_TTL_DAYS', '7')),
} as const
