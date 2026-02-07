import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import path from 'node:path'

export type ShareSnapshot = {
  payload: string
  createdAt: number
  expiresAt: number
}

export interface ShareSnapshotStore {
  get(code: string): Promise<ShareSnapshot | null>
  set(code: string, snapshot: ShareSnapshot): Promise<void>
  setIfAbsent(code: string, snapshot: ShareSnapshot): Promise<boolean>
  has(code: string): Promise<boolean>
  delete(code: string): Promise<void>
  purgeExpired(now: number): Promise<void>
  evictOldest(maxEntries: number): Promise<void>
}

type ShareStoreMode = 'memory' | 'file' | 'redis'

const SHARE_STORE_FILE_VERSION = 1

type ShareStoreFile = {
  version: number
  entries: Array<{
    code: string
    payload: string
    createdAt: number
    expiresAt: number
  }>
}

type RedisClientFactory = (options: { url: string }) => RedisClientCompat

interface RedisMultiCompat {
  del(key: string | string[]): RedisMultiCompat
  zRem(key: string, members: string | string[]): RedisMultiCompat
  zAdd(key: string, members: Array<{ score: number, value: string }>): RedisMultiCompat
  exec(): Promise<unknown>
}

interface RedisClientCompat {
  isOpen: boolean
  connect(): Promise<unknown>
  on(event: 'error', listener: (error: unknown) => void): unknown
  get(key: string): Promise<string | null>
  set(
    key: string,
    value: string,
    options?: { PX?: number, NX?: boolean }
  ): Promise<'OK' | null | unknown>
  exists(key: string): Promise<number>
  zRangeByScore(key: string, min: number | string, max: number | string): Promise<string[]>
  zRange(key: string, start: number, stop: number): Promise<string[]>
  zCard(key: string): Promise<number>
  multi(): RedisMultiCompat
}

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)

const isSnapshot = (value: unknown): value is ShareSnapshot => {
  if (!value || typeof value !== 'object') return false
  const candidate = value as Partial<ShareSnapshot>
  return (
    typeof candidate.payload === 'string' &&
    isFiniteNumber(candidate.createdAt) &&
    isFiniteNumber(candidate.expiresAt)
  )
}

const isFileEntry = (value: unknown): value is ShareStoreFile['entries'][number] => {
  if (!value || typeof value !== 'object') return false
  const candidate = value as Partial<ShareStoreFile['entries'][number]>
  return (
    typeof candidate.code === 'string' &&
    candidate.code.length > 0 &&
    typeof candidate.payload === 'string' &&
    isFiniteNumber(candidate.createdAt) &&
    isFiniteNumber(candidate.expiresAt)
  )
}

const normalizeStoreFile = (raw: unknown): ShareStoreFile => {
  if (!raw || typeof raw !== 'object') {
    return { version: SHARE_STORE_FILE_VERSION, entries: [] }
  }

  const candidate = raw as Partial<ShareStoreFile>
  const entriesRaw = Array.isArray(candidate.entries) ? candidate.entries : []
  const entries = entriesRaw.filter(isFileEntry)

  return {
    version: SHARE_STORE_FILE_VERSION,
    entries
  }
}

const toKeyPrefix = (prefix: string): string => {
  const trimmed = prefix.trim()
  if (!trimmed) return 'omnitwin:share:'
  return trimmed.endsWith(':') ? trimmed : `${trimmed}:`
}

const getSnapshotTtlMs = (snapshot: ShareSnapshot): number => {
  const ttlMs = snapshot.expiresAt - Date.now()
  return Math.max(1, ttlMs)
}

let redisClientFactoryPromise: Promise<RedisClientFactory | null> | null = null

const loadRedisClientFactory = async (): Promise<RedisClientFactory | null> => {
  if (redisClientFactoryPromise) {
    return redisClientFactoryPromise
  }

  redisClientFactoryPromise = (async () => {
    try {
      const dynamicImport = new Function(
        'moduleName',
        'return import(moduleName)'
      ) as (moduleName: string) => Promise<unknown>

      const redisModule = await dynamicImport('redis') as { createClient?: unknown }
      if (!redisModule || typeof redisModule.createClient !== 'function') {
        return null
      }

      return redisModule.createClient as RedisClientFactory
    } catch {
      return null
    }
  })()

  return redisClientFactoryPromise
}

class MemoryShareSnapshotStore implements ShareSnapshotStore {
  protected readonly entries = new Map<string, ShareSnapshot>()

  async get(code: string): Promise<ShareSnapshot | null> {
    const entry = this.entries.get(code)
    return entry ? { ...entry } : null
  }

  async set(code: string, snapshot: ShareSnapshot): Promise<void> {
    if (!isSnapshot(snapshot)) {
      throw new Error('Invalid snapshot payload.')
    }
    this.entries.set(code, { ...snapshot })
  }

  async setIfAbsent(code: string, snapshot: ShareSnapshot): Promise<boolean> {
    if (!isSnapshot(snapshot)) {
      throw new Error('Invalid snapshot payload.')
    }
    if (this.entries.has(code)) return false
    this.entries.set(code, { ...snapshot })
    return true
  }

  async has(code: string): Promise<boolean> {
    return this.entries.has(code)
  }

  async delete(code: string): Promise<void> {
    this.entries.delete(code)
  }

  async purgeExpired(now: number): Promise<void> {
    for (const [code, snapshot] of this.entries) {
      if (snapshot.expiresAt <= now) {
        this.entries.delete(code)
      }
    }
  }

  async evictOldest(maxEntries: number): Promise<void> {
    if (maxEntries <= 0) {
      this.entries.clear()
      return
    }

    while (this.entries.size >= maxEntries) {
      let oldestCode: string | null = null
      let oldestCreatedAt = Number.POSITIVE_INFINITY

      for (const [code, snapshot] of this.entries) {
        if (snapshot.createdAt < oldestCreatedAt) {
          oldestCreatedAt = snapshot.createdAt
          oldestCode = code
        }
      }

      if (!oldestCode) return
      this.entries.delete(oldestCode)
    }
  }
}

class FileShareSnapshotStore extends MemoryShareSnapshotStore {
  private readonly filePath: string
  private isLoaded = false
  private writeQueue: Promise<void> = Promise.resolve()
  private hasWriteFailure = false

  constructor(filePath: string) {
    super()
    this.filePath = filePath
  }

  override async get(code: string): Promise<ShareSnapshot | null> {
    await this.ensureLoaded()
    return super.get(code)
  }

  override async set(code: string, snapshot: ShareSnapshot): Promise<void> {
    await this.ensureLoaded()
    await super.set(code, snapshot)
    await this.flush()
  }

  override async setIfAbsent(code: string, snapshot: ShareSnapshot): Promise<boolean> {
    await this.ensureLoaded()
    const inserted = await super.setIfAbsent(code, snapshot)
    if (inserted) {
      await this.flush()
    }
    return inserted
  }

  override async has(code: string): Promise<boolean> {
    await this.ensureLoaded()
    return super.has(code)
  }

  override async delete(code: string): Promise<void> {
    await this.ensureLoaded()
    const existed = this.entries.has(code)
    await super.delete(code)
    if (existed) {
      await this.flush()
    }
  }

  override async purgeExpired(now: number): Promise<void> {
    await this.ensureLoaded()
    const beforeSize = this.entries.size
    await super.purgeExpired(now)
    if (this.entries.size !== beforeSize) {
      await this.flush()
    }
  }

  override async evictOldest(maxEntries: number): Promise<void> {
    await this.ensureLoaded()
    const beforeSize = this.entries.size
    await super.evictOldest(maxEntries)
    if (this.entries.size !== beforeSize) {
      await this.flush()
    }
  }

  private async ensureLoaded(): Promise<void> {
    if (this.isLoaded) return
    this.isLoaded = true

    try {
      const raw = await readFile(this.filePath, 'utf8')
      const parsed = normalizeStoreFile(JSON.parse(raw))
      for (const entry of parsed.entries) {
        this.entries.set(entry.code, {
          payload: entry.payload,
          createdAt: entry.createdAt,
          expiresAt: entry.expiresAt
        })
      }
    } catch {
      // Missing/unreadable/corrupt file starts empty.
      this.entries.clear()
    }
  }

  private async flush(): Promise<void> {
    if (this.hasWriteFailure) return

    const writeTask = async () => {
      try {
        const dir = path.dirname(this.filePath)
        await mkdir(dir, { recursive: true })

        const data: ShareStoreFile = {
          version: SHARE_STORE_FILE_VERSION,
          entries: [...this.entries.entries()]
            .map(([code, snapshot]) => ({
              code,
              payload: snapshot.payload,
              createdAt: snapshot.createdAt,
              expiresAt: snapshot.expiresAt
            }))
            .sort((a, b) => a.createdAt - b.createdAt)
        }

        const tempPath = `${this.filePath}.tmp`
        await writeFile(tempPath, JSON.stringify(data), 'utf8')
        await rename(tempPath, this.filePath)
      } catch {
        this.hasWriteFailure = true
      }
    }

    this.writeQueue = this.writeQueue.then(writeTask, writeTask)
    await this.writeQueue
  }
}

class RedisShareSnapshotStore implements ShareSnapshotStore {
  private readonly redisUrl: string
  private readonly keyPrefix: string
  private readonly createdIndexKey: string
  private readonly expiresIndexKey: string
  private client: RedisClientCompat | null = null
  private connectPromise: Promise<void> | null = null
  private clientInitPromise: Promise<RedisClientCompat> | null = null

  constructor(redisUrl: string, keyPrefix: string) {
    this.redisUrl = redisUrl
    this.keyPrefix = toKeyPrefix(keyPrefix)
    this.createdIndexKey = `${this.keyPrefix}__idx_created`
    this.expiresIndexKey = `${this.keyPrefix}__idx_expires`
  }

  async get(code: string): Promise<ShareSnapshot | null> {
    const client = await this.ensureConnected()
    const key = this.toSnapshotKey(code)
    const raw = await client.get(key)
    if (!raw) return null

    try {
      const parsed = JSON.parse(raw) as unknown
      if (!isSnapshot(parsed)) {
        await this.delete(code)
        return null
      }
      return parsed
    } catch {
      await this.delete(code)
      return null
    }
  }

  async set(code: string, snapshot: ShareSnapshot): Promise<void> {
    if (!isSnapshot(snapshot)) {
      throw new Error('Invalid snapshot payload.')
    }

    const client = await this.ensureConnected()
    const key = this.toSnapshotKey(code)
    const ttlMs = getSnapshotTtlMs(snapshot)
    const payload = JSON.stringify(snapshot)

    await client.set(key, payload, { PX: ttlMs })
    await client.multi()
      .zAdd(this.createdIndexKey, [{ score: snapshot.createdAt, value: key }])
      .zAdd(this.expiresIndexKey, [{ score: snapshot.expiresAt, value: key }])
      .exec()
  }

  async setIfAbsent(code: string, snapshot: ShareSnapshot): Promise<boolean> {
    if (!isSnapshot(snapshot)) {
      throw new Error('Invalid snapshot payload.')
    }

    const client = await this.ensureConnected()
    const key = this.toSnapshotKey(code)
    const ttlMs = getSnapshotTtlMs(snapshot)
    const payload = JSON.stringify(snapshot)
    const result = await client.set(key, payload, { PX: ttlMs, NX: true })
    if (result !== 'OK') return false

    await client.multi()
      .zAdd(this.createdIndexKey, [{ score: snapshot.createdAt, value: key }])
      .zAdd(this.expiresIndexKey, [{ score: snapshot.expiresAt, value: key }])
      .exec()

    return true
  }

  async has(code: string): Promise<boolean> {
    const client = await this.ensureConnected()
    const key = this.toSnapshotKey(code)
    return (await client.exists(key)) > 0
  }

  async delete(code: string): Promise<void> {
    const client = await this.ensureConnected()
    const key = this.toSnapshotKey(code)

    await client.multi()
      .del(key)
      .zRem(this.createdIndexKey, key)
      .zRem(this.expiresIndexKey, key)
      .exec()
  }

  async purgeExpired(now: number): Promise<void> {
    const client = await this.ensureConnected()
    const expiredKeys = await client.zRangeByScore(this.expiresIndexKey, '-inf', now)
    if (expiredKeys.length === 0) return

    await client.multi()
      .del(expiredKeys)
      .zRem(this.createdIndexKey, expiredKeys)
      .zRem(this.expiresIndexKey, expiredKeys)
      .exec()
  }

  async evictOldest(maxEntries: number): Promise<void> {
    const client = await this.ensureConnected()

    if (maxEntries <= 0) {
      const keys = await client.zRange(this.createdIndexKey, 0, -1)
      const pipeline = client.multi()
      if (keys.length > 0) {
        pipeline.del(keys)
      }
      await pipeline.del(this.createdIndexKey).del(this.expiresIndexKey).exec()
      return
    }

    let count = await client.zCard(this.createdIndexKey)
    while (count >= maxEntries) {
      const oldest = await client.zRange(this.createdIndexKey, 0, 0)
      if (oldest.length === 0) return

      const oldestKey = oldest[0]!
      await client.multi()
        .del(oldestKey)
        .zRem(this.createdIndexKey, oldestKey)
        .zRem(this.expiresIndexKey, oldestKey)
        .exec()

      count = await client.zCard(this.createdIndexKey)
    }
  }

  private toSnapshotKey(code: string): string {
    return `${this.keyPrefix}${code}`
  }

  private async ensureConnected(): Promise<RedisClientCompat> {
    const client = await this.ensureClient()
    if (client.isOpen) return client

    if (!this.connectPromise) {
      this.connectPromise = client.connect().then(() => undefined)
    }

    try {
      await this.connectPromise
    } finally {
      this.connectPromise = null
    }

    return client
  }

  private async ensureClient(): Promise<RedisClientCompat> {
    if (this.client) return this.client
    if (this.clientInitPromise) {
      return this.clientInitPromise
    }

    this.clientInitPromise = (async () => {
      const createRedisClient = await loadRedisClientFactory()
      if (!createRedisClient) {
        throw new Error('SHARE_SNAPSHOT_STORE=redis requires the "redis" package to be installed.')
      }

      const client = createRedisClient({ url: this.redisUrl })
      client.on('error', () => {
        // Errors surface via awaited commands; avoid noisy unhandled error logging.
      })

      this.client = client
      return client
    })()

    try {
      return await this.clientInitPromise
    } finally {
      this.clientInitPromise = null
    }
  }
}

const resolveStoreMode = (): ShareStoreMode => {
  const envValue = process.env.SHARE_SNAPSHOT_STORE?.trim().toLowerCase()
  if (envValue === 'memory' || envValue === 'file' || envValue === 'redis') {
    return envValue
  }

  // Default to file-backed snapshots for single-instance durability.
  return 'file'
}

const resolveFilePath = (): string => {
  const customPath = process.env.SHARE_SNAPSHOT_FILE_PATH?.trim()
  if (customPath && customPath.length > 0) {
    return path.resolve(customPath)
  }

  return path.join(process.cwd(), '.data', 'share-snapshots.json')
}

const resolveRedisUrl = (): string | null => {
  const candidates = [
    process.env.SHARE_SNAPSHOT_REDIS_URL,
    process.env.REDIS_URL
  ]

  for (const candidate of candidates) {
    const value = candidate?.trim()
    if (value) return value
  }

  return null
}

const resolveRedisPrefix = (): string => {
  const prefix = process.env.SHARE_SNAPSHOT_REDIS_PREFIX?.trim()
  return prefix && prefix.length > 0 ? prefix : 'omnitwin:share:'
}

let shareSnapshotStore: ShareSnapshotStore | null = null

export const getShareSnapshotStore = (): ShareSnapshotStore => {
  if (shareSnapshotStore) return shareSnapshotStore

  const mode = resolveStoreMode()
  if (mode === 'redis') {
    const redisUrl = resolveRedisUrl()
    if (!redisUrl) {
      throw new Error('SHARE_SNAPSHOT_STORE=redis requires SHARE_SNAPSHOT_REDIS_URL or REDIS_URL.')
    }

    shareSnapshotStore = new RedisShareSnapshotStore(redisUrl, resolveRedisPrefix())
    return shareSnapshotStore
  }

  if (mode === 'file') {
    shareSnapshotStore = new FileShareSnapshotStore(resolveFilePath())
    return shareSnapshotStore
  }

  shareSnapshotStore = new MemoryShareSnapshotStore()
  return shareSnapshotStore
}
