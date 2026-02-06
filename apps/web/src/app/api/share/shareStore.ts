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
  has(code: string): Promise<boolean>
  delete(code: string): Promise<void>
  purgeExpired(now: number): Promise<void>
  evictOldest(maxEntries: number): Promise<void>
}

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

const resolveStoreMode = (): 'memory' | 'file' => {
  const envValue = process.env.SHARE_SNAPSHOT_STORE?.trim().toLowerCase()
  if (envValue === 'memory' || envValue === 'file') {
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

let shareSnapshotStore: ShareSnapshotStore | null = null

export const getShareSnapshotStore = (): ShareSnapshotStore => {
  if (shareSnapshotStore) return shareSnapshotStore

  const mode = resolveStoreMode()
  if (mode === 'file') {
    shareSnapshotStore = new FileShareSnapshotStore(resolveFilePath())
    return shareSnapshotStore
  }

  shareSnapshotStore = new MemoryShareSnapshotStore()
  return shareSnapshotStore
}
