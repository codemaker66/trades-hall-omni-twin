import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const MAX_PAYLOAD_BYTES = 600_000
const SNAPSHOT_TTL_MS = 7 * 24 * 60 * 60 * 1000
const MAX_SNAPSHOTS = 2000
const CODE_LENGTH = 8
const CODE_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'

type ShareSnapshot = {
  payload: string
  createdAt: number
  expiresAt: number
}

const snapshots = new Map<string, ShareSnapshot>()

const jsonNoStore = (body: unknown, init?: ResponseInit) =>
  NextResponse.json(body, {
    ...init,
    headers: {
      'Cache-Control': 'no-store',
      ...(init?.headers ?? {})
    }
  })

const nowMs = () => Date.now()

const purgeExpiredSnapshots = () => {
  const now = nowMs()
  for (const [code, snapshot] of snapshots) {
    if (snapshot.expiresAt <= now) {
      snapshots.delete(code)
    }
  }
}

const createSnapshotCode = (): string => {
  let code = ''
  for (let i = 0; i < CODE_LENGTH; i += 1) {
    const idx = Math.floor(Math.random() * CODE_ALPHABET.length)
    code += CODE_ALPHABET[idx]
  }
  return code
}

const createUniqueSnapshotCode = (): string => {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    const code = createSnapshotCode()
    if (!snapshots.has(code)) {
      return code
    }
  }

  while (true) {
    const code = createSnapshotCode()
    if (!snapshots.has(code)) {
      return code
    }
  }
}

const evictOldestSnapshotIfNeeded = () => {
  if (snapshots.size < MAX_SNAPSHOTS) return
  const oldest = snapshots.keys().next()
  if (!oldest.done) {
    snapshots.delete(oldest.value)
  }
}

export async function POST(request: NextRequest) {
  purgeExpiredSnapshots()

  let body: unknown
  try {
    body = await request.json()
  } catch {
    return jsonNoStore({ error: 'Invalid JSON body.' }, { status: 400 })
  }

  const payload = (body as { payload?: unknown } | null)?.payload
  if (typeof payload !== 'string' || payload.trim().length === 0) {
    return jsonNoStore({ error: 'Missing payload string.' }, { status: 400 })
  }

  const payloadBytes = new TextEncoder().encode(payload).byteLength
  if (payloadBytes > MAX_PAYLOAD_BYTES) {
    return jsonNoStore(
      { error: `Payload too large (${payloadBytes} bytes). Max is ${MAX_PAYLOAD_BYTES} bytes.` },
      { status: 413 }
    )
  }

  evictOldestSnapshotIfNeeded()

  const createdAt = nowMs()
  const expiresAt = createdAt + SNAPSHOT_TTL_MS
  const code = createUniqueSnapshotCode()

  snapshots.set(code, {
    payload,
    createdAt,
    expiresAt
  })

  return jsonNoStore({
    code,
    createdAt: new Date(createdAt).toISOString(),
    expiresAt: new Date(expiresAt).toISOString()
  })
}

export function GET(request: NextRequest) {
  purgeExpiredSnapshots()

  const rawCode = request.nextUrl.searchParams.get('code')
  if (!rawCode) {
    return jsonNoStore({ error: 'Missing share code.' }, { status: 400 })
  }

  const code = rawCode.trim().toUpperCase()
  if (!/^[A-Z2-9]{6,12}$/.test(code)) {
    return jsonNoStore({ error: 'Invalid share code.' }, { status: 400 })
  }

  const snapshot = snapshots.get(code)
  if (!snapshot) {
    return jsonNoStore({ error: 'Share code not found or expired.' }, { status: 404 })
  }

  if (snapshot.expiresAt <= nowMs()) {
    snapshots.delete(code)
    return jsonNoStore({ error: 'Share code not found or expired.' }, { status: 404 })
  }

  return jsonNoStore({
    payload: snapshot.payload,
    createdAt: new Date(snapshot.createdAt).toISOString(),
    expiresAt: new Date(snapshot.expiresAt).toISOString()
  })
}
