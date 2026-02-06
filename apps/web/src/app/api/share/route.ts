import { NextRequest, NextResponse } from 'next/server'
import { randomBytes } from 'node:crypto'
import { getShareSnapshotStore, type ShareSnapshot } from './shareStore'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const MAX_PAYLOAD_BYTES = 600_000
const SNAPSHOT_TTL_MS = 7 * 24 * 60 * 60 * 1000
const MAX_SNAPSHOTS = 2000
const CODE_LENGTH = 8
const CODE_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'

const shareSnapshotStore = getShareSnapshotStore()

const jsonNoStore = (body: unknown, init?: ResponseInit) =>
  NextResponse.json(body, {
    ...init,
    headers: {
      'Cache-Control': 'no-store',
      ...(init?.headers ?? {})
    }
  })

const nowMs = () => Date.now()

const createSnapshotCode = (): string => {
  const bytes = randomBytes(CODE_LENGTH)
  return [...bytes]
    .map((byte) => CODE_ALPHABET[byte % CODE_ALPHABET.length])
    .join('')
}

const createUniqueSnapshotCode = async (snapshot: ShareSnapshot): Promise<string> => {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    const code = createSnapshotCode()
    const reserved = await shareSnapshotStore.setIfAbsent(code, snapshot)
    if (reserved) {
      return code
    }
  }

  while (true) {
    const code = createSnapshotCode()
    const reserved = await shareSnapshotStore.setIfAbsent(code, snapshot)
    if (reserved) {
      return code
    }
  }
}

export async function POST(request: NextRequest) {
  await shareSnapshotStore.purgeExpired(nowMs())

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

  await shareSnapshotStore.evictOldest(MAX_SNAPSHOTS)

  const createdAt = nowMs()
  const expiresAt = createdAt + SNAPSHOT_TTL_MS
  const snapshot: ShareSnapshot = {
    payload,
    createdAt,
    expiresAt
  }
  const code = await createUniqueSnapshotCode(snapshot)

  return jsonNoStore({
    code,
    createdAt: new Date(createdAt).toISOString(),
    expiresAt: new Date(expiresAt).toISOString()
  })
}

export async function GET(request: NextRequest) {
  await shareSnapshotStore.purgeExpired(nowMs())

  const rawCode = request.nextUrl.searchParams.get('code')
  if (!rawCode) {
    return jsonNoStore({ error: 'Missing share code.' }, { status: 400 })
  }

  const code = rawCode.trim().toUpperCase()
  if (!/^[A-Z2-9]{6,12}$/.test(code)) {
    return jsonNoStore({ error: 'Invalid share code.' }, { status: 400 })
  }

  const snapshot = await shareSnapshotStore.get(code)
  if (!snapshot) {
    return jsonNoStore({ error: 'Share code not found or expired.' }, { status: 404 })
  }

  if (snapshot.expiresAt <= nowMs()) {
    await shareSnapshotStore.delete(code)
    return jsonNoStore({ error: 'Share code not found or expired.' }, { status: 404 })
  }

  return jsonNoStore({
    payload: snapshot.payload,
    createdAt: new Date(snapshot.createdAt).toISOString(),
    expiresAt: new Date(snapshot.expiresAt).toISOString()
  })
}
