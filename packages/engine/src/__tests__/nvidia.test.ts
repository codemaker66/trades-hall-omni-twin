import { describe, it, expect } from 'vitest'

import type { ProjectedVenueState, ProjectedItem } from '../projector'
import { emptyVenueState } from '../projector'

// Cosmos
import {
  serializeScene,
  countByType,
  estimateCapacity,
  formatFurnitureSummary,
  detectLayoutStyle,
  MockCosmosClient,
  createCosmosClient,
} from '../nvidia/cosmos'

// Omniverse
import {
  toOmniverseScene,
  toOmniverseItem,
  computeSceneDiff,
  MockOmniverseClient,
  createOmniverseClient,
} from '../nvidia/omniverse'

// ACE
import {
  buildConciergeContext,
  buildSystemPrompt,
  MockConciergeClient,
  createConciergeClient,
} from '../nvidia/ace'

// ─── Helpers ─────────────────────────────────────────────────────────────────

function makeItem(id: string, type: string, x: number, z: number): ProjectedItem {
  return {
    id,
    furnitureType: type as ProjectedItem['furnitureType'],
    position: [x, 0, z],
    rotation: [0, 0, 0],
    scale: [1, 1, 1],
  }
}

function venueWith(items: ProjectedItem[]): ProjectedVenueState {
  const state = emptyVenueState('venue-1')
  ;(state as { name: string }).name = 'Grand Hall'
  for (const item of items) {
    state.items.set(item.id, item)
  }
  return state
}

const DIMS = { width: 21, depth: 10, height: 6 }

// ─── Cosmos: Scene Serializer ────────────────────────────────────────────────

describe('Cosmos: Scene Serializer', () => {
  it('counts items by type', () => {
    const items = new Map<string, ProjectedItem>()
    items.set('a', makeItem('a', 'chair', 1, 1))
    items.set('b', makeItem('b', 'chair', 2, 2))
    items.set('c', makeItem('c', 'round-table', 5, 5))

    const counts = countByType(items)
    expect(counts.get('chair')).toBe(2)
    expect(counts.get('round-table')).toBe(1)
  })

  it('estimates capacity', () => {
    const counts = new Map([['chair', 50], ['round-table', 10]])
    expect(estimateCapacity(counts)).toBe(130) // 50 + 10*8
  })

  it('formats furniture summary', () => {
    const counts = new Map([['chair', 50], ['round-table', 10]])
    const summary = formatFurnitureSummary(counts)
    expect(summary).toContain('50 chairs')
    expect(summary).toContain('10 round tables')
  })

  it('formats single item type', () => {
    const counts = new Map([['chair', 1]])
    expect(formatFurnitureSummary(counts)).toBe('1 chair')
  })

  it('formats empty venue', () => {
    expect(formatFurnitureSummary(new Map())).toBe('an empty venue')
  })

  it('detects theater style', () => {
    const items = new Map<string, ProjectedItem>()
    for (let i = 0; i < 60; i++) {
      items.set(`c${i}`, makeItem(`c${i}`, 'chair', i, i))
    }
    items.set('s1', makeItem('s1', 'stage', 10, 1))
    expect(detectLayoutStyle(items)).toBe('theater-style')
  })

  it('detects banquet style', () => {
    const items = new Map<string, ProjectedItem>()
    for (let i = 0; i < 8; i++) {
      items.set(`t${i}`, makeItem(`t${i}`, 'round-table', i * 3, 5))
    }
    expect(detectLayoutStyle(items)).toBe('banquet-style')
  })

  it('serializes complete scene description', () => {
    const state = venueWith([
      makeItem('c1', 'chair', 1, 1),
      makeItem('c2', 'chair', 2, 2),
      makeItem('t1', 'round-table', 5, 5),
    ])

    const desc = serializeScene(state, DIMS)
    expect(desc.venueId).toBe('venue-1')
    expect(desc.description).toContain('21m')
    expect(desc.description).toContain('10m')
    expect(desc.furnitureSummary).toBeDefined()
    expect(desc.dimensions).toEqual(DIMS)
  })
})

// ─── Cosmos: Client ──────────────────────────────────────────────────────────

describe('Cosmos: Mock Client', () => {
  it('submits and polls a job', async () => {
    const client = new MockCosmosClient()
    const jobId = await client.submit({
      scene: {
        venueId: 'v1',
        description: 'test venue',
        dimensions: DIMS,
        furnitureSummary: '10 chairs',
      },
      resolution: '720p',
      duration: 10,
      model: 'Predict2.5',
    })

    expect(jobId).toContain('cosmos-job')

    const status = await client.status(jobId)
    expect(status.status).toBe('processing')
    expect(status.progress).toBeGreaterThanOrEqual(0)
    expect(status.progress).toBeLessThanOrEqual(1)
  })

  it('reports not-found for unknown job', async () => {
    const client = new MockCosmosClient()
    const status = await client.status('nonexistent')
    expect(status.status).toBe('failed')
    expect(status.error).toContain('not found')
  })

  it('cancels a job', async () => {
    const client = new MockCosmosClient()
    const jobId = await client.submit({
      scene: { venueId: 'v1', description: 'test', dimensions: DIMS, furnitureSummary: '' },
      resolution: '720p',
      duration: 5,
      model: 'Predict2.5',
    })

    await client.cancel(jobId)
    const status = await client.status(jobId)
    expect(status.status).toBe('failed') // cancelled = not found
  })

  it('createCosmosClient returns mock without env vars', () => {
    const client = createCosmosClient()
    expect(client).toBeInstanceOf(MockCosmosClient)
  })
})

// ─── Omniverse: Scene Sync ───────────────────────────────────────────────────

describe('Omniverse: Scene Sync', () => {
  it('converts item to Omniverse format', () => {
    const item = makeItem('c1', 'chair', 5, 10)
    const ovi = toOmniverseItem(item)
    expect(ovi.id).toBe('c1')
    expect(ovi.type).toBe('chair')
    expect(ovi.position).toEqual([5, 0, 10])
    expect(ovi.material).toBe('fabric_burgundy')
  })

  it('converts full state to Omniverse scene', () => {
    const state = venueWith([
      makeItem('c1', 'chair', 1, 1),
      makeItem('t1', 'round-table', 5, 5),
    ])

    const scene = toOmniverseScene(state, DIMS, 'warm')
    expect(scene.items.length).toBe(2)
    expect(scene.venueDimensions).toEqual(DIMS)
    expect(scene.lighting).toBe('warm')
  })

  it('computes scene diff: add', () => {
    const old = toOmniverseScene(emptyVenueState('v1'), DIMS)
    const state = venueWith([makeItem('c1', 'chair', 1, 1)])
    const curr = toOmniverseScene(state, DIMS)

    const diff = computeSceneDiff(old, curr)
    expect(diff.length).toBe(1)
    expect(diff[0]!.type).toBe('add')
  })

  it('computes scene diff: remove', () => {
    const state = venueWith([makeItem('c1', 'chair', 1, 1)])
    const old = toOmniverseScene(state, DIMS)
    const curr = toOmniverseScene(emptyVenueState('v1'), DIMS)

    const diff = computeSceneDiff(old, curr)
    expect(diff.length).toBe(1)
    expect(diff[0]!.type).toBe('remove')
  })

  it('computes scene diff: update', () => {
    const state1 = venueWith([makeItem('c1', 'chair', 1, 1)])
    const state2 = venueWith([makeItem('c1', 'chair', 5, 5)])
    const old = toOmniverseScene(state1, DIMS)
    const curr = toOmniverseScene(state2, DIMS)

    const diff = computeSceneDiff(old, curr)
    expect(diff.length).toBe(1)
    expect(diff[0]!.type).toBe('update')
  })

  it('empty diff for identical scenes', () => {
    const state = venueWith([makeItem('c1', 'chair', 1, 1)])
    const scene = toOmniverseScene(state, DIMS)
    const diff = computeSceneDiff(scene, scene)
    expect(diff.length).toBe(0)
  })
})

// ─── Omniverse: Streaming ────────────────────────────────────────────────────

describe('Omniverse: Mock Streaming', () => {
  it('connects and gets session', async () => {
    const client = new MockOmniverseClient()
    const session = await client.connect({
      quality: 'high',
      resolution: [1920, 1080],
      maxBitrate: 5000,
    })

    expect(session.sessionId).toContain('omni-session')
    expect(session.status).toBe('active')
    expect(session.streamUrl).toContain('wss://')
  })

  it('syncs scene without error', async () => {
    const client = new MockOmniverseClient()
    const session = await client.connect({
      quality: 'medium',
      resolution: [1280, 720],
      maxBitrate: 3000,
    })

    const state = venueWith([makeItem('c1', 'chair', 1, 1)])
    const scene = toOmniverseScene(state, DIMS)
    await expect(client.syncScene(session.sessionId, scene)).resolves.toBeUndefined()
  })

  it('disconnects session', async () => {
    const client = new MockOmniverseClient()
    const session = await client.connect({
      quality: 'low',
      resolution: [640, 480],
      maxBitrate: 1000,
    })

    await client.disconnect(session.sessionId)
    const updated = await client.getSession(session.sessionId)
    expect(updated?.status).toBe('disconnected')
  })

  it('returns null for unknown session', async () => {
    const client = new MockOmniverseClient()
    const session = await client.getSession('nonexistent')
    expect(session).toBeNull()
  })

  it('createOmniverseClient returns mock', () => {
    const client = createOmniverseClient()
    expect(client).toBeInstanceOf(MockOmniverseClient)
  })
})

// ─── ACE: Context Builder ────────────────────────────────────────────────────

describe('ACE: Context Builder', () => {
  it('builds context from venue state', () => {
    const state = venueWith([
      makeItem('c1', 'chair', 1, 1),
      makeItem('c2', 'chair', 2, 2),
      makeItem('t1', 'round-table', 5, 5),
    ])

    const ctx = buildConciergeContext(state, {
      venueName: 'Grand Hall',
      venueDescription: 'A beautiful heritage hall',
      amenities: ['PA System', 'Kitchen', 'Parking'],
      pricing: '$5,000 per day',
      faq: [{ question: 'Is catering included?', answer: 'Catering can be arranged.' }],
    })

    expect(ctx.venueId).toBe('venue-1')
    expect(ctx.venueName).toBe('Grand Hall')
    expect(ctx.capacity).toBe(10) // 2 chairs + 1 round table (8)
    expect(ctx.amenities).toContain('PA System')
    expect(ctx.floorPlanSummary).toContain('chair')
    expect(ctx.pricing).toBe('$5,000 per day')
    expect(ctx.faq.length).toBe(1)
  })

  it('builds system prompt with context', () => {
    const ctx = buildConciergeContext(venueWith([]), {
      venueName: 'Test Venue',
      amenities: ['WiFi'],
      faq: [{ question: 'Where to park?', answer: 'Rear lot.' }],
    })

    const prompt = buildSystemPrompt(ctx)
    expect(prompt).toContain('Test Venue')
    expect(prompt).toContain('WiFi')
    expect(prompt).toContain('Where to park?')
    expect(prompt).toContain('Rear lot.')
    expect(prompt).toContain('concierge')
  })
})

// ─── ACE: Mock Concierge ─────────────────────────────────────────────────────

describe('ACE: Mock Concierge', () => {
  it('creates a session', () => {
    const client = new MockConciergeClient()
    const ctx = buildConciergeContext(venueWith([]), { venueName: 'Test' })
    const session = client.createSession('venue-1', ctx)

    expect(session.sessionId).toContain('concierge')
    expect(session.status).toBe('active')
    expect(session.messages.length).toBe(0)
  })

  it('responds to capacity question', async () => {
    const client = new MockConciergeClient()
    const ctx = buildConciergeContext(venueWith([]), { venueName: 'Test' })
    const session = client.createSession('venue-1', ctx)

    const { session: updated, response } = await client.sendMessage(session, 'What is the capacity?')
    expect(response.message).toContain('accommodate')
    expect(updated.messages.length).toBe(2) // user + assistant
    expect(updated.messages[0]!.role).toBe('user')
    expect(updated.messages[1]!.role).toBe('assistant')
  })

  it('responds to pricing question', async () => {
    const client = new MockConciergeClient()
    const ctx = buildConciergeContext(venueWith([]), { venueName: 'Test' })
    const session = client.createSession('venue-1', ctx)

    const { response } = await client.sendMessage(session, 'How much does it cost?')
    expect(response.message).toContain('pricing')
  })

  it('suggests actions', async () => {
    const client = new MockConciergeClient()
    const ctx = buildConciergeContext(venueWith([]), { venueName: 'Test' })
    const session = client.createSession('venue-1', ctx)

    const { response } = await client.sendMessage(session, 'Tell me about the venue')
    expect(response.suggestedActions).toBeDefined()
    expect(response.suggestedActions!.length).toBeGreaterThan(0)
  })

  it('ends session', () => {
    const client = new MockConciergeClient()
    const ctx = buildConciergeContext(venueWith([]), { venueName: 'Test' })
    const session = client.createSession('venue-1', ctx)
    const ended = client.endSession(session)
    expect(ended.status).toBe('ended')
  })

  it('createConciergeClient returns mock', () => {
    const client = createConciergeClient()
    expect(client).toBeInstanceOf(MockConciergeClient)
  })
})
