/**
 * Pipeline composition tests and natural transformation tests.
 */

import { describe, test, expect } from 'vitest'
import fc from 'fast-check'
import { ok, err, compose } from '../core'
import type { Morphism } from '../core'
import {
  matchEventToVenue, constrainFloorPlan, assignEventToVenue,
  priceAssignment, scheduleAssignment, validateConfiguration,
} from '../morphisms'
import type {
  VenueSpec, EventSpec, FloorPlan, Constraint, Service, Configuration,
} from '../objects'
import { compatibilityScore, markValidated, cents, minutes, isoDateTime } from '../objects'
import {
  verticalCompose, identityNT, createStrategySwap, verifyStrategyRoundTrip,
} from '../natural-transformation'
import {
  createPersistenceSwap, updateInMemory, updatePostgres,
} from '../persistence-nt'
import type { InMemoryReadModel } from '../persistence-nt'
import {
  compileToWebGL, compileToWebGPU, createRendererSwap,
} from '../renderer-nt'
import type { RenderFrame } from '../renderer-nt'
import {
  createTransportSwap, sendWebSocket, sendWebRTC,
  encodeWebSocket, decodeWebSocket, encodeWebRTC, decodeWebRTC,
} from '../transport-nt'
import type { TransportMessage, WebSocketState, WebRTCState } from '../transport-nt'

// ─── Test Data ──────────────────────────────────────────────────────────────

const testVenue: VenueSpec = {
  id: 'venue-1',
  name: 'Main Hall',
  width: 20,
  depth: 15,
  height: 4,
  maxCapacity: 200,
  exits: [{ x: 0, z: 7.5, width: 1.5, facing: Math.PI }],
  obstacles: [],
  amenities: ['stage', 'av-system', 'lighting', 'wifi'],
  focalPoint: { x: 10, z: 0 },
}

const testEvent: EventSpec = {
  id: 'evt-1',
  name: 'Conference',
  type: 'conference',
  guestCount: 150,
  startTime: isoDateTime('2025-06-15T09:00:00Z'),
  duration: minutes(480),
  requirements: [
    { kind: 'amenity', amenity: 'stage' },
    { kind: 'amenity', amenity: 'av-system' },
  ],
  budget: cents(500000),
}

const testFloorPlan: FloorPlan = {
  venueId: 'venue-1',
  placements: [
    { id: 'chair-1', type: 'chair', x: 5, z: 5, rotation: 0, width: 0.5, depth: 0.5 },
    { id: 'chair-2', type: 'chair', x: 6, z: 5, rotation: 0, width: 0.5, depth: 0.5 },
  ],
  groupings: [],
}

// ─── Morphism Composition Tests ─────────────────────────────────────────────

describe('CT-1: Morphism Composition', () => {
  test('match → constrain → assign pipeline', () => {
    // Step 1: match
    const score = matchEventToVenue(testEvent, testVenue)
    expect(score).toBeGreaterThan(0)
    expect(score).toBeLessThanOrEqual(1)

    // Step 2: constrain
    const constraints: Constraint[] = [
      { kind: 'capacity', maxOccupants: 100 },
    ]
    const validated = constrainFloorPlan(testFloorPlan, constraints)
    expect(validated.ok).toBe(true)

    // Step 3: assign
    if (validated.ok) {
      const assignment = assignEventToVenue(testEvent, testVenue, validated.value)
      expect(assignment.eventId).toBe('evt-1')
      expect(assignment.venueId).toBe('venue-1')
      expect(assignment.allocatedAmenities).toContain('stage')
    }
  })

  test('full pipeline: match → validate', () => {
    const score = matchEventToVenue(testEvent, testVenue)
    expect(score).toBeGreaterThan(0)

    const validated = constrainFloorPlan(testFloorPlan, [])
    expect(validated.ok).toBe(true)
    if (!validated.ok) return

    const assignment = assignEventToVenue(testEvent, testVenue, validated.value)
    const services: Service[] = [{
      id: 'svc-1', name: 'AV', type: 'av',
      baseCost: cents(50000), setupTime: minutes(60), teardownTime: minutes(30),
      requirements: [],
    }]
    const proposal = priceAssignment(assignment, services, cents(10000), minutes(480))
    expect(proposal.totalCost).toBeGreaterThan(0)

    const scheduleEntry = scheduleAssignment(
      assignment, isoDateTime('2025-06-15T09:00:00Z'), minutes(480),
    )

    const config: Configuration = {
      event: testEvent,
      venue: testVenue,
      assignment,
      services,
      schedule: { entries: [scheduleEntry] },
      totalCost: proposal.totalCost,
      status: 'draft',
    }

    const result = validateConfiguration(config)
    expect(result.ok).toBe(true)
  })

  test('constraint violation propagates', () => {
    const constraints: Constraint[] = [
      { kind: 'capacity', maxOccupants: 1 },  // only 1 item allowed
    ]
    const result = constrainFloorPlan(testFloorPlan, constraints)
    expect(result.ok).toBe(false)
    if (!result.ok) {
      expect(result.error).toHaveLength(1)
      expect(result.error[0]!.message).toContain('max is 1')
    }
  })

  test('spacing constraint detects close items', () => {
    const closePlan: FloorPlan = {
      venueId: 'v1',
      placements: [
        { id: 'a', type: 'chair', x: 0, z: 0, rotation: 0, width: 0.5, depth: 0.5 },
        { id: 'b', type: 'chair', x: 0.3, z: 0, rotation: 0, width: 0.5, depth: 0.5 },
      ],
      groupings: [],
    }
    const result = constrainFloorPlan(closePlan, [{ kind: 'spacing', minGap: 1.0 }])
    expect(result.ok).toBe(false)
  })
})

// ─── Natural Transformation Tests ───────────────────────────────────────────

describe('CT-3: Persistence Natural Transformation', () => {
  test('InMemory ↔ Postgres round-trip', () => {
    const swap = createPersistenceSwap<number>()
    const model: InMemoryReadModel<number> = {
      kind: 'in-memory',
      data: new Map([['a', 1], ['b', 2]]),
      version: 5,
    }

    const postgres = swap.forward(model)
    expect(postgres.kind).toBe('postgres')
    expect(postgres.rows).toHaveLength(2)
    expect(postgres.version).toBe(5)

    const backToMemory = swap.backward(postgres)
    expect(backToMemory.kind).toBe('in-memory')
    expect(backToMemory.data.get('a')).toBe(1)
    expect(backToMemory.data.get('b')).toBe(2)
  })

  test('update naturality: update then persist ≡ persist then update', () => {
    const model: InMemoryReadModel<number> = {
      kind: 'in-memory',
      data: new Map([['x', 10]]),
      version: 0,
    }

    // Path 1: update in-memory, then convert to postgres
    const swap = createPersistenceSwap<number>()
    const updater = updateInMemory<number>('x', (v) => (v ?? 0) + 5)
    const path1 = swap.forward(updater(model))

    // Path 2: convert to postgres, then update postgres
    const pgModel = swap.forward(model)
    const pgUpdater = updatePostgres<number>('x', (v) => (v ?? 0) + 5)
    const path2 = pgUpdater(pgModel)

    // Both should have x = 15
    const row1 = path1.rows.find(r => r.id === 'x')
    const row2 = path2.rows.find(r => r.id === 'x')
    expect(row1?.data).toBe(15)
    expect(row2?.data).toBe(15)
  })
})

describe('CT-3: Renderer Natural Transformation', () => {
  const testFrame: RenderFrame = {
    frameId: 1,
    width: 1920,
    height: 1080,
    commands: [
      { kind: 'clear', color: [0, 0, 0, 1] as const },
      { kind: 'drawMesh', meshId: 'cube', transform: new Float64Array(16) },
      { kind: 'setCamera', position: [0, 5, 10] as const, target: [0, 0, 0] as const },
    ],
  }

  test('WebGL compilation produces correct commands', () => {
    const gl = compileToWebGL(testFrame)
    expect(gl.kind).toBe('webgl')
    expect(gl.commands.length).toBeGreaterThanOrEqual(3)
  })

  test('WebGPU compilation produces correct commands', () => {
    const gpu = compileToWebGPU(testFrame)
    expect(gpu.kind).toBe('webgpu')
    expect(gpu.commands.length).toBeGreaterThanOrEqual(3)
  })

  test('WebGL ↔ WebGPU swap preserves command count', () => {
    const swap = createRendererSwap()
    const gl = compileToWebGL(testFrame)
    const gpu = swap.forward(gl)
    expect(gpu.commands.length).toBe(gl.commands.length)

    const backToGL = swap.backward(gpu)
    expect(backToGL.commands.length).toBe(gl.commands.length)
  })
})

describe('CT-3: Transport Natural Transformation', () => {
  const testMsg: TransportMessage = {
    id: 'msg-1',
    type: 'update',
    payload: { x: 1, y: 2 },
    timestamp: 1000000,
    sequence: 42,
  }

  test('WebSocket encode/decode round-trip', () => {
    const frame = encodeWebSocket(testMsg)
    const decoded = decodeWebSocket(frame)
    expect(decoded.id).toBe(testMsg.id)
    expect(decoded.type).toBe(testMsg.type)
    expect(decoded.sequence).toBe(testMsg.sequence)
  })

  test('WebRTC encode/decode round-trip', () => {
    const msg = encodeWebRTC(testMsg)
    const decoded = decodeWebRTC(msg)
    expect(decoded.id).toBe(testMsg.id)
    expect(decoded.sequence).toBe(testMsg.sequence)
  })

  test('WebSocket ↔ WebRTC swap preserves messages', () => {
    const swap = createTransportSwap()
    const wsState: WebSocketState = {
      kind: 'websocket',
      url: 'ws://test',
      messages: [encodeWebSocket(testMsg)],
      sequence: 1,
    }

    const rtcState = swap.forward(wsState)
    expect(rtcState.messages).toHaveLength(1)

    const backToWS = swap.backward(rtcState)
    expect(backToWS.messages).toHaveLength(1)
  })

  test('send operations on both transports', () => {
    const wsState: WebSocketState = { kind: 'websocket', url: 'ws://test', messages: [], sequence: 0 }
    const rtcState: WebRTCState = { kind: 'webrtc', channelId: 'test', messages: [], sequence: 0 }

    const newWS = sendWebSocket(testMsg)(wsState)
    const newRTC = sendWebRTC(testMsg)(rtcState)

    expect(newWS.messages).toHaveLength(1)
    expect(newRTC.messages).toHaveLength(1)
    expect(newWS.sequence).toBe(1)
    expect(newRTC.sequence).toBe(1)
  })
})

describe('CT-3: Natural Transformation Composition', () => {
  test('vertical composition of identity NTs', () => {
    const idNT = identityNT('test')
    const composed = verticalCompose(idNT, idNT)
    expect(composed.source).toBe('test')
    expect(composed.target).toBe('test')
  })
})
