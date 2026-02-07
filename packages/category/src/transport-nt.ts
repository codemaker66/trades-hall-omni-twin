/**
 * CT-3: Transport Natural Transformation
 *
 * η: WebSocketTransport ⇒ WebRTCTransport
 *
 * Naturality guarantees message ordering and delivery semantics
 * are preserved when swapping transport backends.
 */

import type { Morphism } from './core'
import type { StrategySwap } from './natural-transformation'
import { createStrategySwap } from './natural-transformation'

// ─── Message Types ──────────────────────────────────────────────────────────

/** A transport-agnostic message. */
export interface TransportMessage {
  readonly id: string
  readonly type: string
  readonly payload: unknown
  readonly timestamp: number
  readonly sequence: number
}

/** Delivery guarantee levels. */
export type DeliveryGuarantee = 'at-most-once' | 'at-least-once' | 'exactly-once'

// ─── WebSocket Representation ───────────────────────────────────────────────

export interface WebSocketState {
  readonly kind: 'websocket'
  readonly url: string
  readonly messages: readonly WebSocketFrame[]
  readonly sequence: number
}

export interface WebSocketFrame {
  readonly opcode: 'text' | 'binary'
  readonly data: string
  readonly sequence: number
}

// ─── WebRTC Representation ──────────────────────────────────────────────────

export interface WebRTCState {
  readonly kind: 'webrtc'
  readonly channelId: string
  readonly messages: readonly WebRTCMessage[]
  readonly sequence: number
}

export interface WebRTCMessage {
  readonly channelLabel: string
  readonly data: string
  readonly ordered: boolean
  readonly sequence: number
}

// ─── Transport Natural Transformation ───────────────────────────────────────

/**
 * Encode a transport message for WebSocket.
 */
export function encodeWebSocket(msg: TransportMessage): WebSocketFrame {
  return {
    opcode: 'text',
    data: JSON.stringify({ id: msg.id, type: msg.type, payload: msg.payload, timestamp: msg.timestamp }),
    sequence: msg.sequence,
  }
}

/**
 * Decode a WebSocket frame to a transport message.
 */
export function decodeWebSocket(frame: WebSocketFrame): TransportMessage {
  const parsed = JSON.parse(frame.data) as { id: string; type: string; payload: unknown; timestamp: number }
  return {
    id: parsed.id,
    type: parsed.type,
    payload: parsed.payload,
    timestamp: parsed.timestamp,
    sequence: frame.sequence,
  }
}

/**
 * Encode a transport message for WebRTC.
 */
export function encodeWebRTC(msg: TransportMessage, channelLabel = 'data'): WebRTCMessage {
  return {
    channelLabel,
    data: JSON.stringify({ id: msg.id, type: msg.type, payload: msg.payload, timestamp: msg.timestamp }),
    ordered: true,
    sequence: msg.sequence,
  }
}

/**
 * Decode a WebRTC message to a transport message.
 */
export function decodeWebRTC(msg: WebRTCMessage): TransportMessage {
  const parsed = JSON.parse(msg.data) as { id: string; type: string; payload: unknown; timestamp: number }
  return {
    id: parsed.id,
    type: parsed.type,
    payload: parsed.payload,
    timestamp: parsed.timestamp,
    sequence: msg.sequence,
  }
}

/**
 * Create the transport strategy swap: WebSocket ↔ WebRTC.
 */
export function createTransportSwap(): StrategySwap<WebSocketState, WebRTCState> {
  return createStrategySwap(
    'WebSocket',
    'WebRTC',
    websocketToWebRTC,
    webrtcToWebSocket,
  )
}

function websocketToWebRTC(ws: WebSocketState): WebRTCState {
  return {
    kind: 'webrtc',
    channelId: ws.url,
    messages: ws.messages.map(frame => ({
      channelLabel: 'data',
      data: frame.data,
      ordered: true,
      sequence: frame.sequence,
    })),
    sequence: ws.sequence,
  }
}

function webrtcToWebSocket(rtc: WebRTCState): WebSocketState {
  return {
    kind: 'websocket',
    url: rtc.channelId,
    messages: rtc.messages.map(msg => ({
      opcode: 'text' as const,
      data: msg.data,
      sequence: msg.sequence,
    })),
    sequence: rtc.sequence,
  }
}

/**
 * Send a message through a WebSocket state.
 */
export function sendWebSocket(msg: TransportMessage): Morphism<WebSocketState, WebSocketState> {
  return (state) => ({
    ...state,
    messages: [...state.messages, encodeWebSocket(msg)],
    sequence: state.sequence + 1,
  })
}

/**
 * Send a message through a WebRTC state.
 */
export function sendWebRTC(msg: TransportMessage): Morphism<WebRTCState, WebRTCState> {
  return (state) => ({
    ...state,
    messages: [...state.messages, encodeWebRTC(msg)],
    sequence: state.sequence + 1,
  })
}
