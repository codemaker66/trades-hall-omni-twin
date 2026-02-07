/**
 * Shared types for NVIDIA integrations: Cosmos, Omniverse, ACE.
 */

// ─── Common ──────────────────────────────────────────────────────────────────

export type JobStatus = 'queued' | 'processing' | 'complete' | 'failed'

export interface JobResult<T> {
  readonly jobId: string
  readonly status: JobStatus
  readonly progress: number // 0–1
  readonly result?: T
  readonly error?: string
  readonly createdAt: string
  readonly updatedAt: string
}

// ─── Cosmos (T9) ─────────────────────────────────────────────────────────────

/** Scene description for text-based Cosmos generation. */
export interface SceneDescription {
  readonly venueId: string
  /** Natural language description of the venue and layout. */
  readonly description: string
  /** Venue dimensions in meters. */
  readonly dimensions: { width: number; depth: number; height: number }
  /** Summary of furniture placement. */
  readonly furnitureSummary: string
  /** Style hints for generation. */
  readonly style?: string
}

export interface CosmosRequest {
  readonly scene: SceneDescription
  readonly resolution: '720p' | '1080p'
  readonly duration: number // seconds
  readonly model: 'Predict2.5' | 'Transfer2.5'
}

export interface CosmosResult {
  readonly videoUrl: string
  readonly thumbnailUrl: string
  readonly durationMs: number
  readonly resolution: string
}

// ─── Omniverse (T10) ─────────────────────────────────────────────────────────

export interface OmniverseSessionConfig {
  readonly quality: 'low' | 'medium' | 'high' | 'ultra'
  readonly resolution: [number, number]
  readonly maxBitrate: number // kbps
}

export interface OmniverseSession {
  readonly sessionId: string
  readonly streamUrl: string
  readonly status: 'connecting' | 'active' | 'idle' | 'disconnected'
  readonly latencyMs: number
  readonly fps: number
}

/** Furniture item data for Omniverse scene sync. */
export interface OmniverseSceneItem {
  readonly id: string
  readonly type: string
  readonly position: [number, number, number]
  readonly rotation: [number, number, number]
  readonly scale: [number, number, number]
  readonly material?: string
}

export interface OmniverseSceneState {
  readonly items: OmniverseSceneItem[]
  readonly venueDimensions: { width: number; depth: number; height: number }
  readonly lighting: 'warm' | 'cool' | 'natural' | 'dramatic'
}

// ─── ACE (T11) ───────────────────────────────────────────────────────────────

export interface ConciergeContext {
  readonly venueId: string
  readonly venueName: string
  readonly venueDescription: string
  readonly capacity: number
  readonly amenities: string[]
  readonly floorPlanSummary: string
  readonly pricing?: string
  readonly availability?: string[]
  readonly faq: Array<{ question: string; answer: string }>
}

export interface ConciergeMessage {
  readonly role: 'user' | 'assistant'
  readonly content: string
  readonly timestamp: string
}

export interface ConciergeSession {
  readonly sessionId: string
  readonly venueId: string
  readonly messages: ConciergeMessage[]
  readonly mode: 'text' | 'video'
  readonly status: 'active' | 'idle' | 'ended'
}

export interface ConciergeResponse {
  readonly message: string
  readonly audioUrl?: string // for TTS
  readonly suggestedActions?: string[]
}
