/**
 * Omniverse Kit App Streaming session management.
 *
 * Production: connects to Kit App Streaming via WebRTC.
 * Development: mock session with simulated state.
 */

import type { OmniverseSession, OmniverseSessionConfig, OmniverseSceneState } from '../types'

// ─── Session Interface ───────────────────────────────────────────────────────

export interface OmniverseStreamingClient {
  /** Request a new streaming session. */
  connect(config: OmniverseSessionConfig): Promise<OmniverseSession>
  /** Send scene state to the active session. */
  syncScene(sessionId: string, scene: OmniverseSceneState): Promise<void>
  /** Disconnect and release GPU resources. */
  disconnect(sessionId: string): Promise<void>
  /** Get current session status. */
  getSession(sessionId: string): Promise<OmniverseSession | null>
}

// ─── Mock Client ─────────────────────────────────────────────────────────────

export class MockOmniverseClient implements OmniverseStreamingClient {
  private sessions = new Map<string, OmniverseSession>()
  private sessionCounter = 0

  async connect(config: OmniverseSessionConfig): Promise<OmniverseSession> {
    const sessionId = `omni-session-${++this.sessionCounter}`
    const session: OmniverseSession = {
      sessionId,
      streamUrl: `wss://mock-omniverse.nvidia.com/stream/${sessionId}`,
      status: 'active',
      latencyMs: 45,
      fps: config.quality === 'ultra' ? 30 : 60,
    }
    this.sessions.set(sessionId, session)
    return session
  }

  async syncScene(sessionId: string, _scene: OmniverseSceneState): Promise<void> {
    const session = this.sessions.get(sessionId)
    if (!session) throw new Error(`Session not found: ${sessionId}`)
    // Mock: scene sync is a no-op
  }

  async disconnect(sessionId: string): Promise<void> {
    const session = this.sessions.get(sessionId)
    if (session) {
      this.sessions.set(sessionId, { ...session, status: 'disconnected' })
    }
  }

  async getSession(sessionId: string): Promise<OmniverseSession | null> {
    return this.sessions.get(sessionId) ?? null
  }
}

/** Create an Omniverse streaming client. */
export function createOmniverseClient(
  _orchestratorUrl?: string,
): OmniverseStreamingClient {
  // Always return mock for now — production needs Kit App Streaming infra
  return new MockOmniverseClient()
}
