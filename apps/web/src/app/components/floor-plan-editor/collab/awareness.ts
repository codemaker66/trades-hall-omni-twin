/**
 * Presence & Awareness for Floor Plan Collaboration
 *
 * Tracks connected users, their cursor positions, and selected items.
 * Uses the Yjs Awareness protocol.
 */
import { Awareness } from 'y-protocols/awareness'

export interface UserPresence {
  /** User display name */
  name: string
  /** Unique color for this user (assigned by hash) */
  color: string
  /** Cursor position in 2D feet coordinates, null if off-canvas */
  cursor: { x: number; y: number } | null
  /** IDs of items currently selected by this user */
  selectedIds: string[]
  /** Timestamp of last update */
  lastUpdate: number
}

// Deterministic color palette for collaboration (high contrast, colorblind-safe)
const PRESENCE_COLORS = [
  '#e6194b', '#3cb44b', '#4363d8', '#f58231',
  '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
  '#fabed4', '#469990', '#dcbeff', '#9A6324',
]

export function getPresenceColor(clientId: number): string {
  return PRESENCE_COLORS[clientId % PRESENCE_COLORS.length]!
}

/**
 * Set the local user's presence state.
 */
export function setLocalPresence(
  awareness: Awareness,
  data: Partial<UserPresence>,
): void {
  const existing = awareness.getLocalState() as UserPresence | null
  awareness.setLocalState({
    ...existing,
    ...data,
    lastUpdate: Date.now(),
  })
}

/**
 * Get all remote users' presence (excludes local user).
 */
export function getRemotePresences(awareness: Awareness): Map<number, UserPresence> {
  const states = awareness.getStates()
  const localId = awareness.clientID
  const result = new Map<number, UserPresence>()

  states.forEach((state, clientId) => {
    if (clientId !== localId && state && typeof state === 'object' && 'name' in state) {
      result.set(clientId, state as UserPresence)
    }
  })

  return result
}

/**
 * Get all connected users (including local) with their presence data.
 */
export function getAllPresences(awareness: Awareness): Map<number, UserPresence> {
  const states = awareness.getStates()
  const result = new Map<number, UserPresence>()

  states.forEach((state, clientId) => {
    if (state && typeof state === 'object' && 'name' in state) {
      result.set(clientId, state as UserPresence)
    }
  })

  return result
}
