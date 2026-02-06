/**
 * React hook for floor plan collaboration.
 *
 * Manages the Y.Doc lifecycle, WebSocket provider connection,
 * IndexedDB offline persistence, and the Yjs ↔ Zustand bridge.
 */
import { useEffect, useRef, useState, useCallback } from 'react'
import * as Y from 'yjs'
import { WebsocketProvider } from 'y-websocket'
import { IndexeddbPersistence } from 'y-indexeddb'
import { Awareness } from 'y-protocols/awareness'
import { createYjsBridge, type YjsBridge } from './yjsBridge'
import {
  setLocalPresence,
  getRemotePresences,
  getPresenceColor,
  type UserPresence,
} from './awareness'

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected'

export interface CollaborationState {
  /** Current connection status */
  status: ConnectionStatus
  /** Number of connected users (including self) */
  userCount: number
  /** Remote users' presence data */
  remoteUsers: Map<number, UserPresence>
  /** Update local cursor position */
  updateCursor: (x: number, y: number) => void
  /** Clear local cursor (mouse left canvas) */
  clearCursor: () => void
  /** Update local selection */
  updateSelection: (ids: string[]) => void
  /** Disconnect and clean up */
  disconnect: () => void
}

interface UseCollaborationOptions {
  /** Floor plan document ID (room name) */
  roomId: string
  /** User display name */
  userName: string
  /** WebSocket server URL (e.g., "ws://localhost:1234") */
  wsUrl?: string
  /** Enable IndexedDB offline persistence */
  enableOffline?: boolean
}

export function useCollaboration(options: UseCollaborationOptions | null): CollaborationState | null {
  const { roomId, userName, wsUrl, enableOffline = true } = options ?? { roomId: '', userName: '' }
  const [status, setStatus] = useState<ConnectionStatus>('disconnected')
  const [userCount, setUserCount] = useState(1)
  const [remoteUsers, setRemoteUsers] = useState<Map<number, UserPresence>>(new Map())

  const docRef = useRef<Y.Doc | null>(null)
  const wsProviderRef = useRef<WebsocketProvider | null>(null)
  const idbRef = useRef<IndexeddbPersistence | null>(null)
  const bridgeRef = useRef<YjsBridge | null>(null)
  const awarenessRef = useRef<Awareness | null>(null)

  // Set up collaboration
  useEffect(() => {
    if (!options || !roomId) return

    const doc = new Y.Doc()
    docRef.current = doc

    // Yjs ↔ Store bridge
    const bridge = createYjsBridge(doc)
    bridgeRef.current = bridge

    // IndexedDB offline persistence
    let idb: IndexeddbPersistence | undefined
    if (enableOffline) {
      idb = new IndexeddbPersistence(`floorplan-${roomId}`, doc)
      idbRef.current = idb
      idb.on('synced', () => {
        // Pull persisted data into store once loaded
        bridge.pullFromDoc()
      })
    }

    // WebSocket provider (only if URL provided)
    let wsProvider: WebsocketProvider | undefined
    if (wsUrl) {
      wsProvider = new WebsocketProvider(wsUrl, roomId, doc, {
        connect: true,
      })
      wsProviderRef.current = wsProvider

      const awareness = wsProvider.awareness
      awarenessRef.current = awareness

      // Set initial local presence
      setLocalPresence(awareness, {
        name: userName,
        color: getPresenceColor(awareness.clientID),
        cursor: null,
        selectedIds: [],
      })

      // Connection status tracking
      wsProvider.on('status', (event: { status: string }) => {
        if (event.status === 'connected') setStatus('connected')
        else if (event.status === 'connecting') setStatus('connecting')
        else setStatus('disconnected')
      })

      // Awareness change tracking (throttled to ~15Hz via Yjs default)
      const onAwarenessChange = () => {
        const remote = getRemotePresences(awareness)
        setRemoteUsers(new Map(remote))
        setUserCount(awareness.getStates().size)
      }
      awareness.on('change', onAwarenessChange)

      // Sync: if doc already has data from IndexedDB, push to server
      if (idb) {
        idb.on('synced', () => {
          // Doc loaded from IndexedDB — WebSocket will merge via CRDT
        })
      }

      // Wait for initial sync then pull
      wsProvider.on('sync', (synced: boolean) => {
        if (synced) {
          bridge.pullFromDoc()
        }
      })
    }

    // Start bridge observation
    bridge.connect()

    // If no WebSocket and no IndexedDB, push current store to doc
    if (!wsUrl && !enableOffline) {
      bridge.pushToDoc()
    }

    return () => {
      bridge.disconnect()
      wsProvider?.disconnect()
      wsProvider?.destroy()
      idb?.destroy()
      doc.destroy()
      docRef.current = null
      wsProviderRef.current = null
      idbRef.current = null
      bridgeRef.current = null
      awarenessRef.current = null
      setStatus('disconnected')
    }
  }, [roomId, userName, wsUrl, enableOffline, options])

  const updateCursor = useCallback((x: number, y: number) => {
    if (awarenessRef.current) {
      setLocalPresence(awarenessRef.current, { cursor: { x, y } })
    }
  }, [])

  const clearCursor = useCallback(() => {
    if (awarenessRef.current) {
      setLocalPresence(awarenessRef.current, { cursor: null })
    }
  }, [])

  const updateSelection = useCallback((ids: string[]) => {
    if (awarenessRef.current) {
      setLocalPresence(awarenessRef.current, { selectedIds: ids })
    }
  }, [])

  const disconnect = useCallback(() => {
    bridgeRef.current?.disconnect()
    wsProviderRef.current?.disconnect()
  }, [])

  if (!options) return null

  return {
    status,
    userCount,
    remoteUsers,
    updateCursor,
    clearCursor,
    updateSelection,
    disconnect,
  }
}
