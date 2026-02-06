'use client'

import { Scene } from './Scene'
import { TrashBin } from './TrashBin'
import { motion, AnimatePresence } from 'framer-motion'
import { useVenueStore, type FurnitureType, type ProjectImportMode } from '../../store'
import { useState, useEffect, useMemo, useRef, type ChangeEvent } from 'react'

const INLINE_SHARE_HASH_PARAM = 'project'
const SNAPSHOT_SHARE_HASH_PARAM = 'share'
const MAX_INLINE_SHARE_URL_LENGTH = 12000

const encodeProjectPayloadForUrl = (payload: string): string => {
  const bytes = new TextEncoder().encode(payload)
  let binary = ''
  for (const byte of bytes) {
    binary += String.fromCharCode(byte)
  }

  return btoa(binary)
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/g, '')
}

const decodeProjectPayloadFromUrl = (encoded: string): string => {
  const normalized = encoded
    .replace(/-/g, '+')
    .replace(/_/g, '/')
  const padded = normalized + '='.repeat((4 - (normalized.length % 4)) % 4)
  const binary = atob(padded)
  const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0))
  return new TextDecoder().decode(bytes)
}

const copyTextToClipboard = async (text: string): Promise<boolean> => {
  if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text)
      return true
    } catch {
      // Fallback below.
    }
  }

  if (typeof document === 'undefined') {
    return false
  }

  const textarea = document.createElement('textarea')
  textarea.value = text
  textarea.setAttribute('readonly', '')
  textarea.style.position = 'fixed'
  textarea.style.opacity = '0'
  textarea.style.pointerEvents = 'none'
  document.body.appendChild(textarea)
  textarea.select()

  let copied = false
  try {
    copied = document.execCommand('copy')
  } catch {
    copied = false
  } finally {
    document.body.removeChild(textarea)
  }

  return copied
}

export default function VenueViewer() {
  const addItem = useVenueStore((state) => state.addItem)
  const selectedIds = useVenueStore((state) => state.selectedIds)
  const removeItems = useVenueStore((state) => state.removeItems)
  const ungroupItems = useVenueStore((state) => state.ungroupItems)
  const items = useVenueStore((state) => state.items)
  const snappingEnabled = useVenueStore((state) => state.snappingEnabled)
  const toggleSnapping = useVenueStore((state) => state.toggleSnapping)
  const setDraggedItem = useVenueStore((state) => state.setDraggedItem)
  const transformMode = useVenueStore((state) => state.transformMode)
  const toggleTransformMode = useVenueStore((state) => state.toggleTransformMode)
  const rotateSelection = useVenueStore((state) => state.rotateSelection)
  const canUndo = useVenueStore((state) => state.canUndo)
  const canRedo = useVenueStore((state) => state.canRedo)
  const undo = useVenueStore((state) => state.undo)
  const redo = useVenueStore((state) => state.redo)
  const beginHistoryBatch = useVenueStore((state) => state.beginHistoryBatch)
  const endHistoryBatch = useVenueStore((state) => state.endHistoryBatch)

  const openChairPrompt = useVenueStore((state) => state.openChairPrompt)
  const closeChairPrompt = useVenueStore((state) => state.closeChairPrompt)
  const chairPrompt = useVenueStore((state) => state.chairPrompt)
  const inventoryCatalog = useVenueStore((state) => state.inventoryCatalog)
  const inventoryWarning = useVenueStore((state) => state.inventoryWarning)
  const clearInventoryWarning = useVenueStore((state) => state.clearInventoryWarning)
  const updateInventoryItem = useVenueStore((state) => state.updateInventoryItem)
  const scenarios = useVenueStore((state) => state.scenarios)
  const activeScenarioId = useVenueStore((state) => state.activeScenarioId)
  const saveScenario = useVenueStore((state) => state.saveScenario)
  const loadScenario = useVenueStore((state) => state.loadScenario)
  const deleteScenario = useVenueStore((state) => state.deleteScenario)
  const renameScenario = useVenueStore((state) => state.renameScenario)
  const setScenarioStatus = useVenueStore((state) => state.setScenarioStatus)
  const hasInventoryForType = useVenueStore((state) => state.hasInventoryForType)
  const resetProject = useVenueStore((state) => state.resetProject)
  const exportProject = useVenueStore((state) => state.exportProject)
  const importProject = useVenueStore((state) => state.importProject)

  const [chairCount, setChairCount] = useState<number>(8)
  const [scenarioName, setScenarioName] = useState('')
  const [projectTransferNotice, setProjectTransferNotice] = useState<string | null>(null)
  const [projectTransferError, setProjectTransferError] = useState<string | null>(null)
  const [isShortSharePending, setIsShortSharePending] = useState(false)
  const importInputRef = useRef<HTMLInputElement>(null)
  const pendingImportModeRef = useRef<ProjectImportMode>('replace')
  const hasHydratedSharedProjectRef = useRef(false)

  const inventoryUsage = useMemo(() => {
    const usage: Record<FurnitureType, number> = {
      'round-table': 0,
      'trestle-table': 0,
      'chair': 0,
      'platform': 0
    }

    for (const item of items) {
      usage[item.type] += 1
    }

    return usage
  }, [items])

  const layoutMetrics = useMemo(() => {
    let tableCount = 0
    let chairCount = 0
    let platformCount = 0

    for (const item of items) {
      if (item.type === 'round-table' || item.type === 'trestle-table') tableCount += 1
      if (item.type === 'chair') chairCount += 1
      if (item.type === 'platform') platformCount += 1
    }

    return {
      seatCount: chairCount,
      tableCount,
      chairCount,
      platformCount
    }
  }, [items])

  const canAddRoundTable = hasInventoryForType('round-table')
  const canAddTrestleTable = hasInventoryForType('trestle-table')
  const canAddChair = hasInventoryForType('chair')
  const canAddPlatform = hasInventoryForType('platform')

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target?.isContentEditable) return
      const tagName = target?.tagName
      if (tagName === 'INPUT' || tagName === 'TEXTAREA' || tagName === 'SELECT') return

      const key = e.key.toLowerCase()
      const draggedItemType = useVenueStore.getState().draggedItemType
      if ((key === 'q' || key === 'e') && draggedItemType === 'chair') return

      const hasModifier = e.metaKey || e.ctrlKey
      const isUndo = hasModifier && !e.shiftKey && key === 'z'
      const isRedo = hasModifier && (key === 'y' || (e.shiftKey && key === 'z'))
      const isSaveScenario = hasModifier && key === 's'

      if (isUndo) {
        e.preventDefault()
        useVenueStore.getState().undo()
      } else if (isRedo) {
        e.preventDefault()
        useVenueStore.getState().redo()
      } else if (isSaveScenario) {
        e.preventDefault()
        useVenueStore.getState().saveScenario()
      } else if (key === 'q') {
        useVenueStore.getState().rotateSelection(-90)
      } else if (key === 'e') {
        useVenueStore.getState().rotateSelection(90)
      } else if (key === 'delete' || key === 'backspace') {
        const selectedIds = useVenueStore.getState().selectedIds
        if (selectedIds.length > 0) {
          useVenueStore.getState().removeItems(selectedIds)
        }
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  useEffect(() => {
    if (hasHydratedSharedProjectRef.current) return
    hasHydratedSharedProjectRef.current = true

    let cancelled = false
    const hash = window.location.hash
    if (!hash || hash.length < 2) return

    const params = new URLSearchParams(hash.slice(1))
    const snapshotCode = params.get(SNAPSHOT_SHARE_HASH_PARAM)
    const encodedProject = params.get(INLINE_SHARE_HASH_PARAM)
    if (!snapshotCode && !encodedProject) return

    const applyImportedProject = (payload: string, sourceLabel: string): boolean => {
      const result = importProject(payload, { mode: 'replace' })
      if (!result.ok) {
        if (!cancelled) {
          setProjectTransferNotice(null)
          setProjectTransferError(result.message)
        }
        return false
      }

      if (!cancelled) {
        setProjectTransferNotice(`Shared project loaded from ${sourceLabel}.`)
        setProjectTransferError(null)
        setScenarioName('')
      }
      return true
    }

    const hydrateSharedProject = async () => {
      try {
        if (snapshotCode) {
          const response = await fetch(`/api/share?code=${encodeURIComponent(snapshotCode)}`, { cache: 'no-store' })
          const payloadResponse = await response.json().catch(() => null) as { payload?: string, error?: string } | null
          if (!response.ok) {
            if (!cancelled) {
              setProjectTransferNotice(null)
              setProjectTransferError(payloadResponse?.error ?? 'Short share link is invalid or expired.')
            }
            return
          }

          if (typeof payloadResponse?.payload !== 'string') {
            if (!cancelled) {
              setProjectTransferNotice(null)
              setProjectTransferError('Short share link payload is invalid.')
            }
            return
          }

          const imported = applyImportedProject(payloadResponse.payload, 'short link')
          if (!imported || cancelled) return
        } else if (encodedProject) {
          const payload = decodeProjectPayloadFromUrl(encodedProject)
          const imported = applyImportedProject(payload, 'inline link')
          if (!imported || cancelled) return
        }

        const urlWithoutHash = `${window.location.pathname}${window.location.search}`
        window.history.replaceState(null, '', urlWithoutHash)
      } catch {
        if (!cancelled) {
          setProjectTransferNotice(null)
          setProjectTransferError(snapshotCode ? 'Short share link is invalid or expired.' : 'Shared link is invalid or corrupted.')
        }
      }
    }

    void hydrateSharedProject()
    return () => {
      cancelled = true
    }
  }, [importProject])

  const handleAddItemRequest = (type: FurnitureType) => {
    if (type === 'round-table') {
      if (!canAddRoundTable) {
        addItem('round-table', [0, 0, 0], [0, 0, 0], undefined, { recordHistory: false })
        return
      }
      openChairPrompt(type)
      setChairCount(5) // Default based on user request example
    } else {
      addItem(type)
    }
  }

  const handleSaveScenario = () => {
    saveScenario(scenarioName)
    setScenarioName('')
  }

  const handleResetProject = () => {
    const shouldReset = window.confirm('Reset this project? This clears layout, scenarios, and inventory customizations on this device.')
    if (!shouldReset) return
    resetProject()
    setScenarioName('')
    setProjectTransferNotice(null)
    setProjectTransferError(null)
  }

  const handleExportProject = () => {
    try {
      const payload = exportProject()
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
      const filename = `omnitwin-project-${timestamp}.json`
      const blob = new Blob([payload], { type: 'application/json' })
      const url = window.URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = filename
      anchor.style.display = 'none'
      document.body.appendChild(anchor)
      anchor.click()
      document.body.removeChild(anchor)
      window.URL.revokeObjectURL(url)

      setProjectTransferNotice(`Project exported as ${filename}.`)
      setProjectTransferError(null)
    } catch {
      setProjectTransferNotice(null)
      setProjectTransferError('Project export failed. Please try again.')
    }
  }

  const handleShareShortProject = async () => {
    setIsShortSharePending(true)

    try {
      const payload = exportProject()
      const response = await fetch('/api/share', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ payload }),
        cache: 'no-store'
      })

      const shareResponse = await response.json().catch(() => null) as { code?: string, expiresAt?: string, error?: string } | null
      if (!response.ok || typeof shareResponse?.code !== 'string') {
        setProjectTransferNotice(null)
        setProjectTransferError(shareResponse?.error ?? 'Could not create short share link.')
        return
      }

      const shareUrl = `${window.location.origin}${window.location.pathname}${window.location.search}#${SNAPSHOT_SHARE_HASH_PARAM}=${shareResponse.code}`
      const copied = await copyTextToClipboard(shareUrl)
      if (!copied) {
        setProjectTransferNotice(null)
        setProjectTransferError('Could not copy short share link. Use Export instead.')
        return
      }

      const expiresLabel = shareResponse.expiresAt
        ? new Date(shareResponse.expiresAt).toLocaleDateString()
        : null
      setProjectTransferNotice(
        expiresLabel
          ? `Short share link copied. Expires ${expiresLabel}.`
          : 'Short share link copied to clipboard.'
      )
      setProjectTransferError(null)
    } catch {
      setProjectTransferNotice(null)
      setProjectTransferError('Could not create short share link.')
    } finally {
      setIsShortSharePending(false)
    }
  }

  const handleShareInlineProject = async () => {
    try {
      const payload = exportProject()
      const encodedProject = encodeProjectPayloadForUrl(payload)
      const shareUrl = `${window.location.origin}${window.location.pathname}${window.location.search}#${INLINE_SHARE_HASH_PARAM}=${encodedProject}`

      if (shareUrl.length > MAX_INLINE_SHARE_URL_LENGTH) {
        setProjectTransferNotice(null)
        setProjectTransferError('Project is too large to share as a URL. Use Export instead.')
        return
      }

      const copied = await copyTextToClipboard(shareUrl)
      if (!copied) {
        setProjectTransferNotice(null)
        setProjectTransferError('Could not copy share link. Use Export instead.')
        return
      }

      setProjectTransferNotice('Inline share link copied to clipboard.')
      setProjectTransferError(null)
    } catch {
      setProjectTransferNotice(null)
      setProjectTransferError('Could not generate inline share link.')
    }
  }

  const openProjectImportPicker = (mode: ProjectImportMode) => {
    pendingImportModeRef.current = mode
    importInputRef.current?.click()
  }

  const handleImportFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return

    if (pendingImportModeRef.current === 'replace') {
      const shouldReplace = window.confirm('Import and replace the current project on this device?')
      if (!shouldReplace) return
    }

    try {
      const payload = await file.text()
      const result = importProject(payload, { mode: pendingImportModeRef.current })
      if (!result.ok) {
        setProjectTransferNotice(null)
        setProjectTransferError(result.message)
        return
      }

      setProjectTransferNotice(result.message)
      setProjectTransferError(null)
      setScenarioName('')
    } catch {
      setProjectTransferNotice(null)
      setProjectTransferError('Project import failed. Please try again.')
    }
  }

  const handleConfirmChairs = () => {
    if (!chairPrompt) return

    const { tableId } = chairPrompt
    beginHistoryBatch()

    try {
      // If we are reconfiguring an existing table
      if (tableId) {
        const table = items.find(i => i.id === tableId)
        if (table) {
          // 1. Remove old items in the same group (chairs) but keep the table
          const groupItems = items.filter(i => i.groupId === table.groupId && i.id !== tableId)
          removeItems(groupItems.map(i => i.id), { recordHistory: false })

          // 2. Add Chairs around existing table position
          const radius = 1.3
          const center = table.position
          for (let i = 0; i < chairCount; i++) {
            const angle = (i / chairCount) * Math.PI * 2
            const x = center[0] + Math.sin(angle) * radius
            const z = center[2] + Math.cos(angle) * radius
            const rotY = angle + Math.PI
            addItem('chair', [x, 0, z], [0, rotY, 0], table.groupId, { recordHistory: false })
          }
        }
      } else {
        // Standard new table + chairs placement
        if (!hasInventoryForType('round-table')) {
          addItem('round-table', [0, 0, 0], [0, 0, 0], undefined, { recordHistory: false })
          closeChairPrompt()
          return
        }

        // Create Group ID
        const groupId = crypto.randomUUID()

        // 1. Add the Round Table at center (for now)
        const center: [number, number, number] = [0, 0, 0]
        addItem('round-table', center, [0, 0, 0], groupId, { recordHistory: false })

        // 2. Add Chairs evenly spaced
        const radius = 1.3 // Distance from center (Table R ~0.9 + clearance)
        for (let i = 0; i < chairCount; i++) {
          // Calculate angle
          const angle = (i / chairCount) * Math.PI * 2

          // Position
          const x = center[0] + Math.sin(angle) * radius
          const z = center[2] + Math.cos(angle) * radius

          // Rotation: Face Center
          const rotY = angle + Math.PI

          addItem('chair', [x, 0, z], [0, rotY, 0], groupId, { recordHistory: false })
        }
      }
    } finally {
      endHistoryBatch()
    }

    closeChairPrompt()
  }

  // Check if current selection has any groups
  const hasGroupsSelected = selectedIds.some(id => items.find(i => i.id === id)?.groupId)

  const handleClosePrompt = () => closeChairPrompt()

  return (
    <div className="relative w-full h-[100dvh] bg-[#0b0f14] selection:bg-[#6366f1] selection:text-white overflow-hidden">
      {/* 3D Scene Layer */}
      <div className="absolute inset-0 z-0">
        <Scene />
      </div>

      {/* UI Overlay Layer */}
      {/* UI Overlay Layer */}
      {/* UI Overlay Layer */}
      <div className="absolute inset-0 z-10 pointer-events-none p-6 flex flex-col justify-between font-serif">
        {/* Header / HUD */}
        <header className="flex justify-between items-start">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="border-4 border-[#3e2723] p-4 rounded-sm pointer-events-auto shadow-2xl relative overflow-hidden group min-w-[300px]"
            style={{
              backgroundColor: '#2d1b15',
              backgroundImage: `repeating-linear-gradient(90deg, 
                    rgba(255,255,255,0.03) 0px, 
                    rgba(255,255,255,0.03) 1px, 
                    transparent 1px, 
                    transparent 40px, 
                    rgba(0,0,0,0.4) 40px, 
                    rgba(0,0,0,0.4) 42px),
                    linear-gradient(to bottom, rgba(0,0,0,0.2), rgba(0,0,0,0.6))`
            }}
          >
            {/* Iron Studs / Bolts */}
            <div className="absolute top-1 left-1 w-2 h-2 rounded-full bg-[#1a110e] shadow-[inset_0_1px_1px_rgba(255,255,255,0.1)]" />
            <div className="absolute top-1 right-1 w-2 h-2 rounded-full bg-[#1a110e] shadow-[inset_0_1px_1px_rgba(255,255,255,0.1)]" />
            <div className="absolute bottom-1 left-1 w-2 h-2 rounded-full bg-[#1a110e] shadow-[inset_0_1px_1px_rgba(255,255,255,0.1)]" />
            <div className="absolute bottom-1 right-1 w-2 h-2 rounded-full bg-[#1a110e] shadow-[inset_0_1px_1px_rgba(255,255,255,0.1)]" />

            <h1 className="text-2xl font-bold text-[#d7ccc8] tracking-widest flex flex-col items-center drop-shadow-md pb-2 border-b border-[#5d4037]/50">
              OMNITWIN
              <span className="text-[#a1887f] text-xs font-normal tracking-[0.2em] mt-1 uppercase">Trades Hall Planner</span>
            </h1>
          </motion.div>

          {/* Top Right Tools */}
          <div className="flex gap-2 pointer-events-auto">
            <button
              onClick={undo}
              disabled={!canUndo}
              className={`px-4 py-2 rounded-sm border-2 transition-all font-semibold shadow-md ${canUndo
                ? 'bg-[#3e2723] border-[#8d6e63] text-[#d7ccc8] hover:bg-[#4e342e]'
                : 'bg-[#1a0f0a]/70 border-[#2a1b16] text-[#5e4a43] cursor-not-allowed'
                }`}
            >
              Undo
            </button>

            <button
              onClick={redo}
              disabled={!canRedo}
              className={`px-4 py-2 rounded-sm border-2 transition-all font-semibold shadow-md ${canRedo
                ? 'bg-[#3e2723] border-[#8d6e63] text-[#d7ccc8] hover:bg-[#4e342e]'
                : 'bg-[#1a0f0a]/70 border-[#2a1b16] text-[#5e4a43] cursor-not-allowed'
                }`}
            >
              Redo
            </button>

            <button
              onClick={toggleSnapping}
              className={`px-4 py-2 rounded-sm border-2 transition-all font-semibold shadow-md ${snappingEnabled
                ? 'bg-[#3e2723] border-[#8d6e63] text-[#d7ccc8]'
                : 'bg-[#1a0f0a]/90 border-[#3e2723] text-[#795548] hover:bg-[#2d1b15] hover:text-[#a1887f]'
                }`}
            >
              Grid Snap: {snappingEnabled ? 'ON' : 'OFF'}
            </button>

            {hasGroupsSelected && (
              <button
                onClick={toggleTransformMode}
                className={`px-4 py-2 rounded-sm border-2 transition-all font-semibold shadow-md ${transformMode === 'rotate'
                  ? 'bg-[#4e342e] border-[#ffb74d] text-[#ffcc80]'
                  : 'bg-[#1a0f0a]/90 border-[#3e2723] text-[#795548] hover:bg-[#2d1b15] hover:text-[#a1887f]'
                  }`}
              >
                Mode: {transformMode === 'rotate' ? 'ROTATE' : 'MOVE'}
              </button>
            )}

            {transformMode === 'rotate' && hasGroupsSelected && (
              <div className="flex gap-1 bg-[#2d1b15] p-1 rounded-sm border border-[#5d4037]">
                <button
                  onClick={() => rotateSelection(-90)}
                  className="px-3 py-2 bg-[#3e2723] hover:bg-[#4e342e] text-[#ffcc80] rounded-sm font-bold border border-[#ffb74d]/30 transition-colors"
                  title="Rotate -90°"
                >
                  ↺ 90°
                </button>
                <button
                  onClick={() => rotateSelection(-1)}
                  className="px-2 py-2 bg-[#3e2723] hover:bg-[#4e342e] text-[#ffcc80] rounded-sm font-medium border border-[#ffb74d]/30 text-xs transition-colors"
                  title="Rotate -1°"
                >
                  -1°
                </button>
                <button
                  onClick={() => rotateSelection(1)}
                  className="px-2 py-2 bg-[#3e2723] hover:bg-[#4e342e] text-[#ffcc80] rounded-sm font-medium border border-[#ffb74d]/30 text-xs transition-colors"
                  title="Rotate +1°"
                >
                  +1°
                </button>
                <button
                  onClick={() => rotateSelection(90)}
                  className="px-3 py-2 bg-[#3e2723] hover:bg-[#4e342e] text-[#ffcc80] rounded-sm font-bold border border-[#ffb74d]/30 transition-colors"
                  title="Rotate +90°"
                >
                  90° ↻
                </button>
              </div>
            )}

            {hasGroupsSelected && (
              <button
                onClick={() => ungroupItems(selectedIds)}
                className="px-4 py-2 bg-[#3e2723] border-2 border-[#ffb74d] text-[#ffcc80] rounded-sm hover:bg-[#4e342e] transition-all font-semibold shadow-md"
              >
                Ungroup
              </button>
            )}

            {!hasGroupsSelected && selectedIds.length > 1 && (
              <button
                onClick={() => useVenueStore.getState().groupItems(selectedIds)}
                className="px-4 py-2 bg-[#3e2723] border-2 border-[#ffb74d] text-[#ffcc80] rounded-sm hover:bg-[#4e342e] transition-all font-semibold shadow-md"
              >
                Group
              </button>
            )}

            {selectedIds.length > 0 && (
              <button
                onClick={() => removeItems(selectedIds)}
                className="px-4 py-2 bg-[#3e2723] border-2 border-[#ef5350] text-[#ef9a9a] rounded-sm hover:bg-[#4e342e] transition-all font-semibold shadow-md"
              >
                Delete Selected ({selectedIds.length})
              </button>
            )}
          </div>
        </header>

        <motion.aside
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.15 }}
          className="absolute top-28 left-6 w-96 max-h-[calc(100dvh-190px)] overflow-y-auto border-2 border-[#3e2723] bg-[#1a0f0a]/90 p-4 rounded-sm shadow-2xl pointer-events-auto space-y-4"
        >
          <div>
            <h2 className="text-sm font-bold tracking-wider text-[#d7ccc8] uppercase mb-2">Layout Metrics</h2>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="bg-[#2d1b15] border border-[#5d4037] rounded-sm px-3 py-2 text-[#d7ccc8]">Seats: {layoutMetrics.seatCount}</div>
              <div className="bg-[#2d1b15] border border-[#5d4037] rounded-sm px-3 py-2 text-[#d7ccc8]">Tables: {layoutMetrics.tableCount}</div>
              <div className="bg-[#2d1b15] border border-[#5d4037] rounded-sm px-3 py-2 text-[#d7ccc8]">Chairs: {layoutMetrics.chairCount}</div>
              <div className="bg-[#2d1b15] border border-[#5d4037] rounded-sm px-3 py-2 text-[#d7ccc8]">Platforms: {layoutMetrics.platformCount}</div>
            </div>
          </div>

          <div>
            <h2 className="text-sm font-bold tracking-wider text-[#d7ccc8] uppercase mb-2">Inventory</h2>
            <div className="space-y-2">
              {inventoryCatalog.map((inventoryItem) => {
                const used = inventoryUsage[inventoryItem.furnitureType]
                const available = Math.max(0, inventoryItem.quantityTotal - inventoryItem.quantityReserved)
                const usageRatio = available > 0 ? used / available : 1
                const ratio = Math.min(1, usageRatio)
                const barColor = usageRatio > 1 ? 'bg-[#ef5350]' : usageRatio > 0.8 ? 'bg-[#ffb74d]' : 'bg-[#66bb6a]'

                return (
                  <div key={inventoryItem.id} className="bg-[#2d1b15] border border-[#5d4037] rounded-sm p-2">
                    <div className="flex items-center justify-between text-xs text-[#d7ccc8] mb-1">
                      <span>{inventoryItem.name}</span>
                      <span>{used}/{available}</span>
                    </div>
                    <div className="h-2 bg-[#1a110e] rounded-sm overflow-hidden mb-2">
                      <div className={`h-full ${barColor}`} style={{ width: `${ratio * 100}%` }} />
                    </div>
                    <div className="flex items-center justify-between gap-2">
                      <button
                        onClick={() => updateInventoryItem(inventoryItem.id, { quantityTotal: Math.max(0, inventoryItem.quantityTotal - 1) })}
                        className="px-2 py-1 text-xs bg-[#3e2723] border border-[#8d6e63] rounded-sm text-[#d7ccc8] hover:bg-[#4e342e]"
                      >
                        - Stock
                      </button>
                      <button
                        onClick={() => updateInventoryItem(inventoryItem.id, { quantityTotal: inventoryItem.quantityTotal + 1 })}
                        className="px-2 py-1 text-xs bg-[#3e2723] border border-[#8d6e63] rounded-sm text-[#d7ccc8] hover:bg-[#4e342e]"
                      >
                        + Stock
                      </button>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          <div>
            <h2 className="text-sm font-bold tracking-wider text-[#d7ccc8] uppercase mb-2">Scenarios</h2>
            <div className="flex gap-2 mb-2">
              <input
                value={scenarioName}
                onChange={(e) => setScenarioName(e.target.value)}
                placeholder="Scenario name"
                className="flex-1 bg-[#120a07] border border-[#5d4037] rounded-sm px-3 py-2 text-sm text-[#d7ccc8] focus:outline-none focus:border-[#ffb74d]"
              />
              <button
                onClick={handleSaveScenario}
                className="px-3 py-2 text-sm bg-[#3e2723] border border-[#ffb74d] text-[#ffcc80] rounded-sm hover:bg-[#4e342e]"
              >
                Save
              </button>
            </div>

            <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
              {scenarios.length === 0 && (
                <div className="text-xs text-[#a1887f]">No scenarios yet. Save one to branch the layout.</div>
              )}
              {scenarios.map((scenario) => (
                <div
                  key={scenario.id}
                  className={`border rounded-sm p-2 ${activeScenarioId === scenario.id ? 'border-[#ffb74d] bg-[#2d1b15]' : 'border-[#5d4037] bg-[#1a110e]'}`}
                >
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <button
                      onClick={() => loadScenario(scenario.id)}
                      className="text-sm text-left text-[#d7ccc8] hover:text-white truncate"
                    >
                      {scenario.name}
                    </button>
                    <span className="text-[10px] text-[#a1887f]">{new Date(scenario.updatedAt).toLocaleDateString()}</span>
                  </div>

                  <div className="flex items-center gap-2">
                    <select
                      value={scenario.status}
                      onChange={(e) => setScenarioStatus(scenario.id, e.target.value as 'draft' | 'review' | 'approved')}
                      className="flex-1 bg-[#120a07] border border-[#5d4037] rounded-sm px-2 py-1 text-xs text-[#d7ccc8]"
                    >
                      <option value="draft">Draft</option>
                      <option value="review">Review</option>
                      <option value="approved">Approved</option>
                    </select>
                    <button
                      onClick={() => {
                        const nextName = window.prompt('Rename scenario', scenario.name)
                        if (!nextName) return
                        renameScenario(scenario.id, nextName)
                      }}
                      className="px-2 py-1 text-xs bg-[#3e2723] border border-[#8d6e63] rounded-sm text-[#d7ccc8] hover:bg-[#4e342e]"
                    >
                      Rename
                    </button>
                    <button
                      onClick={() => deleteScenario(scenario.id)}
                      className="px-2 py-1 text-xs bg-[#3e2723] border border-[#ef5350] rounded-sm text-[#ef9a9a] hover:bg-[#4e342e]"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-3 flex items-center justify-between gap-2">
              <span className="text-[11px] text-[#a1887f]">Autosave is enabled for this browser.</span>
              <button
                onClick={handleResetProject}
                className="px-2 py-1 text-xs bg-[#3e2723] border border-[#ef5350] rounded-sm text-[#ef9a9a] hover:bg-[#4e342e]"
              >
                Reset Project
              </button>
            </div>

            <input
              ref={importInputRef}
              type="file"
              accept=".json,application/json"
              className="hidden"
              onChange={handleImportFileChange}
            />

            <div className="mt-2 grid grid-cols-5 gap-2">
              <button
                onClick={handleShareShortProject}
                disabled={isShortSharePending}
                className={`px-2 py-1 text-xs bg-[#3e2723] border border-[#4fc3f7] rounded-sm text-[#b3e5fc] hover:bg-[#4e342e] ${isShortSharePending ? 'opacity-60 cursor-not-allowed' : ''}`}
              >
                {isShortSharePending ? 'Sharing...' : 'Share Short'}
              </button>
              <button
                onClick={handleShareInlineProject}
                className="px-2 py-1 text-xs bg-[#3e2723] border border-[#4fc3f7] rounded-sm text-[#b3e5fc] hover:bg-[#4e342e]"
              >
                Share Inline
              </button>
              <button
                onClick={handleExportProject}
                className="px-2 py-1 text-xs bg-[#3e2723] border border-[#8d6e63] rounded-sm text-[#d7ccc8] hover:bg-[#4e342e]"
              >
                Export
              </button>
              <button
                onClick={() => openProjectImportPicker('merge')}
                className="px-2 py-1 text-xs bg-[#3e2723] border border-[#66bb6a] rounded-sm text-[#b9f6ca] hover:bg-[#4e342e]"
              >
                Import Merge
              </button>
              <button
                onClick={() => openProjectImportPicker('replace')}
                className="px-2 py-1 text-xs bg-[#3e2723] border border-[#ffb74d] rounded-sm text-[#ffcc80] hover:bg-[#4e342e]"
              >
                Import Replace
              </button>
            </div>

            {projectTransferNotice && (
              <div className="mt-2 bg-[#1f3324] border border-[#66bb6a] text-[#b9f6ca] rounded-sm px-2 py-1 text-[11px]">
                {projectTransferNotice}
              </div>
            )}

            {projectTransferError && (
              <div className="mt-2 bg-[#3e1f1f] border border-[#ef5350] text-[#ef9a9a] rounded-sm px-2 py-1 text-[11px]">
                {projectTransferError}
              </div>
            )}
          </div>

          {inventoryWarning && (
            <div className="bg-[#3e1f1f] border border-[#ef5350] text-[#ef9a9a] rounded-sm px-3 py-2 text-xs flex items-start justify-between gap-2">
              <span>{inventoryWarning}</span>
              <button onClick={clearInventoryWarning} className="text-[#ef9a9a] hover:text-white">x</button>
            </div>
          )}
        </motion.aside>

        {/* Bottom UI / Toolbar */}
        <footer className="absolute bottom-0 left-0 right-0 flex flex-col items-center justify-end pb-4 pointer-events-none gap-4">
          {/* ... Toolbar ... */}
          {/* ... Toolbar ... */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-1 rounded-t-xl relative z-20 pointer-events-auto"
            style={{
              // Metaphor: This is the outer Gold Frame of the Plaque
              background: 'linear-gradient(to bottom, #bf953f, #fcf6ba, #b38728, #fbf5b7, #aa771c)',
              boxShadow: '0 10px 30px rgba(0,0,0,0.8), 0 0 15px rgba(255, 215, 0, 0.3)'
            }}
          >
            {/* Inner Wood Inlay */}
            <div className="px-8 py-4 rounded-t-lg flex gap-6"
              style={{
                backgroundColor: '#1a0000', // Deepest Mahogany
                backgroundImage: `
                        repeating-linear-gradient(90deg, 
                            rgba(255,255,255,0.03) 0px, 
                            rgba(255,255,255,0.03) 1px, 
                            transparent 1px, 
                            transparent 40px),
                        linear-gradient(to bottom, rgba(50,20,0,0.8), rgba(20,0,0,1))
                    `,
                boxShadow: 'inset 0 2px 20px rgba(0,0,0,1), inset 0 0 0 2px #3e2723'
              }}
            >
              {/* Decorative Corner Ornaments (CSS pseudo-elements concept) */}
              <div className="absolute top-2 left-3 w-3 h-3 rounded-full bg-[#fcf6ba] shadow-[0_2px_4px_rgba(0,0,0,0.5),inset_0_-2px_4px_rgba(184,134,11,1)]" />
              <div className="absolute top-2 right-3 w-3 h-3 rounded-full bg-[#fcf6ba] shadow-[0_2px_4px_rgba(0,0,0,0.5),inset_0_-2px_4px_rgba(184,134,11,1)]" />

              <ToolbarButton
                label="Round Table"
                onClick={() => handleAddItemRequest('round-table')}
                onDragStart={() => setDraggedItem('round-table')}
                disabled={!canAddRoundTable}
              />
              <ToolbarButton
                label="Trestle Table"
                onClick={() => addItem('trestle-table')}
                onDragStart={() => setDraggedItem('trestle-table')}
                disabled={!canAddTrestleTable}
              />
              <ToolbarButton
                label="Chair"
                onClick={() => setDraggedItem('chair')}
                onDragStart={() => setDraggedItem('chair')}
                disabled={!canAddChair}
              />
              <ToolbarButton
                label="Platform"
                onClick={() => addItem('platform')}
                onDragStart={() => setDraggedItem('platform')}
                disabled={!canAddPlatform}
              />
            </div>
          </motion.div>

          {/* TFT Style Trash Bin - Fits underneath */}
          <div className="w-full pointer-events-auto relative z-10">
            <TrashBin />
          </div>
        </footer>

      </div>

      {/* Modal Prompt */}
      <AnimatePresence>
        {chairPrompt && (
          <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm pointer-events-auto">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="bg-[#1a1f26] border border-white/10 p-6 rounded-2xl shadow-2xl w-80"
            >
              <h3 className="text-lg font-semibold text-white mb-4">Add Table Setup</h3>

              <div className="mb-6">
                <label className="block text-sm text-gray-400 mb-2">How many chairs?</label>
                <div className="flex items-center gap-4">
                  <input
                    type="number"
                    min="0"
                    max="20"
                    value={chairCount}
                    onChange={(e) => setChairCount(parseInt(e.target.value) || 0)}
                    className="w-full bg-black/30 border border-white/10 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-[#6366f1]"
                    autoFocus
                  />
                </div>
                <p className="text-xs text-gray-500 mt-2">Placed evenly around the table.</p>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={handleClosePrompt}
                  className="flex-1 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-gray-300 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleConfirmChairs}
                  className="flex-1 px-4 py-2 rounded-lg bg-[#6366f1] hover:bg-[#4f46e5] text-white font-medium transition-colors"
                >
                  Place
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  )
}

const ToolbarButton = ({ label, onClick, onDragStart, disabled = false }: { label: string, onClick: () => void, onDragStart: () => void, disabled?: boolean }) => (
  <button
    disabled={disabled}
    onPointerDown={(e) => {
      if (disabled) return
      // Only left click triggers drag
      if (e.button === 0) onDragStart()
    }}
    onClick={() => {
      if (disabled) return
      onClick()
    }}
    className={`px-6 py-4 rounded-lg transition-transform active:scale-95 flex flex-col items-center gap-1 group touch-none select-none relative overflow-hidden shadow-[0_4px_8px_rgba(0,0,0,0.6)] ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
    style={{
      // Button is a smaller plaque
      background: 'linear-gradient(to bottom, #5d4037, #3e2723)',
      border: '2px solid #b38728', // Gold border
      boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.1), 0 2px 4px rgba(0,0,0,0.5)'
    }}
  >
    {/* Shine effect */}
    <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

    <span className="text-xs font-bold text-[#ffd700] group-hover:text-white relative z-10 font-serif tracking-widest drop-shadow-[0_1px_2px_rgba(0,0,0,0.8)] uppercase border-b border-[#b38728]/30 pb-1">
      {label}
    </span>
  </button>
)
