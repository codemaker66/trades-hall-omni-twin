'use client'

import { Scene } from './Scene'
import { TrashBin } from './TrashBin'
import { motion, AnimatePresence } from 'framer-motion'
import { useVenueStore, FurnitureType } from '../../store'
import { useState, useEffect } from 'react'

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

  const openChairPrompt = useVenueStore((state) => state.openChairPrompt)
  const closeChairPrompt = useVenueStore((state) => state.closeChairPrompt)
  const chairPrompt = useVenueStore((state) => state.chairPrompt)

  const [chairCount, setChairCount] = useState<number>(8)

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === 'INPUT') return

      const key = e.key.toLowerCase()
      if (key === 'q') {
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

  const handleAddItemRequest = (type: FurnitureType) => {
    if (type === 'round-table') {
      openChairPrompt(type)
      setChairCount(5) // Default based on user request example
    } else {
      addItem(type)
    }
  }

  const handleConfirmChairs = () => {
    if (!chairPrompt) return

    const { tableId, type } = chairPrompt

    // If we are reconfiguring an existing table
    if (tableId) {
      const table = items.find(i => i.id === tableId)
      if (table) {
        // 1. Remove old items in the same group (chairs) but keep the table
        const groupItems = items.filter(i => i.groupId === table.groupId && i.id !== tableId)
        removeItems(groupItems.map(i => i.id))

        // 2. Add Chairs around existing table position
        const radius = 1.3
        const center = table.position
        for (let i = 0; i < chairCount; i++) {
          const angle = (i / chairCount) * Math.PI * 2
          const x = center[0] + Math.sin(angle) * radius
          const z = center[2] + Math.cos(angle) * radius
          const rotY = angle + Math.PI
          addItem('chair', [x, 0, z], [0, rotY, 0], table.groupId)
        }
      }
    } else {
      // Standard new table + chairs placement
      // Create Group ID
      const groupId = crypto.randomUUID()

      // 1. Add the Round Table at center (for now)
      const center: [number, number, number] = [0, 0, 0]
      addItem('round-table', center, [0, 0, 0], groupId)

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

        addItem('chair', [x, 0, z], [0, rotY, 0], groupId)
      }
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
              />
              <ToolbarButton
                label="Trestle Table"
                onClick={() => addItem('trestle-table')}
                onDragStart={() => setDraggedItem('trestle-table')}
              />
              <ToolbarButton
                label="Chair"
                onClick={() => setDraggedItem('chair')}
                onDragStart={() => setDraggedItem('chair')}
              />
              <ToolbarButton
                label="Platform"
                onClick={() => addItem('platform')}
                onDragStart={() => setDraggedItem('platform')}
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

const ToolbarButton = ({ label, onClick, onDragStart }: { label: string, onClick: () => void, onDragStart: () => void }) => (
  <button
    onPointerDown={(e) => {
      // Only left click triggers drag
      if (e.button === 0) onDragStart()
    }}
    onClick={onClick}
    className="px-6 py-4 rounded-lg transition-transform active:scale-95 flex flex-col items-center gap-1 group touch-none select-none relative overflow-hidden shadow-[0_4px_8px_rgba(0,0,0,0.6)]"
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
