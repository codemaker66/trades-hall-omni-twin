'use client'

import { motion } from 'framer-motion'
import { useVenueStore, type FurnitureType } from '../../../store'
import { TrashBin } from '../TrashBin'

export const BottomToolbar = () => {
    const addItem = useVenueStore((state) => state.addItem)
    const setDraggedItem = useVenueStore((state) => state.setDraggedItem)
    const hasInventoryForType = useVenueStore((state) => state.hasInventoryForType)
    const openChairPrompt = useVenueStore((state) => state.openChairPrompt)

    const canAddRoundTable = hasInventoryForType('round-table')
    const canAddTrestleTable = hasInventoryForType('trestle-table')
    const canAddChair = hasInventoryForType('chair')
    const canAddPlatform = hasInventoryForType('platform')

    const handleAddItemRequest = (type: FurnitureType) => {
        if (type === 'round-table') {
            if (!canAddRoundTable) {
                addItem('round-table', [0, 0, 0], [0, 0, 0], undefined, { recordHistory: false })
                return
            }
            openChairPrompt(type)
        } else {
            addItem(type)
        }
    }

    return (
        <footer className="absolute bottom-0 left-0 right-0 flex flex-col items-center justify-end pb-4 pointer-events-none gap-4" role="toolbar" aria-label="Furniture palette">
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-1 rounded-t-xl relative z-20 pointer-events-auto"
                style={{
                    background: `linear-gradient(to bottom, var(--color-gold-30), var(--color-gold-80), var(--color-gold-20), var(--color-gold-90), var(--color-gold-10))`,
                    boxShadow: '0 10px 30px rgba(0,0,0,0.8), 0 0 15px rgba(255,215,0,0.3)'
                }}
            >
                <div className="px-8 py-4 rounded-t-lg flex gap-6"
                    style={{
                        backgroundColor: 'var(--color-surface-10)',
                        backgroundImage: `
                            repeating-linear-gradient(90deg,
                                rgba(255,255,255,0.03) 0px,
                                rgba(255,255,255,0.03) 1px,
                                transparent 1px,
                                transparent 40px),
                            linear-gradient(to bottom, rgba(50,20,0,0.8), rgba(20,0,0,1))
                        `,
                        boxShadow: `inset 0 2px 20px rgba(0,0,0,1), inset 0 0 0 2px var(--color-surface-25)`
                    }}
                >
                    <div className="absolute top-2 left-3 w-3 h-3 rounded-full bg-gold-80 shadow-[0_2px_4px_rgba(0,0,0,0.5),inset_0_-2px_4px_rgba(184,134,11,1)]" />
                    <div className="absolute top-2 right-3 w-3 h-3 rounded-full bg-gold-80 shadow-[0_2px_4px_rgba(0,0,0,0.5),inset_0_-2px_4px_rgba(184,134,11,1)]" />

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

            <div className="w-full pointer-events-auto relative z-10">
                <TrashBin />
            </div>
        </footer>
    )
}

const ToolbarButton = ({ label, onClick, onDragStart, disabled = false }: { label: string, onClick: () => void, onDragStart: () => void, disabled?: boolean }) => (
    <button
        disabled={disabled}
        aria-label={`Add ${label}`}
        onPointerDown={(e) => {
            if (disabled) return
            if (e.button === 0) onDragStart()
        }}
        onClick={() => {
            if (disabled) return
            onClick()
        }}
        className={`
            px-6 py-4 rounded-lg transition-transform active:scale-95
            flex flex-col items-center gap-1 group touch-none select-none
            relative overflow-hidden shadow-[0_4px_8px_rgba(0,0,0,0.6)]
            focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-50
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        `}
        style={{
            background: `linear-gradient(to bottom, var(--color-surface-40), var(--color-surface-25))`,
            border: `2px solid var(--color-gold-20)`,
            boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.1), 0 2px 4px rgba(0,0,0,0.5)'
        }}
    >
        <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
        <span className="text-xs font-bold text-gold-50 group-hover:text-white relative z-10 font-serif tracking-widest drop-shadow-[0_1px_2px_rgba(0,0,0,0.8)] uppercase border-b border-gold-20/30 pb-1">
            {label}
        </span>
    </button>
)
