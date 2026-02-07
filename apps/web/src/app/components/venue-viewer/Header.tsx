'use client'

import { motion } from 'framer-motion'
import { useVenueStore } from '../../../store'
import { useShallow } from 'zustand/react/shallow'
import { Button } from '../ui/Button'
import { Tooltip } from '../ui/Tooltip'
import { useAuth } from '../../hooks/useAuth'

export const Header = () => {
    const { selectedIds, items, canUndo, canRedo, snappingEnabled, transformMode } = useVenueStore(
        useShallow((state) => ({
            selectedIds: state.selectedIds,
            items: state.items,
            canUndo: state.canUndo,
            canRedo: state.canRedo,
            snappingEnabled: state.snappingEnabled,
            transformMode: state.transformMode,
        }))
    )
    const undo = useVenueStore((state) => state.undo)
    const redo = useVenueStore((state) => state.redo)
    const toggleSnapping = useVenueStore((state) => state.toggleSnapping)
    const toggleTransformMode = useVenueStore((state) => state.toggleTransformMode)
    const rotateSelection = useVenueStore((state) => state.rotateSelection)
    const removeItems = useVenueStore((state) => state.removeItems)
    const ungroupItems = useVenueStore((state) => state.ungroupItems)

    const { user, loading: authLoading, logout } = useAuth()

    const hasGroupsSelected = selectedIds.some(id => items.find(i => i.id === id)?.groupId)

    return (
        <header className="flex justify-between items-start">
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="border-4 border-surface-25 p-4 rounded-sm pointer-events-auto shadow-2xl relative overflow-hidden group min-w-[300px]"
                style={{
                    backgroundColor: 'var(--color-surface-20)',
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
                <div className="absolute top-1 left-1 w-2 h-2 rounded-full bg-surface-15 shadow-[inset_0_1px_1px_rgba(255,255,255,0.1)]" />
                <div className="absolute top-1 right-1 w-2 h-2 rounded-full bg-surface-15 shadow-[inset_0_1px_1px_rgba(255,255,255,0.1)]" />
                <div className="absolute bottom-1 left-1 w-2 h-2 rounded-full bg-surface-15 shadow-[inset_0_1px_1px_rgba(255,255,255,0.1)]" />
                <div className="absolute bottom-1 right-1 w-2 h-2 rounded-full bg-surface-15 shadow-[inset_0_1px_1px_rgba(255,255,255,0.1)]" />

                <h1 className="text-2xl font-bold text-surface-90 tracking-widest flex flex-col items-center drop-shadow-md pb-2 border-b border-surface-40/50">
                    OMNITWIN
                    <span className="text-surface-80 text-xs font-normal tracking-[0.2em] mt-1 uppercase">Trades Hall Planner</span>
                </h1>
            </motion.div>

            <div className="flex gap-2 pointer-events-auto">
                <Tooltip content="Undo" shortcut="Ctrl+Z">
                    <Button
                        onClick={undo}
                        disabled={!canUndo}
                        aria-label="Undo (Ctrl+Z)"
                    >
                        Undo
                    </Button>
                </Tooltip>

                <Tooltip content="Redo" shortcut="Ctrl+Y">
                    <Button
                        onClick={redo}
                        disabled={!canRedo}
                        aria-label="Redo (Ctrl+Y)"
                    >
                        Redo
                    </Button>
                </Tooltip>

                <Tooltip content="Toggle grid snapping" shortcut="G">
                    <Button
                        variant={snappingEnabled ? 'primary' : 'ghost'}
                        onClick={toggleSnapping}
                        aria-label={`Grid snap: ${snappingEnabled ? 'on' : 'off'}`}
                        aria-pressed={snappingEnabled}
                    >
                        Grid Snap: {snappingEnabled ? 'ON' : 'OFF'}
                    </Button>
                </Tooltip>

                {hasGroupsSelected && (
                    <Tooltip content="Toggle move/rotate mode" shortcut="R">
                        <Button
                            variant={transformMode === 'rotate' ? 'primary' : 'ghost'}
                            onClick={toggleTransformMode}
                            aria-label={`Transform mode: ${transformMode}`}
                            aria-pressed={transformMode === 'rotate'}
                        >
                            Mode: {transformMode === 'rotate' ? 'ROTATE' : 'MOVE'}
                        </Button>
                    </Tooltip>
                )}

                {transformMode === 'rotate' && hasGroupsSelected && (
                    <div className="flex gap-1 bg-surface-20 p-1 rounded-sm border border-surface-40" role="group" aria-label="Rotation controls">
                        <Button size="sm" variant="primary" onClick={() => rotateSelection(-90)} aria-label="Rotate 90 degrees counter-clockwise">
                            ↺ 90°
                        </Button>
                        <Button size="sm" variant="primary" onClick={() => rotateSelection(-1)} aria-label="Rotate 1 degree counter-clockwise">
                            -1°
                        </Button>
                        <Button size="sm" variant="primary" onClick={() => rotateSelection(1)} aria-label="Rotate 1 degree clockwise">
                            +1°
                        </Button>
                        <Button size="sm" variant="primary" onClick={() => rotateSelection(90)} aria-label="Rotate 90 degrees clockwise">
                            90° ↻
                        </Button>
                    </div>
                )}

                {hasGroupsSelected && (
                    <Button
                        variant="primary"
                        onClick={() => ungroupItems(selectedIds)}
                        aria-label="Ungroup selected items"
                    >
                        Ungroup
                    </Button>
                )}

                {!hasGroupsSelected && selectedIds.length > 1 && (
                    <Button
                        variant="primary"
                        onClick={() => useVenueStore.getState().groupItems(selectedIds)}
                        aria-label="Group selected items"
                    >
                        Group
                    </Button>
                )}

                {selectedIds.length > 0 && (
                    <Tooltip content="Delete selection" shortcut="Del">
                        <Button
                            variant="danger"
                            onClick={() => removeItems(selectedIds)}
                            aria-label={`Delete ${selectedIds.length} selected item${selectedIds.length > 1 ? 's' : ''}`}
                        >
                            Delete Selected ({selectedIds.length})
                        </Button>
                    </Tooltip>
                )}

                <div className="ml-2 pl-2 border-l border-surface-40 flex items-center">
                    {authLoading ? (
                        <div className="w-8 h-8 rounded-full bg-surface-25 animate-pulse" />
                    ) : user ? (
                        <div className="flex items-center gap-2">
                            <div
                                className="w-8 h-8 rounded-full bg-accent-60 flex items-center justify-center text-sm font-bold text-surface-0"
                                title={user.name}
                            >
                                {user.name.charAt(0).toUpperCase()}
                            </div>
                            <Button size="sm" variant="ghost" onClick={logout} aria-label="Sign out">
                                Sign out
                            </Button>
                        </div>
                    ) : (
                        <a href="/login">
                            <Button size="sm" variant="ghost" aria-label="Sign in">
                                Sign in
                            </Button>
                        </a>
                    )}
                </div>
            </div>
        </header>
    )
}
