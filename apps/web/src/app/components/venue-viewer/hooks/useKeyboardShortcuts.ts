import { useEffect, useCallback } from 'react'
import { useVenueStore } from '../../../../store'

export type ShortcutCategory = 'Editing' | 'Selection' | 'Tools' | 'Navigation'

export type ShortcutDef = {
    key: string
    modifiers?: ('ctrl' | 'shift')[]
    description: string
    category: ShortcutCategory
    /** Display string for the shortcut (e.g. "Ctrl+Z") */
    display: string
}

/**
 * Central registry of all keyboard shortcuts.
 * Single source of truth — referenced by useKeyboardShortcuts and KeyboardShortcutsHelp.
 */
export const SHORTCUTS: ShortcutDef[] = [
    // Editing
    { key: 'z', modifiers: ['ctrl'], description: 'Undo', category: 'Editing', display: 'Ctrl+Z' },
    { key: 'z', modifiers: ['ctrl', 'shift'], description: 'Redo', category: 'Editing', display: 'Ctrl+Shift+Z' },
    { key: 'y', modifiers: ['ctrl'], description: 'Redo', category: 'Editing', display: 'Ctrl+Y' },
    { key: 's', modifiers: ['ctrl'], description: 'Save scenario', category: 'Editing', display: 'Ctrl+S' },

    // Selection & manipulation
    { key: 'delete', description: 'Delete selection', category: 'Selection', display: 'Del' },
    { key: 'backspace', description: 'Delete selection', category: 'Selection', display: 'Backspace' },
    { key: 'escape', description: 'Deselect all / Cancel tool', category: 'Selection', display: 'Esc' },
    { key: 'a', modifiers: ['ctrl'], description: 'Select all items', category: 'Selection', display: 'Ctrl+A' },
    { key: 'q', description: 'Rotate selection -90°', category: 'Selection', display: 'Q' },
    { key: 'e', description: 'Rotate selection +90°', category: 'Selection', display: 'E' },

    // Tools
    { key: 'g', description: 'Toggle grid snap', category: 'Tools', display: 'G' },
    { key: 'r', description: 'Toggle move/rotate mode', category: 'Tools', display: 'R' },
    { key: '?', description: 'Show keyboard shortcuts', category: 'Tools', display: '?' },

    // Navigation
    { key: 'w', description: 'Pan camera forward', category: 'Navigation', display: 'W' },
    { key: 'a', description: 'Pan camera left', category: 'Navigation', display: 'A' },
    { key: 's', description: 'Pan camera backward', category: 'Navigation', display: 'S' },
    { key: 'd', description: 'Pan camera right', category: 'Navigation', display: 'D' },
]

function shouldIgnoreEvent(e: KeyboardEvent): boolean {
    if (e.defaultPrevented) return true
    const target = e.target as HTMLElement | null
    if (!target) return false
    if (target.isContentEditable) return true
    const tag = target.tagName
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
}

export const useKeyboardShortcuts = () => {
    const handleKeyDown = useCallback((e: KeyboardEvent) => {
        if (shouldIgnoreEvent(e)) return

        const key = e.key.toLowerCase()
        const hasModifier = e.metaKey || e.ctrlKey
        const state = useVenueStore.getState()

        // Chair tool owns Q/E when active
        if ((key === 'q' || key === 'e') && state.draggedItemType === 'chair') return

        // --- Modifier shortcuts ---
        if (hasModifier) {
            if (!e.shiftKey && key === 'z') {
                e.preventDefault()
                state.undo()
                return
            }
            if (key === 'y' || (e.shiftKey && key === 'z')) {
                e.preventDefault()
                state.redo()
                return
            }
            if (key === 's') {
                e.preventDefault()
                state.saveScenario()
                return
            }
            if (key === 'a') {
                e.preventDefault()
                const allIds = state.items.map(i => i.id)
                state.setSelection(allIds)
                return
            }
            return
        }

        // --- Plain key shortcuts ---
        switch (key) {
            case 'q':
                state.rotateSelection(-90)
                break
            case 'e':
                state.rotateSelection(90)
                break
            case 'delete':
            case 'backspace':
                if (state.selectedIds.length > 0) {
                    state.removeItems(state.selectedIds)
                }
                break
            case 'escape':
                if (state.draggedItemType) {
                    // DragPreview handles its own Escape for chair tool
                    return
                }
                if (state.selectedIds.length > 0) {
                    state.setSelection([])
                }
                break
            case 'g':
                state.toggleSnapping()
                break
            case 'r':
                state.toggleTransformMode()
                break
            case '?':
                state.setShortcutsHelpOpen(true)
                break
        }
    }, [])

    useEffect(() => {
        window.addEventListener('keydown', handleKeyDown)
        return () => window.removeEventListener('keydown', handleKeyDown)
    }, [handleKeyDown])
}
