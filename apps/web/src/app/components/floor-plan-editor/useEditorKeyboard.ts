/**
 * Keyboard navigation for the 2D floor plan editor.
 *
 * Handles:
 * - Tab/Shift+Tab: cycle selection through items
 * - Arrow keys: nudge selected items by grid size (Shift = 5x)
 * - Delete/Backspace: remove selected items
 * - R / Shift+R: rotate selection 45° / -45°
 * - Ctrl+Z / Ctrl+Shift+Z: undo / redo
 * - Ctrl+A: select all
 * - Escape: deselect
 */
import { useEffect } from 'react'
import { useFloorPlanStore } from './store'

export function useEditorKeyboard(enabled: boolean) {
  useEffect(() => {
    if (!enabled) return

    const handler = (e: KeyboardEvent) => {
      // Don't capture when user is in an input/textarea
      const tag = (e.target as HTMLElement).tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return

      const state = useFloorPlanStore.getState()

      // ─── Tab: cycle through items ────────────────────────────────────
      if (e.key === 'Tab') {
        if (state.items.length === 0) return
        e.preventDefault()

        const currentId = state.selectedIds[0]
        const currentIdx = currentId
          ? state.items.findIndex((i) => i.id === currentId)
          : -1

        let nextIdx: number
        if (e.shiftKey) {
          nextIdx = currentIdx <= 0 ? state.items.length - 1 : currentIdx - 1
        } else {
          nextIdx = currentIdx >= state.items.length - 1 ? 0 : currentIdx + 1
        }

        const nextItem = state.items[nextIdx]
        if (nextItem) {
          state.setSelection([nextItem.id])
        }
        return
      }

      // ─── Arrow keys: nudge selection ─────────────────────────────────
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
        if (state.selectedIds.length === 0) return
        e.preventDefault()

        const step = e.shiftKey ? state.gridSizeFt * 5 : state.gridSizeFt
        let dx = 0
        let dy = 0
        if (e.key === 'ArrowLeft') dx = -step
        if (e.key === 'ArrowRight') dx = step
        if (e.key === 'ArrowUp') dy = -step
        if (e.key === 'ArrowDown') dy = step

        const updates = state.selectedIds.map((id) => {
          const item = state.items.find((i) => i.id === id)
          if (!item || item.locked) return null
          return { id, changes: { x: item.x + dx, y: item.y + dy } }
        }).filter(Boolean) as { id: string; changes: { x: number; y: number } }[]

        if (updates.length > 0) {
          state.updateItems(updates)
        }
        return
      }

      // ─── Delete / Backspace: remove selection ────────────────────────
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (state.selectedIds.length === 0) return
        e.preventDefault()
        state.removeItems(state.selectedIds)
        return
      }

      // ─── R / Shift+R: rotate ─────────────────────────────────────────
      if (e.key === 'r' || e.key === 'R') {
        if (state.selectedIds.length === 0) return
        e.preventDefault()
        state.rotateSelection(e.shiftKey ? -45 : 45)
        return
      }

      // ─── Ctrl+Z / Ctrl+Shift+Z: undo / redo ─────────────────────────
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault()
        if (e.shiftKey) {
          state.redo()
        } else {
          state.undo()
        }
        return
      }

      // ─── Ctrl+A: select all ──────────────────────────────────────────
      if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
        e.preventDefault()
        state.setSelection(state.items.map((i) => i.id))
        return
      }

      // ─── Escape: deselect ────────────────────────────────────────────
      if (e.key === 'Escape') {
        if (state.selectedIds.length > 0) {
          e.preventDefault()
          state.setSelection([])
        }
      }
    }

    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [enabled])
}
