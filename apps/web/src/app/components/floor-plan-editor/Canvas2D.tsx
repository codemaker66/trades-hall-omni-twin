'use client'

import { useRef, useState, useCallback, useEffect } from 'react'
import { Stage, Layer, Rect } from 'react-konva'
import type Konva from 'konva'
import { Grid2D } from './Grid2D'
import { FurnitureItem2D } from './FurnitureItem2D'
import { SelectionRect } from './SelectionRect'
import { SpacingOverlay } from './SpacingOverlay'
import { useFloorPlanStore, snapToGrid2D } from './store'

export const PIXELS_PER_FOOT = 40

// Module-level ref for export access
let _stageInstance: Konva.Stage | null = null
export function getStageInstance(): Konva.Stage | null {
  return _stageInstance
}

interface Canvas2DProps {
  width: number
  height: number
}

export function Canvas2D({ width, height }: Canvas2DProps) {
  const stageRef = useRef<Konva.Stage>(null)
  const {
    planWidthFt, planHeightFt, items, selectedIds, zoom, panX, panY,
    snapEnabled, gridSizeFt, tool,
    setZoom, setPan, setSelection, updateItems, beginBatch, endBatch,
  } = useFloorPlanStore()

  // Marquee selection state
  const [selRect, setSelRect] = useState({ x: 0, y: 0, w: 0, h: 0, visible: false })
  const selStart = useRef({ x: 0, y: 0 })

  const scale = PIXELS_PER_FOOT * zoom

  // ── Wheel zoom ────────────────────────────────────────────────────────────
  const handleWheel = useCallback((e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault()
    const stage = stageRef.current
    if (!stage) return

    const oldZoom = zoom
    const pointer = stage.getPointerPosition()
    if (!pointer) return

    const zoomDelta = e.evt.deltaY > 0 ? 0.9 : 1.1
    const newZoom = Math.max(0.1, Math.min(5, oldZoom * zoomDelta))

    // Zoom toward pointer
    const mousePointTo = {
      x: (pointer.x - panX) / (PIXELS_PER_FOOT * oldZoom),
      y: (pointer.y - panY) / (PIXELS_PER_FOOT * oldZoom),
    }
    const newPanX = pointer.x - mousePointTo.x * PIXELS_PER_FOOT * newZoom
    const newPanY = pointer.y - mousePointTo.y * PIXELS_PER_FOOT * newZoom

    setZoom(newZoom)
    setPan(newPanX, newPanY)
  }, [zoom, panX, panY, setZoom, setPan])

  // ── Stage mouse down (selection or pan) ───────────────────────────────────
  const handleMouseDown = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    // Ignore if clicking on a furniture item
    if (e.target !== e.currentTarget && e.target.getParent()?.name() !== 'background') return

    const stage = stageRef.current
    if (!stage) return
    const pointer = stage.getPointerPosition()
    if (!pointer) return

    if (tool === 'pan' || e.evt.button === 1) {
      // Middle-click pan handled by stage drag
      return
    }

    // Start marquee selection
    selStart.current = { x: pointer.x, y: pointer.y }
    setSelRect({ x: pointer.x, y: pointer.y, w: 0, h: 0, visible: true })

    if (!e.evt.shiftKey) {
      setSelection([])
    }
  }, [tool, setSelection])

  const handleMouseMove = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    if (!selRect.visible) return
    const stage = stageRef.current
    if (!stage) return
    const pointer = stage.getPointerPosition()
    if (!pointer) return

    setSelRect({
      x: selStart.current.x,
      y: selStart.current.y,
      w: pointer.x - selStart.current.x,
      h: pointer.y - selStart.current.y,
      visible: true,
    })
  }, [selRect.visible])

  const handleMouseUp = useCallback(() => {
    if (!selRect.visible) return

    // Find items within selection rect
    const minX = Math.min(selRect.x, selRect.x + selRect.w)
    const maxX = Math.max(selRect.x, selRect.x + selRect.w)
    const minY = Math.min(selRect.y, selRect.y + selRect.h)
    const maxY = Math.max(selRect.y, selRect.y + selRect.h)

    // Only if a meaningful drag (> 5px)
    if (Math.abs(selRect.w) > 5 || Math.abs(selRect.h) > 5) {
      const selected = items.filter((item) => {
        const ix = panX + item.x * scale
        const iy = panY + item.y * scale
        return ix >= minX && ix <= maxX && iy >= minY && iy <= maxY
      }).map((i) => i.id)
      setSelection(selected)
    }

    setSelRect({ x: 0, y: 0, w: 0, h: 0, visible: false })
  }, [selRect, items, scale, panX, panY, setSelection])

  // ── Item interaction callbacks ─────────────────────────────────────────────
  const handleSelect = useCallback((id: string, shiftKey: boolean) => {
    const current = useFloorPlanStore.getState().selectedIds
    if (shiftKey) {
      if (current.includes(id)) {
        setSelection(current.filter((s) => s !== id))
      } else {
        setSelection([...current, id])
      }
    } else {
      setSelection([id])
    }
  }, [setSelection])

  const handleDragStart = useCallback((id: string) => {
    const current = useFloorPlanStore.getState().selectedIds
    if (!current.includes(id)) {
      setSelection([id])
    }
    beginBatch()
  }, [setSelection, beginBatch])

  const handleDragMove = useCallback((id: string, rawX: number, rawY: number) => {
    const state = useFloorPlanStore.getState()
    let x = rawX
    let y = rawY

    if (state.snapEnabled) {
      x = snapToGrid2D(x, state.gridSizeFt)
      y = snapToGrid2D(y, state.gridSizeFt)
    }

    // If multi-selected, move all selected items by the delta
    const draggedItem = state.items.find((i) => i.id === id)
    if (!draggedItem) return

    const dx = x - draggedItem.x
    const dy = y - draggedItem.y

    const updates = state.selectedIds.map((sid) => ({
      id: sid,
      changes: sid === id
        ? { x, y }
        : {
            x: (state.items.find((i) => i.id === sid)?.x ?? 0) + dx,
            y: (state.items.find((i) => i.id === sid)?.y ?? 0) + dy,
          },
    }))

    updateItems(updates)
  }, [updateItems])

  const handleDragEnd = useCallback(() => {
    endBatch()
  }, [endBatch])

  // ── Stage drag for pan ────────────────────────────────────────────────────
  const handleStageDragEnd = useCallback((e: Konva.KonvaEventObject<DragEvent>) => {
    setPan(e.target.x(), e.target.y())
  }, [setPan])

  // ── Keyboard shortcuts ────────────────────────────────────────────────────
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return

      const state = useFloorPlanStore.getState()
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault()
        state.undo()
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'z' && e.shiftKey) {
        e.preventDefault()
        state.redo()
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'y') {
        e.preventDefault()
        state.redo()
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        if (state.selectedIds.length > 0) {
          e.preventDefault()
          state.removeItems(state.selectedIds)
        }
      } else if (e.key === 'r' || e.key === 'R') {
        if (state.selectedIds.length > 0) {
          e.preventDefault()
          state.rotateSelection(e.shiftKey ? -45 : 45)
        }
      } else if (e.key === 'Escape') {
        state.setSelection([])
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
        e.preventDefault()
        state.setSelection(state.items.map((i) => i.id))
      }
    }

    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  // Expose stage instance for export
  useEffect(() => {
    _stageInstance = stageRef.current
    return () => { _stageInstance = null }
  }, [])

  return (
    <Stage
      ref={stageRef}
      width={width}
      height={height}
      x={panX}
      y={panY}
      draggable={tool === 'pan'}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onDragEnd={handleStageDragEnd}
    >
      {/* Background + Grid */}
      <Layer>
        <Rect
          name="background"
          x={0}
          y={0}
          width={planWidthFt * scale}
          height={planHeightFt * scale}
          fill="#120a07"
          stroke="#3e2723"
          strokeWidth={2}
        />
        <Grid2D
          widthFt={planWidthFt}
          heightFt={planHeightFt}
          gridSizeFt={gridSizeFt}
          scale={scale}
        />
      </Layer>

      {/* Furniture */}
      <Layer>
        {items.map((item) => (
          <FurnitureItem2D
            key={item.id}
            item={item}
            scale={scale}
            selected={selectedIds.includes(item.id)}
            onSelect={handleSelect}
            onDragStart={handleDragStart}
            onDragMove={handleDragMove}
            onDragEnd={handleDragEnd}
          />
        ))}
      </Layer>

      {/* Spacing violations */}
      <Layer>
        <SpacingOverlay items={items} minSpacingFt={3} scale={scale} />
      </Layer>

      {/* Selection overlay */}
      <Layer>
        <SelectionRect
          x={selRect.x - panX}
          y={selRect.y - panY}
          width={selRect.w}
          height={selRect.h}
          visible={selRect.visible}
        />
      </Layer>
    </Stage>
  )
}
