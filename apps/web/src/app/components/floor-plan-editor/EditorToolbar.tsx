'use client'

import { useFloorPlanStore } from './store'
import { getStageInstance } from './Canvas2D'
import { downloadPng, generateLegend } from './exportFloorPlan'

export function EditorToolbar() {
  const {
    zoom, setZoom, tool, setTool, snapEnabled, toggleSnap, gridSizeFt, setGridSize,
    canUndo, canRedo, undo, redo, selectedIds, removeItems, rotateSelection,
  } = useFloorPlanStore()

  const metrics = useFloorPlanStore((s) => s.getMetrics())

  const handleExport = () => {
    const stage = getStageInstance()
    if (!stage) return
    const state = useFloorPlanStore.getState()
    downloadPng({
      stage,
      items: state.items,
      planWidthFt: state.planWidthFt,
      planHeightFt: state.planHeightFt,
      planName: 'floor-plan',
    })
  }

  const handleCopyLegend = () => {
    const state = useFloorPlanStore.getState()
    const legend = generateLegend(state.items, state.planWidthFt, state.planHeightFt)
    void navigator.clipboard.writeText(legend)
  }

  return (
    <div className="h-12 flex items-center justify-between px-4 bg-surface-5 border-b border-surface-25 text-sm">
      {/* Left: Tools */}
      <div className="flex items-center gap-1">
        <button
          onClick={() => setTool('select')}
          className={`px-2 py-1 rounded text-xs font-medium ${tool === 'select' ? 'bg-surface-25 text-gold-50' : 'text-surface-70 hover:text-surface-90'}`}
          title="Select (V)"
        >
          Select
        </button>
        <button
          onClick={() => setTool('pan')}
          className={`px-2 py-1 rounded text-xs font-medium ${tool === 'pan' ? 'bg-surface-25 text-gold-50' : 'text-surface-70 hover:text-surface-90'}`}
          title="Pan (H)"
        >
          Pan
        </button>

        <div className="w-px h-6 bg-surface-25 mx-2" />

        <button
          onClick={undo}
          disabled={!canUndo}
          className="px-2 py-1 rounded text-xs text-surface-70 hover:text-surface-90 disabled:opacity-30"
          title="Undo (Ctrl+Z)"
        >
          Undo
        </button>
        <button
          onClick={redo}
          disabled={!canRedo}
          className="px-2 py-1 rounded text-xs text-surface-70 hover:text-surface-90 disabled:opacity-30"
          title="Redo (Ctrl+Shift+Z)"
        >
          Redo
        </button>

        <div className="w-px h-6 bg-surface-25 mx-2" />

        <button
          onClick={() => rotateSelection(45)}
          disabled={selectedIds.length === 0}
          className="px-2 py-1 rounded text-xs text-surface-70 hover:text-surface-90 disabled:opacity-30"
          title="Rotate 45° (R)"
        >
          Rotate
        </button>
        <button
          onClick={() => removeItems(selectedIds)}
          disabled={selectedIds.length === 0}
          className="px-2 py-1 rounded text-xs text-danger-50 hover:text-danger-70 disabled:opacity-30"
          title="Delete (Del)"
        >
          Delete
        </button>
      </div>

      {/* Center: Metrics */}
      <div className="flex items-center gap-4 text-xs text-surface-60">
        <span>Chairs: <strong className="text-surface-90">{metrics.chairs}</strong></span>
        <span>Tables: <strong className="text-surface-90">{metrics.tables}</strong></span>
        <span>Seats: <strong className="text-gold-50">{metrics.totalSeats}</strong></span>
      </div>

      {/* Right: View controls */}
      <div className="flex items-center gap-2">
        <button
          onClick={toggleSnap}
          className={`px-2 py-1 rounded text-xs font-medium ${snapEnabled ? 'bg-surface-25 text-gold-50' : 'text-surface-60'}`}
          title="Toggle Grid Snap"
        >
          Snap {snapEnabled ? 'ON' : 'OFF'}
        </button>
        <select
          value={gridSizeFt}
          onChange={(e) => setGridSize(Number(e.target.value))}
          className="bg-surface-10 border border-surface-25 text-surface-80 text-xs rounded px-1 py-1"
        >
          <option value={0.5}>6in</option>
          <option value={1}>1ft</option>
          <option value={2}>2ft</option>
        </select>

        <div className="w-px h-6 bg-surface-25 mx-1" />

        <button
          onClick={handleExport}
          className="px-2 py-1 rounded text-xs font-medium text-surface-70 hover:text-surface-90"
          title="Export PNG"
        >
          Export
        </button>
        <button
          onClick={handleCopyLegend}
          className="px-2 py-1 rounded text-xs font-medium text-surface-70 hover:text-surface-90"
          title="Copy legend to clipboard"
        >
          Legend
        </button>

        <div className="w-px h-6 bg-surface-25 mx-1" />

        <button onClick={() => setZoom(zoom / 1.2)} className="text-surface-60 hover:text-surface-90 text-xs px-1">-</button>
        <span className="text-xs text-surface-70 w-10 text-center">{Math.round(zoom * 100)}%</span>
        <button onClick={() => setZoom(zoom * 1.2)} className="text-surface-60 hover:text-surface-90 text-xs px-1">+</button>
      </div>
    </div>
  )
}
