'use client'

import { useRef, useState, useEffect } from 'react'
import dynamic from 'next/dynamic'
import { EditorToolbar } from './EditorToolbar'
import { CatalogSidebar } from './CatalogSidebar'
import { CapacityWarning } from './CapacityWarning'
import { useEditorKeyboard } from './useEditorKeyboard'
import { useAutoSave } from './useAutoSave'
import { SaveIndicator } from './SaveIndicator'

// Konva must be client-only (uses window)
const Canvas2D = dynamic(() => import('./Canvas2D').then((m) => m.Canvas2D), { ssr: false })

// Three.js lazy-loaded — only initialized when user switches to 3D tab
const Scene3DPreview = dynamic(
  () => import('./Scene3DPreview').then((m) => m.Scene3DPreview),
  { ssr: false },
)

type ViewMode = '2d' | '3d'

export function FloorPlanEditor() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ width: 800, height: 600 })
  const [viewMode, setViewMode] = useState<ViewMode>('2d')

  // Global keyboard shortcuts (Tab cycle, arrow nudge, Del, R, Ctrl+Z/A/Esc)
  useEditorKeyboard(viewMode === '2d')

  // Auto-save to localStorage with debounce
  const { status: saveStatus } = useAutoSave()

  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setSize({
          width: Math.floor(entry.contentRect.width),
          height: Math.floor(entry.contentRect.height),
        })
      }
    })

    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  return (
    <div className="flex h-[100dvh] bg-surface-0 text-surface-90">
      {viewMode === '2d' && <CatalogSidebar />}
      <div className="flex flex-col flex-1 min-w-0">
        <div className="h-12 flex items-center bg-surface-5 border-b border-surface-25">
          {/* View mode toggle */}
          <div className="flex items-center border-r border-surface-25 px-2 gap-1">
            <button
              onClick={() => setViewMode('2d')}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                viewMode === '2d'
                  ? 'bg-surface-25 text-gold-50'
                  : 'text-surface-60 hover:text-surface-90'
              }`}
            >
              2D Edit
            </button>
            <button
              onClick={() => setViewMode('3d')}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                viewMode === '3d'
                  ? 'bg-surface-25 text-gold-50'
                  : 'text-surface-60 hover:text-surface-90'
              }`}
            >
              3D View
            </button>
          </div>

          {/* Toolbar fills the rest */}
          {viewMode === '2d' && (
            <div className="flex-1 min-w-0">
              <EditorToolbar />
            </div>
          )}
          {viewMode === '3d' && (
            <div className="flex-1 px-4 text-xs text-surface-60">
              Orbit: right-click drag &middot; Pan: middle-click &middot; Zoom: scroll wheel
            </div>
          )}

          {/* Save indicator */}
          <div className="px-3">
            <SaveIndicator status={saveStatus} />
          </div>
        </div>

        <div ref={containerRef} className="relative flex-1 overflow-hidden">
          {viewMode === '2d' ? (
            <>
              <Canvas2D width={size.width} height={size.height} />
              <CapacityWarning maxCapacity={500} />
            </>
          ) : (
            <Scene3DPreview width={size.width} height={size.height} />
          )}
        </div>
      </div>
    </div>
  )
}
