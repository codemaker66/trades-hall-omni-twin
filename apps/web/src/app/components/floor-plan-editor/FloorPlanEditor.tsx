'use client'

import { useRef, useState, useEffect } from 'react'
import dynamic from 'next/dynamic'
import { EditorToolbar } from './EditorToolbar'
import { CatalogSidebar } from './CatalogSidebar'

// Konva must be client-only (uses window)
const Canvas2D = dynamic(() => import('./Canvas2D').then((m) => m.Canvas2D), { ssr: false })

export function FloorPlanEditor() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ width: 800, height: 600 })

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
      <CatalogSidebar />
      <div className="flex flex-col flex-1 min-w-0">
        <EditorToolbar />
        <div ref={containerRef} className="flex-1 overflow-hidden">
          <Canvas2D width={size.width} height={size.height} />
        </div>
      </div>
    </div>
  )
}
