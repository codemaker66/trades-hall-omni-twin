'use client'

import { useFloorPlanStore, type FurnitureCategory } from './store'
import { templates } from './templates'

interface CatalogEntry {
  name: string
  category: FurnitureCategory
  widthFt: number
  depthFt: number
}

const catalogItems: CatalogEntry[] = [
  { name: 'Round Table 6ft', category: 'table', widthFt: 6, depthFt: 6 },
  { name: 'Round Table 5ft', category: 'table', widthFt: 5, depthFt: 5 },
  { name: 'Rect Table 8x3', category: 'table', widthFt: 8, depthFt: 3 },
  { name: 'Cocktail Table', category: 'table', widthFt: 2, depthFt: 2 },
  { name: 'Chair', category: 'chair', widthFt: 1.5, depthFt: 1.5 },
  { name: 'Stage 8x4', category: 'stage', widthFt: 8, depthFt: 4 },
  { name: 'Stage Riser 4x4', category: 'stage', widthFt: 4, depthFt: 4 },
  { name: 'Podium', category: 'decor', widthFt: 2, depthFt: 2 },
  { name: 'Bar Unit 6ft', category: 'equipment', widthFt: 6, depthFt: 2 },
  { name: 'DJ Booth', category: 'equipment', widthFt: 6, depthFt: 3 },
]

const categoryColors: Record<FurnitureCategory, string> = {
  table: 'bg-gold-30',
  chair: 'bg-indigo-50',
  stage: 'bg-success-50',
  decor: 'bg-warning-50',
  equipment: 'bg-surface-50',
}

export function CatalogSidebar() {
  const addItem = useFloorPlanStore((s) => s.addItem)
  const loadTemplate = useFloorPlanStore((s) => s.loadTemplate)
  const planWidthFt = useFloorPlanStore((s) => s.planWidthFt)
  const planHeightFt = useFloorPlanStore((s) => s.planHeightFt)

  const handleAdd = (entry: CatalogEntry) => {
    addItem({
      name: entry.name,
      category: entry.category,
      x: planWidthFt / 2,
      y: planHeightFt / 2,
      widthFt: entry.widthFt,
      depthFt: entry.depthFt,
      rotation: 0,
    })
  }

  return (
    <div className="w-52 bg-surface-5 border-r border-surface-25 overflow-y-auto flex flex-col">
      <div className="px-3 py-3 border-b border-surface-25">
        <h3 className="text-xs font-semibold text-surface-80 uppercase tracking-wider">Furniture</h3>
      </div>
      <div className="flex-1 p-2 space-y-1 overflow-y-auto">
        {catalogItems.map((entry) => (
          <button
            key={entry.name}
            onClick={() => handleAdd(entry)}
            className="w-full flex items-center gap-2 px-2 py-2 rounded-lg text-left hover:bg-surface-15 transition-colors group"
          >
            <div className={`w-6 h-6 rounded ${categoryColors[entry.category]} opacity-80 flex-shrink-0`} />
            <div className="min-w-0">
              <p className="text-xs font-medium text-surface-80 group-hover:text-surface-90 truncate">{entry.name}</p>
              <p className="text-[10px] text-surface-60">{entry.widthFt}&apos; x {entry.depthFt}&apos;</p>
            </div>
          </button>
        ))}
      </div>

      {/* Templates */}
      <div className="border-t border-surface-25">
        <div className="px-3 py-3 border-b border-surface-25">
          <h3 className="text-xs font-semibold text-surface-80 uppercase tracking-wider">Templates</h3>
        </div>
        <div className="p-2 space-y-1">
          {templates.map((t) => (
            <button
              key={t.id}
              onClick={() => loadTemplate(t.items)}
              className="w-full px-2 py-2 rounded-lg text-left hover:bg-surface-15 transition-colors group"
            >
              <p className="text-xs font-medium text-surface-80 group-hover:text-surface-90">{t.name}</p>
              <p className="text-[10px] text-surface-60">{t.description}</p>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
