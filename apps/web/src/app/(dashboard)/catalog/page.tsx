'use client'

import { useState } from 'react'
import { Card } from '../../components/ui/Card'
import { Badge } from '../../components/ui/Badge'
import { Button } from '../../components/ui/Button'
import { Input } from '../../components/ui/Input'
import { Select } from '../../components/ui/Select'
import { EmptyState } from '../../components/ui/EmptyState'

interface CatalogItem {
  id: string
  name: string
  category: 'table' | 'chair' | 'stage' | 'decor' | 'equipment'
  widthFt: number
  depthFt: number
  heightFt: number
  capacity: number | null
  stackable: boolean
}

const mockCatalog: CatalogItem[] = [
  { id: '1', name: 'Round Table 6ft', category: 'table', widthFt: 6, depthFt: 6, heightFt: 2.5, capacity: 8, stackable: false },
  { id: '2', name: 'Round Table 5ft', category: 'table', widthFt: 5, depthFt: 5, heightFt: 2.5, capacity: 6, stackable: false },
  { id: '3', name: 'Trestle Table 8ft', category: 'table', widthFt: 8, depthFt: 2.5, heightFt: 2.5, capacity: 8, stackable: true },
  { id: '4', name: 'Banquet Chair', category: 'chair', widthFt: 1.5, depthFt: 1.5, heightFt: 3, capacity: 1, stackable: true },
  { id: '5', name: 'Chiavari Chair', category: 'chair', widthFt: 1.3, depthFt: 1.3, heightFt: 3, capacity: 1, stackable: true },
  { id: '6', name: 'Folding Chair', category: 'chair', widthFt: 1.5, depthFt: 1.5, heightFt: 2.8, capacity: 1, stackable: true },
  { id: '7', name: 'Stage Platform 8x4', category: 'stage', widthFt: 8, depthFt: 4, heightFt: 2, capacity: null, stackable: false },
  { id: '8', name: 'Stage Riser 4x4', category: 'stage', widthFt: 4, depthFt: 4, heightFt: 1, capacity: null, stackable: true },
  { id: '9', name: 'Podium', category: 'decor', widthFt: 2, depthFt: 2, heightFt: 4, capacity: null, stackable: false },
  { id: '10', name: 'Bar Unit 6ft', category: 'equipment', widthFt: 6, depthFt: 2, heightFt: 3.5, capacity: null, stackable: false },
  { id: '11', name: 'Cocktail Table', category: 'table', widthFt: 2, depthFt: 2, heightFt: 3.5, capacity: 4, stackable: false },
  { id: '12', name: 'DJ Booth', category: 'equipment', widthFt: 6, depthFt: 3, heightFt: 3, capacity: null, stackable: false },
]

const categories = [
  { value: '', label: 'All Categories' },
  { value: 'table', label: 'Tables' },
  { value: 'chair', label: 'Chairs' },
  { value: 'stage', label: 'Stages' },
  { value: 'decor', label: 'Decor' },
  { value: 'equipment', label: 'Equipment' },
]

const categoryBadge: Record<CatalogItem['category'], 'gold' | 'info' | 'success' | 'warning' | 'default'> = {
  table: 'gold',
  chair: 'info',
  stage: 'success',
  decor: 'warning',
  equipment: 'default',
}

export default function CatalogPage() {
  const [search, setSearch] = useState('')
  const [categoryFilter, setCategoryFilter] = useState('')

  const filtered = mockCatalog.filter((item) => {
    const matchesSearch = item.name.toLowerCase().includes(search.toLowerCase())
    const matchesCategory = !categoryFilter || item.category === categoryFilter
    return matchesSearch && matchesCategory
  })

  return (
    <div className="space-y-6 max-w-6xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-surface-95">Furniture Catalog</h1>
          <p className="text-sm text-surface-60 mt-1">{mockCatalog.length} items in catalog</p>
        </div>
        <Button variant="primary">+ Add Item</Button>
      </div>

      {/* Filters */}
      <div className="flex items-end gap-3">
        <div className="flex-1 max-w-sm">
          <Input
            placeholder="Search catalog..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <div className="w-48">
          <Select
            options={categories}
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
          />
        </div>
      </div>

      {/* Grid */}
      {filtered.length === 0 ? (
        <EmptyState
          title="No items found"
          description="Try adjusting your search or filter."
        />
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {filtered.map((item) => (
            <Card key={item.id} className="hover:border-surface-40 transition-colors cursor-pointer">
              {/* Placeholder thumbnail */}
              <div className="w-full h-28 bg-surface-20 rounded-md mb-3 flex items-center justify-center">
                <svg className="w-10 h-10 text-surface-40" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M21 7.5l-2.25-1.313M21 7.5v2.25m0-2.25l-2.25 1.313M3 7.5l2.25-1.313M3 7.5l2.25 1.313M3 7.5v2.25m9 3l2.25-1.313M12 12.75l-2.25-1.313M12 12.75V15m0 6.75l2.25-1.313M12 21.75V15m0 0l-2.25 1.313M3 16.5v2.25" />
                </svg>
              </div>

              <div className="flex items-start justify-between mb-2">
                <h3 className="text-sm font-semibold text-surface-90">{item.name}</h3>
                <Badge variant={categoryBadge[item.category]}>{item.category}</Badge>
              </div>

              <div className="text-xs text-surface-60 space-y-1">
                <p>{item.widthFt}&apos; x {item.depthFt}&apos; x {item.heightFt}&apos;</p>
                <div className="flex items-center gap-2">
                  {item.capacity && <span>Seats {item.capacity}</span>}
                  {item.stackable && (
                    <Badge variant="default">Stackable</Badge>
                  )}
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
