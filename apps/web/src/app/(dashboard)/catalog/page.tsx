'use client'

import { useState } from 'react'
import { Card } from '../../components/ui/Card'
import { Badge } from '../../components/ui/Badge'
import { Button } from '../../components/ui/Button'
import { Input } from '../../components/ui/Input'
import { Select } from '../../components/ui/Select'
import { EmptyState } from '../../components/ui/EmptyState'
import { SkeletonCard } from '../../components/ui/Skeleton'
import { ErrorAlert } from '../../components/ui/ErrorAlert'
import { useCatalog, type CatalogItem } from '../../hooks/useCatalog'
import { CreateCatalogItemModal } from './CreateCatalogItemModal'

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
  const { items, loading, error, refetch } = useCatalog()
  const [search, setSearch] = useState('')
  const [categoryFilter, setCategoryFilter] = useState('')
  const [showCreate, setShowCreate] = useState(false)

  const filtered = items.filter((item) => {
    const matchesSearch = item.name.toLowerCase().includes(search.toLowerCase())
    const matchesCategory = !categoryFilter || item.category === categoryFilter
    return matchesSearch && matchesCategory
  })

  return (
    <div className="space-y-6 max-w-6xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-surface-95">Furniture Catalog</h1>
          <p className="text-sm text-surface-60 mt-1">{items.length} items in catalog</p>
        </div>
        <Button variant="primary" onClick={() => setShowCreate(true)}>+ Add Item</Button>
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

      {/* Loading */}
      {loading && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {Array.from({ length: 8 }, (_, i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      )}

      {/* Error */}
      {error && <ErrorAlert message={error} onRetry={refetch} />}

      {/* Content */}
      {!loading && !error && (
        filtered.length === 0 ? (
          <EmptyState
            title="No items found"
            description={search || categoryFilter ? 'Try adjusting your search or filter.' : 'Add your first catalog item to get started.'}
            action={!search && !categoryFilter ? <Button variant="primary" onClick={() => setShowCreate(true)}>+ Add Item</Button> : undefined}
          />
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {filtered.map((item) => (
              <Card key={item.id} className="hover:border-surface-40 transition-colors cursor-pointer">
                {/* Thumbnail */}
                <div className="w-full h-28 bg-surface-20 rounded-md mb-3 flex items-center justify-center">
                  {item.thumbnailUrl ? (
                    <img src={item.thumbnailUrl} alt={item.name} className="w-full h-full object-cover rounded-md" />
                  ) : (
                    <svg className="w-10 h-10 text-surface-40" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M21 7.5l-2.25-1.313M21 7.5v2.25m0-2.25l-2.25 1.313M3 7.5l2.25-1.313M3 7.5l2.25 1.313M3 7.5v2.25m9 3l2.25-1.313M12 12.75l-2.25-1.313M12 12.75V15m0 6.75l2.25-1.313M12 21.75V15m0 0l-2.25 1.313M3 16.5v2.25" />
                    </svg>
                  )}
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
        )
      )}
      <CreateCatalogItemModal open={showCreate} onClose={() => setShowCreate(false)} onCreated={refetch} />
    </div>
  )
}
