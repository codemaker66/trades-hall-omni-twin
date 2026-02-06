'use client'

import { useState } from 'react'
import Link from 'next/link'
import { Card } from '../../components/ui/Card'
import { Badge } from '../../components/ui/Badge'
import { Input } from '../../components/ui/Input'
import { EmptyState } from '../../components/ui/EmptyState'
import { Button } from '../../components/ui/Button'

interface Venue {
  id: string
  name: string
  slug: string
  venueType: string
  capacity: number | null
  status: 'draft' | 'published' | 'archived'
  address: string | null
  updatedAt: string
}

// Placeholder data — will be replaced with API call
const mockVenues: Venue[] = [
  {
    id: '1',
    name: 'Trades Hall',
    slug: 'trades-hall',
    venueType: 'ballroom',
    capacity: 500,
    status: 'published',
    address: '54 Victoria St, Melbourne VIC 3000',
    updatedAt: '2026-02-05T10:30:00Z',
  },
  {
    id: '2',
    name: 'Garden Terrace',
    slug: 'garden-terrace',
    venueType: 'outdoor',
    capacity: 200,
    status: 'draft',
    address: '12 Park Lane, Melbourne VIC 3000',
    updatedAt: '2026-02-04T15:00:00Z',
  },
  {
    id: '3',
    name: 'Conference Centre',
    slug: 'conference-centre',
    venueType: 'conference',
    capacity: 300,
    status: 'published',
    address: '88 Collins St, Melbourne VIC 3000',
    updatedAt: '2026-02-03T09:00:00Z',
  },
]

const statusBadge: Record<string, 'success' | 'gold' | 'default'> = {
  published: 'success',
  draft: 'gold',
  archived: 'default',
}

type ViewMode = 'grid' | 'list'

export default function VenuesPage() {
  const [search, setSearch] = useState('')
  const [viewMode, setViewMode] = useState<ViewMode>('grid')

  const filtered = mockVenues.filter(
    (v) =>
      v.name.toLowerCase().includes(search.toLowerCase()) ||
      v.venueType.toLowerCase().includes(search.toLowerCase()),
  )

  return (
    <div className="space-y-6 max-w-6xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-surface-95">Venues</h1>
          <p className="text-sm text-surface-60 mt-1">{mockVenues.length} venues</p>
        </div>
        <Button variant="primary">+ New Venue</Button>
      </div>

      {/* Search + View Toggle */}
      <div className="flex items-center gap-3">
        <div className="flex-1 max-w-sm">
          <Input
            placeholder="Search venues..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <div className="flex border border-surface-25 rounded-lg overflow-hidden">
          <button
            onClick={() => setViewMode('grid')}
            className={`px-3 py-2 text-sm ${viewMode === 'grid' ? 'bg-surface-25 text-surface-90' : 'text-surface-60 hover:text-surface-80'}`}
            aria-label="Grid view"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6A2.25 2.25 0 016 3.75h2.25A2.25 2.25 0 0110.5 6v2.25a2.25 2.25 0 01-2.25 2.25H6a2.25 2.25 0 01-2.25-2.25V6zM3.75 15.75A2.25 2.25 0 016 13.5h2.25a2.25 2.25 0 012.25 2.25V18a2.25 2.25 0 01-2.25 2.25H6A2.25 2.25 0 013.75 18v-2.25zM13.5 6a2.25 2.25 0 012.25-2.25H18A2.25 2.25 0 0120.25 6v2.25A2.25 2.25 0 0118 10.5h-2.25a2.25 2.25 0 01-2.25-2.25V6zM13.5 15.75a2.25 2.25 0 012.25-2.25H18a2.25 2.25 0 012.25 2.25V18A2.25 2.25 0 0118 20.25h-2.25A2.25 2.25 0 0113.5 18v-2.25z" />
            </svg>
          </button>
          <button
            onClick={() => setViewMode('list')}
            className={`px-3 py-2 text-sm ${viewMode === 'list' ? 'bg-surface-25 text-surface-90' : 'text-surface-60 hover:text-surface-80'}`}
            aria-label="List view"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 12h16.5m-16.5 3.75h16.5M3.75 19.5h16.5M5.625 4.5h12.75a1.875 1.875 0 010 3.75H5.625a1.875 1.875 0 010-3.75z" />
            </svg>
          </button>
        </div>
      </div>

      {/* Results */}
      {filtered.length === 0 ? (
        <EmptyState
          title="No venues found"
          description={search ? 'Try adjusting your search.' : 'Create your first venue to get started.'}
          action={!search ? <Button variant="primary">+ New Venue</Button> : undefined}
        />
      ) : viewMode === 'grid' ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map((venue) => (
            <Link key={venue.id} href={`/venues/${venue.id}`}>
              <Card className="hover:border-surface-40 transition-colors cursor-pointer">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h3 className="text-base font-semibold text-surface-90">{venue.name}</h3>
                    <p className="text-xs text-surface-60 capitalize">{venue.venueType}</p>
                  </div>
                  <Badge variant={statusBadge[venue.status]}>{venue.status}</Badge>
                </div>
                {venue.address && (
                  <p className="text-xs text-surface-60 mb-2 truncate">{venue.address}</p>
                )}
                <div className="flex items-center justify-between text-xs text-surface-60">
                  {venue.capacity && <span>Capacity: {venue.capacity}</span>}
                  <span>Updated {new Date(venue.updatedAt).toLocaleDateString()}</span>
                </div>
              </Card>
            </Link>
          ))}
        </div>
      ) : (
        <Card noPadding>
          <div className="divide-y divide-surface-20">
            {filtered.map((venue) => (
              <Link
                key={venue.id}
                href={`/venues/${venue.id}`}
                className="flex items-center justify-between px-5 py-3 hover:bg-surface-15 transition-colors"
              >
                <div className="flex items-center gap-4 min-w-0">
                  <div className="min-w-0">
                    <h3 className="text-sm font-semibold text-surface-90">{venue.name}</h3>
                    <p className="text-xs text-surface-60 capitalize">{venue.venueType}{venue.address ? ` — ${venue.address}` : ''}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3 flex-shrink-0">
                  {venue.capacity && (
                    <span className="text-xs text-surface-60">Cap: {venue.capacity}</span>
                  )}
                  <Badge variant={statusBadge[venue.status]}>{venue.status}</Badge>
                </div>
              </Link>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}
