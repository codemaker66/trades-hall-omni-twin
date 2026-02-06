'use client'

import { useState } from 'react'
import { Card } from '../../components/ui/Card'
import { Badge } from '../../components/ui/Badge'
import { Button } from '../../components/ui/Button'

type ViewMode = 'list' | 'pipeline'

interface Occasion {
  id: string
  name: string
  type: string
  venueName: string
  dateStart: string
  guestCount: number
  status: 'inquiry' | 'confirmed' | 'completed' | 'cancelled'
  budget: number | null
}

const mockOccasions: Occasion[] = [
  { id: '1', name: 'Smith Wedding', type: 'wedding', venueName: 'Trades Hall', dateStart: '2026-06-15T18:00:00Z', guestCount: 200, status: 'confirmed', budget: 35000 },
  { id: '2', name: 'TechCorp Annual Summit', type: 'conference', venueName: 'Conference Centre', dateStart: '2026-07-20T09:00:00Z', guestCount: 300, status: 'inquiry', budget: 15000 },
  { id: '3', name: 'Charity Gala 2026', type: 'gala', venueName: 'Trades Hall', dateStart: '2026-08-10T19:00:00Z', guestCount: 400, status: 'inquiry', budget: 50000 },
  { id: '4', name: 'Product Launch Party', type: 'corporate', venueName: 'Garden Terrace', dateStart: '2026-05-01T17:00:00Z', guestCount: 150, status: 'confirmed', budget: 12000 },
  { id: '5', name: 'Melbourne Jazz Festival', type: 'social', venueName: 'Trades Hall', dateStart: '2026-03-15T20:00:00Z', guestCount: 500, status: 'completed', budget: 25000 },
  { id: '6', name: 'Board Meeting Q2', type: 'corporate', venueName: 'Conference Centre', dateStart: '2026-04-05T10:00:00Z', guestCount: 20, status: 'cancelled', budget: 2000 },
]

const statusBadge: Record<Occasion['status'], 'info' | 'success' | 'gold' | 'danger'> = {
  inquiry: 'info',
  confirmed: 'success',
  completed: 'gold',
  cancelled: 'danger',
}

const pipelineColumns: { key: Occasion['status']; label: string }[] = [
  { key: 'inquiry', label: 'Inquiry' },
  { key: 'confirmed', label: 'Confirmed' },
  { key: 'completed', label: 'Completed' },
  { key: 'cancelled', label: 'Cancelled' },
]

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-AU', { month: 'short', day: 'numeric', year: 'numeric' })
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-AU', { style: 'currency', currency: 'AUD', maximumFractionDigits: 0 }).format(value)
}

export default function EventsPage() {
  const [viewMode, setViewMode] = useState<ViewMode>('list')

  return (
    <div className="space-y-6 max-w-6xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-surface-95">Events</h1>
          <p className="text-sm text-surface-60 mt-1">{mockOccasions.length} occasions across all venues</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex border border-surface-25 rounded-lg overflow-hidden">
            <button
              onClick={() => setViewMode('list')}
              className={`px-3 py-1.5 text-xs font-medium ${viewMode === 'list' ? 'bg-surface-25 text-surface-90' : 'text-surface-60 hover:text-surface-80'}`}
            >
              List
            </button>
            <button
              onClick={() => setViewMode('pipeline')}
              className={`px-3 py-1.5 text-xs font-medium ${viewMode === 'pipeline' ? 'bg-surface-25 text-surface-90' : 'text-surface-60 hover:text-surface-80'}`}
            >
              Pipeline
            </button>
          </div>
          <Button variant="primary">+ New Event</Button>
        </div>
      </div>

      {viewMode === 'list' ? (
        <Card noPadding>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead>
                <tr className="border-b border-surface-25">
                  <th className="px-4 py-3 text-xs font-semibold text-surface-60 uppercase tracking-wider">Event</th>
                  <th className="px-4 py-3 text-xs font-semibold text-surface-60 uppercase tracking-wider">Venue</th>
                  <th className="px-4 py-3 text-xs font-semibold text-surface-60 uppercase tracking-wider">Date</th>
                  <th className="px-4 py-3 text-xs font-semibold text-surface-60 uppercase tracking-wider">Guests</th>
                  <th className="px-4 py-3 text-xs font-semibold text-surface-60 uppercase tracking-wider">Budget</th>
                  <th className="px-4 py-3 text-xs font-semibold text-surface-60 uppercase tracking-wider">Status</th>
                </tr>
              </thead>
              <tbody>
                {mockOccasions.map((occ) => (
                  <tr key={occ.id} className="border-b border-surface-20 hover:bg-surface-15 transition-colors cursor-pointer">
                    <td className="px-4 py-3">
                      <div>
                        <p className="font-medium text-surface-90">{occ.name}</p>
                        <p className="text-xs text-surface-60 capitalize">{occ.type}</p>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-surface-80">{occ.venueName}</td>
                    <td className="px-4 py-3 text-surface-80">{formatDate(occ.dateStart)}</td>
                    <td className="px-4 py-3 text-surface-80">{occ.guestCount}</td>
                    <td className="px-4 py-3 text-surface-80">{occ.budget ? formatCurrency(occ.budget) : '—'}</td>
                    <td className="px-4 py-3">
                      <Badge variant={statusBadge[occ.status]}>{occ.status}</Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      ) : (
        <div className="grid grid-cols-4 gap-4">
          {pipelineColumns.map((col) => {
            const items = mockOccasions.filter((o) => o.status === col.key)
            return (
              <div key={col.key}>
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-semibold text-surface-80">{col.label}</h3>
                  <span className="text-xs text-surface-60 bg-surface-20 px-2 py-0.5 rounded-full">
                    {items.length}
                  </span>
                </div>
                <div className="space-y-2">
                  {items.map((occ) => (
                    <div
                      key={occ.id}
                      className="bg-surface-10 border border-surface-25 rounded-lg p-3 cursor-pointer hover:border-surface-40 transition-colors"
                    >
                      <p className="text-sm font-medium text-surface-90 mb-1">{occ.name}</p>
                      <p className="text-xs text-surface-60 mb-2">{occ.venueName}</p>
                      <div className="flex items-center justify-between text-xs text-surface-60">
                        <span>{formatDate(occ.dateStart)}</span>
                        <span>{occ.guestCount} guests</span>
                      </div>
                      {occ.budget && (
                        <p className="text-xs text-gold-50 mt-1">{formatCurrency(occ.budget)}</p>
                      )}
                    </div>
                  ))}
                  {items.length === 0 && (
                    <div className="text-center py-8 text-xs text-surface-50">
                      No events
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
