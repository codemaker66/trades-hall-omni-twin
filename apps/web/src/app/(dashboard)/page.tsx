'use client'

import Link from 'next/link'
import { Card } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { Skeleton } from '../components/ui/Skeleton'
import { useVenues } from '../hooks/useVenues'
import { useAllOccasions } from '../hooks/useOccasions'

const recentActivity = [
  { id: '1', text: 'New inquiry for Trades Hall — Wedding, Jun 15', time: '2 hours ago', type: 'inquiry' as const },
  { id: '2', text: 'Floor plan updated — Grand Ballroom', time: '5 hours ago', type: 'edit' as const },
  { id: '3', text: 'Booking confirmed — Corporate Conference', time: '1 day ago', type: 'confirmed' as const },
  { id: '4', text: 'New venue added — Garden Terrace', time: '2 days ago', type: 'venue' as const },
]

const badgeVariant: Record<string, 'info' | 'gold' | 'success' | 'default'> = {
  inquiry: 'info',
  edit: 'gold',
  confirmed: 'success',
  venue: 'default',
}

export default function DashboardPage() {
  const { venues, loading: venuesLoading } = useVenues()
  const { occasions, loading: occasionsLoading } = useAllOccasions()

  const loading = venuesLoading || occasionsLoading

  const upcomingEvents = occasions.filter(
    (o) => o.status !== 'cancelled' && o.status !== 'completed' && new Date(o.dateStart) > new Date(),
  )
  const pendingInquiries = occasions.filter((o) => o.status === 'inquiry')
  const confirmedBookings = occasions.filter((o) => o.status === 'confirmed')
  const totalRevenue = confirmedBookings.reduce((sum, o) => sum + (o.budget ?? 0), 0)

  const stats = [
    { label: 'Total Venues', value: String(venues.length), change: '' },
    { label: 'Upcoming Events', value: String(upcomingEvents.length), change: '' },
    { label: 'Pending Inquiries', value: String(pendingInquiries.length), change: '' },
    {
      label: 'Confirmed Bookings',
      value: String(confirmedBookings.length),
      change: totalRevenue > 0
        ? `${new Intl.NumberFormat('en-AU', { style: 'currency', currency: 'AUD', maximumFractionDigits: 0 }).format(totalRevenue)} revenue`
        : '',
    },
  ]

  return (
    <div className="space-y-8 max-w-6xl">
      <div>
        <h1 className="text-2xl font-semibold text-surface-95">Dashboard</h1>
        <p className="text-sm text-surface-60 mt-1">Overview of your venue operations</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {loading
          ? Array.from({ length: 4 }, (_, i) => (
              <Card key={i}>
                <Skeleton width="w-24" height="h-3" />
                <Skeleton width="w-16" height="h-8" className="mt-2" />
                <Skeleton width="w-20" height="h-3" className="mt-3" />
              </Card>
            ))
          : stats.map((stat) => (
              <Card key={stat.label}>
                <p className="text-xs text-surface-60 uppercase tracking-wider">{stat.label}</p>
                <p className="text-3xl font-bold text-surface-95 mt-1">{stat.value}</p>
                {stat.change && <p className="text-xs text-surface-60 mt-2">{stat.change}</p>}
              </Card>
            ))}
      </div>

      {/* Quick Actions */}
      <div className="flex flex-wrap gap-3">
        <Link
          href="/editor"
          className="inline-flex items-center gap-2 px-4 py-2 bg-gold-30 text-surface-5 rounded-lg text-sm font-medium hover:bg-gold-40 transition-colors"
        >
          Open 3D Editor
        </Link>
        <Link
          href="/venues"
          className="inline-flex items-center gap-2 px-4 py-2 bg-surface-25 text-surface-90 border border-surface-40 rounded-lg text-sm font-medium hover:bg-surface-30 transition-colors"
        >
          Manage Venues
        </Link>
        <Link
          href="/catalog"
          className="inline-flex items-center gap-2 px-4 py-2 bg-surface-25 text-surface-90 border border-surface-40 rounded-lg text-sm font-medium hover:bg-surface-30 transition-colors"
        >
          Furniture Catalog
        </Link>
      </div>

      {/* Recent Activity */}
      <Card header="Recent Activity">
        <ul className="divide-y divide-surface-20 -mx-5">
          {recentActivity.map((item) => (
            <li key={item.id} className="flex items-center justify-between px-5 py-3">
              <div className="flex items-center gap-3 min-w-0">
                <Badge variant={badgeVariant[item.type]}>
                  {item.type}
                </Badge>
                <span className="text-sm text-surface-80 truncate">{item.text}</span>
              </div>
              <span className="text-xs text-surface-60 whitespace-nowrap ml-4">{item.time}</span>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  )
}
