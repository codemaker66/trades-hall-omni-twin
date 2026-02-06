'use client'

import { use } from 'react'
import { useState } from 'react'
import Link from 'next/link'
import { Card } from '../../../components/ui/Card'
import { Badge } from '../../../components/ui/Badge'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Select } from '../../../components/ui/Select'

// Placeholder data — will be replaced with API fetch
const mockVenueData = {
  id: '1',
  name: 'Trades Hall',
  slug: 'trades-hall',
  description: 'A magnificent heritage-listed building in the heart of Melbourne, perfect for grand events and conferences.',
  venueType: 'ballroom',
  capacity: 500,
  squareFootage: 8500,
  address: '54 Victoria St, Melbourne VIC 3000',
  pricingModel: 'hourly' as const,
  basePrice: 250,
  status: 'published' as const,
  amenities: ['WiFi', 'Stage', 'AV Equipment', 'Catering Kitchen', 'Parking'],
}

const venueTypes = [
  { value: 'ballroom', label: 'Ballroom' },
  { value: 'conference', label: 'Conference' },
  { value: 'outdoor', label: 'Outdoor' },
  { value: 'theater', label: 'Theater' },
  { value: 'restaurant', label: 'Restaurant' },
  { value: 'warehouse', label: 'Warehouse' },
  { value: 'other', label: 'Other' },
]

const pricingModels = [
  { value: 'hourly', label: 'Hourly' },
  { value: 'daily', label: 'Daily' },
  { value: 'flat', label: 'Flat Rate' },
  { value: 'custom', label: 'Custom' },
]

const statusOptions = [
  { value: 'draft', label: 'Draft' },
  { value: 'published', label: 'Published' },
  { value: 'archived', label: 'Archived' },
]

const statusBadge: Record<string, 'success' | 'gold' | 'default'> = {
  published: 'success',
  draft: 'gold',
  archived: 'default',
}

export default function VenueDetailPage({ params }: { params: Promise<{ venueId: string }> }) {
  const { venueId } = use(params)
  const venue = mockVenueData // In production: fetch by venueId

  const [name, setName] = useState(venue.name)
  const [slug, setSlug] = useState(venue.slug)
  const [description, setDescription] = useState(venue.description)
  const [venueType, setVenueType] = useState(venue.venueType)
  const [capacity, setCapacity] = useState(String(venue.capacity))
  const [squareFootage, setSquareFootage] = useState(String(venue.squareFootage))
  const [address, setAddress] = useState(venue.address)
  const [pricingModel, setPricingModel] = useState(venue.pricingModel)
  const [basePrice, setBasePrice] = useState(String(venue.basePrice))
  const [status, setStatus] = useState(venue.status)

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Breadcrumb + Actions */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm">
          <Link href="/venues" className="text-surface-60 hover:text-surface-80 transition-colors">
            Venues
          </Link>
          <span className="text-surface-40">/</span>
          <span className="text-surface-90 font-medium">{venue.name}</span>
          <Badge variant={statusBadge[status]}>{status}</Badge>
        </div>
        <div className="flex items-center gap-2">
          <Link href={`/editor`}>
            <Button variant="ghost" size="sm">Open in Editor</Button>
          </Link>
          <Button variant="primary" size="sm">Save Changes</Button>
        </div>
      </div>

      {/* Basic Info */}
      <Card header="Basic Information">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Input
            label="Venue Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <Input
            label="Slug"
            value={slug}
            onChange={(e) => setSlug(e.target.value)}
            description="URL-friendly identifier"
          />
          <div className="md:col-span-2">
            <label className="text-sm text-surface-80 block mb-1.5">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
              className="w-full bg-surface-5 border border-white/10 rounded-lg px-4 py-2 text-surface-90 text-sm placeholder:text-surface-60 focus:outline-none focus:border-indigo-50"
            />
          </div>
          <Select
            label="Venue Type"
            options={venueTypes}
            value={venueType}
            onChange={(e) => setVenueType(e.target.value)}
          />
          <Select
            label="Status"
            options={statusOptions}
            value={status}
            onChange={(e) => setStatus(e.target.value as typeof status)}
          />
        </div>
      </Card>

      {/* Location & Capacity */}
      <Card header="Location & Capacity">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="md:col-span-2">
            <Input
              label="Address"
              value={address}
              onChange={(e) => setAddress(e.target.value)}
            />
          </div>
          <Input
            label="Capacity (guests)"
            type="number"
            value={capacity}
            onChange={(e) => setCapacity(e.target.value)}
          />
          <Input
            label="Square Footage"
            type="number"
            value={squareFootage}
            onChange={(e) => setSquareFootage(e.target.value)}
          />
        </div>
      </Card>

      {/* Pricing */}
      <Card header="Pricing">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Select
            label="Pricing Model"
            options={pricingModels}
            value={pricingModel}
            onChange={(e) => setPricingModel(e.target.value as typeof pricingModel)}
          />
          <Input
            label="Base Price ($)"
            type="number"
            value={basePrice}
            onChange={(e) => setBasePrice(e.target.value)}
          />
        </div>
      </Card>

      {/* Amenities */}
      <Card header="Amenities">
        <div className="flex flex-wrap gap-2">
          {venue.amenities.map((amenity) => (
            <Badge key={amenity} variant="gold">{amenity}</Badge>
          ))}
        </div>
        <p className="text-xs text-surface-60 mt-3">Amenity editing will be available in a future update.</p>
      </Card>
    </div>
  )
}
