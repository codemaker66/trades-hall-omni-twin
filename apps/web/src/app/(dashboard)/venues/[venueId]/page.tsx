'use client'

import { use, useState, useEffect } from 'react'
import Link from 'next/link'
import { Card } from '../../../components/ui/Card'
import { Badge } from '../../../components/ui/Badge'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Select } from '../../../components/ui/Select'
import { Skeleton } from '../../../components/ui/Skeleton'
import { ErrorAlert } from '../../../components/ui/ErrorAlert'
import { useVenue } from '../../../hooks/useVenues'
import { apiFetch } from '../../../../lib/api-client'
import { toast } from '../../../components/ui/Toast'

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
  const { venue, loading, error, refetch } = useVenue(venueId)

  const [name, setName] = useState('')
  const [slug, setSlug] = useState('')
  const [description, setDescription] = useState('')
  const [venueType, setVenueType] = useState('')
  const [capacity, setCapacity] = useState('')
  const [squareFootage, setSquareFootage] = useState('')
  const [address, setAddress] = useState('')
  const [pricingModel, setPricingModel] = useState('')
  const [basePrice, setBasePrice] = useState('')
  const [status, setStatus] = useState<'draft' | 'published' | 'archived'>('draft')
  const [saving, setSaving] = useState(false)

  // Populate form when venue data arrives
  useEffect(() => {
    if (venue) {
      setName(venue.name)
      setSlug(venue.slug)
      setDescription(venue.description ?? '')
      setVenueType(venue.venueType ?? '')
      setCapacity(venue.capacity != null ? String(venue.capacity) : '')
      setSquareFootage(venue.squareFootage != null ? String(venue.squareFootage) : '')
      setAddress(venue.address ?? '')
      setPricingModel(venue.pricingModel ?? '')
      setBasePrice(venue.basePrice != null ? String(venue.basePrice) : '')
      setStatus(venue.status)
    }
  }, [venue])

  async function handleSave() {
    setSaving(true)
    const res = await apiFetch(`/venues/${venueId}`, {
      method: 'PATCH',
      body: JSON.stringify({
        name,
        slug,
        description: description || null,
        venueType: venueType || null,
        capacity: capacity ? Number(capacity) : null,
        squareFootage: squareFootage ? Number(squareFootage) : null,
        address: address || null,
        pricingModel: pricingModel || null,
        basePrice: basePrice ? Number(basePrice) : null,
        status,
      }),
    })
    setSaving(false)

    if (res.ok) {
      toast.success('Venue saved successfully.')
      refetch()
    } else {
      toast.error(res.error ?? 'Failed to save venue.')
    }
  }

  if (loading) {
    return (
      <div className="space-y-6 max-w-4xl">
        <Skeleton width="w-48" height="h-6" />
        <div className="bg-surface-10 border border-surface-25 rounded-xl p-5 space-y-4">
          <Skeleton width="w-1/3" height="h-5" />
          <div className="grid grid-cols-2 gap-4">
            <Skeleton height="h-10" />
            <Skeleton height="h-10" />
          </div>
          <Skeleton height="h-20" />
        </div>
        <div className="bg-surface-10 border border-surface-25 rounded-xl p-5 space-y-4">
          <Skeleton width="w-1/4" height="h-5" />
          <div className="grid grid-cols-2 gap-4">
            <Skeleton height="h-10" />
            <Skeleton height="h-10" />
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="max-w-4xl">
        <ErrorAlert message={error} onRetry={refetch} />
      </div>
    )
  }

  if (!venue) return null

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
          <Button variant="primary" size="sm" onClick={handleSave} loading={saving}>
            Save Changes
          </Button>
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
            onChange={(e) => setPricingModel(e.target.value)}
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
          {(venue.amenities ?? []).map((amenity) => (
            <Badge key={amenity} variant="gold">{amenity}</Badge>
          ))}
          {(venue.amenities ?? []).length === 0 && (
            <p className="text-xs text-surface-60">No amenities configured.</p>
          )}
        </div>
        <p className="text-xs text-surface-60 mt-3">Amenity editing will be available in a future update.</p>
      </Card>
    </div>
  )
}
