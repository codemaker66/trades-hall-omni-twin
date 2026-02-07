'use client'

import { useState } from 'react'
import { Modal } from '../../components/ui/Modal'
import { Input } from '../../components/ui/Input'
import { Select } from '../../components/ui/Select'
import { Button } from '../../components/ui/Button'
import { apiFetch } from '../../../lib/api-client'
import { toast } from '../../components/ui/Toast'

interface CreateVenueModalProps {
  open: boolean
  onClose: () => void
  onCreated: () => void
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

const statusOptions = [
  { value: 'draft', label: 'Draft' },
  { value: 'published', label: 'Published' },
]

export function CreateVenueModal({ open, onClose, onCreated }: CreateVenueModalProps) {
  const [name, setName] = useState('')
  const [slug, setSlug] = useState('')
  const [venueType, setVenueType] = useState('ballroom')
  const [capacity, setCapacity] = useState('')
  const [address, setAddress] = useState('')
  const [status, setStatus] = useState('draft')
  const [saving, setSaving] = useState(false)
  const [fieldErrors, setFieldErrors] = useState<Record<string, string[]>>({})

  function reset() {
    setName('')
    setSlug('')
    setVenueType('ballroom')
    setCapacity('')
    setAddress('')
    setStatus('draft')
    setFieldErrors({})
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setSaving(true)
    setFieldErrors({})

    const res = await apiFetch('/venues', {
      method: 'POST',
      body: JSON.stringify({
        name,
        slug: slug || name.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, ''),
        venueType,
        capacity: capacity ? Number(capacity) : null,
        address: address || null,
        status,
      }),
    })

    setSaving(false)

    if (res.ok) {
      toast.success('Venue created successfully.')
      reset()
      onCreated()
      onClose()
    } else {
      if (res.fieldErrors) setFieldErrors(res.fieldErrors)
      toast.error(res.error ?? 'Failed to create venue.')
    }
  }

  return (
    <Modal open={open} onClose={onClose} title="Create Venue" width="w-96">
      <form onSubmit={handleSubmit} className="flex flex-col gap-3">
        <Input
          label="Venue Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
          error={fieldErrors['name']?.[0]}
        />
        <Input
          label="Slug"
          value={slug}
          onChange={(e) => setSlug(e.target.value)}
          description="Leave blank to auto-generate from name"
          error={fieldErrors['slug']?.[0]}
        />
        <Select
          label="Venue Type"
          options={venueTypes}
          value={venueType}
          onChange={(e) => setVenueType(e.target.value)}
        />
        <Input
          label="Capacity"
          type="number"
          value={capacity}
          onChange={(e) => setCapacity(e.target.value)}
        />
        <Input
          label="Address"
          value={address}
          onChange={(e) => setAddress(e.target.value)}
        />
        <Select
          label="Status"
          options={statusOptions}
          value={status}
          onChange={(e) => setStatus(e.target.value)}
        />

        <div className="flex items-center justify-end gap-2 mt-2">
          <Button type="button" variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" variant="primary" loading={saving}>
            Create Venue
          </Button>
        </div>
      </form>
    </Modal>
  )
}
