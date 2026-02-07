'use client'

import { useState } from 'react'
import { Modal } from '../../components/ui/Modal'
import { Input } from '../../components/ui/Input'
import { Select } from '../../components/ui/Select'
import { Button } from '../../components/ui/Button'
import { apiFetch } from '../../../lib/api-client'
import { toast } from '../../components/ui/Toast'
import { useVenues } from '../../hooks/useVenues'

interface CreateEventModalProps {
  open: boolean
  onClose: () => void
  onCreated: () => void
}

const eventTypes = [
  { value: 'wedding', label: 'Wedding' },
  { value: 'corporate', label: 'Corporate' },
  { value: 'social', label: 'Social' },
  { value: 'conference', label: 'Conference' },
  { value: 'gala', label: 'Gala' },
  { value: 'other', label: 'Other' },
]

export function CreateEventModal({ open, onClose, onCreated }: CreateEventModalProps) {
  const { venues } = useVenues()
  const [venueId, setVenueId] = useState('')
  const [name, setName] = useState('')
  const [type, setType] = useState('wedding')
  const [dateStart, setDateStart] = useState('')
  const [dateEnd, setDateEnd] = useState('')
  const [guestCount, setGuestCount] = useState('')
  const [budget, setBudget] = useState('')
  const [saving, setSaving] = useState(false)
  const [fieldErrors, setFieldErrors] = useState<Record<string, string[]>>({})

  const venueOptions = venues.map((v) => ({ value: v.id, label: v.name }))

  function reset() {
    setVenueId('')
    setName('')
    setType('wedding')
    setDateStart('')
    setDateEnd('')
    setGuestCount('')
    setBudget('')
    setFieldErrors({})
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!venueId) {
      toast.error('Please select a venue.')
      return
    }
    setSaving(true)
    setFieldErrors({})

    const res = await apiFetch(`/venues/${venueId}/occasions`, {
      method: 'POST',
      body: JSON.stringify({
        name,
        type,
        dateStart: new Date(dateStart).toISOString(),
        dateEnd: dateEnd ? new Date(dateEnd).toISOString() : new Date(dateStart).toISOString(),
        guestCount: Number(guestCount),
        budget: budget ? Number(budget) : null,
      }),
    })

    setSaving(false)

    if (res.ok) {
      toast.success('Event created successfully.')
      reset()
      onCreated()
      onClose()
    } else {
      if (res.fieldErrors) setFieldErrors(res.fieldErrors)
      toast.error(res.error ?? 'Failed to create event.')
    }
  }

  return (
    <Modal open={open} onClose={onClose} title="Create Event" width="w-96">
      <form onSubmit={handleSubmit} className="flex flex-col gap-3">
        <Select
          label="Venue"
          options={venueOptions}
          value={venueId}
          onChange={(e) => setVenueId(e.target.value)}
          placeholder="Select a venue..."
        />
        <Input
          label="Event Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
          error={fieldErrors['name']?.[0]}
        />
        <Select
          label="Event Type"
          options={eventTypes}
          value={type}
          onChange={(e) => setType(e.target.value)}
        />
        <div className="grid grid-cols-2 gap-2">
          <Input
            label="Start Date"
            type="datetime-local"
            value={dateStart}
            onChange={(e) => setDateStart(e.target.value)}
            required
          />
          <Input
            label="End Date"
            type="datetime-local"
            value={dateEnd}
            onChange={(e) => setDateEnd(e.target.value)}
          />
        </div>
        <Input
          label="Guest Count"
          type="number"
          value={guestCount}
          onChange={(e) => setGuestCount(e.target.value)}
          required
        />
        <Input
          label="Budget (AUD)"
          type="number"
          value={budget}
          onChange={(e) => setBudget(e.target.value)}
          description="Leave blank if TBD"
        />

        <div className="flex items-center justify-end gap-2 mt-2">
          <Button type="button" variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" variant="primary" loading={saving}>
            Create Event
          </Button>
        </div>
      </form>
    </Modal>
  )
}
