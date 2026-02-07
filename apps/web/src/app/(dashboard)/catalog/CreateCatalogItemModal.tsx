'use client'

import { useState } from 'react'
import { Modal } from '../../components/ui/Modal'
import { Input } from '../../components/ui/Input'
import { Select } from '../../components/ui/Select'
import { Button } from '../../components/ui/Button'
import { apiFetch } from '../../../lib/api-client'
import { toast } from '../../components/ui/Toast'

interface CreateCatalogItemModalProps {
  open: boolean
  onClose: () => void
  onCreated: () => void
}

const categories = [
  { value: 'table', label: 'Table' },
  { value: 'chair', label: 'Chair' },
  { value: 'stage', label: 'Stage' },
  { value: 'decor', label: 'Decor' },
  { value: 'equipment', label: 'Equipment' },
]

export function CreateCatalogItemModal({ open, onClose, onCreated }: CreateCatalogItemModalProps) {
  const [name, setName] = useState('')
  const [category, setCategory] = useState('table')
  const [widthFt, setWidthFt] = useState('')
  const [depthFt, setDepthFt] = useState('')
  const [heightFt, setHeightFt] = useState('')
  const [capacity, setCapacity] = useState('')
  const [stackable, setStackable] = useState(false)
  const [saving, setSaving] = useState(false)
  const [fieldErrors, setFieldErrors] = useState<Record<string, string[]>>({})

  function reset() {
    setName('')
    setCategory('table')
    setWidthFt('')
    setDepthFt('')
    setHeightFt('')
    setCapacity('')
    setStackable(false)
    setFieldErrors({})
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setSaving(true)
    setFieldErrors({})

    const res = await apiFetch('/catalog', {
      method: 'POST',
      body: JSON.stringify({
        name,
        category,
        widthFt: Number(widthFt),
        depthFt: Number(depthFt),
        heightFt: Number(heightFt),
        capacity: capacity ? Number(capacity) : null,
        stackable,
      }),
    })

    setSaving(false)

    if (res.ok) {
      toast.success('Catalog item created.')
      reset()
      onCreated()
      onClose()
    } else {
      if (res.fieldErrors) setFieldErrors(res.fieldErrors)
      toast.error(res.error ?? 'Failed to create catalog item.')
    }
  }

  return (
    <Modal open={open} onClose={onClose} title="Add Catalog Item" width="w-96">
      <form onSubmit={handleSubmit} className="flex flex-col gap-3">
        <Input
          label="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
          error={fieldErrors['name']?.[0]}
        />
        <Select
          label="Category"
          options={categories}
          value={category}
          onChange={(e) => setCategory(e.target.value)}
        />
        <div className="grid grid-cols-3 gap-2">
          <Input
            label="Width (ft)"
            type="number"
            value={widthFt}
            onChange={(e) => setWidthFt(e.target.value)}
            required
          />
          <Input
            label="Depth (ft)"
            type="number"
            value={depthFt}
            onChange={(e) => setDepthFt(e.target.value)}
            required
          />
          <Input
            label="Height (ft)"
            type="number"
            value={heightFt}
            onChange={(e) => setHeightFt(e.target.value)}
            required
          />
        </div>
        <Input
          label="Capacity (seats)"
          type="number"
          value={capacity}
          onChange={(e) => setCapacity(e.target.value)}
          description="Leave blank if not applicable"
        />
        <label className="flex items-center gap-2 text-sm text-surface-80 cursor-pointer">
          <input
            type="checkbox"
            checked={stackable}
            onChange={(e) => setStackable(e.target.checked)}
            className="rounded border-surface-40"
          />
          Stackable
        </label>

        <div className="flex items-center justify-end gap-2 mt-2">
          <Button type="button" variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" variant="primary" loading={saving}>
            Add Item
          </Button>
        </div>
      </form>
    </Modal>
  )
}
