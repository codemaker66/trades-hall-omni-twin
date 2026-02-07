'use client'

import { useState, useEffect, useCallback } from 'react'
import { apiFetch } from '../../lib/api-client'

export interface CatalogItem {
  id: string
  name: string
  category: 'table' | 'chair' | 'stage' | 'decor' | 'equipment'
  modelUrl: string | null
  thumbnailUrl: string | null
  widthFt: number
  depthFt: number
  heightFt: number
  capacity: number | null
  stackable: boolean
  createdAt: string
  updatedAt: string
}

export function useCatalog() {
  const [items, setItems] = useState<CatalogItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchCatalog = useCallback(async () => {
    setLoading(true)
    setError(null)
    const res = await apiFetch<CatalogItem[]>('/catalog')
    if (res.ok && res.data) {
      setItems(res.data)
    } else {
      setError(res.error ?? 'Failed to load catalog.')
    }
    setLoading(false)
  }, [])

  useEffect(() => {
    fetchCatalog()
  }, [fetchCatalog])

  return { items, loading, error, refetch: fetchCatalog }
}
