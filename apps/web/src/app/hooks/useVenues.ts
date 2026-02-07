'use client'

import { useState, useEffect, useCallback } from 'react'
import { apiFetch } from '../../lib/api-client'

export interface Venue {
  id: string
  name: string
  slug: string
  description: string | null
  venueType: string | null
  capacity: number | null
  squareFootage: number | null
  address: string | null
  latitude: number | null
  longitude: number | null
  pricingModel: string | null
  basePrice: number | null
  amenities: string[]
  images: string[]
  status: 'draft' | 'published' | 'archived'
  createdAt: string
  updatedAt: string
}

export function useVenues() {
  const [venues, setVenues] = useState<Venue[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchVenues = useCallback(async () => {
    setLoading(true)
    setError(null)
    const res = await apiFetch<Venue[]>('/venues')
    if (res.ok && res.data) {
      setVenues(res.data)
    } else {
      setError(res.error ?? 'Failed to load venues.')
    }
    setLoading(false)
  }, [])

  useEffect(() => {
    fetchVenues()
  }, [fetchVenues])

  return { venues, loading, error, refetch: fetchVenues }
}

export function useVenue(venueId: string) {
  const [venue, setVenue] = useState<Venue | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchVenue = useCallback(async () => {
    setLoading(true)
    setError(null)
    const res = await apiFetch<Venue>(`/venues/${venueId}`)
    if (res.ok && res.data) {
      setVenue(res.data)
    } else {
      setError(res.error ?? 'Failed to load venue.')
    }
    setLoading(false)
  }, [venueId])

  useEffect(() => {
    fetchVenue()
  }, [fetchVenue])

  return { venue, loading, error, refetch: fetchVenue }
}
