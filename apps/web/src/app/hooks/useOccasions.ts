'use client'

import { useState, useEffect, useCallback } from 'react'
import { apiFetch } from '../../lib/api-client'

export interface Occasion {
  id: string
  venueId: string
  venueName: string
  organizerId: string
  name: string
  type: 'wedding' | 'corporate' | 'social' | 'conference' | 'gala' | 'other'
  dateStart: string
  dateEnd: string
  guestCount: number
  status: 'inquiry' | 'confirmed' | 'completed' | 'cancelled'
  budget: number | null
  notes: string | null
  createdAt: string
  updatedAt: string
}

interface VenueSummary {
  id: string
  name: string
}

interface RawOccasion {
  id: string
  venueId: string
  organizerId: string
  name: string
  type: 'wedding' | 'corporate' | 'social' | 'conference' | 'gala' | 'other'
  dateStart: string
  dateEnd: string
  guestCount: number
  status: 'inquiry' | 'confirmed' | 'completed' | 'cancelled'
  budget: number | null
  notes: string | null
  createdAt: string
  updatedAt: string
}

/**
 * Fetch all occasions across all venues the user has access to.
 *
 * Fan-out pattern: fetches venues list, then fetches occasions per venue
 * in parallel. Attaches venue name to each occasion for display.
 */
export function useAllOccasions() {
  const [occasions, setOccasions] = useState<Occasion[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchAll = useCallback(async () => {
    setLoading(true)
    setError(null)

    // Step 1: fetch venues
    const venuesRes = await apiFetch<VenueSummary[]>('/venues')
    if (!venuesRes.ok || !venuesRes.data) {
      setError(venuesRes.error ?? 'Failed to load venues.')
      setLoading(false)
      return
    }

    const venues = venuesRes.data
    if (venues.length === 0) {
      setOccasions([])
      setLoading(false)
      return
    }

    // Step 2: fan-out — fetch occasions per venue in parallel
    const results = await Promise.all(
      venues.map(async (v) => {
        const res = await apiFetch<RawOccasion[]>(
          `/venues/${v.id}/occasions`,
        )
        if (res.ok && res.data) {
          return res.data.map((o) => ({ ...o, venueName: v.name }))
        }
        return []
      }),
    )

    // Step 3: merge and sort by dateStart descending
    const merged = results
      .flat()
      .sort(
        (a, b) =>
          new Date(b.dateStart).getTime() - new Date(a.dateStart).getTime(),
      )

    setOccasions(merged)
    setLoading(false)
  }, [])

  useEffect(() => {
    fetchAll()
  }, [fetchAll])

  return { occasions, loading, error, refetch: fetchAll }
}
