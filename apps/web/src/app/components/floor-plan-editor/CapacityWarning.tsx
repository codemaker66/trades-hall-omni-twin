'use client'

import { useFloorPlanStore } from './store'

interface CapacityWarningProps {
  maxCapacity: number
}

export function CapacityWarning({ maxCapacity }: CapacityWarningProps) {
  const metrics = useFloorPlanStore((s) => s.getMetrics())
  const isOver = metrics.totalSeats > maxCapacity

  if (!isOver) return null

  return (
    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 pointer-events-auto">
      <div className="bg-danger-25 border border-danger-50 text-danger-70 px-4 py-2 rounded-lg text-sm font-medium shadow-lg">
        Capacity warning: {metrics.totalSeats} seats exceeds limit of {maxCapacity}
      </div>
    </div>
  )
}
