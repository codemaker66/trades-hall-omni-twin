'use client'

import { useState, useCallback } from 'react'

/**
 * Constraint Weight Panel — Interactive control for layout energy weights.
 *
 * Features:
 * - 8 weight sliders (overlap, aisle, egress, sightline, capacity, ADA, aesthetic, service)
 * - Logarithmic scale display (weights span 10 to 1e6)
 * - Reset to defaults button
 * - Preset buttons (safety-first, capacity-max, balanced, aesthetic)
 * - Live energy breakdown bar chart
 */

export interface ConstraintWeights {
  overlap: number
  aisle: number
  egress: number
  sightline: number
  capacity: number
  ada: number
  aesthetic: number
  service: number
}

export interface EnergyBreakdown {
  overlap: number
  aisle: number
  egress: number
  sightline: number
  capacity: number
  ada: number
  aesthetic: number
  service: number
}

export interface ConstraintPanelProps {
  weights: ConstraintWeights
  onChange: (weights: ConstraintWeights) => void
  energyBreakdown?: EnergyBreakdown
  title?: string
}

const DEFAULT_WEIGHTS: ConstraintWeights = {
  overlap: 1e6,
  aisle: 1e4,
  egress: 1e6,
  sightline: 100,
  capacity: 1e4,
  ada: 1e6,
  aesthetic: 10,
  service: 50,
}

const PRESETS: Record<string, { label: string; weights: ConstraintWeights }> = {
  balanced: {
    label: 'Balanced',
    weights: { ...DEFAULT_WEIGHTS },
  },
  safety: {
    label: 'Safety First',
    weights: {
      overlap: 1e6, aisle: 1e6, egress: 1e6, sightline: 100,
      capacity: 1e3, ada: 1e6, aesthetic: 1, service: 10,
    },
  },
  capacity: {
    label: 'Max Capacity',
    weights: {
      overlap: 1e5, aisle: 1e3, egress: 1e5, sightline: 10,
      capacity: 1e6, ada: 1e5, aesthetic: 1, service: 10,
    },
  },
  aesthetic: {
    label: 'Aesthetic',
    weights: {
      overlap: 1e5, aisle: 1e4, egress: 1e5, sightline: 1e4,
      capacity: 1e3, ada: 1e5, aesthetic: 1e4, service: 1e3,
    },
  },
}

const CONSTRAINT_META: Array<{
  key: keyof ConstraintWeights
  label: string
  color: string
  description: string
}> = [
  { key: 'overlap', label: 'Overlap', color: '#ef4444', description: 'Furniture collision penalty' },
  { key: 'aisle', label: 'Aisle', color: '#f97316', description: 'IBC 1029.9.2 aisle width compliance' },
  { key: 'egress', label: 'Egress', color: '#eab308', description: 'IBC 1017 exit path clearance' },
  { key: 'sightline', label: 'Sightline', color: '#22c55e', description: 'View to stage/podium quality' },
  { key: 'capacity', label: 'Capacity', color: '#06b6d4', description: 'Seating capacity vs target' },
  { key: 'ada', label: 'ADA', color: '#3b82f6', description: 'Accessibility compliance' },
  { key: 'aesthetic', label: 'Aesthetic', color: '#a855f7', description: 'Symmetry and alignment' },
  { key: 'service', label: 'Service', color: '#ec4899', description: 'Service station access paths' },
]

/** Convert a weight value to slider position (log scale, 0-100) */
function weightToSlider(w: number): number {
  return Math.round((Math.log10(Math.max(w, 1)) / 6) * 100)
}

/** Convert slider position back to weight (log scale) */
function sliderToWeight(s: number): number {
  return Math.pow(10, (s / 100) * 6)
}

/** Format weight for display */
function formatWeight(w: number): string {
  if (w >= 1e6) return `${(w / 1e6).toFixed(w % 1e6 === 0 ? 0 : 1)}M`
  if (w >= 1e3) return `${(w / 1e3).toFixed(w % 1e3 === 0 ? 0 : 1)}K`
  return w.toFixed(0)
}

export function ConstraintPanel({
  weights,
  onChange,
  energyBreakdown,
  title = 'Constraint Weights',
}: ConstraintPanelProps) {
  const [expanded, setExpanded] = useState(true)

  const handleSliderChange = useCallback(
    (key: keyof ConstraintWeights, sliderValue: number) => {
      onChange({ ...weights, [key]: sliderToWeight(sliderValue) })
    },
    [weights, onChange],
  )

  const handlePreset = useCallback(
    (presetKey: string) => {
      const preset = PRESETS[presetKey]
      if (preset) onChange({ ...preset.weights })
    },
    [onChange],
  )

  // Compute max energy term for bar chart scaling
  const maxEnergy = energyBreakdown
    ? Math.max(
        ...CONSTRAINT_META.map(c => (energyBreakdown[c.key] ?? 0) * weights[c.key]),
        1,
      )
    : 1

  return (
    <div className="rounded-lg bg-slate-900 p-4">
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <h3 className="text-sm font-medium text-slate-200">{title}</h3>
        <span className="text-slate-500 text-xs">{expanded ? '\u25B2' : '\u25BC'}</span>
      </div>

      {expanded && (
        <div className="mt-3 space-y-4">
          {/* Presets */}
          <div className="flex gap-2">
            {Object.entries(PRESETS).map(([key, preset]) => (
              <button
                key={key}
                onClick={() => handlePreset(key)}
                className="px-2 py-1 text-[10px] rounded bg-slate-800 text-slate-300 hover:bg-slate-700 transition-colors"
              >
                {preset.label}
              </button>
            ))}
            <button
              onClick={() => onChange({ ...DEFAULT_WEIGHTS })}
              className="px-2 py-1 text-[10px] rounded bg-slate-800 text-slate-400 hover:bg-slate-700 transition-colors ml-auto"
            >
              Reset
            </button>
          </div>

          {/* Weight sliders + energy bars */}
          <div className="space-y-2">
            {CONSTRAINT_META.map(({ key, label, color, description }) => {
              const sliderVal = weightToSlider(weights[key])
              const energyVal = energyBreakdown
                ? (energyBreakdown[key] ?? 0) * weights[key]
                : 0
              const barWidth = energyBreakdown
                ? Math.max((energyVal / maxEnergy) * 100, 0)
                : 0

              return (
                <div key={key} className="group">
                  <div className="flex items-center justify-between text-[10px] mb-0.5">
                    <div className="flex items-center gap-1.5">
                      <div
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: color }}
                      />
                      <span className="text-slate-300 font-medium">{label}</span>
                      <span className="text-slate-600 hidden group-hover:inline">
                        {description}
                      </span>
                    </div>
                    <span className="text-slate-400 font-mono">
                      {formatWeight(weights[key])}
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    <input
                      type="range"
                      min={0}
                      max={100}
                      value={sliderVal}
                      onChange={e => handleSliderChange(key, parseInt(e.target.value, 10))}
                      className="flex-1 h-1 accent-slate-500"
                    />

                    {/* Energy bar */}
                    {energyBreakdown && (
                      <div className="w-20 h-2 bg-slate-800 rounded overflow-hidden">
                        <div
                          className="h-full rounded transition-all duration-200"
                          style={{
                            width: `${barWidth}%`,
                            backgroundColor: color,
                            opacity: 0.7,
                          }}
                        />
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>

          {/* Total energy */}
          {energyBreakdown && (
            <div className="pt-2 border-t border-slate-800 flex justify-between text-xs">
              <span className="text-slate-400">Total Energy</span>
              <span className="text-slate-200 font-mono">
                {CONSTRAINT_META.reduce(
                  (sum, c) => sum + (energyBreakdown[c.key] ?? 0) * weights[c.key],
                  0,
                ).toFixed(0)}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
