'use client'

import { useState } from 'react'
import { FEATURE_FLAG_KEYS, DEFAULT_FLAGS, type FeatureFlagKey } from '@omni-twin/config'
import { useFeatureFlags, useFeatureFlagControls } from '../../hooks/useFeatureFlag'
import { Button } from '../ui/Button'

const FLAG_LABELS: Record<FeatureFlagKey, string> = {
  ENABLE_WEBGPU: 'WebGPU Renderer',
  ENABLE_COLLABORATION: 'Real-time Collaboration',
  ENABLE_AI: 'AI Spatial Planning',
  ENABLE_GAUSSIAN_SPLATTING: 'Gaussian Splatting',
  ENABLE_WASM: 'Rust/WASM Engine',
  ENABLE_ECS: 'ECS Data Model',
  ENABLE_EVENT_SOURCING: 'Event Sourcing',
}

export const FeatureFlagPanel = () => {
  const [open, setOpen] = useState(false)
  const resolved = useFeatureFlags()
  const { overrides, setFlag, clearFlag, clearAll } = useFeatureFlagControls()

  // Only show in development
  if (process.env.NODE_ENV !== 'development') return null

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="fixed bottom-4 right-4 z-50 px-2 py-1 text-xs font-mono bg-surface-20 border border-surface-40 text-surface-60 rounded-sm hover:bg-surface-25 hover:text-surface-80 pointer-events-auto"
        aria-label="Open feature flags panel"
      >
        flags
      </button>
    )
  }

  return (
    <div className="fixed bottom-4 right-4 z-50 w-72 bg-surface-10 border border-surface-30 rounded-sm shadow-2xl pointer-events-auto">
      <div className="flex items-center justify-between px-3 py-2 border-b border-surface-25">
        <span className="text-xs font-bold text-surface-80 tracking-wider uppercase">Feature Flags</span>
        <button
          onClick={() => setOpen(false)}
          className="text-surface-50 hover:text-surface-80 text-sm leading-none"
          aria-label="Close feature flags panel"
        >
          x
        </button>
      </div>

      <div className="px-3 py-2 flex flex-col gap-2">
        {FEATURE_FLAG_KEYS.map((key) => {
          const isOverridden = key in overrides
          const value = resolved[key]

          return (
            <label key={key} className="flex items-center gap-2 cursor-pointer group">
              <input
                type="checkbox"
                checked={value}
                onChange={(e) => setFlag(key, e.target.checked)}
                className="accent-accent-60"
              />
              <span className={`text-xs flex-1 ${value ? 'text-surface-80' : 'text-surface-50'}`}>
                {FLAG_LABELS[key]}
              </span>
              {isOverridden && (
                <button
                  onClick={(e) => {
                    e.preventDefault()
                    clearFlag(key)
                  }}
                  className="text-[10px] text-surface-40 hover:text-surface-70 opacity-0 group-hover:opacity-100"
                  title={`Reset to default (${DEFAULT_FLAGS[key] ? 'on' : 'off'})`}
                >
                  reset
                </button>
              )}
            </label>
          )
        })}
      </div>

      <div className="px-3 py-2 border-t border-surface-25">
        <Button size="sm" variant="ghost" onClick={clearAll} className="w-full text-xs">
          Reset all to defaults
        </Button>
      </div>
    </div>
  )
}
