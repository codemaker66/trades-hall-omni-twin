'use client'

import { useSyncExternalStore, useCallback } from 'react'
import {
  resolveFlag,
  setLocalOverride,
  clearLocalOverride,
  clearAllLocalOverrides,
  readLocalOverrides,
  FEATURE_FLAG_KEYS,
  type FeatureFlagKey,
  type FeatureFlags,
} from '@omni-twin/config'

// Listeners for flag changes — allows re-render when flags are toggled at runtime.
const listeners = new Set<() => void>()

function subscribe(cb: () => void) {
  listeners.add(cb)
  return () => { listeners.delete(cb) }
}

function notifyListeners() {
  for (const cb of listeners) cb()
}

// Snapshot: returns a cache-busted object so useSyncExternalStore triggers re-render.
let snapshotVersion = 0
let cachedSnapshot: FeatureFlags | null = null

function buildSnapshot(): FeatureFlags {
  const out = {} as FeatureFlags
  for (const k of FEATURE_FLAG_KEYS) out[k] = resolveFlag(k)
  return out
}

function getSnapshot(): FeatureFlags {
  if (typeof window === 'undefined') return buildSnapshot()
  if (!cachedSnapshot) cachedSnapshot = buildSnapshot()
  return cachedSnapshot
}

function getServerSnapshot(): FeatureFlags {
  return buildSnapshot()
}

function invalidateSnapshot() {
  snapshotVersion++
  cachedSnapshot = null
  notifyListeners()
}

/** Read a single feature flag. Re-renders when the flag changes at runtime. */
export function useFeatureFlag(key: FeatureFlagKey): boolean {
  const snapshot = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot)
  return snapshot[key]
}

/** Read all resolved feature flags. Re-renders when any flag changes. */
export function useFeatureFlags(): FeatureFlags {
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot)
}

/** Toggle a flag override at runtime (persists to localStorage). */
export function useFeatureFlagControls() {
  const overrides = readLocalOverrides()

  const setFlag = useCallback((key: FeatureFlagKey, value: boolean) => {
    setLocalOverride(key, value)
    invalidateSnapshot()
  }, [])

  const clearFlag = useCallback((key: FeatureFlagKey) => {
    clearLocalOverride(key)
    invalidateSnapshot()
  }, [])

  const clearAll = useCallback(() => {
    clearAllLocalOverrides()
    invalidateSnapshot()
  }, [])

  return { overrides, setFlag, clearFlag, clearAll }
}

// Suppress unused warning — snapshotVersion is read indirectly via invalidation.
void snapshotVersion
