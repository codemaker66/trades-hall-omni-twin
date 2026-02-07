/** Feature flags for progressive feature rollout. */
export interface FeatureFlags {
  ENABLE_WEBGPU: boolean
  ENABLE_COLLABORATION: boolean
  ENABLE_AI: boolean
  ENABLE_GAUSSIAN_SPLATTING: boolean
  ENABLE_WASM: boolean
  ENABLE_ECS: boolean
  ENABLE_EVENT_SOURCING: boolean
}

export type FeatureFlagKey = keyof FeatureFlags

/** All flag keys for iteration. */
export const FEATURE_FLAG_KEYS: FeatureFlagKey[] = [
  'ENABLE_WEBGPU',
  'ENABLE_COLLABORATION',
  'ENABLE_AI',
  'ENABLE_GAUSSIAN_SPLATTING',
  'ENABLE_WASM',
  'ENABLE_ECS',
  'ENABLE_EVENT_SOURCING',
]

/** Default flag values — all advanced features off until implemented. */
export const DEFAULT_FLAGS: FeatureFlags = {
  ENABLE_WEBGPU: false,
  ENABLE_COLLABORATION: false,
  ENABLE_AI: false,
  ENABLE_GAUSSIAN_SPLATTING: false,
  ENABLE_WASM: false,
  ENABLE_ECS: false,
  ENABLE_EVENT_SOURCING: false,
}

function readEnvFlag(key: string): boolean | undefined {
  if (typeof process !== 'undefined' && process.env) {
    const val = process.env[key]
    if (val === 'true' || val === '1') return true
    if (val === 'false' || val === '0') return false
  }
  return undefined
}

const STORAGE_KEY = 'omnitwin_flag_overrides'

/** Read runtime overrides from localStorage (browser only). */
export function readLocalOverrides(): Partial<FeatureFlags> {
  if (typeof window === 'undefined') return {}
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return {}
    return JSON.parse(raw) as Partial<FeatureFlags>
  } catch {
    return {}
  }
}

/** Persist a flag override to localStorage. */
export function setLocalOverride(key: FeatureFlagKey, value: boolean): void {
  if (typeof window === 'undefined') return
  const overrides = readLocalOverrides()
  overrides[key] = value
  localStorage.setItem(STORAGE_KEY, JSON.stringify(overrides))
}

/** Remove a flag override from localStorage (revert to env/default). */
export function clearLocalOverride(key: FeatureFlagKey): void {
  if (typeof window === 'undefined') return
  const overrides = readLocalOverrides()
  delete overrides[key]
  localStorage.setItem(STORAGE_KEY, JSON.stringify(overrides))
}

/** Clear all localStorage flag overrides. */
export function clearAllLocalOverrides(): void {
  if (typeof window === 'undefined') return
  localStorage.removeItem(STORAGE_KEY)
}

/** Resolve a single flag: localStorage override > env override > default. */
export function resolveFlag(key: FeatureFlagKey): boolean {
  const localOverride = readLocalOverrides()[key]
  if (localOverride !== undefined) return localOverride
  return readEnvFlag(`NEXT_PUBLIC_${key}`) ?? DEFAULT_FLAGS[key]
}

/** Resolved feature flags (env overrides > defaults). Does NOT include localStorage overrides. */
export const flags: FeatureFlags = {
  ENABLE_WEBGPU: readEnvFlag('NEXT_PUBLIC_ENABLE_WEBGPU') ?? DEFAULT_FLAGS.ENABLE_WEBGPU,
  ENABLE_COLLABORATION: readEnvFlag('NEXT_PUBLIC_ENABLE_COLLABORATION') ?? DEFAULT_FLAGS.ENABLE_COLLABORATION,
  ENABLE_AI: readEnvFlag('NEXT_PUBLIC_ENABLE_AI') ?? DEFAULT_FLAGS.ENABLE_AI,
  ENABLE_GAUSSIAN_SPLATTING: readEnvFlag('NEXT_PUBLIC_ENABLE_GAUSSIAN_SPLATTING') ?? DEFAULT_FLAGS.ENABLE_GAUSSIAN_SPLATTING,
  ENABLE_WASM: readEnvFlag('NEXT_PUBLIC_ENABLE_WASM') ?? DEFAULT_FLAGS.ENABLE_WASM,
  ENABLE_ECS: readEnvFlag('NEXT_PUBLIC_ENABLE_ECS') ?? DEFAULT_FLAGS.ENABLE_ECS,
  ENABLE_EVENT_SOURCING: readEnvFlag('NEXT_PUBLIC_ENABLE_EVENT_SOURCING') ?? DEFAULT_FLAGS.ENABLE_EVENT_SOURCING,
}
