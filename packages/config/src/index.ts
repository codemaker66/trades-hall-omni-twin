// Shared configuration: feature flags, environment validation, spring presets.
// Three.js-specific config (scene.ts, theme.ts) stays in apps/web/src/config/
// until shared across multiple apps.

export {
  flags,
  resolveFlag,
  setLocalOverride,
  clearLocalOverride,
  clearAllLocalOverrides,
  readLocalOverrides,
  DEFAULT_FLAGS,
  FEATURE_FLAG_KEYS,
  type FeatureFlags,
  type FeatureFlagKey,
} from './flags'
