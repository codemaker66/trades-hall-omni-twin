/**
 * OmniTwin Design Tokens
 *
 * Single source of truth for all visual constants.
 * Every color, spacing, shadow, and animation in the app references this file.
 * No hardcoded hex values should exist outside this file.
 */

// ---------------------------------------------------------------------------
// Color Palette
// ---------------------------------------------------------------------------

/** Dark wood surface scale — from deepest background to lightest panel */
export const surface = {
  0: '#0b0f14',   // deepest background (behind 3D canvas)
  5: '#120a07',   // input backgrounds
  10: '#1a0f0a',  // panel backgrounds
  15: '#1a110e',  // corner accents, inactive scenario
  20: '#2d1b15',  // content panels, active areas
  25: '#3e2723',  // borders, button backgrounds
  30: '#4e342e',  // hover states
  40: '#5d4037',  // subtle dividers
  50: '#6d4c41',  // mid-range (unused but reserved for future)
  60: '#795548',  // inactive text
  70: '#8d6e63',  // active borders
  80: '#a1887f',  // secondary text
  90: '#d7ccc8',  // primary text
  95: '#efebe9',  // bright text (rare)
} as const

/** Gold/brass accent scale — the premium touch */
export const gold = {
  10: '#aa771c',  // darkest gold
  20: '#b38728',  // toolbar button border
  30: '#bf953f',  // gradient start
  40: '#d4af37',  // corner accents
  50: '#ffd700',  // primary gold — toolbar labels, axis color
  60: '#ffb74d',  // active borders, mode indicators
  70: '#ffcc80',  // rotation button text
  80: '#fcf6ba',  // lightest gold — gradient highlights
  90: '#fbf5b7',  // shimmer
} as const

/** Indigo accent — primary interactive elements */
export const indigo = {
  40: '#4f46e5',  // hover state
  50: '#6366f1',  // primary — buttons, ring, selection
  60: '#8b7bff',  // selection highlight emissive
  70: '#b7abff',  // selection outline
  80: '#7c6fff',  // ghost preview emissive
} as const

/** Danger — destructive actions, over-capacity */
export const danger = {
  10: '#3e1f1f',  // error background
  20: '#4a0a0a',  // trash gradient start
  25: '#2a0505',  // trash gradient end
  30: '#5e0d0d',  // trash hover
  40: '#8b0000',  // dark red border
  50: '#ef5350',  // primary danger — delete, over-capacity
  60: '#ff3333',  // bright hover
  70: '#ef9a9a',  // danger text
  80: '#ffcccc',  // light danger text
} as const

/** Success — safe states, confirmations */
export const success = {
  10: '#1f3324',  // success background
  50: '#66bb6a',  // primary success
  80: '#b9f6ca',  // success text
} as const

/** Info — share, links, informational */
export const info = {
  50: '#4fc3f7',  // primary info
  80: '#b3e5fc',  // info text
} as const

/** Warning — near-capacity states */
export const warning = {
  50: '#ffb74d',  // warning indicator (same as gold.60)
} as const

/** Neutrals — grays for overlays, disabled states, subtle elements */
export const neutral = {
  0: '#000000',
  10: '#1a1f26',  // modal background
  50: '#a0a0a0',  // gizmo Z-axis (polished iron)
  100: '#ffffff', // white — gizmo center, bright text
} as const

/** Scene-specific colors (fog, background) */
export const scene = {
  fog: '#1a1005',
  background: '#1a1005',
  ambientLight: '#ffeebb',
  pointLight: '#aaccff',
} as const

/** Overlay opacities — for compositing transparent layers */
export const overlay = {
  /** Glass panel background */
  panelGlass: 'rgba(13, 18, 27, 0.85)',
  /** Modal backdrop */
  backdrop: 'rgba(0, 0, 0, 0.60)',
  /** Input background on dark surface */
  inputDark: 'rgba(0, 0, 0, 0.30)',
  /** Subtle white line patterns */
  linePattern: 'rgba(255, 255, 255, 0.03)',
  /** Shadow line patterns */
  shadowPattern: 'rgba(0, 0, 0, 0.40)',
  /** Selection box fill */
  selectionFill: 'rgba(124, 111, 255, 0.12)',
} as const

// ---------------------------------------------------------------------------
// Typography
// ---------------------------------------------------------------------------

export const fontFamily = {
  sans: '"Space Grotesk", system-ui, sans-serif',
  mono: '"Space Mono", ui-monospace, monospace',
} as const

export const fontSize = {
  xs: '0.75rem',    // 12px
  sm: '0.875rem',   // 14px
  base: '1rem',     // 16px
  lg: '1.125rem',   // 18px
  xl: '1.25rem',    // 20px
  '2xl': '1.5rem',  // 24px
} as const

export const fontWeight = {
  normal: '400',
  medium: '500',
  semibold: '600',
  bold: '700',
} as const

export const letterSpacing = {
  tight: '-0.01em',
  normal: '0',
  wide: '0.05em',
  wider: '0.1em',
  widest: '0.2em',
} as const

// ---------------------------------------------------------------------------
// Spacing (used for padding, margin, gap)
// ---------------------------------------------------------------------------

export const spacing = {
  0.5: '0.125rem',  // 2px
  1: '0.25rem',     // 4px
  1.5: '0.375rem',  // 6px
  2: '0.5rem',      // 8px
  3: '0.75rem',     // 12px
  4: '1rem',        // 16px
  5: '1.25rem',     // 20px
  6: '1.5rem',      // 24px
  8: '2rem',        // 32px
  10: '2.5rem',     // 40px
  12: '3rem',       // 48px
  16: '4rem',       // 64px
} as const

// ---------------------------------------------------------------------------
// Border Radius
// ---------------------------------------------------------------------------

export const radius = {
  sm: '0.25rem',    // 4px — subtle rounding
  md: '0.5rem',     // 8px — buttons, inputs
  lg: '0.75rem',    // 12px — panels
  xl: '1rem',       // 16px — cards
  '2xl': '1.5rem',  // 24px — modals
  full: '9999px',   // pill shape
} as const

// ---------------------------------------------------------------------------
// Shadows
// ---------------------------------------------------------------------------

export const shadow = {
  sm: '0 1px 2px rgba(0, 0, 0, 0.3)',
  md: '0 4px 8px rgba(0, 0, 0, 0.4)',
  lg: '0 8px 16px rgba(0, 0, 0, 0.5)',
  xl: '0 12px 32px rgba(0, 0, 0, 0.6)',
  /** Gold glow for premium elements */
  goldGlow: '0 0 15px rgba(255, 215, 0, 0.3)',
  /** Danger glow for trash/delete areas */
  dangerGlow: '0 0 20px rgba(139, 0, 0, 0.5)',
  /** Inset shadow for carved/recessed panels */
  inset: 'inset 0 2px 20px rgba(0, 0, 0, 1)',
} as const

// ---------------------------------------------------------------------------
// Animations (Framer Motion configs)
// ---------------------------------------------------------------------------

export const animation = {
  /** Default enter animation (opacity + slide) */
  springDefault: { type: 'spring' as const, stiffness: 400, damping: 30 },
  /** Subtle spring for hover/press feedback */
  springSubtle: { type: 'spring' as const, stiffness: 300, damping: 25 },
  /** Duration-based fade */
  fadeDuration: 0.2,
  /** Scale on press (active state) */
  pressScale: 0.95,
  /** Scale on hover for interactive elements */
  hoverScale: 1.05,
  /** Tooltip show delay in ms */
  tooltipDelay: 200,
} as const

// ---------------------------------------------------------------------------
// Z-Index Scale
// ---------------------------------------------------------------------------

export const zIndex = {
  base: 0,
  dropdown: 10,
  sticky: 20,
  overlay: 30,
  modal: 40,
  tooltip: 50,
  toast: 60,
} as const
