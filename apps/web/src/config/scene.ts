/**
 * OmniTwin 3D Scene Constants
 *
 * Single source of truth for all 3D scene values: camera, lighting, physics,
 * furniture dimensions, snap/grid, and selection.
 * No magic numbers should exist in component files — import from here.
 */

import * as THREE from 'three'

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

export const CAMERA = {
  /** Initial camera position [x, y, z] */
  defaultPosition: [0, 5, 8] as [number, number, number],
  /** Field of view in degrees */
  fov: 60,
  /** Device pixel ratio range [min, max] */
  dpr: [1, 2] as [number, number],

  /** RTS orbit controls */
  rts: {
    /** Default yaw angle in degrees */
    defaultYaw: 40,
    /** Default distance from target */
    defaultDistance: 18,
    /** Minimum zoom distance */
    minDistance: 6,
    /** Maximum zoom distance */
    maxDistance: 32,
    /** Minimum pitch angle in degrees */
    minPitch: 32,
    /** Maximum pitch angle in degrees */
    maxPitch: 65,
    /** Edge-pan margin as fraction of viewport (0.02 = 2%) */
    edgeMarginFraction: 0.03,
    /** Camera padding from hall boundaries */
    boundaryPadding: 0.6,
    /** Default pan speed */
    panSpeed: 6,
    /** Default zoom speed */
    zoomSpeed: 10,
  },

  /** RTS damping — controls how "weighty" camera movement feels */
  damping: {
    /** Pan acceleration (higher = snappier) */
    panAccel: 20,
    /** Pan friction (higher = stops faster) */
    panFriction: 16,
    /** Minimum pan scale at closest zoom */
    panZoomScaleMin: 0.7,
    /** Maximum pan scale at farthest zoom */
    panZoomScaleMax: 1.6,
    /** Zoom friction (higher = stops faster) */
    zoomFriction: 14,
    /** Rotation damping */
    rotateDamping: 12,
    /** Mouse rotation drag speed */
    rotateDragSpeed: 0.005,
    /** Pitch damping (vertical smoothing) */
    pitchDamping: 8,
    /** Focus-on-target damping */
    focusDamping: 6,
  },

  /** Zoom input */
  zoom: {
    /** Wheel sensitivity multiplier */
    wheelSensitivity: 0.0015,
    /** Maximum zoom velocity */
    maxVelocity: 12,
  },

  /** MapControls fallback (RTSCameraControlsView) */
  mapControls: {
    minPolarAngle: 0,
    maxPolarAngle: Math.PI / 2 - 0.05,
    minDistance: 1,
    maxDistance: 60,
    dampingFactor: 0.1,
    wasdSpeed: 10,
  },

  /** Frame timing */
  maxDeltaTime: 0.1,
  /** Focus arrival threshold (distance squared) */
  focusArrivalThreshold: 0.0004,

  /** Disabled mouse button sentinel for MapControls */
  DISABLED_MOUSE: -1 as unknown as THREE.MOUSE,
} as const

// ---------------------------------------------------------------------------
// Lighting
// ---------------------------------------------------------------------------

export const LIGHTING = {
  ambient: {
    color: '#ffeebb',
    intensity: 0.5,
  },
  spot: {
    position: [10, 10, 10] as [number, number, number],
    angle: 0.5,
    penumbra: 1,
    intensity: 400,
    shadowBias: -0.0001,
  },
  point: {
    position: [-10, 5, -10] as [number, number, number],
    intensity: 200,
    color: '#aaccff',
  },
  contactShadows: {
    resolution: 1024,
    scale: 50,
    blur: 2,
    opacity: 0.5,
    far: 10,
  },
  fog: {
    color: '#1a1005',
    near: 30,
    far: 120,
  },
  environment: {
    preset: 'city' as const,
    blur: 0.8,
  },
} as const

// ---------------------------------------------------------------------------
// Hall Dimensions
// ---------------------------------------------------------------------------

export const HALL = {
  /** Hall width (X axis, meters) */
  width: 21,
  /** Hall depth (Z axis, meters) */
  depth: 10,
  /** Hall height (Y axis, meters) */
  height: 6,
  /** Wainscot (lower wall panel) height */
  wainscotHeight: 2,
  /** Dome radius on ceiling */
  domeRadius: 4,
  /** Dome geometry segments [width, height] */
  domeSegments: [32, 16] as [number, number],
  /** Ring torus around dome [major, minor, radial, tubular] */
  ringTorus: [4, 0.2, 16, 64] as [number, number, number, number],
} as const

// ---------------------------------------------------------------------------
// Furniture Dimensions (meters)
// ---------------------------------------------------------------------------

export const FURNITURE = {
  'round-table': {
    /** Top: radius, height, radial segments */
    top: { radius: 0.9, height: 0.05, segments: 32 },
    /** Pedestal leg: top radius, bottom radius, height, segments */
    leg: { radiusTop: 0.1, radiusBottom: 0.4, height: 0.74, segments: 16 },
    /** Y positions */
    topY: 0.74,
    legY: 0.37,
    /** Selection highlight bounds */
    selectionCenter: [0, 0.37, 0] as [number, number, number],
    selectionSize: [1.8, 0.74, 1.8] as [number, number, number],
  },
  'trestle-table': {
    /** Top: width, height, depth */
    top: { width: 1.8, height: 0.05, depth: 0.76 },
    /** Leg: radius, height */
    leg: { radius: 0.05, height: 0.74 },
    /** Full dimensions for snapping calculations */
    length: 1.8,
    depth: 0.76,
    topY: 0.74,
    legY: 0.37,
    /** X offset for leg placement */
    legOffsetX: 0.8,
    selectionCenter: [0, 0.3825, 0] as [number, number, number],
    selectionSize: [1.8, 0.765, 0.76] as [number, number, number],
  },
  chair: {
    seat: { width: 0.45, height: 0.05, depth: 0.45 },
    back: { width: 0.45, height: 0.5, depth: 0.05 },
    leg: { radius: 0.03, height: 0.45 },
    seatY: 0.45,
    backY: 0.7,
    backZ: -0.2,
    legY: 0.225,
    /** Leg positions: [x, z] for each corner */
    legPositions: [
      [0.2, 0.2],
      [-0.2, 0.2],
      [0.2, -0.2],
      [-0.2, -0.2],
    ] as [number, number][],
    selectionCenter: [0, 0.475, 0] as [number, number, number],
    selectionSize: [0.45, 0.95, 0.45] as [number, number, number],
  },
  platform: {
    width: 2,
    height: 0.2,
    depth: 1,
    meshY: 0.1,
    selectionCenter: [0, 0.1, 0] as [number, number, number],
    selectionSize: [2, 0.2, 1] as [number, number, number],
  },
} as const

/** Selection highlight scaling factors */
export const SELECTION = {
  /** Box geometry scale relative to furniture bounds (3% larger) */
  boxScale: 1.03,
  /** Edges geometry scale (5% larger) */
  edgesScale: 1.05,
  /** Gizmo size multiplier */
  gizmoSize: 1.2,
  /** Minimum drag distance to count as area selection (meters) */
  minDragThreshold: 0.1,
  /** Floor selection plane Y offset */
  floorPlaneY: 0.02,
  /** Floor selection outline Y offset */
  floorOutlineY: 0.021,
} as const

/** Material properties */
export const MATERIALS = {
  whiteSurface: { color: '#ffffff', roughness: 0.5 },
  darkLeg: { color: '#333333' },
  chairDarkLeg: { color: '#222222' },
  platform: { color: '#111111', roughness: 0.9 },
  chairFabric: { roughness: 0.8, texturePath: './textures/red_fabric_texture.png' },
  dome: { roughness: 0.3, metalness: 0.8, envMapIntensity: 2 },
  glass: { roughness: 0.05, metalness: 0.9, transmission: 0.95, thickness: 0.02 },
} as const

// ---------------------------------------------------------------------------
// Physics & Raycasting
// ---------------------------------------------------------------------------

export const PHYSICS = {
  /** Y origin for downward raycasts (high above scene) */
  stackRayOriginY: 50,
  /** Epsilon for AABB collision overlap detection */
  collisionEpsilon: 0.001,
  /** Movement threshold below which drag updates are skipped */
  movementThreshold: 0.001,
  /** Epsilon for floating-point coordinate comparison */
  coordinateEpsilon: 0.001,
  /** Wall visibility boundary threshold */
  wallBoundaryThreshold: 1,
} as const

// ---------------------------------------------------------------------------
// Snap & Grid
// ---------------------------------------------------------------------------

export const SNAP = {
  /** Default grid size (meters) */
  defaultGrid: 0.5,
  /** Minimum configurable grid size */
  minGrid: 0.1,
  /** Minimum spacing between chairs (prevents overlap) */
  chairMinSpacing: 0.6,
  /** Default chair rotation when placed (-90 degrees, facing forward) */
  defaultChairRotation: -Math.PI / 2,
  /** Default chair count for table setup */
  defaultChairCount: 10,
  /** Rotation step for Q/E keys (90 degrees) */
  rotationStep: Math.PI / 2,
  /** Gizmo rotation snap (45 degrees) */
  gizmoRotationSnap: Math.PI / 4,
  /** Chair placement radius around round table */
  chairTableRadius: 1.3,

  /** Trestle end-to-end snap thresholds */
  trestle: {
    /** Max error along the long axis to trigger snap */
    alongTolerance: 0.25,
    /** Max error across the short axis to trigger snap */
    acrossTolerance: 0.12,
    /** Minimum cos(angle) between tables for snap (cos(6°) ≈ 0.9945) */
    angleCosThreshold: Math.cos(THREE.MathUtils.degToRad(6)),
  },
} as const

// ---------------------------------------------------------------------------
// Ghost Preview
// ---------------------------------------------------------------------------

export const GHOST = {
  /** Ghost material opacity */
  opacity: 0.35,
  /** Ghost emissive intensity */
  emissiveIntensity: 0.6,
  /** Preview ring inner/outer radius */
  ringInner: 0.5,
  ringOuter: 0.6,
  /** Preview ring segments */
  ringSegments: 32,
} as const

// ---------------------------------------------------------------------------
// Wood Texture Generation (Hall.tsx)
// ---------------------------------------------------------------------------

export const WOOD_TEXTURE = {
  /** Canvas resolution (square) */
  resolution: 1024,
  /** Number of planks across texture */
  plankCount: 16,
  /** Number of grain streaks per plank */
  grainPerPlank: 80,
  /** Maximum grain streak length (pixels) */
  maxGrainLength: 300,
  /** Grain height range [min, max] pixels */
  grainHeight: [1, 2] as [number, number],
  /** Grain shadow alpha range [min, max] */
  grainAlpha: [0.1, 0.2] as [number, number],
  /** Plank gap height (pixels) */
  gapHeight: 4,
  /** HSL hue range for wood color [base, variation] */
  hue: [32, 6] as [number, number],
  /** HSL saturation range [base, variation] */
  saturation: [35, 15] as [number, number],
  /** HSL lightness range [base, variation] */
  lightness: [18, 12] as [number, number],
  /** Roughness value range (0-255 scale) [base, variation] */
  roughnessRange: [180, 50] as [number, number],
  /** Texture repeat factor on floor mesh */
  repeat: 4,
  /** Normal map scale (bump depth) */
  normalScale: 0.5,
  /** Environment map intensity for wood sheen */
  envMapIntensity: 0.6,
} as const
