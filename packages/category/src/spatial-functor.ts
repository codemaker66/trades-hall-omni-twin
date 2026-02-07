/**
 * CT-2: Spatial Functor
 *
 * Maps between 2D floor plan representations and 3D scene representations:
 *   F: FloorPlan2D → Scene3D
 *
 * Preserves spatial relationships:
 *   F(move2D(obj, dx, dz)) = move3D(F(obj), dx, 0, dz)
 *   F(rotate2D(obj, θ)) = rotate3D(F(obj), 0, θ, 0)
 *
 * The functor law guarantees 2D↔3D sync is correct by construction.
 */

import type { Morphism } from './core'
import { compose } from './core'

// ─── 2D Floor Plan Objects ──────────────────────────────────────────────────

export interface Object2D {
  readonly id: string
  readonly type: string
  readonly x: number
  readonly z: number
  readonly rotation: number  // radians around Y axis
  readonly width: number
  readonly depth: number
}

export interface FloorPlanScene {
  readonly objects: readonly Object2D[]
  readonly bounds: { readonly width: number; readonly depth: number }
}

/** A 2D operation: a morphism in the 2D category. */
export type Operation2D =
  | { readonly kind: 'move'; readonly dx: number; readonly dz: number }
  | { readonly kind: 'rotate'; readonly dTheta: number }
  | { readonly kind: 'scale'; readonly sx: number; readonly sz: number }

// ─── 3D Scene Objects ───────────────────────────────────────────────────────

export interface Object3D {
  readonly id: string
  readonly type: string
  readonly x: number
  readonly y: number  // height (the added dimension)
  readonly z: number
  readonly rotationX: number
  readonly rotationY: number
  readonly rotationZ: number
  readonly width: number
  readonly height: number
  readonly depth: number
}

export interface Scene3D {
  readonly objects: readonly Object3D[]
  readonly bounds: { readonly width: number; readonly height: number; readonly depth: number }
}

/** A 3D operation: a morphism in the 3D category. */
export type Operation3D =
  | { readonly kind: 'move'; readonly dx: number; readonly dy: number; readonly dz: number }
  | { readonly kind: 'rotate'; readonly dx: number; readonly dy: number; readonly dz: number }
  | { readonly kind: 'scale'; readonly sx: number; readonly sy: number; readonly sz: number }

// ─── Height Mapping ─────────────────────────────────────────────────────────

/** Default heights for furniture types (meters). */
const DEFAULT_HEIGHTS: Record<string, number> = {
  'chair': 0.85,
  'round-table': 0.75,
  'rect-table': 0.75,
  'cocktail-table': 1.1,
  'podium': 1.2,
  'stage': 0.6,
  'bar': 1.1,
}

function getHeight(type: string): number {
  return DEFAULT_HEIGHTS[type] ?? 1.0
}

// ─── Spatial Functor ────────────────────────────────────────────────────────

export const SpatialFunctor = {
  /**
   * Object mapping: 2D floor plan object → 3D scene object.
   * Embeds the 2D plane at y=0, adds height based on furniture type.
   */
  mapObject(obj2D: Object2D): Object3D {
    return {
      id: obj2D.id,
      type: obj2D.type,
      x: obj2D.x,
      y: 0,
      z: obj2D.z,
      rotationX: 0,
      rotationY: obj2D.rotation,
      rotationZ: 0,
      width: obj2D.width,
      height: getHeight(obj2D.type),
      depth: obj2D.depth,
    }
  },

  /**
   * Morphism mapping: 2D operation → 3D operation.
   * Embeds 2D ops into 3D by zeroing out the Y component.
   */
  mapOperation(op2D: Operation2D): Operation3D {
    switch (op2D.kind) {
      case 'move':
        return { kind: 'move', dx: op2D.dx, dy: 0, dz: op2D.dz }
      case 'rotate':
        return { kind: 'rotate', dx: 0, dy: op2D.dTheta, dz: 0 }
      case 'scale':
        return { kind: 'scale', sx: op2D.sx, sy: 1, sz: op2D.sz }
    }
  },

  /**
   * Scene mapping: entire 2D floor plan → 3D scene.
   */
  mapScene(scene2D: FloorPlanScene, wallHeight = 3.0): Scene3D {
    return {
      objects: scene2D.objects.map(SpatialFunctor.mapObject),
      bounds: {
        width: scene2D.bounds.width,
        height: wallHeight,
        depth: scene2D.bounds.depth,
      },
    }
  },

  /**
   * Apply a 2D operation to a 2D object (morphism in source category).
   */
  apply2D(obj: Object2D, op: Operation2D): Object2D {
    switch (op.kind) {
      case 'move':
        return { ...obj, x: obj.x + op.dx, z: obj.z + op.dz }
      case 'rotate':
        return { ...obj, rotation: obj.rotation + op.dTheta }
      case 'scale':
        return { ...obj, width: obj.width * op.sx, depth: obj.depth * op.sz }
    }
  },

  /**
   * Apply a 3D operation to a 3D object (morphism in target category).
   */
  apply3D(obj: Object3D, op: Operation3D): Object3D {
    switch (op.kind) {
      case 'move':
        return { ...obj, x: obj.x + op.dx, y: obj.y + op.dy, z: obj.z + op.dz }
      case 'rotate':
        return {
          ...obj,
          rotationX: obj.rotationX + op.dx,
          rotationY: obj.rotationY + op.dy,
          rotationZ: obj.rotationZ + op.dz,
        }
      case 'scale':
        return {
          ...obj,
          width: obj.width * op.sx,
          height: obj.height * op.sy,
          depth: obj.depth * op.sz,
        }
    }
  },
} as const

/**
 * Verify the functor law for spatial operations:
 *   F(op2 ∘ op1)(F(obj)) ≡ F(op2)(F(op1)(F(obj)))
 */
export function verifySpatialFunctorLaw(
  obj2D: Object2D,
  op1: Operation2D,
  op2: Operation2D,
): boolean {
  // Path 1: apply ops in 2D, then map to 3D
  const after2D = SpatialFunctor.apply2D(SpatialFunctor.apply2D(obj2D, op1), op2)
  const path1 = SpatialFunctor.mapObject(after2D)

  // Path 2: map to 3D, then apply mapped ops
  const mapped3D = SpatialFunctor.mapObject(obj2D)
  const mappedOp1 = SpatialFunctor.mapOperation(op1)
  const mappedOp2 = SpatialFunctor.mapOperation(op2)
  const path2 = SpatialFunctor.apply3D(SpatialFunctor.apply3D(mapped3D, mappedOp1), mappedOp2)

  // Compare (use approximate equality for floating point)
  return approxEqual3D(path1, path2)
}

function approxEqual3D(a: Object3D, b: Object3D, eps = 1e-10): boolean {
  return (
    a.id === b.id &&
    Math.abs(a.x - b.x) < eps &&
    Math.abs(a.y - b.y) < eps &&
    Math.abs(a.z - b.z) < eps &&
    Math.abs(a.rotationY - b.rotationY) < eps &&
    Math.abs(a.width - b.width) < eps &&
    Math.abs(a.depth - b.depth) < eps
  )
}
