/**
 * CT-7: Layout Optimization Adjunction
 *
 * F: Constraints → Layouts  (free functor: generate optimal layout from constraints)
 * G: Layouts → Constraints  (forgetful functor: extract constraints a layout satisfies)
 *
 * The adjunction F ⊣ G means:
 *   F(C) is the BEST layout satisfying constraints C.
 *
 *   For any constraints C and layout L,
 *   Hom(F(C), L) ≅ Hom(C, G(L))
 *   "A morphism from the optimal layout to L"
 *   ≡ "A morphism from the constraints to the constraints L satisfies"
 */

import { createAdjunction } from './adjunction'
import type { Adjunction } from './adjunction'

// ─── Constraint and Layout Types ────────────────────────────────────────────

/** A set of layout constraints. */
export interface LayoutConstraints {
  readonly maxItems: number
  readonly minSpacing: number     // meters
  readonly roomWidth: number      // meters
  readonly roomDepth: number      // meters
  readonly exitClearance: number  // meters
  readonly aisleWidth: number     // meters
}

/** A layout satisfying some constraints. */
export interface OptimizedLayout {
  readonly items: readonly LayoutItem[]
  readonly score: number          // 0-1, quality of the layout
  readonly satisfiedConstraints: LayoutConstraints
}

export interface LayoutItem {
  readonly id: string
  readonly x: number
  readonly z: number
  readonly width: number
  readonly depth: number
  readonly rotation: number
}

// ─── Free Functor: Constraints → Layout ─────────────────────────────────────

/**
 * F: Generate an optimal layout from constraints.
 *
 * This is the "free" construction — given constraints, produce
 * the best layout that satisfies them.
 *
 * Uses a simplified greedy grid placement for demonstration.
 * (In production, this delegates to the constraint solver.)
 */
function generateLayout(constraints: LayoutConstraints): OptimizedLayout {
  const items: LayoutItem[] = []
  const itemWidth = 0.5
  const itemDepth = 0.5
  const spacing = constraints.minSpacing + itemWidth

  // Reserve exit clearance from edges
  const startX = constraints.exitClearance + itemWidth / 2
  const startZ = constraints.exitClearance + itemDepth / 2
  const endX = constraints.roomWidth - constraints.exitClearance - itemWidth / 2
  const endZ = constraints.roomDepth - constraints.exitClearance - itemDepth / 2

  // Grid placement with aisle
  const aisleX = constraints.roomWidth / 2
  let count = 0

  for (let z = startZ; z <= endZ && count < constraints.maxItems; z += spacing) {
    for (let x = startX; x <= endX && count < constraints.maxItems; x += spacing) {
      // Skip aisle zone
      if (Math.abs(x - aisleX) < constraints.aisleWidth / 2) continue

      items.push({
        id: `item-${count}`,
        x,
        z,
        width: itemWidth,
        depth: itemDepth,
        rotation: 0,
      })
      count++
    }
  }

  return {
    items,
    score: items.length / constraints.maxItems,
    satisfiedConstraints: constraints,
  }
}

// ─── Forgetful Functor: Layout → Constraints ────────────────────────────────

/**
 * G: Extract the constraints that a layout satisfies.
 *
 * This is the "forgetful" functor — given a layout, determine
 * what constraints it actually satisfies.
 */
function extractConstraints(layout: OptimizedLayout): LayoutConstraints {
  if (layout.items.length === 0) {
    return layout.satisfiedConstraints
  }

  // Determine actual spacing
  let minDist = Infinity
  for (let i = 0; i < layout.items.length; i++) {
    for (let j = i + 1; j < layout.items.length; j++) {
      const a = layout.items[i]!
      const b = layout.items[j]!
      const dx = a.x - b.x
      const dz = a.z - b.z
      const dist = Math.sqrt(dx * dx + dz * dz) - (a.width + b.width) / 2
      if (dist < minDist) minDist = dist
    }
  }

  // Determine bounding box
  let minX = Infinity, maxX = -Infinity
  let minZ = Infinity, maxZ = -Infinity
  for (const item of layout.items) {
    minX = Math.min(minX, item.x - item.width / 2)
    maxX = Math.max(maxX, item.x + item.width / 2)
    minZ = Math.min(minZ, item.z - item.depth / 2)
    maxZ = Math.max(maxZ, item.z + item.depth / 2)
  }

  const exitClearance = Math.min(
    minX,
    minZ,
    layout.satisfiedConstraints.roomWidth - maxX,
    layout.satisfiedConstraints.roomDepth - maxZ,
  )

  return {
    maxItems: layout.items.length,
    minSpacing: minDist === Infinity ? 0 : Math.max(0, minDist),
    roomWidth: layout.satisfiedConstraints.roomWidth,
    roomDepth: layout.satisfiedConstraints.roomDepth,
    exitClearance: Math.max(0, exitClearance),
    aisleWidth: layout.satisfiedConstraints.aisleWidth,
  }
}

// ─── Layout Adjunction ──────────────────────────────────────────────────────

/**
 * The Layout Optimization Adjunction: F ⊣ G
 *
 * F: LayoutConstraints → OptimizedLayout  (generate optimal layout)
 * G: OptimizedLayout → LayoutConstraints  (extract satisfied constraints)
 *
 * Universal property: F(C) is the best layout satisfying C.
 * For any layout L satisfying C, there is a unique morphism F(C) → L.
 */
export const layoutAdjunction: Adjunction<LayoutConstraints, OptimizedLayout> =
  createAdjunction(
    'Layout ⊣ Constraint',
    generateLayout,
    extractConstraints,
  )

/**
 * Use the adjunction to find the optimal layout for given constraints.
 */
export function optimizeLayout(constraints: LayoutConstraints): OptimizedLayout {
  return layoutAdjunction.leftAdjoint(constraints)
}

/**
 * Use the adjunction to determine what constraints a layout satisfies.
 */
export function satisfiedConstraints(layout: OptimizedLayout): LayoutConstraints {
  return layoutAdjunction.rightAdjoint(layout)
}

/**
 * The unit: embed constraints into the round-trip.
 * η(C) = G(F(C)) — generate a layout, then extract its constraints.
 * The result should be "at least as permissive" as C.
 */
export function constraintRoundTrip(constraints: LayoutConstraints): LayoutConstraints {
  return layoutAdjunction.unit(constraints)
}
