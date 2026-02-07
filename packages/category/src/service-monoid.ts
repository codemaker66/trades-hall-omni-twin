/**
 * CT-4: Service Composition Algebra
 *
 * Models how event services compose using monoidal structure:
 *   Catering ⊗ AV ⊗ Decoration = CompleteEventPackage
 *
 * The tensor product handles:
 *   - Combining pricing (additive with bundle discounts)
 *   - Merging time requirements (union of setup windows)
 *   - Composing space requirements (non-overlapping allocation)
 *   - Joining staff requirements (with conflict detection)
 */

import type { Monoid } from './monoidal'
import type { Cents, Minutes, Service, ServiceType, ServiceRequirement } from './objects'
import { cents, minutes } from './objects'

// ─── Composed Service ───────────────────────────────────────────────────────

/** A composed service bundle — the result of tensoring services. */
export interface ComposedService {
  readonly name: string
  readonly components: readonly Service[]
  readonly totalBaseCost: Cents
  readonly totalSetupTime: Minutes
  readonly totalTeardownTime: Minutes
  readonly allRequirements: readonly ServiceRequirement[]
  readonly discountRate: number  // 0-1, applied to total
}

// ─── Service Monoid ─────────────────────────────────────────────────────────

/** Bundle discount thresholds. */
const BUNDLE_DISCOUNTS: readonly { readonly minServices: number; readonly discount: number }[] = [
  { minServices: 5, discount: 0.15 },
  { minServices: 3, discount: 0.10 },
  { minServices: 2, discount: 0.05 },
]

function bundleDiscount(serviceCount: number): number {
  for (const tier of BUNDLE_DISCOUNTS) {
    if (serviceCount >= tier.minServices) return tier.discount
  }
  return 0
}

/**
 * The empty composed service — identity for tensor.
 */
const emptyComposedService: ComposedService = {
  name: 'Empty',
  components: [],
  totalBaseCost: cents(0),
  totalSetupTime: minutes(0),
  totalTeardownTime: minutes(0),
  allRequirements: [],
  discountRate: 0,
}

/**
 * Tensor product: compose two service bundles.
 *
 * - Costs are additive
 * - Setup times take the max (parallel setup)
 * - Teardown times take the max (parallel teardown)
 * - Requirements are merged (union)
 * - Bundle discount applied based on total component count
 */
function combineComposedServices(a: ComposedService, b: ComposedService): ComposedService {
  const components = [...a.components, ...b.components]
  const totalBaseCost = cents((a.totalBaseCost as number) + (b.totalBaseCost as number))
  const totalSetupTime = minutes(Math.max(a.totalSetupTime as number, b.totalSetupTime as number))
  const totalTeardownTime = minutes(Math.max(a.totalTeardownTime as number, b.totalTeardownTime as number))
  const allRequirements = mergeRequirements(a.allRequirements, b.allRequirements)
  const discount = bundleDiscount(components.length)

  return {
    name: components.length > 0
      ? components.map(s => s.name).join(' + ')
      : 'Empty',
    components,
    totalBaseCost,
    totalSetupTime,
    totalTeardownTime,
    allRequirements,
    discountRate: discount,
  }
}

/**
 * The Service Monoid: compose services with bundle pricing.
 */
export const serviceMonoid: Monoid<ComposedService> = {
  empty: emptyComposedService,
  combine: combineComposedServices,
}

// ─── Service Lifting ────────────────────────────────────────────────────────

/**
 * Lift a single Service into the ComposedService monoid.
 */
export function liftService(service: Service): ComposedService {
  return {
    name: service.name,
    components: [service],
    totalBaseCost: service.baseCost,
    totalSetupTime: service.setupTime,
    totalTeardownTime: service.teardownTime,
    allRequirements: [...service.requirements],
    discountRate: 0,
  }
}

/**
 * Compose multiple services into a bundle using the monoid.
 */
export function bundleServices(services: readonly Service[]): ComposedService {
  return services.reduce(
    (acc, s) => serviceMonoid.combine(acc, liftService(s)),
    serviceMonoid.empty,
  )
}

/**
 * Get the final discounted cost of a composed service.
 */
export function finalCost(composed: ComposedService): Cents {
  const base = composed.totalBaseCost as number
  return cents(Math.round(base * (1 - composed.discountRate)))
}

// ─── Requirement Merging ────────────────────────────────────────────────────

/**
 * Merge service requirements, summing quantities for same kind.
 */
function mergeRequirements(
  a: readonly ServiceRequirement[],
  b: readonly ServiceRequirement[],
): ServiceRequirement[] {
  const merged = new Map<string, ServiceRequirement>()

  for (const req of [...a, ...b]) {
    const key = `${req.kind}:${req.description}`
    const existing = merged.get(key)
    if (existing) {
      merged.set(key, { ...existing, quantity: existing.quantity + req.quantity })
    } else {
      merged.set(key, { ...req })
    }
  }

  return [...merged.values()]
}

// ─── Conflict Detection ─────────────────────────────────────────────────────

export interface ServiceConflict {
  readonly service1: string
  readonly service2: string
  readonly reason: string
  readonly requirement: string
}

/**
 * Detect conflicts between services in a composed bundle.
 * Two services conflict if they require the same exclusive resource.
 */
export function detectConflicts(composed: ComposedService): readonly ServiceConflict[] {
  const conflicts: ServiceConflict[] = []
  const components = composed.components

  for (let i = 0; i < components.length; i++) {
    for (let j = i + 1; j < components.length; j++) {
      const a = components[i]!
      const b = components[j]!

      // Check for space conflicts
      const aSpace = a.requirements.filter(r => r.kind === 'space')
      const bSpace = b.requirements.filter(r => r.kind === 'space')
      for (const ra of aSpace) {
        for (const rb of bSpace) {
          if (ra.description === rb.description) {
            conflicts.push({
              service1: a.name,
              service2: b.name,
              reason: 'Competing for same space',
              requirement: ra.description,
            })
          }
        }
      }

      // Check for power conflicts (exceed capacity)
      const aPower = a.requirements.filter(r => r.kind === 'power').reduce((s, r) => s + r.quantity, 0)
      const bPower = b.requirements.filter(r => r.kind === 'power').reduce((s, r) => s + r.quantity, 0)
      if (aPower + bPower > 100) {  // 100 amps max
        conflicts.push({
          service1: a.name,
          service2: b.name,
          reason: `Combined power draw (${aPower + bPower}A) exceeds venue capacity`,
          requirement: 'power',
        })
      }
    }
  }

  return conflicts
}
