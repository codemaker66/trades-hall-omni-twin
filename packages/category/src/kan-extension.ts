/**
 * CT-5: Kan Extensions
 *
 * A Kan extension formalizes the "best approximation" problem.
 * Given partial information about what a client wants, the left Kan extension
 * computes the best possible match — the mathematically optimal way to do
 * recommendation when you have incomplete data.
 *
 * Left Kan Extension (Lan_F G):
 *   - "Best approximation from below" (optimistic match)
 *   - Uses the coend formula: Lan_F(G)(c) = ∫^a Hom(F(a), c) ⊗ G(a)
 *
 * Right Kan Extension (Ran_F G):
 *   - "Best approximation from above" (conservative match)
 *   - Uses the end formula: Ran_F(G)(c) = ∫_a [Hom(c, F(a)), G(a)]
 *
 * The universal property: the Kan extension is OPTIMAL —
 * no other approximation does better.
 */

import type { Morphism } from './core'

// ─── Kan Extension Interface ────────────────────────────────────────────────

/**
 * A Kan Extension for matching/recommendation problems.
 *
 * @typeParam Pref  - Preference type (partial information)
 * @typeParam Item  - Item type (what we're searching over)
 * @typeParam Score - Scoring type (typically number)
 */
export interface KanExtension<Pref, Item, Score> {
  /**
   * Left Kan extension: optimistic match.
   * For missing preferences, assume the best.
   */
  leftKan(
    preferences: Partial<Pref>,
    items: readonly Item[],
  ): RankedResult<Item, Score>[]

  /**
   * Right Kan extension: conservative match.
   * For missing preferences, assume the worst.
   */
  rightKan(
    preferences: Partial<Pref>,
    items: readonly Item[],
  ): RankedResult<Item, Score>[]
}

export interface RankedResult<Item, Score> {
  readonly item: Item
  readonly score: Score
  readonly factors: readonly ScoringFactor[]
}

export interface ScoringFactor {
  readonly name: string
  readonly weight: number
  readonly score: number
  readonly specified: boolean  // true if this preference was specified
}

// ─── Kan Extension Builder ──────────────────────────────────────────────────

/**
 * Configuration for building a Kan extension.
 */
export interface KanConfig<Pref, Item> {
  /** Individual scoring dimensions. */
  dimensions: readonly KanDimension<Pref, Item>[]

  /** Default weight for unspecified dimensions. */
  defaultWeight: number
}

/**
 * A single scoring dimension for the Kan extension.
 */
export interface KanDimension<Pref, Item> {
  readonly name: string
  readonly weight: number

  /** Extract the preference value (returns undefined if not specified). */
  extractPref(pref: Partial<Pref>): unknown | undefined

  /**
   * Score how well an item matches a preference value.
   * Returns a number in [0, 1].
   */
  score(prefValue: unknown, item: Item): number

  /**
   * Optimistic score when preference is unspecified (for left Kan).
   * Default: 1.0 (assume perfect match).
   */
  optimistic?: number

  /**
   * Conservative score when preference is unspecified (for right Kan).
   * Default: 0.0 (assume no match).
   */
  conservative?: number
}

/**
 * Build a Kan extension from a configuration.
 */
export function buildKanExtension<Pref, Item>(
  config: KanConfig<Pref, Item>,
): KanExtension<Pref, Item, number> {
  return {
    leftKan(preferences, items) {
      return computeKan(config, preferences, items, 'left')
    },
    rightKan(preferences, items) {
      return computeKan(config, preferences, items, 'right')
    },
  }
}

function computeKan<Pref, Item>(
  config: KanConfig<Pref, Item>,
  preferences: Partial<Pref>,
  items: readonly Item[],
  mode: 'left' | 'right',
): RankedResult<Item, number>[] {
  const results: RankedResult<Item, number>[] = items.map(item => {
    const factors: ScoringFactor[] = []
    let totalWeight = 0
    let weightedScore = 0

    for (const dim of config.dimensions) {
      const prefValue = dim.extractPref(preferences)
      const specified = prefValue !== undefined

      let dimScore: number
      if (specified) {
        dimScore = dim.score(prefValue, item)
      } else if (mode === 'left') {
        dimScore = dim.optimistic ?? 1.0
      } else {
        dimScore = dim.conservative ?? 0.0
      }

      factors.push({
        name: dim.name,
        weight: dim.weight,
        score: dimScore,
        specified,
      })

      totalWeight += dim.weight
      weightedScore += dim.weight * dimScore
    }

    const finalScore = totalWeight > 0 ? weightedScore / totalWeight : 0

    return { item, score: finalScore, factors }
  })

  // Sort by score descending
  return results.sort((a, b) => b.score - a.score)
}

// ─── Coend Formula (Theoretical) ────────────────────────────────────────────

/**
 * The coend formula for left Kan extensions:
 *   Lan_F(G)(c) = ∫^a Hom(F(a), c) ⊗ G(a)
 *
 * In concrete terms for venue matching:
 *   For each venue a, compute:
 *     compatibility(preferences, venue_a) × quality(venue_a)
 *   Take the weighted colimit (weighted sum with compatibility as weights)
 *
 * @param homWeights - Hom(F(a), c): compatibility of each item with preferences
 * @param gValues - G(a): quality/intrinsic value of each item
 * @returns The weighted colimit score
 */
export function coendFormula(
  homWeights: readonly number[],
  gValues: readonly number[],
): number {
  if (homWeights.length !== gValues.length || homWeights.length === 0) return 0

  let weightedSum = 0
  let totalWeight = 0

  for (let i = 0; i < homWeights.length; i++) {
    weightedSum += homWeights[i]! * gValues[i]!
    totalWeight += homWeights[i]!
  }

  return totalWeight > 0 ? weightedSum / totalWeight : 0
}

/**
 * The end formula for right Kan extensions:
 *   Ran_F(G)(c) = ∫_a [Hom(c, F(a)), G(a)]
 *
 * Conservative: uses the minimum over compatibility-weighted qualities.
 */
export function endFormula(
  homWeights: readonly number[],
  gValues: readonly number[],
): number {
  if (homWeights.length !== gValues.length || homWeights.length === 0) return 0

  let minWeighted = Infinity

  for (let i = 0; i < homWeights.length; i++) {
    const weight = homWeights[i]!
    if (weight > 0) {
      const weighted = gValues[i]! / weight
      if (weighted < minWeighted) minWeighted = weighted
    }
  }

  return minWeighted === Infinity ? 0 : Math.min(1, minWeighted)
}
