/**
 * Scene serializer: converts floor plan state to a natural language description
 * for NVIDIA Cosmos text-to-world generation.
 *
 * Produces structured scene descriptions from venue state data.
 */

import type { ProjectedVenueState, ProjectedItem } from '../../projector'
import type { SceneDescription } from '../types'

// ─── Furniture Type Labels ───────────────────────────────────────────────────

const FURNITURE_LABELS: Record<string, string> = {
  chair: 'chair',
  'round-table': 'round table',
  'rect-table': 'rectangular table',
  'trestle-table': 'trestle table',
  podium: 'podium',
  stage: 'stage',
  bar: 'bar counter',
}

// ─── Serializer ──────────────────────────────────────────────────────────────

/** Count items by furniture type. */
function countByType(items: Map<string, ProjectedItem>): Map<string, number> {
  const counts = new Map<string, number>()
  for (const [, item] of items) {
    const current = counts.get(item.furnitureType) ?? 0
    counts.set(item.furnitureType, current + 1)
  }
  return counts
}

/** Estimate seating capacity from furniture counts. */
function estimateCapacity(counts: Map<string, number>): number {
  let seats = 0
  seats += (counts.get('chair') ?? 0)
  seats += (counts.get('round-table') ?? 0) * 8
  seats += (counts.get('rect-table') ?? 0) * 6
  seats += (counts.get('trestle-table') ?? 0) * 10
  return seats
}

/** Format furniture counts into a summary sentence. */
function formatFurnitureSummary(counts: Map<string, number>): string {
  const parts: string[] = []

  for (const [type, count] of counts) {
    const label = FURNITURE_LABELS[type] ?? type
    const plural = count > 1 ? 's' : ''
    parts.push(`${count} ${label}${plural}`)
  }

  if (parts.length === 0) return 'an empty venue'
  if (parts.length === 1) return parts[0]!
  const last = parts.pop()
  return `${parts.join(', ')}, and ${last}`
}

/** Detect layout style from item arrangement. */
function detectLayoutStyle(items: Map<string, ProjectedItem>): string {
  const counts = countByType(items)
  const chairs = counts.get('chair') ?? 0
  const roundTables = counts.get('round-table') ?? 0
  const stages = counts.get('stage') ?? 0
  const podiums = counts.get('podium') ?? 0

  if (chairs > 50 && roundTables === 0 && (stages > 0 || podiums > 0)) {
    return 'theater-style'
  }
  if (roundTables > 5) {
    return 'banquet-style'
  }
  if (chairs > 20 && roundTables === 0) {
    return 'classroom or conference-style'
  }
  return 'mixed layout'
}

/**
 * Serialize a venue state into a natural language scene description
 * suitable for NVIDIA Cosmos text-to-world generation.
 */
export function serializeScene(
  state: ProjectedVenueState,
  dimensions: { width: number; depth: number; height: number },
  style?: string,
): SceneDescription {
  const counts = countByType(state.items)
  const capacity = estimateCapacity(counts)
  const layoutStyle = detectLayoutStyle(state.items)
  const furnitureSummary = formatFurnitureSummary(counts)

  const description = [
    `A ${dimensions.width}m × ${dimensions.depth}m ${state.name || 'venue hall'}`,
    `with ${dimensions.height}m ceilings.`,
    `${layoutStyle.charAt(0).toUpperCase() + layoutStyle.slice(1)} arrangement with ${furnitureSummary}.`,
    capacity > 0 ? `Seating capacity: approximately ${capacity} guests.` : '',
    style ?? 'Warm pendant lighting. Polished hardwood floors.',
  ].filter(Boolean).join(' ')

  return {
    venueId: state.venueId,
    description,
    dimensions,
    furnitureSummary,
    style,
  }
}

export { countByType, estimateCapacity, formatFurnitureSummary, detectLayoutStyle }
