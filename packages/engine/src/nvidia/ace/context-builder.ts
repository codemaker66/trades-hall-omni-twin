/**
 * ACE Digital Concierge: RAG context builder.
 *
 * Assembles venue data into a structured context for the LLM-powered
 * venue concierge, including floor plan summaries, pricing, and FAQ.
 */

import type { ProjectedVenueState } from '../../projector'
import type { ConciergeContext } from '../types'
import { countByType, estimateCapacity, formatFurnitureSummary, detectLayoutStyle } from '../cosmos/scene-serializer'

// ─── Context Building ────────────────────────────────────────────────────────

/**
 * Build RAG context for the venue concierge from venue state and metadata.
 */
export function buildConciergeContext(
  state: ProjectedVenueState,
  metadata: {
    venueName: string
    venueDescription?: string
    amenities?: string[]
    pricing?: string
    availability?: string[]
    faq?: Array<{ question: string; answer: string }>
  },
): ConciergeContext {
  const counts = countByType(state.items)
  const capacity = estimateCapacity(counts)
  const layoutStyle = detectLayoutStyle(state.items)
  const summary = formatFurnitureSummary(counts)

  const floorPlanSummary = [
    `Current layout: ${layoutStyle}.`,
    `Furniture: ${summary}.`,
    capacity > 0 ? `Estimated seating capacity: ${capacity} guests.` : 'No seating configured.',
    `Total items: ${state.items.size}.`,
  ].join(' ')

  return {
    venueId: state.venueId,
    venueName: metadata.venueName,
    venueDescription: metadata.venueDescription ?? `${metadata.venueName} venue`,
    capacity,
    amenities: metadata.amenities ?? [],
    floorPlanSummary,
    pricing: metadata.pricing,
    availability: metadata.availability,
    faq: metadata.faq ?? [],
  }
}

// ─── Prompt Template ─────────────────────────────────────────────────────────

/**
 * Generate the system prompt for the venue concierge persona.
 */
export function buildSystemPrompt(context: ConciergeContext): string {
  const lines: string[] = [
    `You are a friendly and knowledgeable concierge for "${context.venueName}".`,
    `You help potential clients learn about the venue and plan their events.`,
    '',
    `## Venue Details`,
    context.venueDescription,
    `Capacity: ${context.capacity} guests.`,
  ]

  if (context.amenities.length > 0) {
    lines.push(`Amenities: ${context.amenities.join(', ')}.`)
  }

  if (context.pricing) {
    lines.push(`Pricing: ${context.pricing}`)
  }

  lines.push('', `## Current Floor Plan`, context.floorPlanSummary)

  if (context.availability && context.availability.length > 0) {
    lines.push('', `## Availability`, `Available dates: ${context.availability.join(', ')}`)
  }

  if (context.faq.length > 0) {
    lines.push('', `## FAQ`)
    for (const qa of context.faq) {
      lines.push(`Q: ${qa.question}`, `A: ${qa.answer}`, '')
    }
  }

  lines.push(
    '',
    `## Instructions`,
    `- Be helpful, concise, and professional.`,
    `- If asked about something outside your knowledge, say so honestly.`,
    `- Suggest booking a tour or contacting the events team for complex queries.`,
    `- You can describe floor plan layouts and suggest configurations.`,
  )

  return lines.join('\n')
}
