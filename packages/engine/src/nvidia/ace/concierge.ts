/**
 * ACE Digital Concierge: text chat session manager.
 *
 * Phase A: Text-only chat using context builder + system prompt.
 * Phase B/C (future): Video chat via NVIDIA ACE/Tokkio.
 */

import type { ConciergeContext, ConciergeSession, ConciergeMessage, ConciergeResponse } from '../types'
import { buildSystemPrompt } from './context-builder'

// ─── Chat Interface ──────────────────────────────────────────────────────────

export interface ConciergeChatClient {
  /** Create a new chat session. */
  createSession(venueId: string, context: ConciergeContext): ConciergeSession
  /** Send a user message and get a response. */
  sendMessage(session: ConciergeSession, message: string): Promise<{ session: ConciergeSession; response: ConciergeResponse }>
  /** End the session. */
  endSession(session: ConciergeSession): ConciergeSession
}

// ─── Mock Chat Client ────────────────────────────────────────────────────────

/**
 * Mock concierge that generates contextual responses without an LLM.
 * Uses keyword matching against the venue context for demo purposes.
 */
export class MockConciergeClient implements ConciergeChatClient {
  private sessionCounter = 0

  createSession(venueId: string, _context: ConciergeContext): ConciergeSession {
    return {
      sessionId: `concierge-${++this.sessionCounter}`,
      venueId,
      messages: [],
      mode: 'text',
      status: 'active',
    }
  }

  async sendMessage(
    session: ConciergeSession,
    message: string,
  ): Promise<{ session: ConciergeSession; response: ConciergeResponse }> {
    const userMsg: ConciergeMessage = {
      role: 'user',
      content: message,
      timestamp: new Date().toISOString(),
    }

    const responseText = this.generateMockResponse(message)
    const assistantMsg: ConciergeMessage = {
      role: 'assistant',
      content: responseText,
      timestamp: new Date().toISOString(),
    }

    const updatedSession: ConciergeSession = {
      ...session,
      messages: [...session.messages, userMsg, assistantMsg],
    }

    return {
      session: updatedSession,
      response: {
        message: responseText,
        suggestedActions: this.suggestActions(message),
      },
    }
  }

  endSession(session: ConciergeSession): ConciergeSession {
    return { ...session, status: 'ended' }
  }

  private generateMockResponse(message: string): string {
    const lower = message.toLowerCase()

    if (lower.includes('capacity') || lower.includes('how many')) {
      return 'Our venue can accommodate various configurations. In theater-style, we seat up to 300 guests. For banquet events, we typically host 200 guests comfortably. Would you like me to describe the different layout options?'
    }

    if (lower.includes('price') || lower.includes('cost') || lower.includes('rate')) {
      return 'Our pricing varies by event type and day of the week. I\'d recommend speaking with our events team for a customized quote. Shall I connect you with them?'
    }

    if (lower.includes('available') || lower.includes('book') || lower.includes('date')) {
      return 'I can help check availability! Please provide your preferred dates and I\'ll look into it for you.'
    }

    if (lower.includes('layout') || lower.includes('floor plan') || lower.includes('setup')) {
      return 'We offer several layout configurations including theater-style, banquet, classroom, cocktail, and U-shape arrangements. Each is optimized for different event types. Would you like details on a specific layout?'
    }

    if (lower.includes('amenity') || lower.includes('feature') || lower.includes('include')) {
      return 'Our venue features include professional lighting, a sound system, built-in bar area, staging, and ample parking. We also offer catering coordination and event planning assistance.'
    }

    return 'Thank you for your interest in our venue! I\'m here to help with any questions about capacity, layouts, pricing, or availability. What would you like to know?'
  }

  private suggestActions(message: string): string[] {
    const lower = message.toLowerCase()
    const actions: string[] = []

    if (!lower.includes('layout')) actions.push('View floor plan layouts')
    if (!lower.includes('price')) actions.push('Get a price quote')
    if (!lower.includes('avail')) actions.push('Check availability')
    if (!lower.includes('tour')) actions.push('Schedule a venue tour')

    return actions.slice(0, 3)
  }
}

/** Create a concierge client. */
export function createConciergeClient(): ConciergeChatClient {
  return new MockConciergeClient()
}

export { buildSystemPrompt }
