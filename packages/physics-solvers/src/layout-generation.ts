/**
 * PS-11: Layout Generation — Templates + LLM + Diffusion Stubs
 *
 * Template-based layout generation (fully functional) for common venue styles.
 * LLM and diffusion generation provide API stubs with template fallbacks.
 *
 * Layout standards:
 * - Theater:    6-8 sq ft/person, 36" between rows, 24" chair width
 * - Banquet:    10-12 sq ft/person, 60" round tables seat 8-10, 60" between tables
 * - Classroom:  18-20 sq ft/person
 * - Cocktail:   6 sq ft/person
 * - Main aisles: >= 54" (4.5ft)
 * - Service aisles: >= 60" (5ft)
 *
 * References:
 * - Feng et al. (2023). "LayoutGPT." NeurIPS. arXiv:2305.15393
 * - Tang et al. (2024). "DiffuScene." CVPR
 * - Shabani et al. (2023). "HouseDiffusion." CVPR. arXiv:2211.13287
 */

import type { FurnitureItem, RoomBoundary, Layout, PRNG } from './types.js'
import { ItemType, createPRNG } from './types.js'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type LayoutStyle = 'theater' | 'banquet' | 'classroom' | 'cocktail' | 'ceremony' | 'conference'

export interface LLMLayoutOptions {
  description: string
  room: RoomBoundary
  apiEndpoint?: string
}

export interface DiffusionLayoutOptions {
  room: RoomBoundary
  itemSpecs: Array<{ type: ItemType; count: number }>
  style: LayoutStyle
  nLayouts: number
  apiEndpoint?: string
}

// ---------------------------------------------------------------------------
// Template-Based Layout Generation
// ---------------------------------------------------------------------------

export function generateTemplateLayout(
  room: RoomBoundary,
  style: LayoutStyle,
  targetCapacity: number,
  seed?: number,
): FurnitureItem[] {
  const rng = createPRNG(seed ?? 42)
  const w = room.width
  const h = room.height

  switch (style) {
    case 'theater': return generateTheater(w, h, targetCapacity, rng)
    case 'banquet': return generateBanquet(w, h, targetCapacity, rng)
    case 'classroom': return generateClassroom(w, h, targetCapacity, rng)
    case 'cocktail': return generateCocktail(w, h, targetCapacity, rng)
    case 'ceremony': return generateCeremony(w, h, targetCapacity, rng)
    case 'conference': return generateConference(w, h, targetCapacity, rng)
  }
}

/** Theater: rows of chairs facing front, center aisle, cross aisles */
function generateTheater(w: number, h: number, capacity: number, _rng: PRNG): FurnitureItem[] {
  const items: FurnitureItem[] = []
  const chairWidth = 2.0   // 24"
  const chairDepth = 2.0
  const rowSpacing = 3.0   // 36"
  const aisleWidth = 4.5   // 54"
  const stageDepth = 8.0

  // Stage at front
  items.push({
    x: w / 2, y: stageDepth / 2 + 1,
    width: Math.min(w * 0.6, 20), depth: stageDepth,
    rotation: 0, itemType: ItemType.Stage, seats: 0,
  })

  // Seating area
  const seatingStartY = stageDepth + 4
  const usableWidth = w - 4 // 2ft margin each side
  const halfWidth = (usableWidth - aisleWidth) / 2
  const chairsPerHalf = Math.floor(halfWidth / chairWidth)
  const chairsPerRow = chairsPerHalf * 2
  const nRows = Math.min(
    Math.ceil(capacity / chairsPerRow),
    Math.floor((h - seatingStartY - 2) / rowSpacing),
  )

  let seated = 0
  for (let row = 0; row < nRows && seated < capacity; row++) {
    const y = seatingStartY + row * rowSpacing

    // Left half
    for (let c = 0; c < chairsPerHalf && seated < capacity; c++) {
      items.push({
        x: 2 + c * chairWidth + chairWidth / 2,
        y,
        width: chairWidth, depth: chairDepth, rotation: 0,
        itemType: ItemType.Chair, seats: 1,
      })
      seated++
    }

    // Right half (after center aisle)
    const rightStart = 2 + chairsPerHalf * chairWidth + aisleWidth
    for (let c = 0; c < chairsPerHalf && seated < capacity; c++) {
      items.push({
        x: rightStart + c * chairWidth + chairWidth / 2,
        y,
        width: chairWidth, depth: chairDepth, rotation: 0,
        itemType: ItemType.Chair, seats: 1,
      })
      seated++
    }
  }

  return items
}

/** Banquet: round tables with chairs, service aisles, optional dance floor */
function generateBanquet(w: number, h: number, capacity: number, _rng: PRNG): FurnitureItem[] {
  const items: FurnitureItem[] = []
  const tableDiameter = 5.0  // 60" round tables
  const seatsPerTable = 8
  const tableSpacing = 10.0  // 60" aisles between tables
  const margin = 4.0

  // Dance floor if capacity > 50
  let danceFloorEndY = margin
  if (capacity > 50) {
    const dfSize = Math.min(w * 0.3, 15)
    items.push({
      x: w / 2, y: margin + dfSize / 2,
      width: dfSize, depth: dfSize,
      rotation: 0, itemType: ItemType.DanceFloor, seats: 0,
    })
    danceFloorEndY = margin + dfSize + 3
  }

  // Bar in corner
  items.push({
    x: w - margin - 3, y: margin + 2,
    width: 6, depth: 3, rotation: 0,
    itemType: ItemType.Bar, seats: 0,
  })

  // Tables in grid
  const nTables = Math.ceil(capacity / seatsPerTable)
  const usableW = w - 2 * margin
  const usableH = h - danceFloorEndY - margin
  const colsMax = Math.floor((usableW + tableSpacing) / (tableDiameter + tableSpacing))
  const cols = Math.min(colsMax, nTables)
  const rows = Math.ceil(nTables / cols)

  let placed = 0
  for (let r = 0; r < rows && placed < nTables; r++) {
    for (let c = 0; c < cols && placed < nTables; c++) {
      const x = margin + (tableDiameter / 2) + c * (tableDiameter + tableSpacing)
      const y = danceFloorEndY + (tableDiameter / 2) + r * (tableDiameter + tableSpacing)
      if (y + tableDiameter / 2 > h - margin) continue

      items.push({
        x, y,
        width: tableDiameter, depth: tableDiameter,
        rotation: 0, itemType: ItemType.RoundTable,
        seats: Math.min(seatsPerTable, capacity - placed * seatsPerTable),
      })
      placed++
    }
  }

  return items
}

/** Classroom: rectangular tables with chairs facing front */
function generateClassroom(w: number, h: number, capacity: number, _rng: PRNG): FurnitureItem[] {
  const items: FurnitureItem[] = []
  const tableWidth = 6.0
  const tableDepth = 2.5
  const chairsPerTable = 2
  const rowSpacing = 5.0
  const colSpacing = 2.0
  const aisleWidth = 4.5
  const margin = 3.0
  const stageDepth = 6.0

  // Podium at front
  items.push({
    x: w / 2, y: margin + 2,
    width: 3, depth: 2,
    rotation: 0, itemType: ItemType.Podium, seats: 0,
  })

  const seatingStartY = margin + stageDepth + 2
  const usableWidth = w - 2 * margin
  const halfWidth = (usableWidth - aisleWidth) / 2
  const tablesPerHalf = Math.max(1, Math.floor(halfWidth / (tableWidth + colSpacing)))

  const nTables = Math.ceil(capacity / chairsPerTable)
  const tablesPerRow = tablesPerHalf * 2
  const nRows = Math.ceil(nTables / tablesPerRow)

  let placed = 0
  for (let row = 0; row < nRows && placed < nTables; row++) {
    const y = seatingStartY + row * rowSpacing

    // Left side tables
    for (let c = 0; c < tablesPerHalf && placed < nTables; c++) {
      const x = margin + c * (tableWidth + colSpacing) + tableWidth / 2
      items.push({
        x, y,
        width: tableWidth, depth: tableDepth,
        rotation: 0, itemType: ItemType.RectTable,
        seats: chairsPerTable,
      })
      placed++
    }

    // Right side tables
    const rightStart = margin + halfWidth + aisleWidth
    for (let c = 0; c < tablesPerHalf && placed < nTables; c++) {
      const x = rightStart + c * (tableWidth + colSpacing) + tableWidth / 2
      items.push({
        x, y,
        width: tableWidth, depth: tableDepth,
        rotation: 0, itemType: ItemType.RectTable,
        seats: chairsPerTable,
      })
      placed++
    }
  }

  return items
}

/** Cocktail: high-tops scattered, bars along walls, open center */
function generateCocktail(w: number, h: number, capacity: number, _rng: PRNG): FurnitureItem[] {
  const items: FurnitureItem[] = []
  const margin = 3.0

  // Bars along two walls
  items.push({
    x: margin + 3, y: h / 2,
    width: 2, depth: 8,
    rotation: 0, itemType: ItemType.Bar, seats: 0,
  })
  items.push({
    x: w - margin - 3, y: h / 2,
    width: 2, depth: 8,
    rotation: 0, itemType: ItemType.Bar, seats: 0,
  })

  // Service station
  items.push({
    x: w / 2, y: h - margin - 2,
    width: 4, depth: 3,
    rotation: 0, itemType: ItemType.ServiceStation, seats: 0,
  })

  // High-top tables scattered in ring around open center
  const nTables = Math.ceil(capacity / 4)
  const centerX = w / 2
  const centerY = h / 2
  const radiusX = (w - 2 * margin - 8) / 2.5
  const radiusY = (h - 2 * margin - 4) / 2.5

  for (let i = 0; i < nTables; i++) {
    const angle = (2 * Math.PI * i) / nTables
    const x = centerX + radiusX * Math.cos(angle)
    const y = centerY + radiusY * Math.sin(angle)
    items.push({
      x, y,
      width: 2.5, depth: 2.5,
      rotation: 0, itemType: ItemType.RoundTable,
      seats: 4,
    })
  }

  return items
}

/** Ceremony: two sections with center aisle, altar/stage at front */
function generateCeremony(w: number, h: number, capacity: number, _rng: PRNG): FurnitureItem[] {
  const items: FurnitureItem[] = []
  const chairWidth = 2.0
  const rowSpacing = 3.0
  const aisleWidth = 5.0
  const margin = 3.0

  // Stage/altar
  items.push({
    x: w / 2, y: margin + 3,
    width: 8, depth: 6,
    rotation: 0, itemType: ItemType.Stage, seats: 0,
  })

  const seatingStart = margin + 10
  const usableWidth = w - 2 * margin
  const halfWidth = (usableWidth - aisleWidth) / 2
  const chairsPerHalf = Math.floor(halfWidth / chairWidth)
  const chairsPerRow = chairsPerHalf * 2
  const nRows = Math.ceil(capacity / chairsPerRow)

  let seated = 0
  for (let row = 0; row < nRows && seated < capacity; row++) {
    const y = seatingStart + row * rowSpacing

    // Left section
    for (let c = 0; c < chairsPerHalf && seated < capacity; c++) {
      items.push({
        x: margin + c * chairWidth + chairWidth / 2, y,
        width: chairWidth, depth: 1.5, rotation: 0,
        itemType: ItemType.Chair, seats: 1,
      })
      seated++
    }

    // Right section
    const rightStart = margin + halfWidth + aisleWidth
    for (let c = 0; c < chairsPerHalf && seated < capacity; c++) {
      items.push({
        x: rightStart + c * chairWidth + chairWidth / 2, y,
        width: chairWidth, depth: 1.5, rotation: 0,
        itemType: ItemType.Chair, seats: 1,
      })
      seated++
    }
  }

  return items
}

/** Conference: U-shaped or rectangular table arrangement */
function generateConference(w: number, h: number, capacity: number, _rng: PRNG): FurnitureItem[] {
  const items: FurnitureItem[] = []
  const tableWidth = 6.0
  const tableDepth = 2.5
  const margin = 4.0

  // Podium at head
  items.push({
    x: w / 2, y: margin + 1.5,
    width: 3, depth: 2,
    rotation: 0, itemType: ItemType.Podium, seats: 0,
  })

  // AV booth at back
  items.push({
    x: w / 2, y: h - margin - 1.5,
    width: 4, depth: 3,
    rotation: 0, itemType: ItemType.AVBooth, seats: 0,
  })

  if (capacity <= 16) {
    // Single rectangular table
    const longTableW = Math.min(w - 2 * margin, capacity * 2.5)
    items.push({
      x: w / 2, y: h / 2,
      width: longTableW, depth: 4,
      rotation: 0, itemType: ItemType.RectTable,
      seats: capacity,
    })
  } else {
    // U-shape: head table + two side tables
    const headLength = Math.min(w - 2 * margin - 4, 20)
    const sideLength = Math.min(h - 2 * margin - 8, 20)

    // Head table
    items.push({
      x: w / 2, y: margin + 5,
      width: headLength, depth: tableDepth,
      rotation: 0, itemType: ItemType.RectTable,
      seats: Math.floor(headLength / 2.5),
    })

    // Left side table
    items.push({
      x: margin + tableDepth / 2, y: margin + 5 + sideLength / 2,
      width: tableDepth, depth: sideLength,
      rotation: 0, itemType: ItemType.RectTable,
      seats: Math.floor(sideLength / 2.5),
    })

    // Right side table
    items.push({
      x: w - margin - tableDepth / 2, y: margin + 5 + sideLength / 2,
      width: tableDepth, depth: sideLength,
      rotation: 0, itemType: ItemType.RectTable,
      seats: Math.floor(sideLength / 2.5),
    })
  }

  return items
}

// ---------------------------------------------------------------------------
// LLM-Based Generation (stub with keyword parsing fallback)
// ---------------------------------------------------------------------------

/**
 * Generate layout via LLM if API available, else parse keywords and
 * delegate to template generator.
 */
export async function generateLayoutLLM(options: LLMLayoutOptions): Promise<FurnitureItem[]> {
  if (options.apiEndpoint) {
    try {
      const res = await fetch(options.apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: buildLayoutGPTPrompt(options.description, options.room),
          room: { width: options.room.width, height: options.room.height },
        }),
      })
      if (res.ok) {
        const data = (await res.json()) as { items: FurnitureItem[] }
        return data.items
      }
    } catch {
      // Fall through to keyword parsing
    }
  }

  // Parse description for style and capacity
  const { style, capacity } = parseDescription(options.description)
  return generateTemplateLayout(options.room, style, capacity)
}

/**
 * Build a LayoutGPT-style CSS spatial prompt.
 */
function buildLayoutGPTPrompt(description: string, room: RoomBoundary): string {
  return [
    `Room dimensions: ${room.width}ft x ${room.height}ft`,
    `Event description: ${description}`,
    'Generate furniture layout using CSS-style positioning:',
    'object { width: ?ft; height: ?ft; left: ?ft; top: ?ft; type: ?; }',
    'Types: chair, round-table, rect-table, stage, bar, podium, dance-floor',
    'Ensure: no overlaps, 36" min aisle width, ADA compliance',
  ].join('\n')
}

function parseDescription(desc: string): { style: LayoutStyle; capacity: number } {
  const lower = desc.toLowerCase()

  // Parse capacity
  const capMatch = lower.match(/(\d+)\s*(?:people|person|guests?|attendees?|seats?)/)
  const capacity = capMatch ? parseInt(capMatch[1]!, 10) : 100

  // Parse style
  let style: LayoutStyle = 'banquet'
  if (lower.includes('theater') || lower.includes('theatre') || lower.includes('presentation')) {
    style = 'theater'
  } else if (lower.includes('classroom') || lower.includes('training') || lower.includes('workshop')) {
    style = 'classroom'
  } else if (lower.includes('cocktail') || lower.includes('standing') || lower.includes('reception')) {
    style = 'cocktail'
  } else if (lower.includes('ceremony') || lower.includes('wedding') || lower.includes('chapel')) {
    style = 'ceremony'
  } else if (lower.includes('conference') || lower.includes('boardroom') || lower.includes('meeting')) {
    style = 'conference'
  }

  return { style, capacity }
}

// ---------------------------------------------------------------------------
// Diffusion-Based Generation (stub with perturbation fallback)
// ---------------------------------------------------------------------------

/**
 * Generate multiple layouts via diffusion model if API available,
 * else generate template + random perturbations.
 */
export async function generateLayoutDiffusion(options: DiffusionLayoutOptions): Promise<FurnitureItem[][]> {
  if (options.apiEndpoint) {
    try {
      const res = await fetch(options.apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          room: { width: options.room.width, height: options.room.height },
          items: options.itemSpecs,
          style: options.style,
          nLayouts: options.nLayouts,
        }),
      })
      if (res.ok) {
        const data = (await res.json()) as { layouts: FurnitureItem[][] }
        return data.layouts
      }
    } catch {
      // Fall through
    }
  }

  // Generate base template + perturbations
  const totalItems = options.itemSpecs.reduce((s, spec) => s + spec.count, 0)
  const base = generateTemplateLayout(options.room, options.style, totalItems)
  const layouts: FurnitureItem[][] = [base]

  for (let i = 1; i < options.nLayouts; i++) {
    const rng = createPRNG(42 + i)
    layouts.push(perturbLayout(base, options.room, 1.0 + i * 0.5, rng))
  }

  return layouts
}

// ---------------------------------------------------------------------------
// Layout Perturbation
// ---------------------------------------------------------------------------

/**
 * Randomly perturb a layout for diversity.
 * Shifts items by up to `magnitude` feet, rotates by up to magnitude*10 degrees.
 */
export function perturbLayout(
  items: FurnitureItem[],
  room: RoomBoundary,
  magnitude: number,
  rng: PRNG,
): FurnitureItem[] {
  return items.map((item) => {
    let x = item.x + (rng.random() - 0.5) * 2 * magnitude
    let y = item.y + (rng.random() - 0.5) * 2 * magnitude
    const rotation = item.rotation + (rng.random() - 0.5) * 2 * magnitude * (Math.PI / 18)

    // Clamp to room boundaries
    x = Math.max(item.width / 2 + 1, Math.min(room.width - item.width / 2 - 1, x))
    y = Math.max(item.depth / 2 + 1, Math.min(room.height - item.depth / 2 - 1, y))

    return { ...item, x, y, rotation }
  })
}
