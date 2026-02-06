/**
 * Pre-built floor plan templates.
 * Each template defines a set of furniture items for a standard layout.
 * Positions are in feet relative to a room origin (0,0).
 */
import type { FurnitureCategory } from './store'

export interface TemplateItem {
  name: string
  category: FurnitureCategory
  x: number
  y: number
  widthFt: number
  depthFt: number
  rotation: number
}

export interface FloorPlanTemplate {
  id: string
  name: string
  description: string
  items: TemplateItem[]
}

// ─── Helper: generate rows of chairs ────────────────────────────────────────

function chairRows(
  startX: number,
  startY: number,
  rows: number,
  seatsPerRow: number,
  spacingX: number,
  spacingY: number,
): TemplateItem[] {
  const items: TemplateItem[] = []
  for (let row = 0; row < rows; row++) {
    for (let seat = 0; seat < seatsPerRow; seat++) {
      items.push({
        name: 'Chair',
        category: 'chair',
        x: startX + seat * spacingX,
        y: startY + row * spacingY,
        widthFt: 1.5,
        depthFt: 1.5,
        rotation: 0,
      })
    }
  }
  return items
}

// ─── Helper: round table with chairs ────────────────────────────────────────

function roundTableWithChairs(cx: number, cy: number, seats: number): TemplateItem[] {
  const items: TemplateItem[] = [
    { name: 'Round Table 6ft', category: 'table', x: cx, y: cy, widthFt: 6, depthFt: 6, rotation: 0 },
  ]
  const radius = 5
  for (let i = 0; i < seats; i++) {
    const angle = (i / seats) * Math.PI * 2 - Math.PI / 2
    items.push({
      name: 'Chair',
      category: 'chair',
      x: cx + Math.cos(angle) * radius,
      y: cy + Math.sin(angle) * radius,
      widthFt: 1.5,
      depthFt: 1.5,
      rotation: (angle * 180) / Math.PI + 90,
    })
  }
  return items
}

// ─── Templates ──────────────────────────────────────────────────────────────

export const templates: FloorPlanTemplate[] = [
  {
    id: 'theater',
    name: 'Theater',
    description: 'Rows of chairs facing a stage. 120 seats.',
    items: [
      { name: 'Stage 8x4', category: 'stage', x: 40, y: 8, widthFt: 16, depthFt: 8, rotation: 0 },
      { name: 'Podium', category: 'decor', x: 40, y: 14, widthFt: 2, depthFt: 2, rotation: 0 },
      ...chairRows(16, 20, 10, 12, 2.2, 3),
    ],
  },
  {
    id: 'banquet',
    name: 'Banquet Rounds',
    description: 'Round tables with 8 chairs each. 80 seats.',
    items: [
      ...roundTableWithChairs(15, 15, 8),
      ...roundTableWithChairs(35, 15, 8),
      ...roundTableWithChairs(55, 15, 8),
      ...roundTableWithChairs(15, 32, 8),
      ...roundTableWithChairs(35, 32, 8),
      ...roundTableWithChairs(55, 32, 8),
      ...roundTableWithChairs(15, 49, 8),
      ...roundTableWithChairs(35, 49, 8),
      ...roundTableWithChairs(55, 49, 8),
      ...roundTableWithChairs(55, 49, 8), // 10th table
    ],
  },
  {
    id: 'classroom',
    name: 'Classroom',
    description: 'Rectangular tables with chairs facing front. 40 seats.',
    items: [
      { name: 'Stage 8x4', category: 'stage', x: 40, y: 6, widthFt: 12, depthFt: 4, rotation: 0 },
      // 5 rows of 4 tables with 2 chairs each
      ...[0, 1, 2, 3, 4].flatMap((row) =>
        [0, 1, 2, 3].flatMap((col) => [
          { name: 'Rect Table 8x3', category: 'table' as const, x: 12 + col * 16, y: 16 + row * 7, widthFt: 8, depthFt: 3, rotation: 0 },
          { name: 'Chair', category: 'chair' as const, x: 9 + col * 16, y: 16 + row * 7, widthFt: 1.5, depthFt: 1.5, rotation: 0 },
          { name: 'Chair', category: 'chair' as const, x: 15 + col * 16, y: 16 + row * 7, widthFt: 1.5, depthFt: 1.5, rotation: 0 },
        ]),
      ),
    ],
  },
  {
    id: 'cocktail',
    name: 'Cocktail',
    description: 'Standing cocktail tables with a bar. 60 capacity.',
    items: [
      { name: 'Bar Unit 6ft', category: 'equipment', x: 40, y: 6, widthFt: 6, depthFt: 2, rotation: 0 },
      { name: 'DJ Booth', category: 'equipment', x: 40, y: 45, widthFt: 6, depthFt: 3, rotation: 0 },
      // Scattered cocktail tables
      ...[
        [15, 15], [30, 12], [50, 15], [65, 12],
        [15, 28], [35, 25], [55, 28], [65, 25],
        [15, 38], [30, 35], [50, 38], [65, 35],
      ].map(([x, y]) => ({
        name: 'Cocktail Table',
        category: 'table' as const,
        x: x!,
        y: y!,
        widthFt: 2,
        depthFt: 2,
        rotation: 0,
      })),
    ],
  },
  {
    id: 'u-shape',
    name: 'U-Shape',
    description: 'U-shaped table arrangement for meetings. 24 seats.',
    items: [
      // Bottom row (8 tables)
      ...[0, 1, 2, 3, 4, 5, 6, 7].map((i) => ({
        name: 'Rect Table 8x3',
        category: 'table' as const,
        x: 12 + i * 8,
        y: 35,
        widthFt: 8,
        depthFt: 3,
        rotation: 0,
      })),
      // Left column
      ...[0, 1, 2].map((i) => ({
        name: 'Rect Table 8x3',
        category: 'table' as const,
        x: 12,
        y: 14 + i * 7,
        widthFt: 3,
        depthFt: 8,
        rotation: 90,
      })),
      // Right column
      ...[0, 1, 2].map((i) => ({
        name: 'Rect Table 8x3',
        category: 'table' as const,
        x: 68,
        y: 14 + i * 7,
        widthFt: 3,
        depthFt: 8,
        rotation: 90,
      })),
      // Chairs around outside
      ...chairRows(12, 40, 1, 8, 8, 0),
      ...chairRows(6, 14, 3, 1, 0, 7),
      ...chairRows(74, 14, 3, 1, 0, 7),
    ],
  },
  {
    id: 'ceremony',
    name: 'Ceremony',
    description: 'Two sections of chairs with center aisle. 100 seats.',
    items: [
      { name: 'Stage 8x4', category: 'stage', x: 40, y: 6, widthFt: 8, depthFt: 4, rotation: 0 },
      // Left section (5 rows x 10 seats)
      ...chairRows(8, 16, 5, 10, 2.2, 3),
      // Right section (5 rows x 10 seats)
      ...chairRows(44, 16, 5, 10, 2.2, 3),
    ],
  },
  {
    id: 'boardroom',
    name: 'Boardroom',
    description: 'Large table with chairs around. 16 seats.',
    items: [
      // Long central table (3 tables end-to-end)
      { name: 'Rect Table 8x3', category: 'table', x: 32, y: 25, widthFt: 8, depthFt: 3, rotation: 0 },
      { name: 'Rect Table 8x3', category: 'table', x: 40, y: 25, widthFt: 8, depthFt: 3, rotation: 0 },
      { name: 'Rect Table 8x3', category: 'table', x: 48, y: 25, widthFt: 8, depthFt: 3, rotation: 0 },
      // Chairs on each side (7 + 7 + 1 each end)
      ...chairRows(30, 21, 1, 7, 3.5, 0),
      ...chairRows(30, 29, 1, 7, 3.5, 0),
      { name: 'Chair', category: 'chair', x: 26, y: 25, widthFt: 1.5, depthFt: 1.5, rotation: 90 },
      { name: 'Chair', category: 'chair', x: 54, y: 25, widthFt: 1.5, depthFt: 1.5, rotation: -90 },
    ],
  },
]
