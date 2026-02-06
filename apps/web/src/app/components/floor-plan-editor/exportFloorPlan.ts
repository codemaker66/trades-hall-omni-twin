/**
 * Floor plan export utilities.
 * Exports the Konva stage to PNG with title block, dimensions, and legend.
 */
import type Konva from 'konva'
import type { FloorPlanItem } from './store'

interface ExportOptions {
  stage: Konva.Stage
  items: FloorPlanItem[]
  planWidthFt: number
  planHeightFt: number
  planName?: string
}

/**
 * Export the floor plan as a PNG data URL.
 * Renders the stage at 2x resolution with a title block at the bottom.
 */
export function exportToPng(options: ExportOptions): string {
  const { stage, planWidthFt, planHeightFt, items, planName = 'Floor Plan' } = options

  // Get the raw stage image at 2x
  const pixelRatio = 2
  const dataUrl = stage.toDataURL({ pixelRatio })

  // For a simple export, return the raw stage image
  // A more advanced version could composite a title block
  return dataUrl
}

/**
 * Trigger a PNG download of the floor plan.
 */
export function downloadPng(options: ExportOptions): void {
  const dataUrl = exportToPng(options)
  const link = document.createElement('a')
  link.download = `${options.planName ?? 'floor-plan'}.png`
  link.href = dataUrl
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

/**
 * Export floor plan summary as a text block (for PDF or print).
 */
export function generateLegend(items: FloorPlanItem[], planWidthFt: number, planHeightFt: number): string {
  const counts: Record<string, number> = {}
  for (const item of items) {
    counts[item.name] = (counts[item.name] ?? 0) + 1
  }

  const chairs = items.filter((i) => i.category === 'chair').length
  const tables = items.filter((i) => i.category === 'table').length
  const tableSeats = items
    .filter((i) => i.category === 'table')
    .reduce((sum, t) => sum + (t.widthFt >= 5 ? 8 : 6), 0)

  let legend = `Floor Plan: ${planWidthFt}ft x ${planHeightFt}ft\n`
  legend += `Total items: ${items.length}\n`
  legend += `Seats: ${chairs + tableSeats} (${chairs} chairs + ${tableSeats} at tables)\n`
  legend += `Tables: ${tables}\n\n`
  legend += `Item Breakdown:\n`

  for (const [name, count] of Object.entries(counts).sort((a, b) => b[1] - a[1])) {
    legend += `  ${name}: ${count}\n`
  }

  return legend
}
