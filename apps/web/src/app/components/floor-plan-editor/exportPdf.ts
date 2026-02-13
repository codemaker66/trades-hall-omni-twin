/**
 * PDF export for floor plan layouts.
 *
 * Uses jspdf + the existing Konva stage to generate a professional PDF with:
 *   - Floor plan image (from Konva stage at 2x resolution)
 *   - Layout metrics table (chairs, tables, stages, total seats)
 *   - Item breakdown legend
 *   - Plan dimensions footer
 */
import { jsPDF } from 'jspdf'
import { getStageInstance } from './Canvas2D'
import { useFloorPlanStore, type FloorPlanItem } from './store'

// ── Metrics generation ───────────────────────────────────────────────────────

interface PdfMetrics {
  chairs: number
  tables: number
  stages: number
  totalSeats: number
  totalItems: number
  planWidthFt: number
  planHeightFt: number
  itemBreakdown: Array<{ name: string; count: number }>
}

function gatherMetrics(items: FloorPlanItem[], planWidthFt: number, planHeightFt: number): PdfMetrics {
  let chairs = 0
  let tables = 0
  let stages = 0

  const counts: Record<string, number> = {}

  for (const item of items) {
    counts[item.name] = (counts[item.name] ?? 0) + 1

    if (item.category === 'chair') chairs++
    else if (item.category === 'table') tables++
    else if (item.category === 'stage') stages++
  }

  const tableSeats = items
    .filter((i) => i.category === 'table')
    .reduce((sum, t) => sum + (t.widthFt >= 5 ? 8 : 6), 0)

  const itemBreakdown = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => ({ name, count }))

  return {
    chairs,
    tables,
    stages,
    totalSeats: chairs + tableSeats,
    totalItems: items.length,
    planWidthFt,
    planHeightFt,
    itemBreakdown,
  }
}

// ── PDF generation ───────────────────────────────────────────────────────────

export async function exportFloorPlanPdf(): Promise<void> {
  const stage = getStageInstance()
  if (!stage) {
    throw new Error('Canvas stage not available. Switch to 2D view first.')
  }

  const state = useFloorPlanStore.getState()
  const metrics = gatherMetrics(state.items, state.planWidthFt, state.planHeightFt)

  // Capture stage as PNG at 2x resolution
  const imageDataUrl = stage.toDataURL({ pixelRatio: 2 })

  // Create A4 landscape PDF (297mm x 210mm)
  const pdf = new jsPDF({ orientation: 'landscape', unit: 'mm', format: 'a4' })
  const pageWidth = pdf.internal.pageSize.getWidth()
  const pageHeight = pdf.internal.pageSize.getHeight()
  const margin = 15

  // ── Header ─────────────────────────────────────────────────────────────
  pdf.setFontSize(16)
  pdf.setFont('helvetica', 'bold')
  pdf.text('Floor Plan Layout', margin, margin + 5)

  pdf.setFontSize(9)
  pdf.setFont('helvetica', 'normal')
  pdf.setTextColor(120, 120, 120)
  const dateStr = new Date().toLocaleDateString('en-AU', {
    year: 'numeric', month: 'long', day: 'numeric',
  })
  pdf.text(dateStr, pageWidth - margin, margin + 5, { align: 'right' })
  pdf.setTextColor(0, 0, 0)

  // ── Floor plan image ───────────────────────────────────────────────────
  const imageTop = margin + 12
  const availableWidth = pageWidth - margin * 2
  const availableHeight = pageHeight - imageTop - 55 // Reserve space for metrics

  // Maintain aspect ratio
  const stageWidth = stage.width()
  const stageHeight = stage.height()
  const aspectRatio = stageWidth / stageHeight
  let imgWidth = availableWidth
  let imgHeight = imgWidth / aspectRatio

  if (imgHeight > availableHeight) {
    imgHeight = availableHeight
    imgWidth = imgHeight * aspectRatio
  }

  const imgX = margin + (availableWidth - imgWidth) / 2
  pdf.addImage(imageDataUrl, 'PNG', imgX, imageTop, imgWidth, imgHeight)

  // Light border around image
  pdf.setDrawColor(200, 200, 200)
  pdf.setLineWidth(0.3)
  pdf.rect(imgX, imageTop, imgWidth, imgHeight)

  // ── Metrics table ──────────────────────────────────────────────────────
  const tableTop = imageTop + imgHeight + 8
  const colWidth = 40
  const rowHeight = 7

  // Header row
  pdf.setFillColor(245, 245, 245)
  pdf.rect(margin, tableTop, colWidth * 4, rowHeight, 'F')
  pdf.setFontSize(8)
  pdf.setFont('helvetica', 'bold')

  const headers = ['Chairs', 'Tables', 'Stages', 'Total Seats']
  const values = [metrics.chairs, metrics.tables, metrics.stages, metrics.totalSeats]

  for (let i = 0; i < headers.length; i++) {
    const x = margin + colWidth * i + colWidth / 2
    pdf.text(headers[i]!, x, tableTop + 5, { align: 'center' })
  }

  // Values row
  pdf.setFont('helvetica', 'normal')
  pdf.setFontSize(10)
  for (let i = 0; i < values.length; i++) {
    const x = margin + colWidth * i + colWidth / 2
    pdf.text(String(values[i]), x, tableTop + rowHeight + 5, { align: 'center' })
  }

  // ── Item breakdown (right side) ────────────────────────────────────────
  if (metrics.itemBreakdown.length > 0) {
    const breakdownX = margin + colWidth * 4 + 15
    let breakdownY = tableTop + 2

    pdf.setFontSize(8)
    pdf.setFont('helvetica', 'bold')
    pdf.text('Item Breakdown', breakdownX, breakdownY)
    breakdownY += 5

    pdf.setFont('helvetica', 'normal')
    pdf.setFontSize(7)
    const maxItems = Math.min(metrics.itemBreakdown.length, 8)
    for (let i = 0; i < maxItems; i++) {
      const entry = metrics.itemBreakdown[i]!
      pdf.text(`${entry.name}: ${entry.count}`, breakdownX, breakdownY)
      breakdownY += 3.5
    }
    if (metrics.itemBreakdown.length > maxItems) {
      pdf.text(`... and ${metrics.itemBreakdown.length - maxItems} more`, breakdownX, breakdownY)
    }
  }

  // ── Footer ─────────────────────────────────────────────────────────────
  pdf.setFontSize(7)
  pdf.setTextColor(150, 150, 150)
  pdf.text(
    `Plan dimensions: ${metrics.planWidthFt}ft × ${metrics.planHeightFt}ft  |  ${metrics.totalItems} items  |  Generated by OmniTwin`,
    pageWidth / 2,
    pageHeight - 8,
    { align: 'center' },
  )

  // ── Download ───────────────────────────────────────────────────────────
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
  pdf.save(`floor-plan-${timestamp}.pdf`)
}
