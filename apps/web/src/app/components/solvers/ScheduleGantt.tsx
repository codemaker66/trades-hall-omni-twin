'use client'

import { useMemo, useState, useCallback, type MouseEvent } from 'react'

/**
 * Schedule Gantt Chart — Visualizes event/room scheduling as a Gantt timeline.
 *
 * Features:
 * - Rows = rooms, columns = time slots
 * - Color-coded events with labels
 * - Hover tooltip with event details
 * - Time scale with grid lines
 * - Conflict highlighting (overlapping events in same room)
 */

export interface GanttEvent {
  id: string
  name: string
  roomId: string
  startTime: number
  endTime: number
  guests: number
  color?: string
}

export interface GanttRoom {
  id: string
  name: string
  capacity: number
}

export interface ScheduleGanttProps {
  events: GanttEvent[]
  rooms: GanttRoom[]
  startHour?: number
  endHour?: number
  width?: number
  rowHeight?: number
  title?: string
  onEventClick?: (eventId: string) => void
}

const EVENT_COLORS = [
  '#3b82f6', '#ef4444', '#22c55e', '#f97316', '#a855f7',
  '#06b6d4', '#ec4899', '#eab308', '#6366f1', '#14b8a6',
]

const PADDING = { top: 40, right: 16, bottom: 24, left: 120 }

export function ScheduleGantt({
  events,
  rooms,
  startHour = 8,
  endHour = 22,
  width = 900,
  rowHeight = 40,
  title = 'Schedule',
  onEventClick,
}: ScheduleGanttProps) {
  const [hoverEvent, setHoverEvent] = useState<GanttEvent | null>(null)
  const [hoverPos, setHoverPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 })

  const totalHours = endHour - startHour
  const timeW = width - PADDING.left - PADDING.right
  const height = PADDING.top + rooms.length * rowHeight + PADDING.bottom

  const xScale = useCallback(
    (hour: number) => PADDING.left + ((hour - startHour) / totalHours) * timeW,
    [startHour, totalHours, timeW],
  )

  // Detect conflicts (overlapping events in the same room)
  const conflicts = useMemo(() => {
    const set = new Set<string>()
    for (let i = 0; i < events.length; i++) {
      for (let j = i + 1; j < events.length; j++) {
        const a = events[i]!
        const b = events[j]!
        if (a.roomId === b.roomId && a.startTime < b.endTime && b.startTime < a.endTime) {
          set.add(a.id)
          set.add(b.id)
        }
      }
    }
    return set
  }, [events])

  // Assign colors to events by index
  const colorMap = useMemo(() => {
    const map = new Map<string, string>()
    events.forEach((e, i) => {
      map.set(e.id, e.color ?? EVENT_COLORS[i % EVENT_COLORS.length]!)
    })
    return map
  }, [events])

  const onMouseMove = useCallback((e: MouseEvent<SVGSVGElement>) => {
    setHoverPos({ x: e.clientX, y: e.clientY })
  }, [])

  return (
    <div className="rounded-lg bg-slate-900 p-4">
      <h3 className="text-sm font-medium text-slate-200 mb-3">{title}</h3>

      <div className="relative">
        <svg
          width={width}
          height={height}
          className="select-none"
          onMouseMove={onMouseMove}
        >
          {/* Time grid lines + labels */}
          {Array.from({ length: totalHours + 1 }, (_, i) => {
            const hour = startHour + i
            const x = xScale(hour)
            return (
              <g key={hour}>
                <line
                  x1={x} y1={PADDING.top}
                  x2={x} y2={PADDING.top + rooms.length * rowHeight}
                  stroke="rgb(51,65,85)"
                  strokeDasharray={i === 0 || i === totalHours ? undefined : '2,3'}
                />
                <text
                  x={x} y={PADDING.top - 8}
                  textAnchor="middle"
                  className="fill-slate-400 text-[10px]"
                >
                  {hour.toString().padStart(2, '0')}:00
                </text>
              </g>
            )
          })}

          {/* Room rows */}
          {rooms.map((room, ri) => {
            const y = PADDING.top + ri * rowHeight
            return (
              <g key={room.id}>
                {/* Row background (alternating) */}
                <rect
                  x={PADDING.left} y={y}
                  width={timeW} height={rowHeight}
                  fill={ri % 2 === 0 ? 'rgba(30,41,59,0.5)' : 'rgba(30,41,59,0.3)'}
                />
                {/* Row divider */}
                <line
                  x1={PADDING.left} y1={y + rowHeight}
                  x2={PADDING.left + timeW} y2={y + rowHeight}
                  stroke="rgb(51,65,85)" strokeWidth={0.5}
                />
                {/* Room label */}
                <text
                  x={PADDING.left - 8} y={y + rowHeight / 2 + 4}
                  textAnchor="end"
                  className="fill-slate-300 text-[11px]"
                >
                  {room.name}
                </text>
                <text
                  x={PADDING.left - 8} y={y + rowHeight / 2 + 15}
                  textAnchor="end"
                  className="fill-slate-500 text-[9px]"
                >
                  cap: {room.capacity}
                </text>
              </g>
            )
          })}

          {/* Events */}
          {events.map(event => {
            const roomIdx = rooms.findIndex(r => r.id === event.roomId)
            if (roomIdx === -1) return null

            const y = PADDING.top + roomIdx * rowHeight + 4
            const x1 = xScale(event.startTime)
            const x2 = xScale(event.endTime)
            const barW = Math.max(x2 - x1, 2)
            const barH = rowHeight - 8
            const isConflict = conflicts.has(event.id)
            const fill = colorMap.get(event.id) ?? '#6b7280'

            return (
              <g
                key={event.id}
                className="cursor-pointer"
                onClick={() => onEventClick?.(event.id)}
                onMouseEnter={() => setHoverEvent(event)}
                onMouseLeave={() => setHoverEvent(null)}
              >
                <rect
                  x={x1} y={y}
                  width={barW} height={barH}
                  rx={3}
                  fill={fill}
                  opacity={0.85}
                  stroke={isConflict ? '#ef4444' : 'none'}
                  strokeWidth={isConflict ? 2 : 0}
                />
                {/* Event label (only if bar wide enough) */}
                {barW > 40 && (
                  <text
                    x={x1 + 4} y={y + barH / 2 + 4}
                    className="fill-white text-[10px] font-medium pointer-events-none"
                  >
                    {event.name.length > barW / 6
                      ? event.name.slice(0, Math.floor(barW / 6)) + '...'
                      : event.name}
                  </text>
                )}
                {/* Conflict indicator */}
                {isConflict && (
                  <text
                    x={x1 + barW - 4} y={y + 10}
                    textAnchor="end"
                    className="fill-red-300 text-[9px] font-bold pointer-events-none"
                  >
                    !
                  </text>
                )}
              </g>
            )
          })}

          {/* X-axis label */}
          <text
            x={PADDING.left + timeW / 2}
            y={height - 4}
            textAnchor="middle"
            className="fill-slate-400 text-[10px]"
          >
            Time (hours)
          </text>
        </svg>

        {/* Hover tooltip */}
        {hoverEvent && (
          <div
            className="fixed z-50 rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-xs text-slate-200 shadow-lg pointer-events-none"
            style={{ left: hoverPos.x + 12, top: hoverPos.y - 40 }}
          >
            <div className="font-medium">{hoverEvent.name}</div>
            <div className="text-slate-400 mt-0.5">
              {hoverEvent.startTime.toFixed(1)}h &ndash; {hoverEvent.endTime.toFixed(1)}h
            </div>
            <div className="text-slate-400">
              Guests: {hoverEvent.guests}
              {conflicts.has(hoverEvent.id) && (
                <span className="text-red-400 ml-2">CONFLICT</span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-3 mt-3 text-[10px]">
        {events.map(e => (
          <div key={e.id} className="flex items-center gap-1">
            <div
              className="w-2.5 h-2.5 rounded-sm"
              style={{ backgroundColor: colorMap.get(e.id) }}
            />
            <span className="text-slate-400">{e.name}</span>
          </div>
        ))}
        {conflicts.size > 0 && (
          <div className="flex items-center gap-1">
            <div className="w-2.5 h-2.5 rounded-sm border-2 border-red-500" />
            <span className="text-red-400">Conflict</span>
          </div>
        )}
      </div>
    </div>
  )
}
