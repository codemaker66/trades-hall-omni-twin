'use client'

import { Group, Rect, Circle, Text } from 'react-konva'
import type { FloorPlanItem } from './store'

interface FurnitureItem2DProps {
  item: FloorPlanItem
  scale: number
  selected: boolean
  onSelect: (id: string, shiftKey: boolean) => void
  onDragStart: (id: string) => void
  onDragMove: (id: string, x: number, y: number) => void
  onDragEnd: (id: string) => void
}

const categoryColors: Record<string, string> = {
  table: '#bf953f',
  chair: '#6366f1',
  stage: '#66bb6a',
  decor: '#ffb74d',
  equipment: '#8d6e63',
}

export function FurnitureItem2D({
  item,
  scale,
  selected,
  onSelect,
  onDragStart,
  onDragMove,
  onDragEnd,
}: FurnitureItem2DProps) {
  const w = item.widthFt * scale
  const d = item.depthFt * scale
  const color = categoryColors[item.category] ?? '#8d6e63'
  const isRound = item.category === 'table' && Math.abs(item.widthFt - item.depthFt) < 0.5

  return (
    <Group
      x={item.x * scale}
      y={item.y * scale}
      rotation={item.rotation}
      draggable={!item.locked}
      onClick={(e) => onSelect(item.id, e.evt.shiftKey)}
      onTap={(e) => onSelect(item.id, false)}
      onDragStart={() => onDragStart(item.id)}
      onDragMove={(e) => {
        onDragMove(item.id, e.target.x() / scale, e.target.y() / scale)
      }}
      onDragEnd={(e) => {
        onDragEnd(item.id)
        // Snap position back to Konva node
        e.target.x(item.x * scale)
        e.target.y(item.y * scale)
      }}
      offsetX={w / 2}
      offsetY={d / 2}
    >
      {isRound ? (
        <Circle
          x={w / 2}
          y={d / 2}
          radius={w / 2}
          fill={color}
          opacity={0.8}
          stroke={selected ? '#ffd700' : '#1a0f0a'}
          strokeWidth={selected ? 2 : 1}
        />
      ) : (
        <Rect
          width={w}
          height={d}
          fill={color}
          opacity={0.8}
          cornerRadius={2}
          stroke={selected ? '#ffd700' : '#1a0f0a'}
          strokeWidth={selected ? 2 : 1}
        />
      )}
      <Text
        text={item.name}
        x={isRound ? 0 : 0}
        y={isRound ? d / 2 - 5 : d / 2 - 5}
        width={w}
        align="center"
        fontSize={Math.min(10, w * 0.15)}
        fill="#1a0f0a"
        listening={false}
      />
    </Group>
  )
}
