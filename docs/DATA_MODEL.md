# Project Omni-Twin Data Model

Types are written in a TypeScript-style shorthand for clarity.

## Venue
```
Venue {
  id: string
  name: string
  timezone: string
  address?: string
  floorPlan: {
    width: number
    depth: number
    unit: "ft" | "m"
    gridSize: number
    obstacles?: { id: string; name: string; x: number; y: number; width: number; depth: number }[]
  }
  createdAt: string
  updatedAt: string
}
```

## InventoryItem
```
InventoryItem {
  id: string
  venueId: string
  name: string
  category: "table" | "chair" | "other"
  dimensions: { width: number; depth: number; height?: number; unit: "ft" | "m" }
  seats?: number
  quantityTotal: number
  quantityReserved: number
  tags?: string[]
  createdAt: string
  updatedAt: string
}
```

## Layout
```
Layout {
  id: string
  venueId: string
  name: string
  status: "draft" | "saved" | "archived"
  items: LayoutItem[]
  metrics: {
    seatCount: number
    tableCount: number
    chairCount: number
  }
  notes?: string
  createdAt: string
  updatedAt: string
}

LayoutItem {
  id: string
  inventoryItemId: string
  position: { x: number; y: number }
  rotation: number
  scale?: number
  snapped: boolean
}
```

## Scenario
```
Scenario {
  id: string
  venueId: string
  name: string
  baseLayoutId: string
  layoutId: string
  status: "draft" | "review" | "approved"
  summary: {
    seatCount: number
    inventoryUsage: { inventoryItemId: string; used: number }[]
  }
  createdAt: string
  updatedAt: string
}
```
