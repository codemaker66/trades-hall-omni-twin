'use client'

import type { ReactNode } from 'react'

export interface Column<T> {
  key: string
  header: string
  render: (row: T) => ReactNode
  className?: string
}

interface DataTableProps<T> {
  columns: Column<T>[]
  data: T[]
  rowKey: (row: T) => string
  onRowClick?: (row: T) => void
  emptyMessage?: string
}

export function DataTable<T>({ columns, data, rowKey, onRowClick, emptyMessage = 'No data' }: DataTableProps<T>) {
  if (data.length === 0) {
    return (
      <div className="text-center py-12 text-surface-60 text-sm">
        {emptyMessage}
      </div>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm text-left">
        <thead>
          <tr className="border-b border-surface-25">
            {columns.map((col) => (
              <th
                key={col.key}
                className={`px-4 py-3 text-xs font-semibold text-surface-60 uppercase tracking-wider ${col.className ?? ''}`}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row) => (
            <tr
              key={rowKey(row)}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
              className={`
                border-b border-surface-20 text-surface-80
                ${onRowClick ? 'cursor-pointer hover:bg-surface-15 transition-colors' : ''}
              `.trim()}
            >
              {columns.map((col) => (
                <td key={col.key} className={`px-4 py-3 ${col.className ?? ''}`}>
                  {col.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
