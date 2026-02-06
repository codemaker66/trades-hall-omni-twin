'use client'

import { forwardRef, type SelectHTMLAttributes } from 'react'

interface SelectOption {
  value: string
  label: string
}

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: string
  options: SelectOption[]
  placeholder?: string
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ label, options, placeholder, className = '', id, ...props }, ref) => {
    const selectId = id ?? label?.toLowerCase().replace(/\s+/g, '-')

    return (
      <div className="flex flex-col gap-1">
        {label && (
          <label htmlFor={selectId} className="text-xs text-surface-80 font-medium">
            {label}
          </label>
        )}
        <select
          ref={ref}
          id={selectId}
          className={`
            bg-surface-5 border border-surface-25 text-surface-90 text-sm rounded-sm
            px-3 py-2 focus:outline-none focus:ring-1 focus:ring-indigo-50
            disabled:opacity-50 disabled:cursor-not-allowed
            appearance-none bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2212%22%20height%3D%2212%22%20viewBox%3D%220%200%2012%2012%22%3E%3Cpath%20fill%3D%22%23a1887f%22%20d%3D%22M6%208L1%203h10z%22%2F%3E%3C%2Fsvg%3E')]
            bg-[length:12px] bg-[right_8px_center] bg-no-repeat pr-8
            ${className}
          `.trim()}
          {...props}
        >
          {placeholder && (
            <option value="" disabled>
              {placeholder}
            </option>
          )}
          {options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>
    )
  },
)

Select.displayName = 'Select'
