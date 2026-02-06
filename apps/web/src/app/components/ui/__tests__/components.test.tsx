import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { Card } from '../Card'
import { Badge } from '../Badge'
import { Avatar } from '../Avatar'
import { DataTable, type Column } from '../DataTable'
import { EmptyState } from '../EmptyState'
import { Skeleton, SkeletonRow, SkeletonCard } from '../Skeleton'
import { Select } from '../Select'

// ─── Card ────────────────────────────────────────────────────────────────────

describe('Card', () => {
  it('renders children', () => {
    render(<Card>Card content</Card>)
    expect(screen.getByText('Card content')).toBeTruthy()
  })

  it('renders header', () => {
    render(<Card header="Title">Body</Card>)
    expect(screen.getByText('Title')).toBeTruthy()
  })

  it('renders footer', () => {
    render(<Card footer={<button>OK</button>}>Body</Card>)
    expect(screen.getByRole('button', { name: 'OK' })).toBeTruthy()
  })

  it('applies noPadding', () => {
    const { container } = render(<Card noPadding>Body</Card>)
    const bodyDiv = container.querySelector('.p-5')
    expect(bodyDiv).toBeNull()
  })

  it('passes className', () => {
    const { container } = render(<Card className="extra-class">Body</Card>)
    expect(container.firstElementChild?.classList.contains('extra-class')).toBe(true)
  })
})

// ─── Badge ───────────────────────────────────────────────────────────────────

describe('Badge', () => {
  it('renders text', () => {
    render(<Badge>Active</Badge>)
    expect(screen.getByText('Active')).toBeTruthy()
  })

  it('applies variant classes', () => {
    const { container } = render(<Badge variant="success">OK</Badge>)
    const span = container.firstElementChild
    expect(span?.className).toContain('text-success-50')
  })

  it('defaults to default variant', () => {
    const { container } = render(<Badge>Tag</Badge>)
    const span = container.firstElementChild
    expect(span?.className).toContain('bg-surface-25')
  })
})

// ─── Avatar ──────────────────────────────────────────────────────────────────

describe('Avatar', () => {
  it('renders initials when no src', () => {
    render(<Avatar name="John Doe" />)
    expect(screen.getByText('JD')).toBeTruthy()
  })

  it('renders single initial for single name', () => {
    render(<Avatar name="Jane" />)
    expect(screen.getByText('J')).toBeTruthy()
  })

  it('renders image when src provided', () => {
    render(<Avatar name="John" src="/avatar.jpg" />)
    const img = screen.getByRole('img') as HTMLImageElement
    expect(img.src).toContain('/avatar.jpg')
    expect(img.alt).toBe('John')
  })

  it('has aria-label on fallback', () => {
    render(<Avatar name="Jane Doe" />)
    expect(screen.getByLabelText('Jane Doe')).toBeTruthy()
  })
})

// ─── DataTable ───────────────────────────────────────────────────────────────

describe('DataTable', () => {
  interface TestRow { id: string; name: string; age: number }

  const columns: Column<TestRow>[] = [
    { key: 'name', header: 'Name', render: (r) => r.name },
    { key: 'age', header: 'Age', render: (r) => r.age },
  ]

  const data: TestRow[] = [
    { id: '1', name: 'Alice', age: 30 },
    { id: '2', name: 'Bob', age: 25 },
  ]

  it('renders headers and rows', () => {
    render(<DataTable columns={columns} data={data} rowKey={(r) => r.id} />)
    expect(screen.getByText('Name')).toBeTruthy()
    expect(screen.getByText('Age')).toBeTruthy()
    expect(screen.getByText('Alice')).toBeTruthy()
    expect(screen.getByText('Bob')).toBeTruthy()
  })

  it('shows empty message when no data', () => {
    render(<DataTable columns={columns} data={[]} rowKey={(r) => r.id} emptyMessage="Nothing here" />)
    expect(screen.getByText('Nothing here')).toBeTruthy()
  })

  it('calls onRowClick', () => {
    const onClick = vi.fn()
    render(<DataTable columns={columns} data={data} rowKey={(r) => r.id} onRowClick={onClick} />)
    fireEvent.click(screen.getByText('Alice'))
    expect(onClick).toHaveBeenCalledWith(data[0])
  })
})

// ─── EmptyState ──────────────────────────────────────────────────────────────

describe('EmptyState', () => {
  it('renders title', () => {
    render(<EmptyState title="No results" />)
    expect(screen.getByText('No results')).toBeTruthy()
  })

  it('renders description', () => {
    render(<EmptyState title="Empty" description="Try again later" />)
    expect(screen.getByText('Try again later')).toBeTruthy()
  })

  it('renders action', () => {
    render(<EmptyState title="Empty" action={<button>Create</button>} />)
    expect(screen.getByRole('button', { name: 'Create' })).toBeTruthy()
  })
})

// ─── Skeleton ────────────────────────────────────────────────────────────────

describe('Skeleton', () => {
  it('renders with aria-hidden', () => {
    const { container } = render(<Skeleton />)
    expect(container.firstElementChild?.getAttribute('aria-hidden')).toBe('true')
  })

  it('applies custom width and height', () => {
    const { container } = render(<Skeleton width="w-1/2" height="h-8" />)
    const el = container.firstElementChild
    expect(el?.className).toContain('w-1/2')
    expect(el?.className).toContain('h-8')
  })

  it('renders circle variant', () => {
    const { container } = render(<Skeleton circle />)
    expect(container.firstElementChild?.className).toContain('rounded-full')
  })
})

describe('SkeletonRow', () => {
  it('renders correct number of columns', () => {
    const { container } = render(<SkeletonRow columns={3} />)
    const skeletons = container.querySelectorAll('[aria-hidden="true"]')
    expect(skeletons.length).toBe(3)
  })
})

describe('SkeletonCard', () => {
  it('renders without error', () => {
    const { container } = render(<SkeletonCard />)
    expect(container.firstElementChild).toBeTruthy()
  })
})

// ─── Select ──────────────────────────────────────────────────────────────────

describe('Select', () => {
  const options = [
    { value: 'a', label: 'Alpha' },
    { value: 'b', label: 'Beta' },
  ]

  it('renders options', () => {
    render(<Select options={options} />)
    expect(screen.getByText('Alpha')).toBeTruthy()
    expect(screen.getByText('Beta')).toBeTruthy()
  })

  it('renders label', () => {
    render(<Select options={options} label="Pick one" />)
    expect(screen.getByText('Pick one')).toBeTruthy()
  })

  it('renders placeholder', () => {
    render(<Select options={options} placeholder="Choose..." />)
    expect(screen.getByText('Choose...')).toBeTruthy()
  })

  it('calls onChange', () => {
    const onChange = vi.fn()
    render(<Select options={options} onChange={onChange} />)
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'b' } })
    expect(onChange).toHaveBeenCalled()
  })
})
