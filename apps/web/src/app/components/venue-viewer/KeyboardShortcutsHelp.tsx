'use client'

import { useVenueStore } from '../../../store'
import { Modal } from '../ui/Modal'
import { SHORTCUTS, type ShortcutCategory } from './hooks/useKeyboardShortcuts'

const CATEGORY_ORDER: ShortcutCategory[] = ['Editing', 'Selection', 'Tools', 'Navigation']

/** Deduplicate shortcuts that map to the same action (e.g. Del + Backspace → "Del / Backspace") */
function getGroupedShortcuts() {
    const seen = new Map<string, { description: string; displays: string[]; category: ShortcutCategory }>()

    for (const s of SHORTCUTS) {
        const key = `${s.category}:${s.description}`
        const existing = seen.get(key)
        if (existing) {
            existing.displays.push(s.display)
        } else {
            seen.set(key, { description: s.description, displays: [s.display], category: s.category })
        }
    }

    return Array.from(seen.values())
}

export const KeyboardShortcutsHelp = () => {
    const open = useVenueStore((state) => state.shortcutsHelpOpen)
    const setOpen = useVenueStore((state) => state.setShortcutsHelpOpen)

    const grouped = getGroupedShortcuts()

    return (
        <Modal open={open} onClose={() => setOpen(false)} title="Keyboard Shortcuts">
            <div className="space-y-4 max-h-[60vh] overflow-y-auto pr-1">
                {CATEGORY_ORDER.map((category) => {
                    const items = grouped.filter(s => s.category === category)
                    if (items.length === 0) return null

                    return (
                        <section key={category}>
                            <h3 className="text-xs font-bold tracking-wider text-surface-80 uppercase mb-2">{category}</h3>
                            <div className="space-y-1">
                                {items.map((item) => (
                                    <div key={item.description} className="flex items-center justify-between py-1.5 px-2 rounded-sm hover:bg-surface-15">
                                        <span className="text-sm text-surface-90">{item.description}</span>
                                        <div className="flex gap-1">
                                            {item.displays.map((d, i) => (
                                                <span key={d}>
                                                    {i > 0 && <span className="text-surface-60 text-xs mx-1">/</span>}
                                                    <kbd className="inline-block min-w-[24px] text-center px-1.5 py-0.5 text-xs font-mono bg-surface-20 border border-surface-40 rounded text-surface-90 shadow-sm">
                                                        {d}
                                                    </kbd>
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </section>
                    )
                })}
            </div>
            <p className="text-[11px] text-surface-60 mt-4 text-center">
                Press <kbd className="px-1 py-0.5 text-[10px] font-mono bg-surface-20 border border-surface-40 rounded">?</kbd> to toggle this overlay
            </p>
        </Modal>
    )
}
