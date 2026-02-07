'use client'

import { useVenueStore } from '../../../store'
import { useState } from 'react'
import { Modal } from '../ui/Modal'
import { Input } from '../ui/Input'
import { Button } from '../ui/Button'

export const ChairPromptModal = () => {
    const chairPrompt = useVenueStore((state) => state.chairPrompt)
    const closeChairPrompt = useVenueStore((state) => state.closeChairPrompt)
    const items = useVenueStore((state) => state.items)
    const addItem = useVenueStore((state) => state.addItem)
    const removeItems = useVenueStore((state) => state.removeItems)
    const hasInventoryForType = useVenueStore((state) => state.hasInventoryForType)
    const beginHistoryBatch = useVenueStore((state) => state.beginHistoryBatch)
    const endHistoryBatch = useVenueStore((state) => state.endHistoryBatch)

    const [chairCount, setChairCount] = useState<number>(10)

    const handleConfirmChairs = () => {
        if (!chairPrompt) return

        const { tableId } = chairPrompt
        beginHistoryBatch()

        try {
            if (tableId) {
                const table = items.find(i => i.id === tableId)
                if (table) {
                    const groupItems = items.filter(i => i.groupId === table.groupId && i.id !== tableId)
                    removeItems(groupItems.map(i => i.id), { recordHistory: false })

                    const radius = 1.3
                    const center = table.position
                    for (let i = 0; i < chairCount; i++) {
                        const angle = (i / chairCount) * Math.PI * 2
                        const x = center[0] + Math.sin(angle) * radius
                        const z = center[2] + Math.cos(angle) * radius
                        const rotY = angle + Math.PI
                        addItem('chair', [x, 0, z], [0, rotY, 0], table.groupId, { recordHistory: false })
                    }
                }
            } else {
                if (!hasInventoryForType('round-table')) {
                    addItem('round-table', [0, 0, 0], [0, 0, 0], undefined, { recordHistory: false })
                    closeChairPrompt()
                    return
                }

                const groupId = crypto.randomUUID()
                const center: [number, number, number] = [0, 0, 0]
                addItem('round-table', center, [0, 0, 0], groupId, { recordHistory: false })

                const radius = 1.3
                for (let i = 0; i < chairCount; i++) {
                    const angle = (i / chairCount) * Math.PI * 2
                    const x = center[0] + Math.sin(angle) * radius
                    const z = center[2] + Math.cos(angle) * radius
                    const rotY = angle + Math.PI
                    addItem('chair', [x, 0, z], [0, rotY, 0], groupId, { recordHistory: false })
                }
            }
        } finally {
            endHistoryBatch()
        }

        closeChairPrompt()
    }

    return (
        <Modal
            open={!!chairPrompt}
            onClose={closeChairPrompt}
            title="Add Table Setup"
        >
            <div className="mb-6">
                <Input
                    label="How many chairs?"
                    description="Placed evenly around the table."
                    type="number"
                    min={0}
                    max={20}
                    value={chairCount}
                    onChange={(e) => setChairCount(parseInt(e.target.value) || 0)}
                    autoFocus
                />
            </div>
            <div className="flex gap-2">
                <Button variant="ghost" onClick={closeChairPrompt} className="flex-1">
                    Cancel
                </Button>
                <Button variant="primary" onClick={handleConfirmChairs} className="flex-1">
                    Place
                </Button>
            </div>
        </Modal>
    )
}
