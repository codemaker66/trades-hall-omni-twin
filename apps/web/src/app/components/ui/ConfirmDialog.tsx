'use client'

import { Modal } from './Modal'
import { Button } from './Button'

interface ConfirmDialogProps {
    open: boolean
    onClose: () => void
    onConfirm: () => void
    title: string
    description: string
    /** Label for the confirm button (default: "Confirm") */
    confirmLabel?: string
    /** Label for the cancel button (default: "Cancel") */
    cancelLabel?: string
    /** Use danger styling for the confirm button */
    destructive?: boolean
}

export const ConfirmDialog = ({
    open,
    onClose,
    onConfirm,
    title,
    description,
    confirmLabel = 'Confirm',
    cancelLabel = 'Cancel',
    destructive = false,
}: ConfirmDialogProps) => {
    const handleConfirm = () => {
        onConfirm()
        onClose()
    }

    return (
        <Modal open={open} onClose={onClose} title={title}>
            <p className="text-sm text-surface-80 mb-6">{description}</p>
            <div className="flex gap-2">
                <Button variant="ghost" onClick={onClose} className="flex-1">
                    {cancelLabel}
                </Button>
                <Button
                    variant={destructive ? 'danger' : 'primary'}
                    onClick={handleConfirm}
                    className="flex-1"
                >
                    {confirmLabel}
                </Button>
            </div>
        </Modal>
    )
}
