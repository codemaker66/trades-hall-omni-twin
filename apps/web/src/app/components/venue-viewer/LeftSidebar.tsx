'use client'

import { motion } from 'framer-motion'
import { useVenueStore, type ProjectImportMode } from '../../../store'
import { useShallow } from 'zustand/react/shallow'
import { useMemo, useRef, useState, type ChangeEvent } from 'react'
import { computeInventoryUsage, computeLayoutMetrics } from '../../../store/helpers'
import { copyTextToClipboard } from './utils/clipboard'
import {
    encodeProjectPayloadForUrl,
    INLINE_SHARE_HASH_PARAM, SNAPSHOT_SHARE_HASH_PARAM, MAX_INLINE_SHARE_URL_LENGTH
} from './utils/urlEncoding'
import { Button } from '../ui/Button'
import { Input } from '../ui/Input'
import { Modal } from '../ui/Modal'
import { ConfirmDialog } from '../ui/ConfirmDialog'
import { toast } from '../ui/Toast'

interface LeftSidebarProps {
    projectNotice: string | null
    setProjectNotice: (v: string | null) => void
    projectError: string | null
    setProjectError: (v: string | null) => void
}

export const LeftSidebar = ({ projectNotice, setProjectNotice, projectError, setProjectError }: LeftSidebarProps) => {
    const { items, inventoryCatalog, inventoryWarning, scenarios, activeScenarioId } = useVenueStore(
        useShallow((state) => ({
            items: state.items,
            inventoryCatalog: state.inventoryCatalog,
            inventoryWarning: state.inventoryWarning,
            scenarios: state.scenarios,
            activeScenarioId: state.activeScenarioId,
        }))
    )
    const clearInventoryWarning = useVenueStore((state) => state.clearInventoryWarning)
    const updateInventoryItem = useVenueStore((state) => state.updateInventoryItem)
    const saveScenario = useVenueStore((state) => state.saveScenario)
    const loadScenario = useVenueStore((state) => state.loadScenario)
    const deleteScenario = useVenueStore((state) => state.deleteScenario)
    const renameScenario = useVenueStore((state) => state.renameScenario)
    const setScenarioStatus = useVenueStore((state) => state.setScenarioStatus)
    const resetProject = useVenueStore((state) => state.resetProject)
    const exportProject = useVenueStore((state) => state.exportProject)
    const importProject = useVenueStore((state) => state.importProject)

    const [scenarioName, setScenarioName] = useState('')
    const [isShortSharePending, setIsShortSharePending] = useState(false)
    const importInputRef = useRef<HTMLInputElement>(null)
    const pendingImportModeRef = useRef<ProjectImportMode>('replace')

    // Dialog state
    const [resetConfirmOpen, setResetConfirmOpen] = useState(false)
    const [importConfirmOpen, setImportConfirmOpen] = useState(false)
    const pendingImportFileRef = useRef<File | null>(null)
    const [renameModal, setRenameModal] = useState<{ id: string; name: string } | null>(null)

    const inventoryUsage = useMemo(() => computeInventoryUsage(items), [items])
    const layoutMetrics = useMemo(() => computeLayoutMetrics(items), [items])

    const handleSaveScenario = () => {
        saveScenario(scenarioName)
        setScenarioName('')
    }

    const handleResetProject = () => {
        setResetConfirmOpen(true)
    }

    const confirmReset = () => {
        resetProject()
        setScenarioName('')
        setProjectNotice(null)
        setProjectError(null)
        toast.success('Project reset successfully.')
    }

    const handleExportProject = () => {
        try {
            const payload = exportProject()
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
            const filename = `omnitwin-project-${timestamp}.json`
            const blob = new Blob([payload], { type: 'application/json' })
            const url = window.URL.createObjectURL(blob)
            const anchor = document.createElement('a')
            anchor.href = url
            anchor.download = filename
            anchor.style.display = 'none'
            document.body.appendChild(anchor)
            anchor.click()
            document.body.removeChild(anchor)
            window.URL.revokeObjectURL(url)

            setProjectNotice(null)
            setProjectError(null)
            toast.success(`Project exported as ${filename}.`)
        } catch {
            toast.error('Project export failed. Please try again.')
        }
    }

    const handleShareShortProject = async () => {
        setIsShortSharePending(true)
        try {
            const payload = exportProject()
            const response = await fetch('/api/share', {
                method: 'POST',
                headers: { 'content-type': 'application/json' },
                body: JSON.stringify({ payload }),
                cache: 'no-store'
            })

            const shareResponse = await response.json().catch(() => null) as { code?: string, expiresAt?: string, error?: string } | null
            if (!response.ok || typeof shareResponse?.code !== 'string') {
                toast.error(shareResponse?.error ?? 'Could not create short share link.')
                return
            }

            const shareUrl = `${window.location.origin}${window.location.pathname}${window.location.search}#${SNAPSHOT_SHARE_HASH_PARAM}=${shareResponse.code}`
            const copied = await copyTextToClipboard(shareUrl)
            if (!copied) {
                toast.error('Could not copy short share link. Use Export instead.')
                return
            }

            const expiresLabel = shareResponse.expiresAt
                ? new Date(shareResponse.expiresAt).toLocaleDateString()
                : null
            toast.success(
                expiresLabel
                    ? `Short share link copied. Expires ${expiresLabel}.`
                    : 'Short share link copied to clipboard.'
            )
        } catch {
            toast.error('Could not create short share link.')
        } finally {
            setIsShortSharePending(false)
        }
    }

    const handleShareInlineProject = async () => {
        try {
            const payload = exportProject()
            const encodedProject = encodeProjectPayloadForUrl(payload)
            const shareUrl = `${window.location.origin}${window.location.pathname}${window.location.search}#${INLINE_SHARE_HASH_PARAM}=${encodedProject}`

            if (shareUrl.length > MAX_INLINE_SHARE_URL_LENGTH) {
                toast.error('Project is too large to share as a URL. Use Export instead.')
                return
            }

            const copied = await copyTextToClipboard(shareUrl)
            if (!copied) {
                toast.error('Could not copy share link. Use Export instead.')
                return
            }

            toast.success('Inline share link copied to clipboard.')
        } catch {
            toast.error('Could not generate inline share link.')
        }
    }

    const openProjectImportPicker = (mode: ProjectImportMode) => {
        pendingImportModeRef.current = mode
        importInputRef.current?.click()
    }

    const handleImportFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0]
        event.target.value = ''
        if (!file) return

        if (pendingImportModeRef.current === 'replace') {
            pendingImportFileRef.current = file
            setImportConfirmOpen(true)
            return
        }

        await processImport(file)
    }

    const processImport = async (file: File) => {
        try {
            const payload = await file.text()
            const result = importProject(payload, { mode: pendingImportModeRef.current })
            if (!result.ok) {
                toast.error(result.message)
                return
            }

            toast.success(result.message)
            setScenarioName('')
        } catch {
            toast.error('Project import failed. Please try again.')
        }
    }

    const confirmImport = async () => {
        const file = pendingImportFileRef.current
        pendingImportFileRef.current = null
        if (file) await processImport(file)
    }

    const handleRenameConfirm = () => {
        if (!renameModal || !renameModal.name.trim()) return
        renameScenario(renameModal.id, renameModal.name.trim())
        setRenameModal(null)
    }

    return (
        <>
            <motion.aside
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.15 }}
                className="absolute top-28 left-6 w-96 max-h-[calc(100dvh-190px)] overflow-y-auto border-2 border-surface-25 bg-surface-10/90 p-4 rounded-sm shadow-2xl pointer-events-auto space-y-4"
                role="region"
                aria-label="Project sidebar"
            >
                {/* Layout Metrics */}
                <section aria-label="Layout metrics">
                    <h2 className="text-sm font-bold tracking-wider text-surface-90 uppercase mb-2">Layout Metrics</h2>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                        <div className="bg-surface-20 border border-surface-40 rounded-sm px-3 py-2 text-surface-90">Seats: {layoutMetrics.seatCount}</div>
                        <div className="bg-surface-20 border border-surface-40 rounded-sm px-3 py-2 text-surface-90">Tables: {layoutMetrics.tableCount}</div>
                        <div className="bg-surface-20 border border-surface-40 rounded-sm px-3 py-2 text-surface-90">Chairs: {layoutMetrics.chairCount}</div>
                        <div className="bg-surface-20 border border-surface-40 rounded-sm px-3 py-2 text-surface-90">Platforms: {layoutMetrics.platformCount}</div>
                    </div>
                </section>

                {/* Inventory */}
                <section aria-label="Inventory management">
                    <h2 className="text-sm font-bold tracking-wider text-surface-90 uppercase mb-2">Inventory</h2>
                    <div className="space-y-2">
                        {inventoryCatalog.map((inventoryItem) => {
                            const used = inventoryUsage[inventoryItem.furnitureType]
                            const available = Math.max(0, inventoryItem.quantityTotal - inventoryItem.quantityReserved)
                            const usageRatio = available > 0 ? used / available : 1
                            const ratio = Math.min(1, usageRatio)
                            const barColor = usageRatio > 1 ? 'bg-danger-50' : usageRatio > 0.8 ? 'bg-warning-50' : 'bg-success-50'

                            return (
                                <div key={inventoryItem.id} className="bg-surface-20 border border-surface-40 rounded-sm p-2">
                                    <div className="flex items-center justify-between text-xs text-surface-90 mb-1">
                                        <span>{inventoryItem.name}</span>
                                        <span aria-label={`${used} of ${available} used`}>{used}/{available}</span>
                                    </div>
                                    <div
                                        className="h-2 bg-surface-15 rounded-sm overflow-hidden mb-2"
                                        role="progressbar"
                                        aria-valuenow={used}
                                        aria-valuemax={available}
                                        aria-label={`${inventoryItem.name} usage`}
                                    >
                                        <div className={`h-full ${barColor}`} style={{ width: `${ratio * 100}%` }} />
                                    </div>
                                    <div className="flex items-center justify-between gap-2">
                                        <Button
                                            size="sm"
                                            onClick={() => updateInventoryItem(inventoryItem.id, { quantityTotal: Math.max(0, inventoryItem.quantityTotal - 1) })}
                                            aria-label={`Decrease ${inventoryItem.name} stock`}
                                        >
                                            - Stock
                                        </Button>
                                        <Button
                                            size="sm"
                                            onClick={() => updateInventoryItem(inventoryItem.id, { quantityTotal: inventoryItem.quantityTotal + 1 })}
                                            aria-label={`Increase ${inventoryItem.name} stock`}
                                        >
                                            + Stock
                                        </Button>
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                </section>

                {/* Scenarios */}
                <section aria-label="Scenario management">
                    <h2 className="text-sm font-bold tracking-wider text-surface-90 uppercase mb-2">Scenarios</h2>
                    <div className="flex gap-2 mb-2">
                        <div className="flex-1">
                            <Input
                                value={scenarioName}
                                onChange={(e) => setScenarioName(e.target.value)}
                                placeholder="Scenario name"
                            />
                        </div>
                        <Button variant="primary" size="sm" onClick={handleSaveScenario} aria-label="Save scenario" className="self-start">
                            Save
                        </Button>
                    </div>

                    <div className="space-y-2 max-h-48 overflow-y-auto pr-1" role="list" aria-label="Saved scenarios">
                        {scenarios.length === 0 && (
                            <div className="text-xs text-surface-80">No scenarios yet. Save one to branch the layout.</div>
                        )}
                        {scenarios.map((scenario) => (
                            <div
                                key={scenario.id}
                                role="listitem"
                                className={`border rounded-sm p-2 ${activeScenarioId === scenario.id ? 'border-gold-60 bg-surface-20' : 'border-surface-40 bg-surface-15'}`}
                            >
                                <div className="flex items-center justify-between gap-2 mb-2">
                                    <button
                                        onClick={() => loadScenario(scenario.id)}
                                        className="text-sm text-left text-surface-90 hover:text-white truncate focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-indigo-50 rounded-sm"
                                        aria-label={`Load scenario: ${scenario.name}`}
                                    >
                                        {scenario.name}
                                    </button>
                                    <span className="text-[10px] text-surface-80">{new Date(scenario.updatedAt).toLocaleDateString()}</span>
                                </div>

                                <div className="flex items-center gap-2">
                                    <select
                                        value={scenario.status}
                                        onChange={(e) => setScenarioStatus(scenario.id, e.target.value as 'draft' | 'review' | 'approved')}
                                        className="flex-1 bg-surface-5 border border-surface-40 rounded-sm px-2 py-1 text-xs text-surface-90 focus:outline-none focus:border-indigo-50"
                                        aria-label={`Status for ${scenario.name}`}
                                    >
                                        <option value="draft">Draft</option>
                                        <option value="review">Review</option>
                                        <option value="approved">Approved</option>
                                    </select>
                                    <Button
                                        size="sm"
                                        onClick={() => setRenameModal({ id: scenario.id, name: scenario.name })}
                                        aria-label={`Rename ${scenario.name}`}
                                    >
                                        Rename
                                    </Button>
                                    <Button
                                        size="sm"
                                        variant="danger"
                                        onClick={() => deleteScenario(scenario.id)}
                                        aria-label={`Delete ${scenario.name}`}
                                    >
                                        Delete
                                    </Button>
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="mt-3 flex items-center justify-between gap-2">
                        <span className="text-[11px] text-surface-80">Autosave is enabled for this browser.</span>
                        <Button size="sm" variant="danger" onClick={handleResetProject} aria-label="Reset project">
                            Reset Project
                        </Button>
                    </div>

                    <input
                        ref={importInputRef}
                        type="file"
                        accept=".json,application/json"
                        className="hidden"
                        onChange={handleImportFileChange}
                        aria-hidden="true"
                    />

                    <div className="mt-2 grid grid-cols-5 gap-2">
                        <Button size="sm" onClick={handleShareShortProject} loading={isShortSharePending} aria-label="Share short link">
                            {isShortSharePending ? 'Sharing...' : 'Share Short'}
                        </Button>
                        <Button size="sm" onClick={handleShareInlineProject} aria-label="Share inline link">
                            Share Inline
                        </Button>
                        <Button size="sm" onClick={handleExportProject} aria-label="Export project">
                            Export
                        </Button>
                        <Button size="sm" onClick={() => openProjectImportPicker('merge')} aria-label="Import and merge project">
                            Import Merge
                        </Button>
                        <Button size="sm" variant="primary" onClick={() => openProjectImportPicker('replace')} aria-label="Import and replace project">
                            Import Replace
                        </Button>
                    </div>

                    {projectNotice && (
                        <div className="mt-2 bg-success-10 border border-success-50 text-success-80 rounded-sm px-2 py-1 text-[11px]" role="status">
                            {projectNotice}
                        </div>
                    )}

                    {projectError && (
                        <div className="mt-2 bg-danger-10 border border-danger-50 text-danger-70 rounded-sm px-2 py-1 text-[11px]" role="alert">
                            {projectError}
                        </div>
                    )}
                </section>

                {inventoryWarning && (
                    <div className="bg-danger-10 border border-danger-50 text-danger-70 rounded-sm px-3 py-2 text-xs flex items-start justify-between gap-2" role="alert">
                        <span>{inventoryWarning}</span>
                        <button onClick={clearInventoryWarning} className="text-danger-70 hover:text-white" aria-label="Dismiss inventory warning">
                            ✕
                        </button>
                    </div>
                )}
            </motion.aside>

            {/* Confirm dialogs */}
            <ConfirmDialog
                open={resetConfirmOpen}
                onClose={() => setResetConfirmOpen(false)}
                onConfirm={confirmReset}
                title="Reset Project"
                description="This clears the layout, scenarios, and inventory customizations on this device. This action cannot be undone."
                confirmLabel="Reset"
                destructive
            />

            <ConfirmDialog
                open={importConfirmOpen}
                onClose={() => { setImportConfirmOpen(false); pendingImportFileRef.current = null }}
                onConfirm={confirmImport}
                title="Import and Replace"
                description="This will replace the current project on this device with the imported file. This action cannot be undone."
                confirmLabel="Replace"
                destructive
            />

            {/* Rename modal */}
            <Modal
                open={!!renameModal}
                onClose={() => setRenameModal(null)}
                title="Rename Scenario"
            >
                <div className="mb-4">
                    <Input
                        label="Scenario name"
                        value={renameModal?.name ?? ''}
                        onChange={(e) => setRenameModal((prev) => prev ? { ...prev, name: e.target.value } : null)}
                        autoFocus
                        onKeyDown={(e) => { if (e.key === 'Enter') handleRenameConfirm() }}
                    />
                </div>
                <div className="flex gap-2">
                    <Button variant="ghost" onClick={() => setRenameModal(null)} className="flex-1">
                        Cancel
                    </Button>
                    <Button variant="primary" onClick={handleRenameConfirm} className="flex-1">
                        Rename
                    </Button>
                </div>
            </Modal>
        </>
    )
}
