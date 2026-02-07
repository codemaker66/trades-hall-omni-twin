import { useEffect, useRef, useState } from 'react'
import { useVenueStore } from '../../../../store'
import { SNAPSHOT_SHARE_HASH_PARAM, INLINE_SHARE_HASH_PARAM, decodeProjectPayloadFromUrl } from '../utils/urlEncoding'

export const useSharedProjectLoader = () => {
    const importProject = useVenueStore((state) => state.importProject)
    const hasHydratedRef = useRef(false)
    const [notice, setNotice] = useState<string | null>(null)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        if (hasHydratedRef.current) return
        hasHydratedRef.current = true

        let cancelled = false
        const hash = window.location.hash
        if (!hash || hash.length < 2) return

        const params = new URLSearchParams(hash.slice(1))
        const snapshotCode = params.get(SNAPSHOT_SHARE_HASH_PARAM)
        const encodedProject = params.get(INLINE_SHARE_HASH_PARAM)
        if (!snapshotCode && !encodedProject) return

        const applyImportedProject = (payload: string, sourceLabel: string): boolean => {
            const result = importProject(payload, { mode: 'replace' })
            if (!result.ok) {
                if (!cancelled) {
                    setNotice(null)
                    setError(result.message)
                }
                return false
            }

            if (!cancelled) {
                setNotice(`Shared project loaded from ${sourceLabel}.`)
                setError(null)
            }
            return true
        }

        const hydrateSharedProject = async () => {
            try {
                if (snapshotCode) {
                    const response = await fetch(`/api/share?code=${encodeURIComponent(snapshotCode)}`, { cache: 'no-store' })
                    const payloadResponse = await response.json().catch(() => null) as { payload?: string, error?: string } | null
                    if (!response.ok) {
                        if (!cancelled) {
                            setNotice(null)
                            setError(payloadResponse?.error ?? 'Short share link is invalid or expired.')
                        }
                        return
                    }

                    if (typeof payloadResponse?.payload !== 'string') {
                        if (!cancelled) {
                            setNotice(null)
                            setError('Short share link payload is invalid.')
                        }
                        return
                    }

                    const imported = applyImportedProject(payloadResponse.payload, 'short link')
                    if (!imported || cancelled) return
                } else if (encodedProject) {
                    const payload = decodeProjectPayloadFromUrl(encodedProject)
                    const imported = applyImportedProject(payload, 'inline link')
                    if (!imported || cancelled) return
                }

                const urlWithoutHash = `${window.location.pathname}${window.location.search}`
                window.history.replaceState(null, '', urlWithoutHash)
            } catch {
                if (!cancelled) {
                    setNotice(null)
                    setError(snapshotCode ? 'Short share link is invalid or expired.' : 'Shared link is invalid or corrupted.')
                }
            }
        }

        void hydrateSharedProject()
        return () => {
            cancelled = true
        }
    }, [importProject])

    return { notice, setNotice, error, setError }
}
