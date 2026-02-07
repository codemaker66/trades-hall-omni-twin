export const INLINE_SHARE_HASH_PARAM = 'project'
export const SNAPSHOT_SHARE_HASH_PARAM = 'share'
export const MAX_INLINE_SHARE_URL_LENGTH = 12000

export const encodeProjectPayloadForUrl = (payload: string): string => {
    const bytes = new TextEncoder().encode(payload)
    let binary = ''
    for (const byte of bytes) {
        binary += String.fromCharCode(byte)
    }

    return btoa(binary)
        .replace(/\+/g, '-')
        .replace(/\//g, '_')
        .replace(/=+$/g, '')
}

export const decodeProjectPayloadFromUrl = (encoded: string): string => {
    const normalized = encoded
        .replace(/-/g, '+')
        .replace(/_/g, '/')
    const padded = normalized + '='.repeat((4 - (normalized.length % 4)) % 4)
    const binary = atob(padded)
    const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0))
    return new TextDecoder().decode(bytes)
}
