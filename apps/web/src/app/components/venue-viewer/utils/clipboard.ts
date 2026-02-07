export const copyTextToClipboard = async (text: string): Promise<boolean> => {
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
        try {
            await navigator.clipboard.writeText(text)
            return true
        } catch {
            // Fallback below.
        }
    }

    if (typeof document === 'undefined') {
        return false
    }

    const textarea = document.createElement('textarea')
    textarea.value = text
    textarea.setAttribute('readonly', '')
    textarea.style.position = 'fixed'
    textarea.style.opacity = '0'
    textarea.style.pointerEvents = 'none'
    document.body.appendChild(textarea)
    textarea.select()

    let copied = false
    try {
        copied = document.execCommand('copy')
    } catch {
        copied = false
    } finally {
        document.body.removeChild(textarea)
    }

    return copied
}
