/**
 * Pure selection computation functions.
 * Zero React, zero store — testable with plain objects.
 */

/**
 * Given clicked item IDs and modifier keys, compute the final selection.
 *
 * - No modifier: replace selection with targetIds
 * - Shift/Ctrl (isMultiSelect): toggle targetIds in current selection
 */
export function applySelectionModifier(
    currentSelection: readonly string[],
    targetIds: readonly string[],
    isMultiSelect: boolean,
): string[] {
    if (!isMultiSelect) return [...targetIds]

    const allSelected = targetIds.every((id) => currentSelection.includes(id))
    if (allSelected) {
        return currentSelection.filter((id) => !targetIds.includes(id))
    }
    return [...new Set([...currentSelection, ...targetIds])]
}

/**
 * Expand a single item click to include its group members.
 * Ctrl-click bypasses group expansion (selects individual item only).
 */
export function resolveClickTargetIds(
    clickedId: string,
    groupId: string | undefined,
    isCtrlClick: boolean,
    items: readonly { id: string; groupId?: string }[],
): string[] {
    if (!groupId || isCtrlClick) return [clickedId]
    return items.filter((i) => i.groupId === groupId).map((i) => i.id)
}

/**
 * Find the "anchor" item in a multi-select drag.
 * The anchor is the lowest item (by bottom edge), preferring platforms when tied.
 */
export function findDragAnchor(
    selectedItems: readonly { id: string; type: string; position: [number, number, number] }[],
    bottomOffsets: ReadonlyMap<string, number>,
): { id: string; type: string; position: [number, number, number] } {
    const first = selectedItems[0]
    if (!first) return undefined as never // caller guarantees non-empty array

    return selectedItems.reduce((lowest, item) => {
        const lowestOffset = bottomOffsets.get(lowest.id) ?? 0
        const itemOffset = bottomOffsets.get(item.id) ?? 0
        const lowestBottom = lowest.position[1] - lowestOffset
        const itemBottom = item.position[1] - itemOffset
        if (itemBottom === lowestBottom && item.type === 'platform') return item
        return itemBottom < lowestBottom ? item : lowest
    }, first)
}
