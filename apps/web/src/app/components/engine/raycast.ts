/**
 * Shared raycast utilities for 3D scene interaction.
 * Consolidates the duplicated "raycast down + walk up to userData.id" pattern.
 */
import * as THREE from 'three'

/**
 * Walk up the parent chain from an object to find the nearest
 * ancestor with userData.id set. Returns that ancestor or null.
 */
export function findRootItemGroup(object: THREE.Object3D): THREE.Object3D | null {
    let current: THREE.Object3D | null = object
    while (current) {
        if (current.userData?.id) return current
        current = current.parent
    }
    return null
}

/**
 * Traverse a scene to find a group by its userData.id.
 */
export function findGroupById(scene: THREE.Object3D, userId: string): THREE.Group | null {
    let found: THREE.Group | null = null
    scene.traverse((obj) => {
        if (found) return
        if (obj.userData?.id === userId) found = obj as THREE.Group
    })
    return found
}

/**
 * Raycast straight down from a point to find the stack height.
 * Used for "Lego physics" — items stacking on platforms.
 *
 * @param origin       Ray origin (high above the target position)
 * @param direction    Ray direction (typically [0, -1, 0])
 * @param raycaster    Pre-allocated raycaster
 * @param scene        Scene to raycast into
 * @param excludeIds   IDs to skip (selected items, self)
 * @param itemType     Type of the item being placed (affects stacking rules)
 * @param items        All venue items (for type lookup)
 */
export function raycastDownForStack(
    origin: THREE.Vector3,
    direction: THREE.Vector3,
    raycaster: THREE.Raycaster,
    scene: THREE.Object3D,
    excludeIds: ReadonlySet<string>,
    itemType: string,
    items: readonly { id: string; type: string }[],
): { y: number; platformRoot: THREE.Object3D | null } {
    raycaster.set(origin, direction)
    const intersects = raycaster.intersectObjects(scene.children, true)

    let targetY = 0
    let platformRoot: THREE.Object3D | null = null

    for (const hit of intersects) {
        const rootGroup = findRootItemGroup(hit.object)
        if (!rootGroup?.userData.id) continue
        if (excludeIds.has(rootGroup.userData.id)) continue

        const hitItem = items.find((i) => i.id === rootGroup!.userData.id)
        if (!hitItem) continue

        // Platform stacking rules:
        // - Platforms only stack on platforms
        // - Non-platforms only stack on platforms (not on other furniture)
        if (itemType === 'platform' && hitItem.type !== 'platform') continue
        if (itemType !== 'platform' && hitItem.type !== 'platform') continue

        targetY = hit.point.y
        if (hitItem.type === 'platform') platformRoot = rootGroup
        break
    }

    return { y: targetY, platformRoot }
}

/**
 * Compute local-space bounding box of a Three.js object hierarchy,
 * excluding any children with userData.skipBounds = true.
 */
export function getLocalBounds(root: THREE.Object3D): THREE.Box3 {
    const box = new THREE.Box3()
    root.updateWorldMatrix(true, false)
    const inv = root.matrixWorld.clone().invert()

    root.traverse((obj) => {
        if (obj.userData?.skipBounds) return
        const mesh = obj as THREE.Mesh
        if (!mesh.isMesh || !mesh.geometry) return
        const geom = mesh.geometry
        if (!geom.boundingBox) geom.computeBoundingBox()
        if (!geom.boundingBox) return
        const localBox = geom.boundingBox.clone()
        localBox.applyMatrix4(mesh.matrixWorld)
        localBox.applyMatrix4(inv)
        box.union(localBox)
    })

    return box
}
