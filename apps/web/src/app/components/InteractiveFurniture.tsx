import React, { useRef, useEffect } from 'react'
import { useVenueStore, FurnitureType } from '../../store'
import { RoundTable6ft, TrestleTable6ft, Chair, Platform } from './Furniture'
import * as THREE from 'three'
import { useThree } from '@react-three/fiber'

interface InteractiveFurnitureProps {
    id: string
    groupId?: string
    type: FurnitureType
    position: [number, number, number]
    rotation: [number, number, number]
    onRegister: (id: string, ref: THREE.Group) => void
}

export const InteractiveFurniture = ({ id, groupId, type, position, rotation, onRegister }: InteractiveFurnitureProps) => {
    const { scene } = useThree()
    const selectedIds = useVenueStore((state) => state.selectedIds)
    const setSelection = useVenueStore((state) => state.setSelection)
    const setIsDragging = useVenueStore((state) => state.setIsDragging)
    const updateItem = useVenueStore((state) => state.updateItem)
    const snappingEnabled = useVenueStore((state) => state.snappingEnabled)
    const snapGrid = useVenueStore((state) => state.snapGrid)

    // Check if selected
    const isSelected = selectedIds.includes(id)

    // Determine which model to render
    const Component =
        type === 'round-table' ? RoundTable6ft :
            type === 'trestle-table' ? TrestleTable6ft :
                type === 'chair' ? Chair :
                    Platform

    // Drag State
    const dragOffset = useRef(new THREE.Vector3())
    const isDraggingRef = useRef(false)
    const groupRef = useRef<THREE.Group>(null)
    const objectBounds = useRef({ centerY: 0, bottomOffset: 0, halfX: 0, halfZ: 0, ready: false })
    const stackRayOriginY = 50

    // Register ref with parent on mount
    useEffect(() => {
        if (groupRef.current) {
            onRegister(id, groupRef.current)
        }
    }, [id, onRegister])

    const openChairPrompt = useVenueStore((state) => state.openChairPrompt)
    const ensureObjectBounds = () => {
        if (objectBounds.current.ready || !groupRef.current) return

        const box = new THREE.Box3().setFromObject(groupRef.current)
        if (box.isEmpty()) return

        const size = new THREE.Vector3()
        const center = new THREE.Vector3()
        box.getSize(size)
        box.getCenter(center)

        objectBounds.current.centerY = center.y - groupRef.current.position.y
        objectBounds.current.bottomOffset = groupRef.current.position.y - box.min.y
        objectBounds.current.halfX = size.x / 2
        objectBounds.current.halfZ = size.z / 2
        objectBounds.current.ready = true
    }

    return (
        <group
            ref={groupRef}
            position={position}
            rotation={rotation}
            onDoubleClick={(e) => {
                if (type === 'round-table') {
                    e.stopPropagation()
                    openChairPrompt(type, id)
                }
            }}
            onClick={(e) => {
                if (isDraggingRef.current) return
                e.stopPropagation()

                const state = useVenueStore.getState()
                let idsToTarget = [id]
                const isCtrl = e.ctrlKey || e.metaKey
                if (groupId && !isCtrl) {
                    idsToTarget = state.items.filter(i => i.groupId === groupId).map(i => i.id)
                }

                const isMultiSelect = e.shiftKey || isCtrl
                if (isMultiSelect) {
                    const allSelected = idsToTarget.every(i => state.selectedIds.includes(i))
                    if (allSelected) setSelection(state.selectedIds.filter(i => !idsToTarget.includes(i)))
                    else setSelection([...new Set([...state.selectedIds, ...idsToTarget])])
                } else {
                    setSelection(idsToTarget)
                }
            }}
            onPointerDown={(e) => {
                if (e.button !== 0) return
                e.stopPropagation()

                const target = e.target as HTMLElement
                target.setPointerCapture(e.pointerId)

                isDraggingRef.current = true
                setIsDragging(true)
                ensureObjectBounds()

                // Calculate Drag Offset relative to the Floor Plane (Y=0)
                // This prevents "jumping" when we start dragging from a non-zero height (like on a stack)
                // We project the click onto the floor, compare to object position's distinct X/Z.

                const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)
                const intersect = new THREE.Vector3()
                const raycaster = new THREE.Raycaster()
                raycaster.setFromCamera(e.pointer, e.camera)
                raycaster.ray.intersectPlane(plane, intersect)

                // Record the offset between the exact mouse ray hitting the floor 
                // and the object's current X/Z position.
                // We ignore Y because we will control Y explicitly via stacking logic.
                if (intersect) {
                    dragOffset.current.set(intersect.x - position[0], 0, intersect.z - position[2])
                }
            }}
            onPointerUp={(e) => {
                if (isDraggingRef.current) {
                    e.stopPropagation()
                    const target = e.target as HTMLElement
                    target.releasePointerCapture(e.pointerId)
                    isDraggingRef.current = false
                    setIsDragging(false)
                }
            }}
            onPointerMove={(e) => {
                if (isDraggingRef.current) {
                    e.stopPropagation()
                    const state = useVenueStore.getState()

                    // 1. Raycast to Floor Plane (Y=0) to get "Candidate X/Z"
                    const raycaster = new THREE.Raycaster()
                    raycaster.setFromCamera(e.pointer, e.camera)
                    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)
                    const intersectPoint = new THREE.Vector3()
                    raycaster.ray.intersectPlane(plane, intersectPoint)

                    if (intersectPoint) {
                        // Candidate Position based on drag offset
                        const rawNewX = intersectPoint.x - dragOffset.current.x
                        const rawNewZ = intersectPoint.z - dragOffset.current.z

                        // 2. Snapping (X/Z first so height uses final snapped position)
                        let finalX = rawNewX
                        let finalZ = rawNewZ

                        if (snappingEnabled && snapGrid > 0) {
                            finalX = Math.round(finalX / snapGrid) * snapGrid
                            finalZ = Math.round(finalZ / snapGrid) * snapGrid
                        }

                        // 3. Logic: Height Calculation via Downward Raycast (Lego Physics)
                        let targetY = 0
                        ensureObjectBounds()

                        const downRay = new THREE.Raycaster()
                        const downDir = new THREE.Vector3(0, -1, 0)
                        const rayOrigin = new THREE.Vector3()
                        const originY = type === 'chair'
                            ? (groupRef.current?.position.y ?? position[1]) + objectBounds.current.centerY
                            : stackRayOriginY

                        rayOrigin.set(finalX, originY, finalZ)
                        downRay.set(rayOrigin, downDir)

                        const downIntersects = downRay.intersectObjects(scene.children, true)

                        for (const hit of downIntersects) {
                            // Ignore self or selection
                            let current: THREE.Object3D | null = hit.object
                            let rootGroup: THREE.Object3D | null = null

                            while (current) {
                                if (current.userData && current.userData.id) {
                                    rootGroup = current
                                    break
                                }
                                current = current.parent
                            }

                            if (!rootGroup || !rootGroup.userData.id) continue
                            if (rootGroup === groupRef.current) continue
                            if (state.selectedIds.includes(rootGroup.userData.id)) continue

                            const hitItem = state.items.find(i => i.id === rootGroup!.userData.id)
                            if (!hitItem) continue

                            if (type === 'platform' && hitItem.type !== 'platform') continue
                            if (type !== 'platform' && hitItem.type !== 'platform') continue

                            targetY = hit.point.y
                            break // Closest valid surface for this ray
                        }

                        // 4. Apply offset
                        let finalY = targetY
                        if (type === 'chair' || type === 'platform') {
                            finalY += objectBounds.current.bottomOffset
                        }

                        // 5. Update Store
                        const deltaX = finalX - position[0]
                        const deltaY = finalY - position[1] // Absolute set logic
                        const deltaZ = finalZ - position[2]

                        if (Math.abs(deltaX) < 0.001 && Math.abs(deltaZ) < 0.001 && Math.abs(deltaY) < 0.001) return

                        const selectedItems = state.items.filter(i => state.selectedIds.includes(i.id))
                        const updates = selectedItems.map(item => {
                            // Maintain relative Y structure? 
                            // Or Flatten? "Stacking" implies flattening onto the target.
                            // If we move a group of chairs, we want them to Land on the platform.
                            // If they were at same height, they land at same height.
                            // If we move a stack of platforms, the base lands on target, upper ones follow.

                            // For complex multi-select, calculating "Base Y" of selection is hard.
                            // Simplified: Apply deltaY to all.
                            // This means if I pick up a Chair (Y=0) and Table (Y=0) and move to Platform (Target=0.2),
                            // DeltaY = 0.2. Both move to 0.2. Correct.

                            // What if I pick a Stack (P1=0, P2=0.2) and move to Floor?
                            // Lead item P1 (Target=0). DeltaY = 0.
                            // P2 stays at 0.2 relative? Wait.
                            // If P1 moves 0->0, P2 moves 0.2->0.2. Correct.

                            return {
                                id: item.id,
                                changes: {
                                    position: [
                                        item.position[0] + deltaX,
                                        item.position[1] + deltaY,
                                        item.position[2] + deltaZ
                                    ] as [number, number, number]
                                }
                            }
                        })

                        state.updateItems(updates)
                    }
                }
            }}
            userData={{ id }}
        >
            <Component />

            {/* Visual Feedback for Selection */}
            {isSelected && (
                <mesh position={[0, 1.5, 0]} raycast={() => null}>
                    <sphereGeometry args={[0.1]} />
                    <meshBasicMaterial color="#6366f1" transparent opacity={0.5} />
                </mesh>
            )}
        </group>
    )
}
