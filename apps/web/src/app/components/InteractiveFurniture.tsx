import React, { useRef, useEffect, useState } from 'react'
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
    const selectionBottomOffsets = useRef<{ [id: string]: number }>({})
    const trestleDims = { length: 1.8, depth: 0.76 }
    const trestleSnap = { along: 0.25, across: 0.12, angleCos: Math.cos(THREE.MathUtils.degToRad(6)) }

    // Register ref with parent on mount
    useEffect(() => {
        if (groupRef.current) {
            onRegister(id, groupRef.current)
        }
    }, [id, onRegister])

    const openChairPrompt = useVenueStore((state) => state.openChairPrompt)
    const [selectionBounds, setSelectionBounds] = useState<{ center: THREE.Vector3, size: THREE.Vector3 } | null>(null)
    const boxesOverlap = (a: THREE.Box3, b: THREE.Box3, eps = 0.001) => {
        return (
            a.max.x > b.min.x + eps &&
            a.min.x < b.max.x - eps &&
            a.max.y > b.min.y + eps &&
            a.min.y < b.max.y - eps &&
            a.max.z > b.min.z + eps &&
            a.min.z < b.max.z - eps
        )
    }
    const getLocalBounds = (root: THREE.Object3D) => {
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

    useEffect(() => {
        if (!isSelected || !groupRef.current) {
            setSelectionBounds(null)
            return
        }

        const box = getLocalBounds(groupRef.current)
        if (box.isEmpty()) return

        const center = new THREE.Vector3()
        const size = new THREE.Vector3()
        box.getCenter(center)
        box.getSize(size)

        setSelectionBounds({ center, size })
    }, [isSelected, type])
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
                const state = useVenueStore.getState()
                selectionBottomOffsets.current = {}

                if (state.selectedIds.length > 1) {
                    state.selectedIds.forEach((selectedId) => {
                        const obj = scene.getObjectByProperty('userData.id', selectedId)
                        if (!obj) return
                        const box = new THREE.Box3().setFromObject(obj)
                        selectionBottomOffsets.current[selectedId] = box.isEmpty() ? 0 : obj.position.y - box.min.y
                    })
                }

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

                        const downRay = new THREE.Raycaster()
                        const downDir = new THREE.Vector3(0, -1, 0)
                        const rayOrigin = new THREE.Vector3()

                        let deltaY = 0
                        const selectedItems = state.items.filter(i => state.selectedIds.includes(i.id))

                        if (type === 'trestle-table' && selectedItems.length === 1) {
                            const longAxis = new THREE.Vector3(1, 0, 0).applyAxisAngle(new THREE.Vector3(0, 1, 0), rotation[1]).normalize()
                            const shortAxis = new THREE.Vector3(-longAxis.z, 0, longAxis.x)
                            const selfPos = new THREE.Vector3(finalX, 0, finalZ)
                            let bestSnap: { x: number, z: number, score: number } | null = null

                            state.items.forEach((item) => {
                                if (item.id === id || item.type !== 'trestle-table') return

                                const otherAxis = new THREE.Vector3(1, 0, 0).applyAxisAngle(new THREE.Vector3(0, 1, 0), item.rotation[1]).normalize()
                                if (Math.abs(longAxis.dot(otherAxis)) < trestleSnap.angleCos) return

                                const otherShort = new THREE.Vector3(-otherAxis.z, 0, otherAxis.x)
                                const delta = new THREE.Vector3(selfPos.x - item.position[0], 0, selfPos.z - item.position[2])
                                const along = delta.dot(otherAxis)
                                const across = delta.dot(otherShort)
                                const alongErr = Math.abs(Math.abs(along) - trestleDims.length)
                                const acrossErr = Math.abs(across)

                                if (alongErr > trestleSnap.along || acrossErr > trestleSnap.across) return

                                const sign = along >= 0 ? 1 : -1
                                const snapPos = new THREE.Vector3(
                                    item.position[0] + otherAxis.x * trestleDims.length * sign,
                                    0,
                                    item.position[2] + otherAxis.z * trestleDims.length * sign
                                )
                                const score = alongErr + acrossErr
                                if (!bestSnap || score < bestSnap.score) {
                                    bestSnap = { x: snapPos.x, z: snapPos.z, score }
                                }
                            })

                            if (bestSnap) {
                                finalX = bestSnap.x
                                finalZ = bestSnap.z
                            }
                        }

                        if (selectedItems.length > 1) {
                            const anchor = selectedItems.reduce((lowest, item) => {
                                const lowestOffset = selectionBottomOffsets.current[lowest.id] ?? 0
                                const itemOffset = selectionBottomOffsets.current[item.id] ?? 0
                                const lowestBottom = lowest.position[1] - lowestOffset
                                const itemBottom = item.position[1] - itemOffset
                                if (itemBottom === lowestBottom && item.type === 'platform') return item
                                return itemBottom < lowestBottom ? item : lowest
                            }, selectedItems[0])

                            const anchorCandidateX = anchor.position[0] + (finalX - position[0])
                            const anchorCandidateZ = anchor.position[2] + (finalZ - position[2])
                            const anchorOffset = selectionBottomOffsets.current[anchor.id] ?? 0

                            rayOrigin.set(anchorCandidateX, stackRayOriginY, anchorCandidateZ)
                            downRay.set(rayOrigin, downDir)

                            let targetY = 0
                            const downIntersects = downRay.intersectObjects(scene.children, true)
                            for (const hit of downIntersects) {
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
                                if (state.selectedIds.includes(rootGroup.userData.id)) continue

                                const hitItem = state.items.find(i => i.id === rootGroup!.userData.id)
                                if (!hitItem) continue

                                if (anchor.type === 'platform' && hitItem.type !== 'platform') continue
                                if (anchor.type !== 'platform' && hitItem.type !== 'platform') continue

                                targetY = hit.point.y
                                break
                            }

                            const finalAnchorY = targetY + anchorOffset
                            deltaY = finalAnchorY - anchor.position[1]
                        } else {
                            // 3. Logic: Height Calculation via Downward Raycast (Lego Physics)
                            let targetY = 0
                            ensureObjectBounds()

                            const originY = type === 'chair'
                                ? (groupRef.current?.position.y ?? position[1]) + objectBounds.current.centerY
                                : stackRayOriginY

                            rayOrigin.set(finalX, originY, finalZ)
                            downRay.set(rayOrigin, downDir)

                        const downIntersects = downRay.intersectObjects(scene.children, true)
                        let platformRoot: THREE.Object3D | null = null

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
                            if (hitItem.type === 'platform') {
                                platformRoot = rootGroup
                            }
                            break // Closest valid surface for this ray
                        }

                        // 4. Apply offset
                        let finalY = targetY
                        if (type === 'chair' || type === 'platform') {
                            finalY += objectBounds.current.bottomOffset
                        }

                        if (type === 'trestle-table' && platformRoot && groupRef.current) {
                            const platformBox = getLocalBounds(platformRoot)
                            if (!platformBox.isEmpty()) {
                                const platformQuat = new THREE.Quaternion()
                                const trestleQuat = new THREE.Quaternion()
                                platformRoot.getWorldQuaternion(platformQuat)
                                groupRef.current.getWorldQuaternion(trestleQuat)

                                const relQuat = platformQuat.clone().invert().multiply(trestleQuat)
                                const axisX = new THREE.Vector3(1, 0, 0).applyQuaternion(relQuat)
                                const axisZ = new THREE.Vector3(0, 0, 1).applyQuaternion(relQuat)

                                const halfLen = trestleDims.length / 2
                                const halfDepth = trestleDims.depth / 2
                                const halfX = Math.abs(axisX.x) * halfLen + Math.abs(axisZ.x) * halfDepth
                                const halfZ = Math.abs(axisX.z) * halfLen + Math.abs(axisZ.z) * halfDepth

                                const localPos = platformRoot.worldToLocal(new THREE.Vector3(finalX, finalY, finalZ))
                                const minX = platformBox.min.x + halfX
                                const maxX = platformBox.max.x - halfX
                                const minZ = platformBox.min.z + halfZ
                                const maxZ = platformBox.max.z - halfZ

                                if (minX <= maxX) localPos.x = THREE.MathUtils.clamp(localPos.x, minX, maxX)
                                else localPos.x = (platformBox.min.x + platformBox.max.x) / 2

                                if (minZ <= maxZ) localPos.z = THREE.MathUtils.clamp(localPos.z, minZ, maxZ)
                                else localPos.z = (platformBox.min.z + platformBox.max.z) / 2

                                const clampedWorld = platformRoot.localToWorld(localPos)
                                finalX = clampedWorld.x
                                finalZ = clampedWorld.z
                            }
                        }

                        deltaY = finalY - position[1]
                    }

                        // 5. Update Store
                        const deltaX = finalX - position[0]
                        const deltaZ = finalZ - position[2]

                        if (Math.abs(deltaX) < 0.001 && Math.abs(deltaZ) < 0.001 && Math.abs(deltaY) < 0.001) return

                        const movingTrestles = selectedItems.filter(item => item.type === 'trestle-table')
                        if (movingTrestles.length > 0) {
                            const others = state.items.filter(item => item.type === 'trestle-table' && !state.selectedIds.includes(item.id))
                            const deltaVec = new THREE.Vector3(deltaX, deltaY, deltaZ)

                            for (const moving of movingTrestles) {
                                const movingObj = scene.getObjectByProperty('userData.id', moving.id)
                                if (!movingObj) continue
                                const movingBox = new THREE.Box3().setFromObject(movingObj).translate(deltaVec)

                                for (const other of others) {
                                    const otherObj = scene.getObjectByProperty('userData.id', other.id)
                                    if (!otherObj) continue
                                    const otherBox = new THREE.Box3().setFromObject(otherObj)

                                    if (boxesOverlap(movingBox, otherBox)) return
                                }
                            }
                        }

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
            {isSelected && selectionBounds && (
                <group position={[selectionBounds.center.x, selectionBounds.center.y, selectionBounds.center.z]} userData={{ skipBounds: true }}>
                    <mesh raycast={() => null}>
                        <boxGeometry args={[
                            selectionBounds.size.x * 1.03,
                            selectionBounds.size.y * 1.03,
                            selectionBounds.size.z * 1.03
                        ]} />
                        <meshStandardMaterial
                            color="#8b7bff"
                            emissive="#8b7bff"
                            emissiveIntensity={0.9}
                            transparent
                            opacity={0.12}
                            depthWrite={false}
                        />
                    </mesh>
                    <lineSegments raycast={() => null}>
                        <edgesGeometry args={[new THREE.BoxGeometry(
                            selectionBounds.size.x * 1.05,
                            selectionBounds.size.y * 1.05,
                            selectionBounds.size.z * 1.05
                        )]} />
                        <lineBasicMaterial color="#b7abff" transparent opacity={0.95} />
                    </lineSegments>
                </group>
            )}
        </group>
    )
}
