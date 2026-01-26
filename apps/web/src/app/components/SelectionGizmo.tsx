'use client'

import React, { useRef, useEffect, useState } from 'react'
import { TransformControls } from '@react-three/drei'
import { useVenueStore } from '../../store'
import * as THREE from 'three'

interface SelectionGizmoProps {
    itemRefs: React.MutableRefObject<{ [id: string]: THREE.Group }>
}

export const SelectionGizmo = ({ itemRefs }: SelectionGizmoProps) => {
    const selectedIds = useVenueStore((state) => state.selectedIds)
    const updateItems = useVenueStore((state) => state.updateItems)
    const snappingEnabled = useVenueStore((state) => state.snappingEnabled)
    const snapGrid = useVenueStore((state) => state.snapGrid)
    const transformMode = useVenueStore((state) => state.transformMode)

    const pivotRef = useRef<THREE.Group>(null)
    const [gizmoVisible, setGizmoVisible] = useState(false)
    const pivotBaseY = useRef(0)
    const initialYs = useRef<{ [id: string]: number }>({})

    // Store initial offsets relative to pivot when drag starts
    const initialPositions = useRef<{ [id: string]: THREE.Vector3 }>({})
    // Store initial rotations relative to pivot
    const initialRotations = useRef<{ [id: string]: THREE.Euler }>({})

    // Update Pivot Position when selection changes
    useEffect(() => {
        if (selectedIds.length === 0 || !pivotRef.current) {
            setGizmoVisible(false)
            return
        }

        // Calculate Centroid
        const centroid = new THREE.Vector3()
        let count = 0

        selectedIds.forEach(id => {
            const obj = itemRefs.current[id]
            if (obj) {
                centroid.add(obj.position)
                count++
            }
        })

        if (count > 0) {
            centroid.divideScalar(count)
            pivotRef.current.position.copy(centroid)
            pivotRef.current.rotation.set(0, 0, 0)
            setGizmoVisible(true)
        } else {
            setGizmoVisible(false)
        }
    }, [selectedIds, itemRefs])

    return (
        <>
            {/* Dummy object to attach controls to */}
            <group ref={pivotRef} />

            {gizmoVisible && (
                <TransformControls
                    ref={(obj) => {
                        if (obj) {
                            // Reskin Gizmo to Medieval Colors
                            // X-Axis (Red) -> Gold
                            // Z-Axis (Blue) -> Iron/Steel
                            // Y-Axis (Green) -> Hidden anyway
                            // obj is the TransformControls instance, which contains the gizmo as children
                            obj.traverse((child: any) => {
                                if (child.isMesh || child.isLine) {
                                    // Identify axis by color or name (usually name is 'X', 'Y', 'Z', 'XY' etc but sometimes internal names)
                                    // Three.js TransformControls assigns names like 'X', 'Y', 'Z', 'E', 'ArrowX', etc.

                                    if (child.name.includes('X')) {
                                        if (child.name === 'XZ' || child.name === 'XY' || child.name === 'YZ' || child.name.includes('plane')) {
                                            // Hide Plane Handles ("Garish Squares")
                                            if (child.material) {
                                                child.material.visible = false
                                                child.visible = false
                                            }
                                        } else if (child.material) {
                                            child.material.color.set('#ffd700') // Bright Gold
                                            child.material.opacity = 1
                                        }
                                    } else if (child.name.includes('Z')) {
                                        if (child.name === 'XZ' || child.name === 'YZ' || child.name.includes('plane')) {
                                            if (child.material) {
                                                child.material.visible = false
                                                child.visible = false
                                            }
                                        } else if (child.material) {
                                            child.material.color.set('#a0a0a0') // Polished Iron
                                            child.material.opacity = 1
                                        }
                                    } else if (child.name.includes('Y') || child.name === 'XZ' || child.name === 'XY' || child.name === 'YZ') {
                                        // Hide Y and any remaining Planes
                                        if (child.material) {
                                            child.material.visible = false
                                            child.visible = false
                                        }
                                    } else {
                                        // Center / Other
                                        if (child.material) child.material.color.set('#ffffff')
                                    }
                                }
                            })
                        }
                    }}
                    object={pivotRef}
                    mode={transformMode}
                    showY={false} // Lock vertical movement
                    size={1.2} // Slightly larger
                    rotationSnap={snappingEnabled ? Math.PI / 4 : null}

                    onMouseDown={() => {
                        // Capture initial relative positions
                        const pivotPos = pivotRef.current?.position.clone() || new THREE.Vector3()
                        initialPositions.current = {}
                        initialRotations.current = {}
                        pivotBaseY.current = pivotPos.y
                        initialYs.current = {}

                        selectedIds.forEach(id => {
                            const obj = itemRefs.current[id]
                            if (obj) {
                                // Store offset: ObjPos - PivotPos
                                initialPositions.current[id] = obj.position.clone().sub(pivotPos)
                                // Store initial rotation
                                initialRotations.current[id] = obj.rotation.clone()
                                initialYs.current[id] = obj.position.y
                            }
                        })
                    }}

                    onObjectChange={() => {
                        // Sync objects to Pivot
                        if (!pivotRef.current) return

                        // Lock Pivot Y to its starting value to avoid vertical drift
                        pivotRef.current.position.y = pivotBaseY.current

                        if (transformMode === 'translate' && snappingEnabled && snapGrid > 0) {
                            pivotRef.current.position.x = Math.round(pivotRef.current.position.x / snapGrid) * snapGrid
                            pivotRef.current.position.z = Math.round(pivotRef.current.position.z / snapGrid) * snapGrid
                        }
                        const newPivotPos = pivotRef.current.position
                        const pivotRotY = pivotRef.current.rotation.y

                        selectedIds.forEach(id => {
                            const obj = itemRefs.current[id]
                            const offset = initialPositions.current[id]
                            const baseRot = initialRotations.current[id]

                            if (obj && offset && baseRot) {
                                if (transformMode === 'rotate') {
                                    // Rotate the offset vector by pivot rotation
                                    const rotatedOffset = offset.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), pivotRotY)
                                    obj.position.copy(newPivotPos).add(rotatedOffset)

                                    // Rotate object itself
                                    obj.rotation.y = baseRot.y + pivotRotY
                                } else {
                                    // Translate Mode
                                    obj.position.copy(newPivotPos).add(offset)
                                }

                                if (initialYs.current[id] !== undefined) {
                                    obj.position.y = initialYs.current[id]
                                }
                            }
                        })

                        // Clamp to floor after positions are applied (handles plane-drag drift)
                        let minBottom = Infinity
                        selectedIds.forEach(id => {
                            const obj = itemRefs.current[id]
                            if (!obj) return
                            const box = new THREE.Box3().setFromObject(obj)
                            if (!box.isEmpty() && box.min.y < minBottom) minBottom = box.min.y
                        })

                        if (minBottom < 0) {
                            const adjustY = -minBottom
                            pivotRef.current.position.y += adjustY
                            pivotBaseY.current = pivotRef.current.position.y

                            selectedIds.forEach(id => {
                                const obj = itemRefs.current[id]
                                if (obj) obj.position.y += adjustY
                                if (initialYs.current[id] !== undefined) initialYs.current[id] += adjustY
                            })
                        }
                    }}

                    onMouseUp={() => {
                        // Commit changes
                        const updates = selectedIds.map(id => {
                            const obj = itemRefs.current[id]
                            if (!obj) return null
                            return {
                                id,
                                changes: {
                                    position: [obj.position.x, obj.position.y, obj.position.z] as [number, number, number],
                                    rotation: [obj.rotation.x, obj.rotation.y, obj.rotation.z] as [number, number, number]
                                }
                            }
                        }).filter(Boolean) as any[]

                        updateItems(updates)
                    }}
                />
            )}
        </>
    )
}
