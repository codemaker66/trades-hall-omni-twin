'use client'

import React, { useRef, useEffect, useState } from 'react'
import { TransformControls } from '@react-three/drei'
import { useVenueStore } from '../../store'
import * as THREE from 'three'
import { _box3A, _yAxis } from './threePool'

interface SelectionGizmoProps {
    itemRefs: React.MutableRefObject<{ [id: string]: THREE.Group }>
}

type SelectionUpdate = {
    id: string
    changes: {
        position: [number, number, number]
        rotation: [number, number, number]
    }
}

const setMaterialColor = (material: THREE.Material, color: string) => {
    const colorMaterial = material as THREE.Material & { color?: THREE.Color }
    if (colorMaterial.color) {
        colorMaterial.color.set(color)
    }
}

export const SelectionGizmo = ({ itemRefs }: SelectionGizmoProps) => {
    const selectedIds = useVenueStore((state) => state.selectedIds)
    const updateItems = useVenueStore((state) => state.updateItems)
    const beginHistoryBatch = useVenueStore((state) => state.beginHistoryBatch)
    const endHistoryBatch = useVenueStore((state) => state.endHistoryBatch)
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
                            obj.traverse((child: THREE.Object3D) => {
                                const axisNode = child as THREE.Object3D & {
                                    isMesh?: boolean
                                    isLine?: boolean
                                    material?: THREE.Material | THREE.Material[]
                                    name: string
                                    visible: boolean
                                }

                                if (axisNode.isMesh || axisNode.isLine) {
                                    // Identify axis by color or name (usually name is 'X', 'Y', 'Z', 'XY' etc but sometimes internal names)
                                    // Three.js TransformControls assigns names like 'X', 'Y', 'Z', 'E', 'ArrowX', etc.
                                    const materials = Array.isArray(axisNode.material)
                                        ? axisNode.material
                                        : axisNode.material
                                            ? [axisNode.material]
                                            : []

                                    if (axisNode.name.includes('X')) {
                                        if (axisNode.name === 'XZ' || axisNode.name === 'XY' || axisNode.name === 'YZ' || axisNode.name.includes('plane')) {
                                            // Hide Plane Handles ("Garish Squares")
                                            materials.forEach((material) => {
                                                material.visible = false
                                            })
                                            axisNode.visible = false
                                        } else {
                                            materials.forEach((material) => {
                                                setMaterialColor(material, '#ffd700') // Bright Gold
                                                material.opacity = 1
                                            })
                                        }
                                    } else if (axisNode.name.includes('Z')) {
                                        if (axisNode.name === 'XZ' || axisNode.name === 'YZ' || axisNode.name.includes('plane')) {
                                            materials.forEach((material) => {
                                                material.visible = false
                                            })
                                            axisNode.visible = false
                                        } else {
                                            materials.forEach((material) => {
                                                setMaterialColor(material, '#a0a0a0') // Polished Iron
                                                material.opacity = 1
                                            })
                                        }
                                    } else if (axisNode.name.includes('Y') || axisNode.name === 'XZ' || axisNode.name === 'XY' || axisNode.name === 'YZ') {
                                        // Hide Y and any remaining Planes
                                        materials.forEach((material) => {
                                            material.visible = false
                                        })
                                        axisNode.visible = false
                                    } else {
                                        // Center / Other
                                        materials.forEach((material) => {
                                            setMaterialColor(material, '#ffffff')
                                        })
                                    }
                                }
                            })
                        }
                    }}
                    object={pivotRef.current ?? undefined}
                    mode={transformMode}
                    showY={false} // Lock vertical movement
                    size={1.2} // Slightly larger
                    rotationSnap={snappingEnabled ? Math.PI / 4 : null}

                    onMouseDown={() => {
                        beginHistoryBatch()
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
                                    // Rotate the offset vector by pivot rotation (reuse _yAxis from pool)
                                    const rotatedOffset = offset.clone().applyAxisAngle(_yAxis, pivotRotY)
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
                            _box3A.setFromObject(obj)
                            if (!_box3A.isEmpty() && _box3A.min.y < minBottom) minBottom = _box3A.min.y
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
                        const updates = selectedIds.map<SelectionUpdate | null>(id => {
                            const obj = itemRefs.current[id]
                            if (!obj) return null
                            return {
                                id,
                                changes: {
                                    position: [obj.position.x, obj.position.y, obj.position.z] as [number, number, number],
                                    rotation: [obj.rotation.x, obj.rotation.y, obj.rotation.z] as [number, number, number]
                                }
                            }
                        }).filter((update): update is SelectionUpdate => update !== null)

                        updateItems(updates)
                        endHistoryBatch()
                    }}
                />
            )}
        </>
    )
}
