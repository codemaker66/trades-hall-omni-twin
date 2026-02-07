import React, { useRef, useEffect, useMemo } from 'react'
import { useVenueStore, FurnitureType } from '../../store'
import { RoundTable6ft, TrestleTable6ft, Chair, Platform } from './Furniture'
import * as THREE from 'three'
import { useThree } from '@react-three/fiber'
import { _raycaster, _raycaster2, _plane, _vec3A, _vec3B, _vec3C, _vec3D, _vec3E, _box3A, _box3B, _quatA, _quatB, _downDir, _yAxis } from './threePool'
import { FURNITURE, PHYSICS, SNAP, SELECTION } from '../../config/scene'
import { snapToGrid, computeTrestleSnap, boxesOverlap, clampToPlatformBounds, type TrestleSnapConfig } from './engine/drag'
import { applySelectionModifier, resolveClickTargetIds, findDragAnchor } from './engine/selection'
import { findGroupById, raycastDownForStack, getLocalBounds } from './engine/raycast'

interface InteractiveFurnitureProps {
    id: string
    groupId?: string
    type: FurnitureType
    position: [number, number, number]
    rotation: [number, number, number]
    onRegister: (id: string, ref: THREE.Group | null) => void
}

const trestleSnapConfig: TrestleSnapConfig = {
    length: FURNITURE['trestle-table'].length,
    depth: FURNITURE['trestle-table'].depth,
    alongTolerance: SNAP.trestle.alongTolerance,
    acrossTolerance: SNAP.trestle.acrossTolerance,
    angleCos: SNAP.trestle.angleCosThreshold,
}

type DragSession = {
    active: boolean
    initialPositions: Map<string, [number, number, number]>
    groups: Map<string, THREE.Group>
    originX: number
    originZ: number
    offsetX: number
    offsetZ: number
    bottomOffsets: Map<string, number>
}

export const InteractiveFurniture = React.memo(({ id, groupId, type, position, rotation, onRegister }: InteractiveFurnitureProps) => {
    const { scene } = useThree()
    const selectedIds = useVenueStore((state) => state.selectedIds)
    const setSelection = useVenueStore((state) => state.setSelection)
    const setIsDragging = useVenueStore((state) => state.setIsDragging)
    const snappingEnabled = useVenueStore((state) => state.snappingEnabled)
    const snapGrid = useVenueStore((state) => state.snapGrid)

    const isSelected = selectedIds.includes(id)

    const Component =
        type === 'round-table' ? RoundTable6ft :
            type === 'trestle-table' ? TrestleTable6ft :
                type === 'chair' ? Chair :
                    Platform

    const isDraggingRef = useRef(false)
    const groupRef = useRef<THREE.Group>(null)
    const objectBounds = useRef({ centerY: 0, bottomOffset: 0, halfX: 0, halfZ: 0, ready: false })

    const drag = useRef<DragSession>({
        active: false,
        initialPositions: new Map(),
        groups: new Map(),
        originX: 0, originZ: 0,
        offsetX: 0, offsetZ: 0,
        bottomOffsets: new Map()
    })

    useEffect(() => {
        if (groupRef.current) onRegister(id, groupRef.current)
        return () => { onRegister(id, null) }
    }, [id, onRegister])

    useEffect(() => {
        if (groupRef.current && !isDraggingRef.current) {
            groupRef.current.position.set(position[0], position[1], position[2])
            groupRef.current.rotation.set(rotation[0], rotation[1], rotation[2])
        }
    }, [position, rotation])

    const openChairPrompt = useVenueStore((state) => state.openChairPrompt)

    const selectionHighlightGeom = useMemo(() => {
        if (!isSelected) return null
        const sb = FURNITURE[type]
        return {
            box: new THREE.BoxGeometry(sb.selectionSize[0] * SELECTION.boxScale, sb.selectionSize[1] * SELECTION.boxScale, sb.selectionSize[2] * SELECTION.boxScale),
            edges: new THREE.BoxGeometry(sb.selectionSize[0] * SELECTION.edgesScale, sb.selectionSize[1] * SELECTION.edgesScale, sb.selectionSize[2] * SELECTION.edgesScale)
        }
    }, [type, isSelected])

    const selectionBounds = isSelected ? { center: FURNITURE[type].selectionCenter, size: FURNITURE[type].selectionSize } : null

    const ensureObjectBounds = () => {
        if (objectBounds.current.ready || !groupRef.current) return

        _box3A.setFromObject(groupRef.current)
        if (_box3A.isEmpty()) return

        const size = _vec3A
        const center = _vec3B
        _box3A.getSize(size)
        _box3A.getCenter(center)

        objectBounds.current.centerY = center.y - groupRef.current.position.y
        objectBounds.current.bottomOffset = groupRef.current.position.y - _box3A.min.y
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
                const isCtrl = e.ctrlKey || e.metaKey
                const idsToTarget = resolveClickTargetIds(id, groupId, isCtrl, state.items)
                const isMultiSelect = e.shiftKey || isCtrl
                const newSelection = applySelectionModifier(state.selectedIds, idsToTarget, isMultiSelect)
                setSelection(newSelection)
            }}
            onPointerDown={(e) => {
                if (e.button !== 0) return
                e.stopPropagation()

                const target = e.target as HTMLElement
                target.setPointerCapture(e.pointerId)

                isDraggingRef.current = true
                setIsDragging(true)
                useVenueStore.getState().beginHistoryBatch()
                ensureObjectBounds()

                const state = useVenueStore.getState()
                const selectedItems = state.items.filter(i => state.selectedIds.includes(i.id))

                const d = drag.current
                d.active = true
                d.initialPositions.clear()
                d.groups.clear()
                d.bottomOffsets.clear()
                d.originX = position[0]
                d.originZ = position[2]

                for (const item of selectedItems) {
                    d.initialPositions.set(item.id, [item.position[0], item.position[1], item.position[2]])
                    const group = item.id === id
                        ? groupRef.current
                        : findGroupById(scene, item.id)
                    if (group) d.groups.set(item.id, group)
                }

                if (selectedItems.length > 1) {
                    for (const item of selectedItems) {
                        const group = d.groups.get(item.id)
                        if (!group) continue
                        _box3A.setFromObject(group)
                        d.bottomOffsets.set(item.id, _box3A.isEmpty() ? 0 : group.position.y - _box3A.min.y)
                    }
                }

                _plane.normal.set(0, 1, 0)
                _plane.constant = 0
                _raycaster.setFromCamera(e.pointer, e.camera)
                _raycaster.ray.intersectPlane(_plane, _vec3A)

                if (_vec3A) {
                    d.offsetX = _vec3A.x - position[0]
                    d.offsetZ = _vec3A.z - position[2]
                }
            }}
            onPointerUp={(e) => {
                if (!isDraggingRef.current) return
                e.stopPropagation()
                const target = e.target as HTMLElement
                target.releasePointerCapture(e.pointerId)

                isDraggingRef.current = false
                const d = drag.current
                d.active = false

                const state = useVenueStore.getState()
                const updates: { id: string, changes: { position: [number, number, number] } }[] = []
                for (const [itemId, group] of d.groups) {
                    updates.push({
                        id: itemId,
                        changes: { position: [group.position.x, group.position.y, group.position.z] }
                    })
                }

                if (updates.length > 0) {
                    state.updateItems(updates, { recordHistory: false })
                }

                setIsDragging(false)
                useVenueStore.getState().endHistoryBatch()

                d.initialPositions.clear()
                d.groups.clear()
                d.bottomOffsets.clear()
            }}
            onPointerMove={(e) => {
                if (!isDraggingRef.current) return
                e.stopPropagation()

                const d = drag.current
                if (!d.active) return

                const state = useVenueStore.getState()

                // 1. Raycast to floor plane
                _raycaster.setFromCamera(e.pointer, e.camera)
                _plane.normal.set(0, 1, 0)
                _plane.constant = 0
                const hit = _raycaster.ray.intersectPlane(_plane, _vec3A)
                if (!hit) return

                const rawNewX = hit.x - d.offsetX
                const rawNewZ = hit.z - d.offsetZ

                // 2. Snapping
                let finalX = rawNewX
                let finalZ = rawNewZ

                if (snappingEnabled && snapGrid > 0) {
                    const snapped = snapToGrid(finalX, finalZ, snapGrid)
                    finalX = snapped.x
                    finalZ = snapped.z
                }

                let deltaY = 0
                const selectedItems = state.items.filter(i => state.selectedIds.includes(i.id))

                // Trestle end-to-end snapping
                if (type === 'trestle-table' && selectedItems.length === 1) {
                    const otherTrestles = state.items
                        .filter(item => item.id !== id && item.type === 'trestle-table')
                    const snap = computeTrestleSnap(
                        finalX, finalZ, rotation[1],
                        otherTrestles, trestleSnapConfig,
                        _vec3B, _vec3C, _vec3D, _yAxis,
                    )
                    if (snap) {
                        finalX = snap.x
                        finalZ = snap.z
                    }
                }

                // 3. Height calculation
                const excludeIds = new Set(state.selectedIds)

                if (selectedItems.length > 1) {
                    const anchor = findDragAnchor(selectedItems, d.bottomOffsets)
                    const anchorInitial = d.initialPositions.get(anchor.id)
                    if (anchorInitial) {
                        const anchorCandidateX = anchorInitial[0] + (finalX - d.originX)
                        const anchorCandidateZ = anchorInitial[2] + (finalZ - d.originZ)
                        const anchorOffset = d.bottomOffsets.get(anchor.id) ?? 0

                        _vec3B.set(anchorCandidateX, PHYSICS.stackRayOriginY, anchorCandidateZ)
                        const stackResult = raycastDownForStack(
                            _vec3B, _downDir, _raycaster2, scene,
                            excludeIds, anchor.type, state.items,
                        )

                        const finalAnchorY = stackResult.y + anchorOffset
                        deltaY = finalAnchorY - anchorInitial[1]
                    }
                } else {
                    ensureObjectBounds()

                    const originY = type === 'chair'
                        ? (groupRef.current?.position.y ?? position[1]) + objectBounds.current.centerY
                        : PHYSICS.stackRayOriginY

                    _vec3B.set(finalX, originY, finalZ)
                    excludeIds.add(id)
                    const stackResult = raycastDownForStack(
                        _vec3B, _downDir, _raycaster2, scene,
                        excludeIds, type, state.items,
                    )

                    let finalY = stackResult.y
                    if (type === 'chair' || type === 'platform') {
                        finalY += objectBounds.current.bottomOffset
                    }

                    // Platform clamping for trestle tables
                    if (type === 'trestle-table' && stackResult.platformRoot && groupRef.current) {
                        const platformBox = getLocalBounds(stackResult.platformRoot)
                        const clamped = clampToPlatformBounds(
                            finalX, finalY, finalZ,
                            stackResult.platformRoot, groupRef.current, platformBox,
                            trestleSnapConfig.length, trestleSnapConfig.depth,
                            _quatA, _quatB, _vec3C, _vec3D, _vec3E,
                        )
                        if (clamped) {
                            finalX = clamped.x
                            finalZ = clamped.z
                        }
                    }

                    deltaY = finalY - position[1]
                }

                // 4. Compute deltas from origin
                const deltaX = finalX - d.originX
                const deltaZ = finalZ - d.originZ

                if (Math.abs(deltaX) < PHYSICS.movementThreshold && Math.abs(deltaZ) < PHYSICS.movementThreshold && Math.abs(deltaY) < PHYSICS.movementThreshold) return

                // 5. Collision detection for trestle tables
                const movingTrestles = selectedItems.filter(item => item.type === 'trestle-table')
                if (movingTrestles.length > 0) {
                    const others = state.items.filter(item => item.type === 'trestle-table' && !state.selectedIds.includes(item.id))

                    for (const moving of movingTrestles) {
                        const movingGroup = d.groups.get(moving.id)
                        if (!movingGroup) continue
                        const initPos = d.initialPositions.get(moving.id)
                        if (!initPos) continue
                        _box3A.setFromObject(movingGroup)
                        const visualDelta = _vec3C.set(
                            initPos[0] + deltaX - movingGroup.position.x,
                            initPos[1] + deltaY - movingGroup.position.y,
                            initPos[2] + deltaZ - movingGroup.position.z
                        )
                        _box3A.translate(visualDelta)

                        for (const other of others) {
                            const otherGroup = findGroupById(scene, other.id)
                            if (!otherGroup) continue
                            _box3B.setFromObject(otherGroup)

                            if (boxesOverlap(_box3A, _box3B, PHYSICS.collisionEpsilon)) return
                        }
                    }
                }

                // 6. DIRECTLY MOVE Three.js groups (NO store update!)
                for (const [itemId, group] of d.groups) {
                    const initPos = d.initialPositions.get(itemId)
                    if (!initPos) continue
                    group.position.set(
                        initPos[0] + deltaX,
                        initPos[1] + deltaY,
                        initPos[2] + deltaZ
                    )
                }
            }}
            userData={{ id }}
        >
            <Component />

            {isSelected && selectionBounds && selectionHighlightGeom && (
                <group position={selectionBounds.center} userData={{ skipBounds: true }}>
                    <mesh raycast={() => null} geometry={selectionHighlightGeom.box}>
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
                        <edgesGeometry args={[selectionHighlightGeom.edges]} />
                        <lineBasicMaterial color="#b7abff" transparent opacity={0.95} />
                    </lineSegments>
                </group>
            )}
        </group>
    )
})
