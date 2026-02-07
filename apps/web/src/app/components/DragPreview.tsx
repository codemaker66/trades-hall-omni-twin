import { useFrame, useThree } from '@react-three/fiber'
import { useRef, useEffect, useState } from 'react'
import { useVenueStore } from '../../store'
import { useShallow } from 'zustand/react/shallow'
import * as THREE from 'three'
import { RoundTable6ft, TrestleTable6ft, Chair, Platform } from './Furniture'
import { _raycaster, _raycaster2, _plane, _vec3A, _vec3B, _vec3C, _vec2A, _downDir } from './threePool'
import { PHYSICS, SNAP, GHOST } from '../../config/scene'
import { snapChairPosition, snapToGrid, getChairSpacing, computeGridPositions } from './engine/drag'
import { findRootItemGroup } from './engine/raycast'
import React from 'react'

type EmissiveMaterial = THREE.Material & {
    emissive: THREE.Color
    emissiveIntensity: number
}

export const DragPreview = () => {
    const { draggedItemType, snapGrid, snappingEnabled } = useVenueStore(
        useShallow((state) => ({
            draggedItemType: state.draggedItemType,
            snapGrid: state.snapGrid,
            snappingEnabled: state.snappingEnabled,
        }))
    )
    const setDraggedItem = useVenueStore((state) => state.setDraggedItem)
    const addItem = useVenueStore((state) => state.addItem)
    const beginHistoryBatch = useVenueStore((state) => state.beginHistoryBatch)
    const endHistoryBatch = useVenueStore((state) => state.endHistoryBatch)

    const groupRef = useRef<THREE.Group>(null)
    const objectBounds = useRef({ centerY: 0, bottomOffset: 0, halfX: 0, halfZ: 0, ready: false })
    const areaDragRef = useRef<{ active: boolean, start: THREE.Vector3 | null, end: THREE.Vector3 | null }>({
        active: false,
        start: null,
        end: null
    })
    const [previewPositions, setPreviewPositions] = useState<[number, number, number][]>([])
    const [position, setPosition] = useState<[number, number, number] | null>(null)
    const [chairRotation, setChairRotation] = useState(SNAP.defaultChairRotation)

    const { camera, raycaster, pointer, scene, gl } = useThree()

    const Component =
        draggedItemType === 'round-table' ? RoundTable6ft :
            draggedItemType === 'trestle-table' ? TrestleTable6ft :
                draggedItemType === 'chair' ? Chair :
                    draggedItemType === 'platform' ? Platform : null

    useEffect(() => {
        objectBounds.current.ready = false
    }, [draggedItemType])

    const ensureObjectBounds = () => {
        if (objectBounds.current.ready || !groupRef.current) return

        _vec3A.set(0, 0, 0)
        _vec3B.set(0, 0, 0)
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

    const applyGhostMaterials = (root: THREE.Object3D) => {
        root.traverse((child) => {
            const mesh = child as THREE.Mesh
            if (!mesh.isMesh || !mesh.material) return

            const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
            materials.forEach((mat) => {
                if (!mat) return
                if ('emissive' in mat && 'emissiveIntensity' in mat) {
                    const emissiveMat = mat as EmissiveMaterial
                    emissiveMat.emissive = new THREE.Color('#7c6fff')
                    emissiveMat.emissiveIntensity = GHOST.emissiveIntensity
                }
                mat.transparent = true
                mat.opacity = GHOST.opacity
                mat.depthWrite = false
            })

            mesh.castShadow = false
            mesh.receiveShadow = false
            mesh.raycast = () => null
        })
    }

    const getPointFromEvent = (e: PointerEvent) => {
        const target = e.target as HTMLElement
        if (!target) return null
        const rect = target.getBoundingClientRect()
        if (!rect.width || !rect.height) return null
        const x = ((e.clientX - rect.left) / rect.width) * 2 - 1
        const y = -((e.clientY - rect.top) / rect.height) * 2 + 1
        _vec2A.set(x, y)
        raycaster.setFromCamera(_vec2A, camera)
        _plane.normal.set(0, 1, 0)
        _plane.constant = 0
        return raycaster.ray.intersectPlane(_plane, _vec3A) ? _vec3A.clone() : null
    }

    const getChairY = (x: number, z: number) => {
        ensureObjectBounds()
        const chairCenterY = (groupRef.current?.position.y ?? 0) + objectBounds.current.centerY
        _vec3C.set(x, chairCenterY, z)
        _raycaster2.set(_vec3C, _downDir)

        const downIntersects = _raycaster2.intersectObjects(scene.children, true)
        const state = useVenueStore.getState()
        let targetY = 0

        for (const hit of downIntersects) {
            const rootGroup = findRootItemGroup(hit.object)
            if (!rootGroup?.userData.id) continue

            const hitItem = state.items.find(i => i.id === rootGroup!.userData.id)
            if (!hitItem) continue

            if (hitItem.type === 'platform') {
                targetY = hit.point.y
                break
            }
        }

        return targetY + objectBounds.current.bottomOffset
    }

    const buildPreviewGrid = (start: THREE.Vector3, end: THREE.Vector3) => {
        const spacing = getChairSpacing(snapGrid, snappingEnabled)
        const gridPositions = computeGridPositions(start.x, start.z, end.x, end.z, spacing)
        return gridPositions.map(({ x, z }) => {
            const y = getChairY(x, z)
            return [x, y, z] as [number, number, number]
        })
    }

    // Update Ghost Position
    useFrame(() => {
        if (!draggedItemType) return

        raycaster.setFromCamera(pointer, camera)
        _plane.normal.set(0, 1, 0)
        _plane.constant = 0
        const target = raycaster.ray.intersectPlane(_plane, _vec3A)

        if (target) {
            if (draggedItemType === 'chair') {
                const snapped = snapChairPosition(target.x, target.z, snapGrid, snappingEnabled)
                target.x = snapped.x
                target.z = snapped.z
            } else if (snappingEnabled) {
                const snapped = snapToGrid(target.x, target.z, snapGrid)
                target.x = snapped.x
                target.z = snapped.z
            }

            let targetY = 0

            if (draggedItemType === 'chair' || draggedItemType === 'platform') {
                ensureObjectBounds()

                const state = useVenueStore.getState()

                const originY = draggedItemType === 'chair'
                    ? (groupRef.current?.position.y ?? 0) + objectBounds.current.centerY
                    : PHYSICS.stackRayOriginY

                _vec3B.set(target.x, originY, target.z)
                _raycaster2.set(_vec3B, _downDir)

                const downIntersects = _raycaster2.intersectObjects(scene.children, true)

                for (const hit of downIntersects) {
                    const rootGroup = findRootItemGroup(hit.object)
                    if (!rootGroup?.userData.id) continue

                    const hitItem = state.items.find(i => i.id === rootGroup!.userData.id)
                    if (!hitItem) continue

                    if (draggedItemType === 'platform' && hitItem.type !== 'platform') continue
                    if (draggedItemType !== 'platform' && hitItem.type !== 'platform') continue

                    targetY = hit.point.y
                    break
                }

                targetY += objectBounds.current.bottomOffset
            }

            target.y = targetY

            setPosition([target.x, target.y, target.z])
            if (groupRef.current) {
                groupRef.current.position.set(target.x, target.y, target.z)
                if (draggedItemType === 'chair') {
                    groupRef.current.rotation.y = chairRotation
                } else {
                    groupRef.current.rotation.y = 0
                }
            }
        } else {
            setPosition(null)
        }
    })

    // Handle Drop (Global Pointer Up)
    useEffect(() => {
        if (groupRef.current && draggedItemType) {
            applyGhostMaterials(groupRef.current)
        }

        const cancelChairTool = () => {
            areaDragRef.current.active = false
            areaDragRef.current.start = null
            areaDragRef.current.end = null
            setPreviewPositions([])
            setPosition(null)
            setDraggedItem(null)
        }

        const handleKeyDown = (e: KeyboardEvent) => {
            if (draggedItemType !== 'chair') return
            const key = e.key.toLowerCase()
            if (key === 'escape') {
                e.preventDefault()
                cancelChairTool()
                return
            }
            if (key !== 'q' && key !== 'e') return
            e.preventDefault()
            setChairRotation((prev) => (key === 'q' ? prev - SNAP.rotationStep : prev + SNAP.rotationStep))
        }

        const handlePointerDown = (e: PointerEvent) => {
            if (draggedItemType !== 'chair') return
            if (e.button === 2) {
                e.preventDefault()
                cancelChairTool()
                return
            }
            if (e.button !== 0) return
            const isCanvas = (e.target as HTMLElement).nodeName === 'CANVAS'
            if (!isCanvas) return
            const hit = getPointFromEvent(e)
            if (!hit) return
            areaDragRef.current.active = true
            areaDragRef.current.start = hit.clone()
            areaDragRef.current.end = hit.clone()
            setPreviewPositions(buildPreviewGrid(hit, hit))
        }

        const handlePointerMove = (e: PointerEvent) => {
            if (!areaDragRef.current.active) return
            const hit = getPointFromEvent(e)
            if (!hit) return
            areaDragRef.current.end = hit.clone()
            if (areaDragRef.current.start) {
                setPreviewPositions(buildPreviewGrid(areaDragRef.current.start, hit))
            }
        }

        const handlePointerUp = (e: PointerEvent) => {
            if (!draggedItemType) return

            const isCanvas = (e.target as HTMLElement).nodeName === 'CANVAS'

            if (draggedItemType === 'chair' && areaDragRef.current.active) {
                areaDragRef.current.active = false
                const start = areaDragRef.current.start
                const end = areaDragRef.current.end
                areaDragRef.current.start = null
                areaDragRef.current.end = null
                setPreviewPositions([])

                if (isCanvas && start && end) {
                    beginHistoryBatch()
                    try {
                        const spacing = getChairSpacing(snapGrid, snappingEnabled)
                        const gridPositions = computeGridPositions(start.x, start.z, end.x, end.z, spacing)

                        for (const { x, z } of gridPositions) {
                            const y = getChairY(x, z)
                            addItem('chair', [x, y, z], [0, chairRotation, 0], undefined, { recordHistory: false })
                        }
                    } finally {
                        endHistoryBatch()
                    }
                }

                setPosition(null)
                return
            }

            if (draggedItemType === 'chair' && !areaDragRef.current.active && !isCanvas) {
                return
            }

            if (isCanvas && position) {
                if (draggedItemType === 'chair') {
                    addItem('chair', position, [0, chairRotation, 0])
                    setPosition(null)
                    setPreviewPositions([])
                } else {
                    addItem(draggedItemType, position)
                    setDraggedItem(null)
                    setPosition(null)
                    setPreviewPositions([])
                }
            } else {
                if (draggedItemType !== 'chair') {
                    setDraggedItem(null)
                }
                setPosition(null)
                setPreviewPositions([])
            }
        }

        const handleContextMenu = (e: MouseEvent) => {
            if (draggedItemType !== 'chair') return
            e.preventDefault()
            cancelChairTool()
        }

        const element = gl.domElement
        element.addEventListener('pointerdown', handlePointerDown)
        element.addEventListener('pointermove', handlePointerMove)
        element.addEventListener('contextmenu', handleContextMenu)
        window.addEventListener('keydown', handleKeyDown)
        window.addEventListener('pointerup', handlePointerUp)
        return () => {
            element.removeEventListener('pointerdown', handlePointerDown)
            element.removeEventListener('pointermove', handlePointerMove)
            element.removeEventListener('contextmenu', handleContextMenu)
            window.removeEventListener('keydown', handleKeyDown)
            window.removeEventListener('pointerup', handlePointerUp)
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [draggedItemType, position, addItem, setDraggedItem, snappingEnabled, snapGrid, chairRotation, gl])

    if (!draggedItemType || !Component) return null

    return (
        <group>
            <group ref={groupRef}>
                <Component />
                <mesh position={[0, 1, 0]} />

                <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.05, 0]}>
                    <ringGeometry args={[GHOST.ringInner, GHOST.ringOuter, GHOST.ringSegments]} />
                    <meshBasicMaterial color="#6366f1" transparent opacity={0.6} side={THREE.DoubleSide} />
                </mesh>
            </group>

            {draggedItemType === 'chair' && previewPositions.map((pos, index) => (
                <GhostChairInstance key={`ghost-${index}`} position={pos} rotationY={chairRotation} applyGhostMaterials={applyGhostMaterials} />
            ))}
        </group>
    )
}

const GhostChairInstance = React.memo(({ position, rotationY, applyGhostMaterials }: { position: [number, number, number], rotationY: number, applyGhostMaterials: (root: THREE.Object3D) => void }) => {
    const ref = useRef<THREE.Group>(null)

    useEffect(() => {
        if (ref.current) applyGhostMaterials(ref.current)
    }, [applyGhostMaterials])

    return (
        <group ref={ref} position={position} rotation={[0, rotationY, 0]}>
            <Chair />
        </group>
    )
})
