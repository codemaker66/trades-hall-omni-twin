import { useFrame, useThree } from '@react-three/fiber'
import { useRef, useEffect, useState } from 'react'
import { useVenueStore } from '../../store'
import * as THREE from 'three'
import { RoundTable6ft, TrestleTable6ft, Chair, Platform } from './Furniture'

export const DragPreview = () => {
    const draggedItemType = useVenueStore((state) => state.draggedItemType)
    const setDraggedItem = useVenueStore((state) => state.setDraggedItem)
    const addItem = useVenueStore((state) => state.addItem)
    const snapGrid = useVenueStore((state) => state.snapGrid)
    const snappingEnabled = useVenueStore((state) => state.snappingEnabled)

    const groupRef = useRef<THREE.Group>(null)
    const objectBounds = useRef({ centerY: 0, bottomOffset: 0, halfX: 0, halfZ: 0, ready: false })
    const stackRayOriginY = 50
    const areaDragRef = useRef<{ active: boolean, start: THREE.Vector3 | null, end: THREE.Vector3 | null }>({
        active: false,
        start: null,
        end: null
    })
    const [previewPositions, setPreviewPositions] = useState<[number, number, number][]>([])
    const [position, setPosition] = useState<[number, number, number] | null>(null)
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0) // Floor plane at y=0

    const { camera, raycaster, pointer, scene, gl } = useThree()

    // Determine Component
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
                if ('emissive' in mat) {
                    mat.emissive = new THREE.Color('#7c6fff')
                    mat.emissiveIntensity = 0.6
                }
                mat.transparent = true
                mat.opacity = 0.35
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
        raycaster.setFromCamera({ x, y }, camera)
        const hit = new THREE.Vector3()
        return raycaster.ray.intersectPlane(plane, hit) ? hit : null
    }

    const getChairY = (x: number, z: number) => {
        ensureObjectBounds()
        const downRay = new THREE.Raycaster()
        const chairCenterY = (groupRef.current?.position.y ?? 0) + objectBounds.current.centerY
        downRay.set(new THREE.Vector3(x, chairCenterY, z), new THREE.Vector3(0, -1, 0))

        const downIntersects = downRay.intersectObjects(scene.children, true)
        const state = useVenueStore.getState()
        let targetY = 0

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
        const minX = Math.min(start.x, end.x)
        const maxX = Math.max(start.x, end.x)
        const minZ = Math.min(start.z, end.z)
        const maxZ = Math.max(start.z, end.z)
        const spacing = Math.max(0.6, snappingEnabled ? snapGrid : 0.6)
        const startX = Math.round(minX / spacing) * spacing
        const startZ = Math.round(minZ / spacing) * spacing
        const endX = Math.round(maxX / spacing) * spacing
        const endZ = Math.round(maxZ / spacing) * spacing
        const positions: [number, number, number][] = []

        for (let x = startX; x <= endX + 0.001; x += spacing) {
            for (let z = startZ; z <= endZ + 0.001; z += spacing) {
                const y = getChairY(x, z)
                positions.push([x, y, z])
            }
        }

        return positions
    }

    // Update Ghost Position
    useFrame(() => {
        if (!draggedItemType) return

        raycaster.setFromCamera(pointer, camera)
        const target = new THREE.Vector3()

        // Raycast against infinite floor plane
        if (raycaster.ray.intersectPlane(plane, target)) {

            // Apply Snapping
            if (snappingEnabled) {
                target.x = Math.round(target.x / snapGrid) * snapGrid
                target.z = Math.round(target.z / snapGrid) * snapGrid
            }

            let targetY = 0

            if (draggedItemType === 'chair' || draggedItemType === 'platform') {
                ensureObjectBounds()

                const downRay = new THREE.Raycaster()
                const downDir = new THREE.Vector3(0, -1, 0)
                const rayOrigin = new THREE.Vector3()
                const state = useVenueStore.getState()

                const originY = draggedItemType === 'chair'
                    ? (groupRef.current?.position.y ?? 0) + objectBounds.current.centerY
                    : stackRayOriginY

                rayOrigin.set(target.x, originY, target.z)
                downRay.set(rayOrigin, downDir)

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

        const handlePointerDown = (e: PointerEvent) => {
            if (draggedItemType !== 'chair') return
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

            // Prevent dropping if mouse is released over UI elements (buttons, etc)
            // We assume the canvas is the target when clicking on the 3D scene.
            const isCanvas = (e.target as HTMLElement).nodeName === 'CANVAS'

            if (draggedItemType === 'chair' && areaDragRef.current.active) {
                areaDragRef.current.active = false
                const start = areaDragRef.current.start
                const end = areaDragRef.current.end
                areaDragRef.current.start = null
                areaDragRef.current.end = null
                setPreviewPositions([])

                if (isCanvas && start && end) {
                    const minX = Math.min(start.x, end.x)
                    const maxX = Math.max(start.x, end.x)
                    const minZ = Math.min(start.z, end.z)
                    const maxZ = Math.max(start.z, end.z)
                    const spacing = Math.max(0.6, snappingEnabled ? snapGrid : 0.6)
                    const startX = Math.round(minX / spacing) * spacing
                    const startZ = Math.round(minZ / spacing) * spacing
                    const endX = Math.round(maxX / spacing) * spacing
                    const endZ = Math.round(maxZ / spacing) * spacing

                    for (let x = startX; x <= endX + 0.001; x += spacing) {
                        for (let z = startZ; z <= endZ + 0.001; z += spacing) {
                            const y = getChairY(x, z)
                            addItem('chair', [x, y, z])
                        }
                    }
                }

                setDraggedItem(null)
                setPosition(null)
                return
            }

            if (isCanvas && position) {
                addItem(draggedItemType, position)
                setDraggedItem(null)
                setPosition(null)
                setPreviewPositions([])
            } else {
                // Cancel drag if dropped on UI or off-screen
                setDraggedItem(null)
                setPosition(null)
                setPreviewPositions([])
            }
        }

        const element = gl.domElement
        element.addEventListener('pointerdown', handlePointerDown)
        element.addEventListener('pointermove', handlePointerMove)
        window.addEventListener('pointerup', handlePointerUp)
        return () => {
            element.removeEventListener('pointerdown', handlePointerDown)
            element.removeEventListener('pointermove', handlePointerMove)
            window.removeEventListener('pointerup', handlePointerUp)
        }
    }, [draggedItemType, position, addItem, setDraggedItem, snappingEnabled, snapGrid, gl])

    if (!draggedItemType || !Component) return null

    return (
        <group>
            <group ref={groupRef}>
                <Component />
                {/* Override Material for Ghost Effect */}
                <mesh position={[0, 1, 0]}> {/* Just a visual indicator helper if needed, but lets try traversing */}
                </mesh>

                {/* Visual Helper Ring */}
                <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.05, 0]}>
                    <ringGeometry args={[0.5, 0.6, 32]} />
                    <meshBasicMaterial color="#6366f1" transparent opacity={0.6} side={THREE.DoubleSide} />
                </mesh>
            </group>

            {draggedItemType === 'chair' && previewPositions.map((pos, index) => (
                <GhostChairInstance key={`ghost-${index}`} position={pos} applyGhostMaterials={applyGhostMaterials} />
            ))}
        </group>
    )
}

const GhostChairInstance = ({ position, applyGhostMaterials }: { position: [number, number, number], applyGhostMaterials: (root: THREE.Object3D) => void }) => {
    const ref = useRef<THREE.Group>(null)

    useEffect(() => {
        if (ref.current) applyGhostMaterials(ref.current)
    }, [applyGhostMaterials])

    return (
        <group ref={ref} position={position}>
            <Chair />
        </group>
    )
}
