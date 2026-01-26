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
    const [position, setPosition] = useState<[number, number, number] | null>(null)
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0) // Floor plane at y=0

    const { camera, raycaster, pointer, scene } = useThree()

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
        const handlePointerUp = (e: PointerEvent) => {
            if (!draggedItemType) return

            // Prevent dropping if mouse is released over UI elements (buttons, etc)
            // We assume the canvas is the target when clicking on the 3D scene.
            const isCanvas = (e.target as HTMLElement).nodeName === 'CANVAS'

            if (isCanvas && position) {
                addItem(draggedItemType, position)
                setDraggedItem(null)
                setPosition(null)
            } else {
                // Cancel drag if dropped on UI or off-screen
                setDraggedItem(null)
                setPosition(null)
            }
        }

        window.addEventListener('pointerup', handlePointerUp)
        return () => window.removeEventListener('pointerup', handlePointerUp)
    }, [draggedItemType, position, addItem, setDraggedItem])

    if (!draggedItemType || !Component) return null

    return (
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
    )
}
