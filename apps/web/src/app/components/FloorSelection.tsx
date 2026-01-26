'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { useVenueStore } from '../../store'

export const FloorSelection = () => {
    const { camera, gl } = useThree()
    const setSelection = useVenueStore((state) => state.setSelection)
    const draggedItemType = useVenueStore((state) => state.draggedItemType)

    const [active, setActive] = useState(false)
    const [start, setStart] = useState<THREE.Vector3 | null>(null)
    const [end, setEnd] = useState<THREE.Vector3 | null>(null)
    const activeRef = useRef(false)
    const startRef = useRef<THREE.Vector3 | null>(null)
    const endRef = useRef<THREE.Vector3 | null>(null)

    const plane = useMemo(() => new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), [])
    const raycaster = useMemo(() => new THREE.Raycaster(), [])
    const pointer = useRef(new THREE.Vector2())

    const setPointerFromEvent = (e: PointerEvent) => {
        const rect = gl.domElement.getBoundingClientRect()
        pointer.current.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
        pointer.current.y = -((e.clientY - rect.top) / rect.height) * 2 + 1
    }

    const getFloorPoint = (e: PointerEvent) => {
        setPointerFromEvent(e)
        raycaster.setFromCamera(pointer.current, camera)
        const hit = new THREE.Vector3()
        return raycaster.ray.intersectPlane(plane, hit) ? hit : null
    }

    useEffect(() => {
        if (draggedItemType) return

        const handlePointerDown = (e: PointerEvent) => {
            if (e.button !== 0) return
            const state = useVenueStore.getState()
            if (state.draggedItemType) return
            if (state.isDragging) return
            const hit = getFloorPoint(e)
            if (!hit) return
            activeRef.current = true
            startRef.current = hit.clone()
            endRef.current = hit.clone()
            setActive(true)
            setStart(hit.clone())
            setEnd(hit.clone())
        }

        const handlePointerMove = (e: PointerEvent) => {
            if (!activeRef.current) return
            const hit = getFloorPoint(e)
            if (!hit) return
            endRef.current = hit.clone()
            setEnd(hit.clone())
        }

        const handlePointerUp = (e: PointerEvent) => {
            if (!activeRef.current || !startRef.current || !endRef.current) return
            activeRef.current = false
            setActive(false)

            const startPoint = startRef.current
            const endPoint = endRef.current
            const minX = Math.min(startPoint.x, endPoint.x)
            const maxX = Math.max(startPoint.x, endPoint.x)
            const minZ = Math.min(startPoint.z, endPoint.z)
            const maxZ = Math.max(startPoint.z, endPoint.z)
            const width = Math.abs(maxX - minX)
            const depth = Math.abs(maxZ - minZ)

            if (width < 0.1 && depth < 0.1) {
                if (!e.shiftKey && !e.ctrlKey && !e.metaKey && !e.altKey) {
                    setSelection([])
                }
                startRef.current = null
                endRef.current = null
                setStart(null)
                setEnd(null)
                return
            }

            const state = useVenueStore.getState()
            const items = state.items
            const selectedIds = state.selectedIds
            const expandGroups = (ids: string[]) => {
                const expanded = new Set<string>()
                ids.forEach((id) => {
                    const item = items.find(i => i.id === id)
                    if (item?.groupId) {
                        items.filter(i => i.groupId === item.groupId).forEach(groupItem => expanded.add(groupItem.id))
                    } else {
                        expanded.add(id)
                    }
                })
                return expanded
            }

            const ids = items
                .filter(item => (
                    item.position[0] >= minX &&
                    item.position[0] <= maxX &&
                    item.position[2] >= minZ &&
                    item.position[2] <= maxZ
                ))
                .map(item => item.id)

            if (e.altKey) {
                const removeSet = expandGroups(ids)
                const nextIds = selectedIds.filter(id => !removeSet.has(id))
                setSelection(nextIds)
            } else if (e.shiftKey || e.ctrlKey || e.metaKey) {
                const addSet = expandGroups(ids)
                setSelection(Array.from(new Set([...selectedIds, ...addSet])))
            } else {
                setSelection(Array.from(expandGroups(ids)))
            }

            startRef.current = null
            endRef.current = null
            setStart(null)
            setEnd(null)
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
    }, [camera, draggedItemType, gl, raycaster])

    if (!active || !start || !end) return null

    const centerX = (start.x + end.x) / 2
    const centerZ = (start.z + end.z) / 2
    const width = Math.abs(end.x - start.x)
    const depth = Math.abs(end.z - start.z)

    return (
        <group>
            <mesh position={[centerX, 0.02, centerZ]} rotation={[-Math.PI / 2, 0, 0]}>
                <planeGeometry args={[width, depth]} />
                <meshBasicMaterial color="#7c6fff" transparent opacity={0.12} />
            </mesh>
            <lineSegments position={[centerX, 0.021, centerZ]} rotation={[-Math.PI / 2, 0, 0]}>
                <edgesGeometry args={[new THREE.PlaneGeometry(width, depth)]} />
                <lineBasicMaterial color="#b7abff" transparent opacity={0.9} />
            </lineSegments>
        </group>
    )
}
