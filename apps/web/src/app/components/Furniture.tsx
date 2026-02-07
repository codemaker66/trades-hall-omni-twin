'use client'

import React from 'react'
import * as THREE from 'three'

interface FurnitureProps {
    position?: [number, number, number]
    rotation?: [number, number, number]
}

// Shared geometries (created once, reused by all instances)
const roundTableTopGeom = new THREE.CylinderGeometry(0.9, 0.9, 0.05, 32)
const roundTableLegGeom = new THREE.CylinderGeometry(0.1, 0.4, 0.74, 16)
const trestleTableTopGeom = new THREE.BoxGeometry(1.8, 0.05, 0.76)
const trestleTableLegGeom = new THREE.CylinderGeometry(0.05, 0.05, 0.74)
const chairSeatGeom = new THREE.BoxGeometry(0.45, 0.05, 0.45)
const chairBackGeom = new THREE.BoxGeometry(0.45, 0.5, 0.05)
const chairLegGeom = new THREE.CylinderGeometry(0.03, 0.03, 0.45)
const platformGeom = new THREE.BoxGeometry(2, 0.2, 1)

// Shared materials
const whiteSurface = new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.5 })
const darkLeg = new THREE.MeshStandardMaterial({ color: '#333' })
const chairDarkLeg = new THREE.MeshStandardMaterial({ color: '#222' })
const platformMat = new THREE.MeshStandardMaterial({ color: '#111', roughness: 0.9 })

// Lazy-loaded chair fabric material (texture loaded once)
let chairFabricMat: THREE.MeshStandardMaterial | null = null
function getChairFabricMaterial(): THREE.MeshStandardMaterial {
    if (!chairFabricMat) {
        const texture = new THREE.TextureLoader().load('./textures/red_fabric_texture.png')
        chairFabricMat = new THREE.MeshStandardMaterial({ map: texture, color: 'white', roughness: 0.8 })
    }
    return chairFabricMat
}

// 6ft Round Table (Diameter ~1.8m)
export const RoundTable6ft = React.memo(({ position = [0, 0, 0], rotation = [0, 0, 0] }: FurnitureProps) => {
    return (
        <group position={position} rotation={rotation}>
            <mesh position={[0, 0.74, 0]} geometry={roundTableTopGeom} material={whiteSurface} castShadow receiveShadow />
            <mesh position={[0, 0.37, 0]} geometry={roundTableLegGeom} material={darkLeg} castShadow />
        </group>
    )
})

// 6ft Trestle Table (1.8m x 0.76m)
export const TrestleTable6ft = React.memo(({ position = [0, 0, 0], rotation = [0, 0, 0] }: FurnitureProps) => {
    return (
        <group position={position} rotation={rotation}>
            <mesh position={[0, 0.74, 0]} geometry={trestleTableTopGeom} material={whiteSurface} castShadow receiveShadow />
            <group>
                <mesh position={[0.8, 0.37, 0]} geometry={trestleTableLegGeom} material={darkLeg} castShadow />
                <mesh position={[-0.8, 0.37, 0]} geometry={trestleTableLegGeom} material={darkLeg} castShadow />
            </group>
        </group>
    )
})

// Simple Chair
export const Chair = React.memo(({ position = [0, 0, 0], rotation = [0, 0, 0] }: FurnitureProps) => {
    const fabricMat = React.useMemo(() => getChairFabricMaterial(), [])

    return (
        <group position={position} rotation={rotation}>
            <mesh position={[0, 0.45, 0]} geometry={chairSeatGeom} material={fabricMat} castShadow receiveShadow />
            <mesh position={[0, 0.7, -0.2]} geometry={chairBackGeom} material={fabricMat} castShadow receiveShadow />
            <mesh position={[0.2, 0.225, 0.2]} geometry={chairLegGeom} material={chairDarkLeg} castShadow />
            <mesh position={[-0.2, 0.225, 0.2]} geometry={chairLegGeom} material={chairDarkLeg} castShadow />
            <mesh position={[0.2, 0.225, -0.2]} geometry={chairLegGeom} material={chairDarkLeg} castShadow />
            <mesh position={[-0.2, 0.225, -0.2]} geometry={chairLegGeom} material={chairDarkLeg} castShadow />
        </group>
    )
})

// Stage Platform (Scalable)
export const Platform = React.memo(({ position = [0, 0, 0], rotation = [0, 0, 0] }: FurnitureProps) => {
    return (
        <group position={position} rotation={rotation}>
            <mesh position={[0, 0.1, 0]} geometry={platformGeom} material={platformMat} castShadow receiveShadow />
        </group>
    )
})
