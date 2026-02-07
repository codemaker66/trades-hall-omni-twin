'use client'

import React, { useRef } from 'react'
import * as THREE from 'three'
import { useFrame } from '@react-three/fiber'

const hash01 = (...values: number[]) => {
    let seed = 0
    for (let i = 0; i < values.length; i++) {
        seed += values[i]! * (i % 2 === 0 ? 12.9898 : 78.233)
    }

    const value = Math.sin(seed) * 43758.5453123
    return value - Math.floor(value)
}

export const Hall = () => {
    // Dimensions from floorplan
    const width = 21 // X-axis
    const depth = 10 // Z-axis
    const height = 6
    const wainscotHeight = 2

    // Procedural Wood Texture Generation (Color, Normal, Roughness)
    // eslint-disable-next-line react-hooks/preserve-manual-memoization
    const { woodTexture, normalMap, roughnessMap } = React.useMemo(() => {
        if (typeof document === 'undefined') return { woodTexture: null, normalMap: null, roughnessMap: null }

        const width = 1024
        const height = 1024

        const canvasColor = document.createElement('canvas')
        canvasColor.width = width
        canvasColor.height = height
        const ctxColor = canvasColor.getContext('2d')

        const canvasNormal = document.createElement('canvas')
        canvasNormal.width = width
        canvasNormal.height = height
        const ctxNormal = canvasNormal.getContext('2d')

        const canvasRough = document.createElement('canvas')
        canvasRough.width = width
        canvasRough.height = height
        const ctxRough = canvasRough.getContext('2d')

        if (!ctxColor || !ctxNormal || !ctxRough) return { woodTexture: null, normalMap: null, roughnessMap: null }

        // Base backgrounds
        ctxColor.fillStyle = '#3d2817'
        ctxColor.fillRect(0, 0, width, height)

        ctxNormal.fillStyle = '#8080ff' // Flat normal color (128, 128, 255)
        ctxNormal.fillRect(0, 0, width, height)

        ctxRough.fillStyle = '#cc' // Base roughness (0.8)
        ctxRough.fillRect(0, 0, width, height)

        const numPlanks = 16 // Power of 2 for clean division (1024/16 = 64px)
        const plankHeight = height / numPlanks

        for (let i = 0; i < numPlanks; i++) {
            const y = i * plankHeight

            // Color Variation - Authentic Dark Oak (Less Orange)
            const hue = 32 + hash01(i, 0) * 6 // 32-38 (Yellow-Brown/Sepia)
            const sat = 35 + hash01(i, 1) * 15 // Reduced saturation (was 55+)
            const light = 18 + hash01(i, 2) * 12 // Darker (was 25+)
            ctxColor.fillStyle = `hsl(${hue}, ${sat}%, ${light}%)`
            ctxColor.fillRect(0, y, width, plankHeight)

            // Roughness Variation per plank
            const roughVal = 180 + hash01(i, 3) * 50 // 0-255 range. higher = rougher.
            const roughHex = Math.floor(roughVal).toString(16).padStart(2, '0')
            ctxRough.fillStyle = `#${roughHex}${roughHex}${roughHex}`
            ctxRough.fillRect(0, y, width, plankHeight)

            // Grain (with Horizontal Wrapping for seamless tiling)
            for (let j = 0; j < 80; j++) {
                const grainLen = hash01(i, j, 0) * 300
                const grainX = hash01(i, j, 1) * width
                const grainY = y + hash01(i, j, 2) * plankHeight
                const grainH = 1 + hash01(i, j, 3) * 2

                ctxColor.fillStyle = `rgba(0,0,0, ${0.1 + hash01(i, j, 4) * 0.2})`
                // Draw grain
                ctxColor.fillRect(grainX, grainY, grainLen, grainH)
                // Wrap grain if it goes off right edge
                if (grainX + grainLen > width) {
                    ctxColor.fillRect(grainX - width, grainY, grainLen, grainH)
                }

                // Grain texture for Roughness Map
                ctxRough.fillStyle = '#ff' // White = rough
                ctxRough.fillRect(grainX, grainY, grainLen, grainH)
                if (grainX + grainLen > width) {
                    ctxRough.fillRect(grainX - width, grainY, grainLen, grainH)
                }
            }

            // Gaps
            const gapY = y + plankHeight - 4

            // Color Gap (Dark)
            ctxColor.fillStyle = '#0a0502'
            ctxColor.fillRect(0, gapY, width, 4)

            // Normal Gap (Steep curve simulated)
            ctxNormal.fillStyle = 'rgb(128, 0, 255)'
            ctxNormal.fillRect(0, gapY, width, 2)
            ctxNormal.fillStyle = 'rgb(128, 255, 255)'
            ctxNormal.fillRect(0, gapY + 2, width, 2)

            // Roughness Gap (Very rough / dust)
            ctxRough.fillStyle = '#ff'
            ctxRough.fillRect(0, gapY, width, 4)
        }

        const texture = new THREE.CanvasTexture(canvasColor)
        const normal = new THREE.CanvasTexture(canvasNormal)
        const rough = new THREE.CanvasTexture(canvasRough)

        texture.wrapS = texture.wrapT = THREE.RepeatWrapping
        normal.wrapS = normal.wrapT = THREE.RepeatWrapping
        rough.wrapS = rough.wrapT = THREE.RepeatWrapping

        const repeat = 4
        texture.repeat.set(repeat, repeat)
        normal.repeat.set(repeat, repeat)
        rough.repeat.set(repeat, repeat)

        // Color space fixes
        texture.colorSpace = THREE.SRGBColorSpace

        return { woodTexture: texture, normalMap: normal, roughnessMap: rough }
    }, [])

    return (
        <group>
            {/* FLOOR - Luxurious Polished Dark Walnut */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
                <planeGeometry args={[width, depth]} />
                <meshStandardMaterial
                    map={woodTexture}
                    normalMap={normalMap}
                    normalScale={new THREE.Vector2(0.5, 0.5)} // Subtle depth
                    roughnessMap={roughnessMap}
                    roughness={1} // Base, modulated by map
                    color="#ffffff"
                    metalness={0.0}
                    envMapIntensity={0.6} // Return some shine, but controlled by roughness map now
                />
            </mesh>



            {/* WALLS - WRAPPED IN SMART LOGIC */}

            {/* Top Wall (North) - Negative Z */}
            <SmartWall direction="north" boundary={-depth / 2}>
                <group position={[0, height / 2, -depth / 2]} rotation={[0, 0, 0]}>
                    <WallSegment width={width} height={height} wainscotHeight={wainscotHeight} />
                    <DoorFrame position={[-7, -height / 2 + 1.1, 0.15]} />
                    <DoorFrame position={[0, -height / 2 + 1.1, 0.15]} />
                    <DoorFrame position={[7, -height / 2 + 1.1, 0.15]} />
                </group>
            </SmartWall>

            {/* Bottom Wall (South) - Positive Z */}
            <SmartWall direction="south" boundary={depth / 2}>
                <group position={[0, height / 2, depth / 2]} rotation={[0, Math.PI, 0]}>
                    <WallSegment width={width} height={height} wainscotHeight={wainscotHeight} />
                    <WindowFrame position={[-7, 0, 0.15]} />
                    <WindowFrame position={[-3.5, 0, 0.15]} />
                    <WindowFrame position={[0, 0, 0.15]} />
                    <WindowFrame position={[3.5, 0, 0.15]} />
                    <WindowFrame position={[7, 0, 0.15]} />
                </group>
            </SmartWall>

            {/* Left Wall (West) - Negative X */}
            <SmartWall direction="west" boundary={-width / 2}>
                <group position={[-width / 2, height / 2, 0]} rotation={[0, Math.PI / 2, 0]}>
                    <WallSegment width={depth} height={height} wainscotHeight={wainscotHeight} />
                </group>
            </SmartWall>

            {/* Right Wall (East) - Positive X */}
            <SmartWall direction="east" boundary={width / 2}>
                <group position={[width / 2, height / 2, 0]} rotation={[0, -Math.PI / 2, 0]}>
                    <WallSegment width={depth} height={height} wainscotHeight={wainscotHeight} />
                    {/* Removed door near North corner per user request */}
                    <DoorFrame position={[3, -height / 2 + 1.1, 0.15]} scale={0.8} />
                </group>
            </SmartWall>

            {/* DOME - Copper/Gold */}
            <mesh position={[0, height - 0.1, 0]} receiveShadow>
                <sphereGeometry args={[4, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2]} />
                <meshStandardMaterial
                    color="#bba14f" // Gold-ish
                    side={THREE.BackSide}
                    roughness={0.3}
                    metalness={0.8}
                    envMapIntensity={2}
                />
            </mesh>
            <mesh position={[0, height - 0.1, 0]} rotation={[Math.PI / 2, 0, 0]}>
                <torusGeometry args={[4, 0.2, 16, 64]} />
                <meshStandardMaterial color="#3B2317" roughness={0.2} />
            </mesh>
        </group>
    )
}

// Logic to hide walls when camera is behind them
const SmartWall = ({ children, direction, boundary }: { children: React.ReactNode, direction: 'north' | 'south' | 'east' | 'west', boundary: number }) => {
    const groupRef = useRef<THREE.Group>(null)
    const prevHidden = useRef<boolean | null>(null)

    useFrame(({ camera }) => {
        if (!groupRef.current) return

        let shouldHide = false
        if (direction === 'south' && camera.position.z > boundary + 1) shouldHide = true
        if (direction === 'north' && camera.position.z < boundary - 1) shouldHide = true
        if (direction === 'east' && camera.position.x > boundary + 1) shouldHide = true
        if (direction === 'west' && camera.position.x < boundary - 1) shouldHide = true

        // Only traverse children when visibility state changes
        if (shouldHide === prevHidden.current) return
        prevHidden.current = shouldHide

        groupRef.current.traverse((child) => {
            if (child instanceof THREE.Mesh && child.material) {
                child.visible = !shouldHide
                child.material.transparent = shouldHide
                child.material.opacity = shouldHide ? 0 : 1
                child.material.depthWrite = !shouldHide
                child.material.needsUpdate = true
            }
        })
    })

    return <group ref={groupRef}>{children}</group>
}

interface WallSegmentProps {
    width: number
    height: number
    wainscotHeight: number
}

const WallSegment = ({ width, height, wainscotHeight }: WallSegmentProps) => {
    const upperHeight = height - wainscotHeight
    return (
        <group>
            {/* Lower Part (Premium Mahogany) */}
            <mesh position={[0, -height / 2 + wainscotHeight / 2, 0]} receiveShadow castShadow>
                <boxGeometry args={[width, wainscotHeight, 0.2]} />
                <meshStandardMaterial
                    color="#3d2010"
                    roughness={0.2}
                    metalness={0.1}
                />
            </mesh>
            {/* Upper Part (Matte Venetian Plaster) */}
            <mesh position={[0, wainscotHeight / 2, 0]} receiveShadow castShadow>
                <boxGeometry args={[width, upperHeight, 0.2]} />
                <meshStandardMaterial
                    color="#f0e6d2" // Warm Cream
                    roughness={0.9}
                    metalness={0.0}
                />
            </mesh>
        </group>
    )
}

const DoorFrame = ({ position, scale = 1 }: { position: [number, number, number], scale?: number }) => (
    <group position={position} scale={scale}>
        <mesh receiveShadow>
            <boxGeometry args={[1.8, 2.2, 0.1]} />
            <meshStandardMaterial color="#1a110a" roughness={0.2} />
        </mesh>
        {/* Door panels with Gold Handles (simulated) */}
        <mesh position={[-0.45, 0, 0.06]}>
            <boxGeometry args={[0.8, 2, 0.05]} />
            <meshStandardMaterial color="#2b1a10" roughness={0.3} />
        </mesh>
        <mesh position={[0.45, 0, 0.06]}>
            <boxGeometry args={[0.8, 2, 0.05]} />
            <meshStandardMaterial color="#2b1a10" roughness={0.3} />
        </mesh>
    </group>
)

const WindowFrame = ({ position }: { position: [number, number, number] }) => (
    <group position={position}>
        <mesh receiveShadow>
            <boxGeometry args={[2.0, 3.5, 0.1]} />
            <meshStandardMaterial color="#1a110a" roughness={0.2} />
        </mesh>
        {/* Glass - Physically Correct */}
        <mesh position={[0, 0, 0.06]}>
            <boxGeometry args={[1.8, 3.3, 0.05]} />
            <meshPhysicalMaterial
                color="white"
                transmission={0.95}
                opacity={1}
                metalness={0}
                roughness={0.0}
                ior={1.5}
                thickness={0.1}
                specularIntensity={1}
                envMapIntensity={1}
                transparent
            />
        </mesh>
    </group>
)
