'use client'

import { useMemo, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Text } from '@react-three/drei'
import * as THREE from 'three'

/**
 * 3D Option Value Surface
 *
 * Interactive R3F surface: option value vs (spot price x time to expiry).
 * Rotate, zoom, hover for exact values.
 * Uses Viridis-like color mapping on value.
 */

export interface OptionSurfaceProps {
  /** Spot prices (x-axis values) */
  spotPrices: number[]
  /** Times to expiry (z-axis values) */
  timesToExpiry: number[]
  /** Option values: grid[spotIdx][timeIdx] */
  values: number[][]
  /** Title */
  title?: string
  /** Chart dimensions */
  width?: number
  height?: number
}

export function OptionSurface({
  spotPrices,
  timesToExpiry,
  values,
  title = 'Option Value Surface',
  width = 640,
  height = 480,
}: OptionSurfaceProps) {
  return (
    <div style={{ width, height }} className="relative rounded-lg overflow-hidden bg-slate-900">
      <div className="absolute top-2 left-1/2 -translate-x-1/2 z-10 text-xs text-slate-300 font-medium">
        {title}
      </div>
      <Canvas camera={{ position: [3, 3, 3], fov: 50 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 8, 3]} intensity={0.8} />
        <SurfaceMesh
          spotPrices={spotPrices}
          timesToExpiry={timesToExpiry}
          values={values}
        />
        <OrbitControls enableDamping dampingFactor={0.1} />
        <AxisLabels />
      </Canvas>
    </div>
  )
}

function AxisLabels() {
  return (
    <>
      <Text position={[0, -0.15, 1.2]} fontSize={0.08} color="#94a3b8">
        Spot Price →
      </Text>
      <Text position={[1.2, -0.15, 0]} fontSize={0.08} color="#94a3b8" rotation={[0, -Math.PI / 2, 0]}>
        Time to Expiry →
      </Text>
      <Text position={[-0.2, 0.5, 0]} fontSize={0.08} color="#94a3b8" rotation={[0, 0, Math.PI / 2]}>
        Value ↑
      </Text>
    </>
  )
}

function SurfaceMesh({
  spotPrices,
  timesToExpiry,
  values,
}: {
  spotPrices: number[]
  timesToExpiry: number[]
  values: number[][]
}) {
  const meshRef = useRef<THREE.Mesh>(null)

  const { geometry, material } = useMemo(() => {
    const nX = spotPrices.length
    const nZ = timesToExpiry.length

    // Normalize to unit cube
    const minSpot = Math.min(...spotPrices)
    const maxSpot = Math.max(...spotPrices)
    const minTime = Math.min(...timesToExpiry)
    const maxTime = Math.max(...timesToExpiry)

    let minVal = Infinity
    let maxVal = -Infinity
    for (let i = 0; i < nX; i++) {
      for (let j = 0; j < nZ; j++) {
        const v = values[i]?.[j] ?? 0
        minVal = Math.min(minVal, v)
        maxVal = Math.max(maxVal, v)
      }
    }

    const rangeSpot = maxSpot - minSpot || 1
    const rangeTime = maxTime - minTime || 1
    const rangeVal = maxVal - minVal || 1

    const geom = new THREE.BufferGeometry()
    const positions = new Float32Array(nX * nZ * 3)
    const colors = new Float32Array(nX * nZ * 3)

    for (let i = 0; i < nX; i++) {
      for (let j = 0; j < nZ; j++) {
        const idx = (i * nZ + j) * 3
        const x = (spotPrices[i]! - minSpot) / rangeSpot - 0.5
        const z = (timesToExpiry[j]! - minTime) / rangeTime - 0.5
        const y = ((values[i]?.[j] ?? 0) - minVal) / rangeVal

        positions[idx] = x
        positions[idx + 1] = y
        positions[idx + 2] = z

        // Viridis-inspired colormap
        const t = (values[i]?.[j] ?? 0 - minVal) / rangeVal
        const [r, g, b] = viridis(t)
        colors[idx] = r
        colors[idx + 1] = g
        colors[idx + 2] = b
      }
    }

    // Build index buffer for triangles
    const indices: number[] = []
    for (let i = 0; i < nX - 1; i++) {
      for (let j = 0; j < nZ - 1; j++) {
        const a = i * nZ + j
        const b = (i + 1) * nZ + j
        const c = (i + 1) * nZ + (j + 1)
        const d = i * nZ + (j + 1)
        indices.push(a, b, c, a, c, d)
      }
    }

    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    geom.setIndex(indices)
    geom.computeVertexNormals()

    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      flatShading: false,
      metalness: 0.1,
      roughness: 0.7,
    })

    return { geometry: geom, material: mat }
  }, [spotPrices, timesToExpiry, values])

  useFrame(() => {
    // Could add slow rotation or hover effects here
  })

  return <mesh ref={meshRef} geometry={geometry} material={material} />
}

/** Approximate Viridis colormap [0,1] → [r,g,b] in [0,1] */
function viridis(t: number): [number, number, number] {
  const c = Math.max(0, Math.min(1, t))
  // Piecewise linear approximation of Viridis
  const r = Math.max(0, Math.min(1, -0.05 + 1.5 * c * c))
  const g = Math.max(0, Math.min(1, 0.03 + 0.8 * c - 0.3 * c * c))
  const b = Math.max(0, Math.min(1, 0.53 - 0.7 * c + 0.4 * c * c))
  return [r, g, b]
}
