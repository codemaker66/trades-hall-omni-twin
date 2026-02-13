'use client'

import { useMemo, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Text } from '@react-three/drei'
import * as THREE from 'three'

/**
 * 3D Energy Landscape Surface
 *
 * Interactive R3F surface showing energy as a function of two selected
 * layout parameters (e.g., table spacing x rotation).
 * Shows the multimodal landscape that SA navigates.
 * Current state plotted as a moving point. Temperature shown as glow radius.
 * SA trajectory plotted as a path on the surface.
 */

export interface EnergyLandscapeProps {
  energyGrid: number[][]
  xValues: number[]
  zValues: number[]
  xLabel?: string
  zLabel?: string
  trajectory?: Array<{ x: number; z: number; energy: number }>
  currentState?: { x: number; z: number; energy: number }
  temperature?: number
  width?: number
  height?: number
  title?: string
}

export function EnergyLandscape({
  energyGrid,
  xValues,
  zValues,
  xLabel = 'Parameter 1',
  zLabel = 'Parameter 2',
  trajectory,
  currentState,
  temperature = 1,
  width = 640,
  height = 480,
  title = 'Energy Landscape',
}: EnergyLandscapeProps) {
  return (
    <div style={{ width, height }} className="relative rounded-lg overflow-hidden bg-slate-900">
      <div className="absolute top-2 left-1/2 -translate-x-1/2 z-10 text-xs text-slate-300 font-medium">
        {title}
      </div>
      <Canvas camera={{ position: [2.5, 2.5, 2.5], fov: 50 }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 8, 3]} intensity={0.8} />
        <SurfaceMesh energyGrid={energyGrid} xValues={xValues} zValues={zValues} />
        {trajectory && trajectory.length > 1 && (
          <TrajectoryLine
            trajectory={trajectory}
            xValues={xValues}
            zValues={zValues}
            energyGrid={energyGrid}
          />
        )}
        {currentState && (
          <CurrentStateMarker
            state={currentState}
            xValues={xValues}
            zValues={zValues}
            energyGrid={energyGrid}
            temperature={temperature}
          />
        )}
        <OrbitControls enableDamping dampingFactor={0.1} />
        <Text position={[0, -0.15, 0.7]} fontSize={0.06} color="#94a3b8">
          {xLabel} →
        </Text>
        <Text position={[0.7, -0.15, 0]} fontSize={0.06} color="#94a3b8" rotation={[0, -Math.PI / 2, 0]}>
          {zLabel} →
        </Text>
        <Text position={[-0.15, 0.4, 0]} fontSize={0.06} color="#94a3b8" rotation={[0, 0, Math.PI / 2]}>
          Energy ↑
        </Text>
      </Canvas>
    </div>
  )
}

function viridis(t: number): [number, number, number] {
  const c = Math.max(0, Math.min(1, t))
  const r = Math.max(0, Math.min(1, -0.05 + 1.5 * c * c))
  const g = Math.max(0, Math.min(1, 0.03 + 0.8 * c - 0.3 * c * c))
  const b = Math.max(0, Math.min(1, 0.53 - 0.7 * c + 0.4 * c * c))
  return [r, g, b]
}

function normalizeGrid(grid: number[][], xLen: number, zLen: number) {
  let minE = Infinity, maxE = -Infinity
  for (let i = 0; i < xLen; i++) {
    for (let j = 0; j < zLen; j++) {
      const v = grid[i]?.[j] ?? 0
      if (v < minE) minE = v
      if (v > maxE) maxE = v
    }
  }
  return { minE, maxE, range: maxE - minE || 1 }
}

function SurfaceMesh({ energyGrid, xValues, zValues }: {
  energyGrid: number[][]; xValues: number[]; zValues: number[]
}) {
  const { geometry, material } = useMemo(() => {
    const nX = xValues.length
    const nZ = zValues.length
    const { minE, range } = normalizeGrid(energyGrid, nX, nZ)

    const geom = new THREE.BufferGeometry()
    const positions = new Float32Array(nX * nZ * 3)
    const colors = new Float32Array(nX * nZ * 3)

    for (let i = 0; i < nX; i++) {
      for (let j = 0; j < nZ; j++) {
        const idx = (i * nZ + j) * 3
        const x = i / Math.max(1, nX - 1) - 0.5
        const z = j / Math.max(1, nZ - 1) - 0.5
        const y = ((energyGrid[i]?.[j] ?? 0) - minE) / range * 0.8

        positions[idx] = x
        positions[idx + 1] = y
        positions[idx + 2] = z

        const t = ((energyGrid[i]?.[j] ?? 0) - minE) / range
        const [r, g, b] = viridis(t)
        colors[idx] = r
        colors[idx + 1] = g
        colors[idx + 2] = b
      }
    }

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
      vertexColors: true, side: THREE.DoubleSide, flatShading: false,
      metalness: 0.1, roughness: 0.7,
    })
    return { geometry: geom, material: mat }
  }, [energyGrid, xValues, zValues])

  return <mesh geometry={geometry} material={material} />
}

function TrajectoryLine({ trajectory, xValues, zValues, energyGrid }: {
  trajectory: Array<{ x: number; z: number; energy: number }>
  xValues: number[]; zValues: number[]; energyGrid: number[][]
}) {
  const geometry = useMemo(() => {
    const nX = xValues.length; const nZ = zValues.length
    const { minE, range } = normalizeGrid(energyGrid, nX, nZ)
    const xMin = xValues[0]!; const xMax = xValues[nX - 1]!
    const zMin = zValues[0]!; const zMax = zValues[nZ - 1]!

    const points = trajectory.map(pt => new THREE.Vector3(
      (pt.x - xMin) / (xMax - xMin || 1) - 0.5,
      (pt.energy - minE) / range * 0.8 + 0.01,
      (pt.z - zMin) / (zMax - zMin || 1) - 0.5,
    ))
    const geom = new THREE.BufferGeometry().setFromPoints(points)
    return geom
  }, [trajectory, xValues, zValues, energyGrid])

  return (
    <primitive object={new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: '#22d3ee', linewidth: 2 }))} />
  )
}

function CurrentStateMarker({ state, xValues, zValues, energyGrid, temperature }: {
  state: { x: number; z: number; energy: number }
  xValues: number[]; zValues: number[]; energyGrid: number[][]
  temperature: number
}) {
  const ref = useRef<THREE.Mesh>(null)
  const nX = xValues.length; const nZ = zValues.length
  const { minE, range } = normalizeGrid(energyGrid, nX, nZ)
  const xMin = xValues[0]!; const xMax = xValues[nX - 1]!
  const zMin = zValues[0]!; const zMax = zValues[nZ - 1]!

  const pos: [number, number, number] = [
    (state.x - xMin) / (xMax - xMin || 1) - 0.5,
    (state.energy - minE) / range * 0.8 + 0.02,
    (state.z - zMin) / (zMax - zMin || 1) - 0.5,
  ]
  const radius = 0.01 + Math.min(0.05, temperature * 0.001)

  useFrame(({ clock }) => {
    if (ref.current) {
      ref.current.scale.setScalar(1 + 0.1 * Math.sin(clock.elapsedTime * 3))
    }
  })

  return (
    <mesh ref={ref} position={pos}>
      <sphereGeometry args={[radius, 16, 16]} />
      <meshStandardMaterial color="#f43f5e" emissive="#f43f5e" emissiveIntensity={0.5} />
    </mesh>
  )
}
