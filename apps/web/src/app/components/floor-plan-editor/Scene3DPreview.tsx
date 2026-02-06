'use client'

import { Canvas, useThree, useFrame } from '@react-three/fiber'
import { Environment, ContactShadows, MapControls } from '@react-three/drei'
import { useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { useFloorPlanStore } from './store'
import { floorPlanTo3D, floorDimensions3D, type Scene3DItem } from './coordinateBridge'

// ─── Shared geometries & materials (module-level, created once) ──────────────

const roundTableTopGeom = new THREE.CylinderGeometry(0.9, 0.9, 0.05, 32)
const roundTableLegGeom = new THREE.CylinderGeometry(0.1, 0.4, 0.74, 16)
const trestleTableTopGeom = new THREE.BoxGeometry(1.8, 0.05, 0.76)
const trestleTableLegGeom = new THREE.CylinderGeometry(0.05, 0.05, 0.74)
const chairSeatGeom = new THREE.BoxGeometry(0.45, 0.05, 0.45)
const chairBackGeom = new THREE.BoxGeometry(0.45, 0.5, 0.05)
const chairLegGeom = new THREE.CylinderGeometry(0.03, 0.03, 0.45)
const platformGeom = new THREE.BoxGeometry(1, 0.2, 1) // scaled per item

const whiteMat = new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.5 })
const darkLegMat = new THREE.MeshStandardMaterial({ color: '#333' })
const chairLegMat = new THREE.MeshStandardMaterial({ color: '#222' })
const platformMat = new THREE.MeshStandardMaterial({ color: '#111', roughness: 0.9 })
const chairFabricMat = new THREE.MeshStandardMaterial({ color: '#c0392b', roughness: 0.8 })

// ─── Item rendering ──────────────────────────────────────────────────────────

function PreviewItem({ item }: { item: Scene3DItem }) {
  const pos = item.position
  const rot = item.rotation

  switch (item.type) {
    case 'round-table':
      return (
        <group position={pos} rotation={rot}>
          <mesh position={[0, 0.74, 0]} geometry={roundTableTopGeom} material={whiteMat} castShadow receiveShadow />
          <mesh position={[0, 0.37, 0]} geometry={roundTableLegGeom} material={darkLegMat} castShadow />
        </group>
      )
    case 'trestle-table':
      return (
        <group position={pos} rotation={rot}>
          <mesh position={[0, 0.74, 0]} geometry={trestleTableTopGeom} material={whiteMat} castShadow receiveShadow />
          <mesh position={[0.8, 0.37, 0]} geometry={trestleTableLegGeom} material={darkLegMat} castShadow />
          <mesh position={[-0.8, 0.37, 0]} geometry={trestleTableLegGeom} material={darkLegMat} castShadow />
        </group>
      )
    case 'chair':
      return (
        <group position={pos} rotation={rot}>
          <mesh position={[0, 0.45, 0]} geometry={chairSeatGeom} material={chairFabricMat} castShadow receiveShadow />
          <mesh position={[0, 0.7, -0.2]} geometry={chairBackGeom} material={chairFabricMat} castShadow />
          <mesh position={[0.2, 0.225, 0.2]} geometry={chairLegGeom} material={chairLegMat} castShadow />
          <mesh position={[-0.2, 0.225, 0.2]} geometry={chairLegGeom} material={chairLegMat} castShadow />
          <mesh position={[0.2, 0.225, -0.2]} geometry={chairLegGeom} material={chairLegMat} castShadow />
          <mesh position={[-0.2, 0.225, -0.2]} geometry={chairLegGeom} material={chairLegMat} castShadow />
        </group>
      )
    case 'platform': {
      const scaleX = item.widthM / 2
      const scaleZ = item.depthM / 1
      return (
        <group position={pos} rotation={rot}>
          <mesh position={[0, 0.1, 0]} scale={[scaleX, 1, scaleZ]} geometry={platformGeom} material={platformMat} castShadow receiveShadow />
        </group>
      )
    }
    default: {
      // Fallback: colored box for decor/equipment
      const color = item.category === 'decor' ? '#e67e22' : '#7f8c8d'
      return (
        <group position={pos} rotation={rot}>
          <mesh position={[0, 0.4, 0]}>
            <boxGeometry args={[item.widthM, 0.8, item.depthM]} />
            <meshStandardMaterial color={color} roughness={0.6} />
          </mesh>
        </group>
      )
    }
  }
}

// ─── Floor plane ─────────────────────────────────────────────────────────────

function FloorPlane({ widthM, depthM }: { widthM: number; depthM: number }) {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
      <planeGeometry args={[widthM, depthM]} />
      <meshStandardMaterial color="#2c1810" roughness={0.9} />
    </mesh>
  )
}

// ─── Camera presets ──────────────────────────────────────────────────────────

export interface CameraPreset {
  id: string
  name: string
  position: [number, number, number]
  target: [number, number, number]
}

export function getCameraPresets(widthM: number, depthM: number): CameraPreset[] {
  const maxDim = Math.max(widthM, depthM)
  return [
    {
      id: 'top-down',
      name: 'Top Down',
      position: [0, maxDim * 1.2, 0.01],
      target: [0, 0, 0],
    },
    {
      id: 'entrance',
      name: 'Entrance',
      position: [0, 2, depthM / 2 + 3],
      target: [0, 0.5, 0],
    },
    {
      id: 'stage',
      name: 'Stage View',
      position: [0, 2, -(depthM / 2 + 3)],
      target: [0, 0.5, 0],
    },
    {
      id: 'perspective',
      name: 'Perspective',
      position: [widthM / 2, maxDim * 0.6, depthM / 2],
      target: [0, 0, 0],
    },
  ]
}

// ─── Smooth camera animator ──────────────────────────────────────────────────

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
}

interface CameraAnimatorProps {
  target: { position: [number, number, number]; lookAt: [number, number, number] } | null
  onComplete: () => void
}

function CameraAnimator({ target, onComplete }: CameraAnimatorProps) {
  const { camera } = useThree()
  const progressRef = useRef(0)
  const startPos = useRef(new THREE.Vector3())
  const endPos = useRef(new THREE.Vector3())
  const startTarget = useRef(new THREE.Vector3())
  const endTarget = useRef(new THREE.Vector3())
  const isAnimating = useRef(false)
  const prevTarget = useRef<typeof target>(null)

  // When target changes, start animation
  if (target && target !== prevTarget.current) {
    startPos.current.copy(camera.position)
    endPos.current.set(...target.position)
    // Compute current look-at from camera direction
    const dir = new THREE.Vector3()
    camera.getWorldDirection(dir)
    startTarget.current.copy(camera.position).add(dir.multiplyScalar(10))
    endTarget.current.set(...target.lookAt)
    progressRef.current = 0
    isAnimating.current = true
    prevTarget.current = target
  }

  useFrame(() => {
    if (!isAnimating.current) return
    progressRef.current = Math.min(1, progressRef.current + 0.035)
    const t = easeInOutCubic(progressRef.current)

    camera.position.lerpVectors(startPos.current, endPos.current, t)
    const lx = startTarget.current.x + (endTarget.current.x - startTarget.current.x) * t
    const ly = startTarget.current.y + (endTarget.current.y - startTarget.current.y) * t
    const lz = startTarget.current.z + (endTarget.current.z - startTarget.current.z) * t
    camera.lookAt(lx, ly, lz)

    if (progressRef.current >= 1) {
      isAnimating.current = false
      onComplete()
    }
  })

  return (
    <MapControls
      enableDamping
      dampingFactor={0.1}
      minPolarAngle={0}
      maxPolarAngle={Math.PI / 2 - 0.05}
      minDistance={1}
      maxDistance={80}
    />
  )
}

// ─── Main 3D Preview ─────────────────────────────────────────────────────────

interface Scene3DPreviewProps {
  width: number
  height: number
}

export function Scene3DPreview({ width, height }: Scene3DPreviewProps) {
  const items = useFloorPlanStore((s) => s.items)
  const planWidthFt = useFloorPlanStore((s) => s.planWidthFt)
  const planHeightFt = useFloorPlanStore((s) => s.planHeightFt)

  const items3D = useMemo(
    () => floorPlanTo3D(items, planWidthFt, planHeightFt),
    [items, planWidthFt, planHeightFt],
  )

  const floor = useMemo(
    () => floorDimensions3D(planWidthFt, planHeightFt),
    [planWidthFt, planHeightFt],
  )

  const [cameraTarget, setCameraTarget] = useState<{
    position: [number, number, number]
    lookAt: [number, number, number]
  } | null>(null)

  const presets = useMemo(
    () => getCameraPresets(floor.widthM, floor.depthM),
    [floor.widthM, floor.depthM],
  )

  const maxDim = Math.max(floor.widthM, floor.depthM)

  return (
    <div className="relative" style={{ width, height }}>
      <Canvas
        shadows
        dpr={[1, 2]}
        camera={{
          position: [floor.widthM / 2, maxDim * 0.6, floor.depthM / 2],
          fov: 60,
        }}
      >
        <fog attach="fog" args={['#1a1005', maxDim * 2, maxDim * 5]} />
        <color attach="background" args={['#1a1005']} />
        <Environment preset="city" background={false} blur={0.8} />

        <ambientLight intensity={0.5} color="#ffeebb" />
        <spotLight position={[10, 10, 10]} angle={0.5} penumbra={1} intensity={400} castShadow shadow-bias={-0.0001} />
        <pointLight position={[-10, 5, -10]} intensity={200} color="#aaccff" />
        <ContactShadows position={[0, 0.01, 0]} resolution={1024} scale={maxDim * 2} blur={2} opacity={0.5} far={10} />

        <FloorPlane widthM={floor.widthM} depthM={floor.depthM} />

        {items3D.map((item) => (
          <PreviewItem key={item.id} item={item} />
        ))}

        <CameraAnimator target={cameraTarget} onComplete={() => setCameraTarget(null)} />
      </Canvas>

      {/* Camera preset buttons */}
      <div className="absolute top-3 right-3 flex flex-col gap-1">
        {presets.map((preset) => (
          <button
            key={preset.id}
            onClick={() => setCameraTarget({ position: preset.position, lookAt: preset.target })}
            className="px-2 py-1 rounded text-xs font-medium bg-surface-10/80 backdrop-blur text-surface-80 hover:bg-surface-25 hover:text-surface-90 transition-colors"
          >
            {preset.name}
          </button>
        ))}
      </div>

      {/* Item count overlay */}
      <div className="absolute bottom-3 left-3 text-xs text-surface-60 bg-surface-5/80 backdrop-blur rounded px-2 py-1">
        {items3D.length} items in scene
      </div>
    </div>
  )
}
