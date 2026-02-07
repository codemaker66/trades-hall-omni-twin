'use client'

import { Canvas } from '@react-three/fiber'
import { Environment, ContactShadows, Html, useProgress } from '@react-three/drei'
import { Hall } from './Hall'
import { InteractiveFurniture } from './InteractiveFurniture'
import { RTSCameraControls } from './RTSCameraControlsView'
import { DragPreview } from './DragPreview'
import { SelectionGizmo } from './SelectionGizmo'
import { FloorSelection } from './FloorSelection'
import { SceneErrorBoundary } from './SceneErrorBoundary'
import { useVenueStore } from '../../store'
import { Suspense, useCallback, useRef } from 'react'
import * as THREE from 'three'

const Loader = () => {
    const { progress } = useProgress()
    return (
        <Html center>
            <div className="text-center">
                <div className="text-[#d7ccc8] text-sm font-medium mb-2">Loading Scene</div>
                <div className="w-48 h-1 bg-[#3e2723] rounded-full overflow-hidden">
                    <div
                        className="h-full bg-[#8d6e63] rounded-full transition-all duration-300"
                        style={{ width: `${progress}%` }}
                    />
                </div>
                <div className="text-[#8d6e63] text-xs mt-1">{progress.toFixed(0)}%</div>
            </div>
        </Html>
    )
}

export const Scene = () => {
    const items = useVenueStore((state) => state.items)
    const itemRefs = useRef<{ [id: string]: THREE.Group }>({})
    const handleRegister = useCallback((id: string, ref: THREE.Group | null) => {
        if (ref) {
            itemRefs.current[id] = ref
            return
        }

        delete itemRefs.current[id]
    }, [])

    return (
        <SceneErrorBoundary>
            <Canvas
                shadows
                dpr={[1, 2]}
                camera={{ position: [0, 5, 8], fov: 60 }}
                className="canvas"
            >
                <Suspense fallback={<Loader />}>
                    {/* Premium Atmosphere & Lighting */}
                    <fog attach="fog" args={['#1a1005', 30, 120]} />
                    <color attach="background" args={['#1a1005']} />

                    <Environment preset="city" background={false} blur={0.8} />

                    <ambientLight intensity={0.5} color="#ffeebb" />
                    <spotLight
                        position={[10, 10, 10]}
                        angle={0.5}
                        penumbra={1}
                        intensity={400}
                        castShadow
                        shadow-bias={-0.0001}
                    />
                    <pointLight position={[-10, 5, -10]} intensity={200} color="#aaccff" />

                    <ContactShadows position={[0, 0.01, 0]} resolution={1024} scale={50} blur={2} opacity={0.5} far={10} color="#000000" />

                    <DragPreview />
                    <FloorSelection />

                    <group position={[0, 0, 0]}>
                        <Hall />
                        {items.map((item) => (
                            <InteractiveFurniture
                                key={item.id}
                                {...item}
                                onRegister={handleRegister}
                            />
                        ))}
                    </group>

                    <RTSCameraControls />
                    <SelectionGizmo itemRefs={itemRefs} />
                </Suspense>
            </Canvas>
        </SceneErrorBoundary>
    )
}
