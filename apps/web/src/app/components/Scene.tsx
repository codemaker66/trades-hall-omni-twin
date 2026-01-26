'use client'

import { Canvas } from '@react-three/fiber'
import { Environment, ContactShadows } from '@react-three/drei'
// import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing'
import { Hall } from './Hall'
import { InteractiveFurniture } from './InteractiveFurniture'
import { RTSCameraControls } from './RTSCameraControls'
import { DragPreview } from './DragPreview'
import { SelectionGizmo } from './SelectionGizmo'
import { FloorSelection } from './FloorSelection'
import { useVenueStore } from '../../store'
import { Suspense, useRef } from 'react'
import * as THREE from 'three'

export const Scene = () => {
    const items = useVenueStore((state) => state.items)
    // Registry of object refs for centralized control
    const itemRefs = useRef<{ [id: string]: THREE.Group }>({})

    return (
        <Canvas
            shadows
            dpr={[1, 2]}
            camera={{ position: [0, 5, 8], fov: 60 }} // Closer, feels larger. Reset FOV to 60 to lessen distortion if close.
            className="canvas"
        >
            <Suspense fallback={null}>
                {/* Premium Atmosphere & Lighting (50 Million Dollar Look) */}
                {/* Premium Atmosphere & Lighting (50 Million Dollar Look) */}
                <fog attach="fog" args={['#1a1005', 30, 120]} /> {/* Dark Warm Brown Fog to match wood */}
                <color attach="background" args={['#1a1005']} />

                {/* High-end Reflection Probe (Lobby/Studio feel) */}
                <Environment preset="city" background={false} blur={0.8} />

                {/* Warm, Cinematic Lighting */}
                <ambientLight intensity={0.5} color="#ffeebb" />
                <spotLight
                    position={[10, 10, 10]}
                    angle={0.5}
                    penumbra={1}
                    intensity={400}
                    castShadow
                    shadow-bias={-0.0001}
                />
                <pointLight position={[-10, 5, -10]} intensity={200} color="#aaccff" /> {/* Cool fill light */}

                {/* Soft Contact Shadows for grounding (Essential for 'realism') */}
                <ContactShadows position={[0, 0.01, 0]} resolution={1024} scale={50} blur={2} opacity={0.5} far={10} color="#000000" />

                <DragPreview />
                <FloorSelection />

                <group position={[0, 0, 0]}>
                    <Hall />
                    {items.map((item) => (
                        <InteractiveFurniture
                            key={item.id}
                            {...item}
                            onRegister={(id, ref) => {
                                itemRefs.current[id] = ref
                            }}
                        />
                    ))}
                </group>

                <RTSCameraControls />
                <SelectionGizmo itemRefs={itemRefs} />

                {/* <EffectComposer>
                    <Bloom luminanceThreshold={1} mipmapBlur intensity={0.4} radius={0.4} />
                    <Vignette eskil={false} offset={0.1} darkness={0.9} />
                </EffectComposer> */}
            </Suspense>
        </Canvas>
    )
}
