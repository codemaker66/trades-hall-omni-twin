import { Canvas, useThree } from '@react-three/fiber'
import { Environment, ContactShadows, OrbitControls } from '@react-three/drei'
import { Selection, EffectComposer, Outline } from '@react-three/postprocessing'
import { ReactiveGrid } from './ReactiveGrid'
import { JuicyObject } from './JuicyObject'
import { Suspense } from 'react'

const PostProcessing = () => {
    const isWebGL2 = useThree((state) => state.gl.capabilities.isWebGL2)

    return (
        <EffectComposer autoClear={false} multisampling={isWebGL2 ? 8 : 0}>
            <Outline blur edgeStrength={100} width={1000} visibleEdgeColor="#6366f1" hiddenEdgeColor="#6366f1" />
        </EffectComposer>
    )
}

export const Scene = () => {
    return (
        <Canvas
            shadows
            dpr={[1, 2]}
            camera={{ position: [5, 5, 5], fov: 45 }}
            className="canvas"
        >
            <Suspense fallback={null}>
                <Environment preset="city" />

                <color attach="background" args={['#0b0f14']} />

                <group position={[0, -0.5, 0]}>
                    <ReactiveGrid />

                    <Selection>
                        <PostProcessing />

                        {/* Sample Juicy Objects */}
                        <JuicyObject position={[0, 0.5, 0]} color="#f472b6" />
                        <JuicyObject position={[2, 0.5, -1]} color="#6366f1" shape="cylinder" />
                        <JuicyObject position={[-2, 0.5, 1]} color="#f2d060" shape="sphere" />
                    </Selection>

                    <ContactShadows
                        resolution={1024}
                        scale={50}
                        blur={2}
                        opacity={0.5}
                        far={10}
                        color="#000000"
                    />
                </group>

                <OrbitControls
                    makeDefault
                    minPolarAngle={0}
                    maxPolarAngle={Math.PI / 2.2}
                    enablePan={false}
                    zoomSpeed={0.5}
                    dampingFactor={0.05}
                />
            </Suspense>
        </Canvas>
    )
}
