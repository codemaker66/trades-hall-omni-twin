import type { ThreeElement } from '@react-three/fiber'
import type { ShaderMaterial } from 'three'

declare module '@react-three/fiber' {
    interface ThreeElements {
        gridShaderMaterial: ThreeElement<typeof ShaderMaterial>
    }
}
