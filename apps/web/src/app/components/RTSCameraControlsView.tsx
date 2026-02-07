'use client'

import { useRef, useEffect } from 'react'
import { useThree, useFrame } from '@react-three/fiber'
import { MapControls } from '@react-three/drei'
import { useVenueStore } from '../../store'
import * as THREE from 'three'
import type { MapControls as MapControlsImpl } from 'three-stdlib'
import { CAMERA } from '../../config/scene'

// Scratch vectors — reused every frame to avoid GC
const _forward = new THREE.Vector3()
const _right = new THREE.Vector3()
const _moveVec = new THREE.Vector3()

export const RTSCameraControls = () => {
    const controlsRef = useRef<MapControlsImpl | null>(null)
    const { camera } = useThree()
    const isDragging = useVenueStore((state) => state.isDragging)

    const keys = useRef({
        w: false,
        a: false,
        s: false,
        d: false,
    })

    useEffect(() => {
        const onKeyDown = (e: KeyboardEvent) => {
            if ((e.target as HTMLElement).tagName === 'INPUT') return

            switch (e.key.toLowerCase()) {
                case 'w': keys.current.w = true; break
                case 'a': keys.current.a = true; break
                case 's': keys.current.s = true; break
                case 'd': keys.current.d = true; break
            }
        }

        const onKeyUp = (e: KeyboardEvent) => {
            switch (e.key.toLowerCase()) {
                case 'w': keys.current.w = false; break
                case 'a': keys.current.a = false; break
                case 's': keys.current.s = false; break
                case 'd': keys.current.d = false; break
            }
        }

        window.addEventListener('keydown', onKeyDown)
        window.addEventListener('keyup', onKeyUp)

        return () => {
            window.removeEventListener('keydown', onKeyDown)
            window.removeEventListener('keyup', onKeyUp)
        }
    }, [])

    useFrame((_, delta) => {
        if (!controlsRef.current) return

        const speed = CAMERA.mapControls.wasdSpeed * delta
        const { w, a, s, d } = keys.current

        if (w || a || s || d) {
            _forward.set(0, 0, -1).applyQuaternion(camera.quaternion)
            _forward.y = 0
            _forward.normalize()

            _right.set(1, 0, 0).applyQuaternion(camera.quaternion)
            _right.y = 0
            _right.normalize()

            _moveVec.set(0, 0, 0)
            if (w) _moveVec.add(_forward)
            if (s) _moveVec.sub(_forward)
            if (d) _moveVec.add(_right)
            if (a) _moveVec.sub(_right)

            if (_moveVec.lengthSq() > 0) {
                _moveVec.normalize().multiplyScalar(speed)
                camera.position.add(_moveVec)
                controlsRef.current.target.add(_moveVec)
            }
        }
    })

    return (
        <MapControls
            ref={controlsRef}
            enabled={!isDragging}
            makeDefault
            minPolarAngle={CAMERA.mapControls.minPolarAngle}
            maxPolarAngle={CAMERA.mapControls.maxPolarAngle}
            minDistance={CAMERA.mapControls.minDistance}
            maxDistance={CAMERA.mapControls.maxDistance}
            enableDamping={true}
            dampingFactor={CAMERA.mapControls.dampingFactor}
            screenSpacePanning={false}
            mouseButtons={{
                LEFT: CAMERA.DISABLED_MOUSE,
                MIDDLE: THREE.MOUSE.DOLLY,
                RIGHT: THREE.MOUSE.ROTATE
            }}
        />
    )
}
