import * as THREE from 'three'

/**
 * Shared scratch THREE.js objects for reuse in hot paths (event handlers, useFrame).
 * Never store references to these across frames — they are overwritten on the next call.
 * Each consumer should use them within a single synchronous scope.
 */

export const _raycaster = new THREE.Raycaster()
export const _raycaster2 = new THREE.Raycaster() // for nested usage (e.g. getChairY inside buildPreviewGrid)

export const _plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)

export const _vec3A = new THREE.Vector3()
export const _vec3B = new THREE.Vector3()
export const _vec3C = new THREE.Vector3()
export const _vec3D = new THREE.Vector3()
export const _vec3E = new THREE.Vector3()

export const _vec2A = new THREE.Vector2()

export const _box3A = new THREE.Box3()
export const _box3B = new THREE.Box3()

export const _quatA = new THREE.Quaternion()
export const _quatB = new THREE.Quaternion()

export const _downDir = new THREE.Vector3(0, -1, 0)
export const _yAxis = new THREE.Vector3(0, 1, 0)
