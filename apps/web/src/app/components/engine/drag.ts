/**
 * Pure drag computation functions.
 * Zero React, zero store — explicit inputs, explicit outputs.
 * Testable with unit tests using mocked Three.js objects.
 */
import * as THREE from 'three'

// ---------------------------------------------------------------------------
// Grid Snapping
// ---------------------------------------------------------------------------

/** Snap a position to a regular grid. */
export function snapToGrid(x: number, z: number, grid: number): { x: number; z: number } {
    return {
        x: Math.round(x / grid) * grid,
        z: Math.round(z / grid) * grid,
    }
}

/** Chair-specific snap: minimum 0.6 spacing, so single chairs align with area-dragged grids. */
export function snapChairPosition(
    x: number,
    z: number,
    snapGrid: number,
    snappingEnabled: boolean,
): { x: number; z: number } {
    const spacing = getChairSpacing(snapGrid, snappingEnabled)
    return {
        x: Math.round(x / spacing) * spacing,
        z: Math.round(z / spacing) * spacing,
    }
}

/** Compute the effective chair spacing (always >= 0.6). */
export function getChairSpacing(snapGrid: number, snappingEnabled: boolean): number {
    return Math.max(0.6, snappingEnabled ? snapGrid : 0.6)
}

// ---------------------------------------------------------------------------
// Area Drag Grid
// ---------------------------------------------------------------------------

/** Compute grid positions for chair area drag. */
export function computeGridPositions(
    startX: number,
    startZ: number,
    endX: number,
    endZ: number,
    spacing: number,
): { x: number; z: number }[] {
    const minX = Math.min(startX, endX)
    const maxX = Math.max(startX, endX)
    const minZ = Math.min(startZ, endZ)
    const maxZ = Math.max(startZ, endZ)
    const gridStartX = Math.round(minX / spacing) * spacing
    const gridStartZ = Math.round(minZ / spacing) * spacing
    const gridEndX = Math.round(maxX / spacing) * spacing
    const gridEndZ = Math.round(maxZ / spacing) * spacing
    const positions: { x: number; z: number }[] = []

    for (let x = gridStartX; x <= gridEndX + 0.001; x += spacing) {
        for (let z = gridStartZ; z <= gridEndZ + 0.001; z += spacing) {
            positions.push({ x, z })
        }
    }

    return positions
}

// ---------------------------------------------------------------------------
// Collision Detection
// ---------------------------------------------------------------------------

/** AABB overlap test with epsilon tolerance. */
export function boxesOverlap(a: THREE.Box3, b: THREE.Box3, eps = 0.001): boolean {
    return (
        a.max.x > b.min.x + eps &&
        a.min.x < b.max.x - eps &&
        a.max.y > b.min.y + eps &&
        a.min.y < b.max.y - eps &&
        a.max.z > b.min.z + eps &&
        a.min.z < b.max.z - eps
    )
}

// ---------------------------------------------------------------------------
// Trestle End-to-End Snapping
// ---------------------------------------------------------------------------

export interface TrestleSnapConfig {
    length: number
    depth: number
    alongTolerance: number
    acrossTolerance: number
    angleCos: number
}

export interface TrestleSnapResult {
    x: number
    z: number
}

/**
 * Try to snap a trestle table end-to-end with nearby trestles.
 * Returns snapped position or null if no snap found.
 *
 * Pass scratch vectors to avoid per-frame allocations.
 */
export function computeTrestleSnap(
    candidateX: number,
    candidateZ: number,
    rotationY: number,
    otherTrestles: readonly { position: [number, number, number]; rotation: [number, number, number] }[],
    config: TrestleSnapConfig,
    longAxis: THREE.Vector3,
    otherAxis: THREE.Vector3,
    otherShort: THREE.Vector3,
    yAxis: THREE.Vector3,
): TrestleSnapResult | null {
    longAxis.set(1, 0, 0).applyAxisAngle(yAxis, rotationY).normalize()

    let best: (TrestleSnapResult & { score: number }) | null = null

    for (const other of otherTrestles) {
        otherAxis.set(1, 0, 0).applyAxisAngle(yAxis, other.rotation[1]).normalize()
        if (Math.abs(longAxis.dot(otherAxis)) < config.angleCos) continue

        otherShort.set(-otherAxis.z, 0, otherAxis.x)
        const deltaX = candidateX - other.position[0]
        const deltaZ = candidateZ - other.position[2]
        const along = deltaX * otherAxis.x + deltaZ * otherAxis.z
        const across = deltaX * otherShort.x + deltaZ * otherShort.z
        const alongErr = Math.abs(Math.abs(along) - config.length)
        const acrossErr = Math.abs(across)

        if (alongErr > config.alongTolerance || acrossErr > config.acrossTolerance) continue

        const sign = along >= 0 ? 1 : -1
        const snapX = other.position[0] + otherAxis.x * config.length * sign
        const snapZ = other.position[2] + otherAxis.z * config.length * sign
        const score = alongErr + acrossErr

        if (!best || score < best.score) {
            best = { x: snapX, z: snapZ, score }
        }
    }

    return best ? { x: best.x, z: best.z } : null
}

// ---------------------------------------------------------------------------
// Platform Clamping
// ---------------------------------------------------------------------------

/**
 * Clamp a trestle table position so it stays within a platform's bounds.
 * Returns the clamped world-space x, z or null if no clamping needed.
 */
export function clampToPlatformBounds(
    finalX: number,
    finalY: number,
    finalZ: number,
    platformRoot: THREE.Object3D,
    itemGroup: THREE.Object3D,
    platformBox: THREE.Box3,
    trestleLength: number,
    trestleDepth: number,
    quatA: THREE.Quaternion,
    quatB: THREE.Quaternion,
    vecC: THREE.Vector3,
    vecD: THREE.Vector3,
    vecE: THREE.Vector3,
): { x: number; z: number } | null {
    if (platformBox.isEmpty()) return null

    platformRoot.getWorldQuaternion(quatA)
    itemGroup.getWorldQuaternion(quatB)

    const relQuat = quatA.invert().multiply(quatB)
    const axisX = vecC.set(1, 0, 0).applyQuaternion(relQuat)
    const axisZ = vecD.set(0, 0, 1).applyQuaternion(relQuat)

    const halfLen = trestleLength / 2
    const halfDepth = trestleDepth / 2
    const halfX = Math.abs(axisX.x) * halfLen + Math.abs(axisZ.x) * halfDepth
    const halfZ = Math.abs(axisX.z) * halfLen + Math.abs(axisZ.z) * halfDepth

    const localPos = platformRoot.worldToLocal(vecE.set(finalX, finalY, finalZ))
    const minX = platformBox.min.x + halfX
    const maxX = platformBox.max.x - halfX
    const minZ = platformBox.min.z + halfZ
    const maxZ = platformBox.max.z - halfZ

    if (minX <= maxX) localPos.x = THREE.MathUtils.clamp(localPos.x, minX, maxX)
    else localPos.x = (platformBox.min.x + platformBox.max.x) / 2

    if (minZ <= maxZ) localPos.z = THREE.MathUtils.clamp(localPos.z, minZ, maxZ)
    else localPos.z = (platformBox.min.z + platformBox.max.z) / 2

    const clampedWorld = platformRoot.localToWorld(localPos)
    return { x: clampedWorld.x, z: clampedWorld.z }
}
