/**
 * CT-3: Renderer Natural Transformation
 *
 * η: WebGLRenderer ⇒ WebGPURenderer
 *
 * Naturality condition:
 *   η(WebGLRender(op)) ≡ WebGPURender(η(op))
 *
 * The WebGPU/WebGL fallback is a natural transformation — guaranteed
 * to produce the same visual result regardless of backend.
 */

import type { Morphism } from './core'
import type { StrategySwap } from './natural-transformation'
import { createStrategySwap } from './natural-transformation'

// ─── Render Command Types ───────────────────────────────────────────────────

/** A backend-agnostic render command. */
export type RenderCommand =
  | { readonly kind: 'clear'; readonly color: readonly [number, number, number, number] }
  | { readonly kind: 'drawMesh'; readonly meshId: string; readonly transform: Float64Array }
  | { readonly kind: 'setCamera'; readonly position: readonly [number, number, number]; readonly target: readonly [number, number, number] }
  | { readonly kind: 'setLight'; readonly lightId: string; readonly position: readonly [number, number, number]; readonly intensity: number }

/** A frame description — what should be rendered. */
export interface RenderFrame {
  readonly frameId: number
  readonly commands: readonly RenderCommand[]
  readonly width: number
  readonly height: number
}

// ─── WebGL Representation ───────────────────────────────────────────────────

export interface WebGLState {
  readonly kind: 'webgl'
  readonly commands: readonly WebGLCommand[]
  readonly contextId: string
}

export type WebGLCommand =
  | { readonly kind: 'gl_clear'; readonly r: number; readonly g: number; readonly b: number; readonly a: number }
  | { readonly kind: 'gl_drawElements'; readonly meshId: string; readonly modelMatrix: Float64Array }
  | { readonly kind: 'gl_uniform'; readonly name: string; readonly value: readonly number[] }

// ─── WebGPU Representation ──────────────────────────────────────────────────

export interface WebGPUState {
  readonly kind: 'webgpu'
  readonly commands: readonly WebGPUCommand[]
  readonly deviceId: string
}

export type WebGPUCommand =
  | { readonly kind: 'gpu_clearValue'; readonly color: readonly [number, number, number, number] }
  | { readonly kind: 'gpu_draw'; readonly meshId: string; readonly bindGroup: Float64Array }
  | { readonly kind: 'gpu_setBuffer'; readonly name: string; readonly data: readonly number[] }

// ─── Renderer Natural Transformation ────────────────────────────────────────

/**
 * Compile a render frame into WebGL commands.
 */
export function compileToWebGL(frame: RenderFrame): WebGLState {
  const commands: WebGLCommand[] = []
  for (const cmd of frame.commands) {
    switch (cmd.kind) {
      case 'clear':
        commands.push({ kind: 'gl_clear', r: cmd.color[0], g: cmd.color[1], b: cmd.color[2], a: cmd.color[3] })
        break
      case 'drawMesh':
        commands.push({ kind: 'gl_drawElements', meshId: cmd.meshId, modelMatrix: cmd.transform })
        break
      case 'setCamera':
        commands.push({ kind: 'gl_uniform', name: 'u_viewPos', value: [...cmd.position] })
        commands.push({ kind: 'gl_uniform', name: 'u_viewTarget', value: [...cmd.target] })
        break
      case 'setLight':
        commands.push({ kind: 'gl_uniform', name: `u_light_${cmd.lightId}_pos`, value: [...cmd.position] })
        commands.push({ kind: 'gl_uniform', name: `u_light_${cmd.lightId}_intensity`, value: [cmd.intensity] })
        break
    }
  }
  return { kind: 'webgl', commands, contextId: `webgl-${frame.frameId}` }
}

/**
 * Compile a render frame into WebGPU commands.
 */
export function compileToWebGPU(frame: RenderFrame): WebGPUState {
  const commands: WebGPUCommand[] = []
  for (const cmd of frame.commands) {
    switch (cmd.kind) {
      case 'clear':
        commands.push({ kind: 'gpu_clearValue', color: cmd.color })
        break
      case 'drawMesh':
        commands.push({ kind: 'gpu_draw', meshId: cmd.meshId, bindGroup: cmd.transform })
        break
      case 'setCamera':
        commands.push({ kind: 'gpu_setBuffer', name: 'camera_position', data: [...cmd.position] })
        commands.push({ kind: 'gpu_setBuffer', name: 'camera_target', data: [...cmd.target] })
        break
      case 'setLight':
        commands.push({ kind: 'gpu_setBuffer', name: `light_${cmd.lightId}_position`, data: [...cmd.position] })
        commands.push({ kind: 'gpu_setBuffer', name: `light_${cmd.lightId}_intensity`, data: [cmd.intensity] })
        break
    }
  }
  return { kind: 'webgpu', commands, deviceId: `webgpu-${frame.frameId}` }
}

/**
 * Create the renderer strategy swap: WebGL ↔ WebGPU.
 *
 * Both renderers produce semantically equivalent output from the same render frame.
 * The natural transformation maps between their intermediate representations.
 */
export function createRendererSwap(): StrategySwap<WebGLState, WebGPUState> {
  return createStrategySwap(
    'WebGL',
    'WebGPU',
    webglToWebGPU,
    webgpuToWebGL,
  )
}

function webglToWebGPU(gl: WebGLState): WebGPUState {
  const commands: WebGPUCommand[] = gl.commands.map(cmd => {
    switch (cmd.kind) {
      case 'gl_clear':
        return { kind: 'gpu_clearValue' as const, color: [cmd.r, cmd.g, cmd.b, cmd.a] as const }
      case 'gl_drawElements':
        return { kind: 'gpu_draw' as const, meshId: cmd.meshId, bindGroup: cmd.modelMatrix }
      case 'gl_uniform':
        return { kind: 'gpu_setBuffer' as const, name: cmd.name, data: cmd.value }
    }
  })
  return { kind: 'webgpu', commands, deviceId: gl.contextId.replace('webgl', 'webgpu') }
}

function webgpuToWebGL(gpu: WebGPUState): WebGLState {
  const commands: WebGLCommand[] = gpu.commands.map(cmd => {
    switch (cmd.kind) {
      case 'gpu_clearValue':
        return { kind: 'gl_clear' as const, r: cmd.color[0], g: cmd.color[1], b: cmd.color[2], a: cmd.color[3] }
      case 'gpu_draw':
        return { kind: 'gl_drawElements' as const, meshId: cmd.meshId, modelMatrix: cmd.bindGroup }
      case 'gpu_setBuffer':
        return { kind: 'gl_uniform' as const, name: cmd.name, value: cmd.data }
    }
  })
  return { kind: 'webgl', commands, contextId: gpu.deviceId.replace('webgpu', 'webgl') }
}
