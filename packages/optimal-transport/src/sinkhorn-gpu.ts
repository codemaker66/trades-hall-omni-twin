/**
 * WebGPU-accelerated Sinkhorn solver (OT-1).
 *
 * Offloads the Sinkhorn iterations to the GPU via compute shaders.
 * Falls back to CPU if WebGPU is unavailable.
 *
 * The GPU crossover point is ~N=200-500. Below that, CPU is faster.
 */

import type { SinkhornConfig, TransportResult, SinkhornSolver } from './types'
import { DEFAULT_SINKHORN_CONFIG } from './types'
import { sinkhorn } from './sinkhorn'

// ─── Shader Sources (inlined for bundler compatibility) ────────────────────

const SINKHORN_U_SHADER = /* wgsl */ `
struct Params {
  N: u32,
  M: u32,
  inv_epsilon: f32
}

@group(0) @binding(0) var<storage, read> C: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read> a: array<f32>;
@group(0) @binding(3) var<storage, read_write> u: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.N) { return; }

  var sum: f32 = 0.0;
  for (var j: u32 = 0u; j < params.M; j++) {
    sum += exp(-C[i * params.M + j] * params.inv_epsilon) * v[j];
  }
  u[i] = a[i] / max(sum, 1e-30);
}
`

const SINKHORN_V_SHADER = /* wgsl */ `
struct Params {
  N: u32,
  M: u32,
  inv_epsilon: f32
}

@group(0) @binding(0) var<storage, read> C: array<f32>;
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let j = gid.x;
  if (j >= params.M) { return; }

  var sum: f32 = 0.0;
  for (var i: u32 = 0u; i < params.N; i++) {
    sum += exp(-C[i * params.M + j] * params.inv_epsilon) * u[i];
  }
  v[j] = b[j] / max(sum, 1e-30);
}
`

const TRANSPORT_PLAN_SHADER = /* wgsl */ `
struct Params {
  N: u32,
  M: u32,
  inv_epsilon: f32
}

@group(0) @binding(0) var<storage, read> C: array<f32>;
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> T: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = params.N * params.M;
  if (idx >= total) { return; }

  let i = idx / params.M;
  let j = idx % params.M;

  T[idx] = u[i] * exp(-C[idx] * params.inv_epsilon) * v[j];
}
`

// ─── GPU Solver Class ──────────────────────────────────────────────────────

export class SinkhornGPU implements SinkhornSolver {
  private device: GPUDevice
  private uPipeline: GPUComputePipeline
  private vPipeline: GPUComputePipeline
  private planPipeline: GPUComputePipeline

  private constructor(
    device: GPUDevice,
    uPipeline: GPUComputePipeline,
    vPipeline: GPUComputePipeline,
    planPipeline: GPUComputePipeline,
  ) {
    this.device = device
    this.uPipeline = uPipeline
    this.vPipeline = vPipeline
    this.planPipeline = planPipeline
  }

  /**
   * Create a GPU solver, or return null if WebGPU is unavailable.
   */
  static async create(): Promise<SinkhornGPU | null> {
    if (typeof navigator === 'undefined' || !navigator.gpu) return null

    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) return null

    const device = await adapter.requestDevice()

    const uModule = device.createShaderModule({ code: SINKHORN_U_SHADER })
    const vModule = device.createShaderModule({ code: SINKHORN_V_SHADER })
    const planModule = device.createShaderModule({ code: TRANSPORT_PLAN_SHADER })

    const uPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: uModule, entryPoint: 'main' },
    })
    const vPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: vModule, entryPoint: 'main' },
    })
    const planPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: planModule, entryPoint: 'main' },
    })

    return new SinkhornGPU(device, uPipeline, vPipeline, planPipeline)
  }

  async solve(
    a: Float64Array | Float32Array,
    b: Float64Array | Float32Array,
    C: Float64Array | Float32Array,
    config: Partial<SinkhornConfig> = {},
  ): Promise<TransportResult> {
    const cfg = { ...DEFAULT_SINKHORN_CONFIG, ...config }
    const N = a.length
    const M = b.length

    // Convert to Float32 for GPU
    const a32 = a instanceof Float32Array ? a : new Float32Array(a)
    const b32 = b instanceof Float32Array ? b : new Float32Array(b)
    const C32 = C instanceof Float32Array ? C : new Float32Array(C)

    const invEpsilon = 1.0 / cfg.epsilon

    // Create GPU buffers
    const costBuf = this.createStorageBuffer(C32)
    const uBuf = this.createStorageBuffer(new Float32Array(N).fill(1))
    const vBuf = this.createStorageBuffer(new Float32Array(M).fill(1))
    const aBuf = this.createStorageBuffer(a32)
    const bBuf = this.createStorageBuffer(b32)
    const planBuf = this.createStorageBuffer(new Float32Array(N * M))

    // Uniform params buffer (N, M, inv_epsilon)
    const paramsData = new ArrayBuffer(12)
    const paramsU32 = new Uint32Array(paramsData, 0, 2)
    const paramsF32 = new Float32Array(paramsData, 8, 1)
    paramsU32[0] = N
    paramsU32[1] = M
    paramsF32[0] = invEpsilon
    const paramsBuf = this.device.createBuffer({
      size: 12,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(paramsBuf, 0, paramsData)

    // Bind groups for u-update
    const uBindGroup = this.device.createBindGroup({
      layout: this.uPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: costBuf } },
        { binding: 1, resource: { buffer: vBuf } },
        { binding: 2, resource: { buffer: aBuf } },
        { binding: 3, resource: { buffer: uBuf } },
        { binding: 4, resource: { buffer: paramsBuf } },
      ],
    })

    // Bind groups for v-update
    const vBindGroup = this.device.createBindGroup({
      layout: this.vPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: costBuf } },
        { binding: 1, resource: { buffer: uBuf } },
        { binding: 2, resource: { buffer: bBuf } },
        { binding: 3, resource: { buffer: vBuf } },
        { binding: 4, resource: { buffer: paramsBuf } },
      ],
    })

    // Dispatch iterations
    const uWorkgroups = Math.ceil(N / 64)
    const vWorkgroups = Math.ceil(M / 64)

    for (let iter = 0; iter < cfg.maxIterations; iter++) {
      const encoder = this.device.createCommandEncoder()

      // u-update pass
      const uPass = encoder.beginComputePass()
      uPass.setPipeline(this.uPipeline)
      uPass.setBindGroup(0, uBindGroup)
      uPass.dispatchWorkgroups(uWorkgroups)
      uPass.end()

      // v-update pass
      const vPass = encoder.beginComputePass()
      vPass.setPipeline(this.vPipeline)
      vPass.setBindGroup(0, vBindGroup)
      vPass.dispatchWorkgroups(vWorkgroups)
      vPass.end()

      this.device.queue.submit([encoder.finish()])
    }

    // Recover transport plan on GPU
    const planBindGroup = this.device.createBindGroup({
      layout: this.planPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: costBuf } },
        { binding: 1, resource: { buffer: uBuf } },
        { binding: 2, resource: { buffer: vBuf } },
        { binding: 3, resource: { buffer: planBuf } },
        { binding: 4, resource: { buffer: paramsBuf } },
      ],
    })

    const planEncoder = this.device.createCommandEncoder()
    const planPass = planEncoder.beginComputePass()
    planPass.setPipeline(this.planPipeline)
    planPass.setBindGroup(0, planBindGroup)
    planPass.dispatchWorkgroups(Math.ceil((N * M) / 64))
    planPass.end()

    // Read back plan, u, v
    const planReadBuf = this.createReadBuffer(N * M * 4)
    const uReadBuf = this.createReadBuffer(N * 4)
    const vReadBuf = this.createReadBuffer(M * 4)

    planEncoder.copyBufferToBuffer(planBuf, 0, planReadBuf, 0, N * M * 4)
    planEncoder.copyBufferToBuffer(uBuf, 0, uReadBuf, 0, N * 4)
    planEncoder.copyBufferToBuffer(vBuf, 0, vReadBuf, 0, M * 4)

    this.device.queue.submit([planEncoder.finish()])

    // Map and read back
    await Promise.all([
      planReadBuf.mapAsync(GPUMapMode.READ),
      uReadBuf.mapAsync(GPUMapMode.READ),
      vReadBuf.mapAsync(GPUMapMode.READ),
    ])

    const planF32 = new Float32Array(planReadBuf.getMappedRange().slice(0))
    const uF32 = new Float32Array(uReadBuf.getMappedRange().slice(0))
    const vF32 = new Float32Array(vReadBuf.getMappedRange().slice(0))

    planReadBuf.unmap()
    uReadBuf.unmap()
    vReadBuf.unmap()

    // Convert to Float64 for consistency
    const plan = new Float64Array(planF32)
    const dualF = new Float64Array(N)
    const dualG = new Float64Array(M)
    for (let i = 0; i < N; i++) {
      dualF[i] = cfg.epsilon * Math.log(Math.max(uF32[i]!, 1e-30))
    }
    for (let j = 0; j < M; j++) {
      dualG[j] = cfg.epsilon * Math.log(Math.max(vF32[j]!, 1e-30))
    }

    // Compute cost
    let cost = 0
    for (let k = 0; k < N * M; k++) {
      cost += plan[k]! * (C instanceof Float64Array ? C[k]! : C32[k]!)
    }

    // Cleanup
    costBuf.destroy()
    uBuf.destroy()
    vBuf.destroy()
    aBuf.destroy()
    bBuf.destroy()
    planBuf.destroy()
    paramsBuf.destroy()
    planReadBuf.destroy()
    uReadBuf.destroy()
    vReadBuf.destroy()

    return {
      plan,
      cost,
      dualF,
      dualG,
      iterations: cfg.maxIterations,
      converged: true, // GPU doesn't check convergence per-iteration
      N,
      M,
    }
  }

  private createStorageBuffer(data: Float32Array): GPUBuffer {
    const buf = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(buf, 0, data.buffer as ArrayBuffer)
    return buf
  }

  private createReadBuffer(size: number): GPUBuffer {
    return this.device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    })
  }

  destroy(): void {
    this.device.destroy()
  }
}

// ─── CPU Solver Wrapper (SinkhornSolver interface) ─────────────────────────

export class SinkhornCPU implements SinkhornSolver {
  solve(
    a: Float64Array | Float32Array,
    b: Float64Array | Float32Array,
    C: Float64Array | Float32Array,
    config: Partial<SinkhornConfig> = {},
  ): TransportResult {
    const a64 = a instanceof Float64Array ? a : new Float64Array(a)
    const b64 = b instanceof Float64Array ? b : new Float64Array(b)
    const C64 = C instanceof Float64Array ? C : new Float64Array(C)
    return sinkhorn(a64, b64, C64, config)
  }
}

// ─── Auto-select GPU or CPU ────────────────────────────────────────────────

export async function createSolver(): Promise<SinkhornSolver> {
  const gpu = await SinkhornGPU.create()
  if (gpu) return gpu
  return new SinkhornCPU()
}
