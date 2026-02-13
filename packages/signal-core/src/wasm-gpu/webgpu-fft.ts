// ---------------------------------------------------------------------------
// SP-12: WebGPU FFT Interface Types
// ---------------------------------------------------------------------------
// Stockham auto-sort FFT algorithm for WebGPU compute shaders.
// log₂(N) dispatches with butterfly operations.
// 1M-point: ~1-5ms GPU (excl. transfer) vs ~50-100ms pure JS.

import type { WebGPUFFTConfig } from '../types.js';

/**
 * Interface for a WebGPU-backed FFT module.
 * The actual implementation would use WGSL compute shaders.
 */
export interface WebGPUFFTModule {
  /** Check if WebGPU is available */
  isAvailable(): boolean;
  /** Initialize GPU resources (adapter, device, pipelines) */
  initialize(): Promise<void>;
  /** Compute forward FFT on GPU */
  fftForward(input: Float32Array): Promise<Float32Array>;
  /** Compute inverse FFT on GPU */
  fftInverse(input: Float32Array): Promise<Float32Array>;
  /** Release GPU resources */
  dispose(): void;
}

/**
 * WGSL shader source for Stockham auto-sort FFT.
 * Each dispatch performs one butterfly stage.
 */
export const STOCKHAM_FFT_WGSL = /* wgsl */ `
struct FFTParams {
  n: u32,
  stage: u32,
  direction: f32, // 1.0 for forward, -1.0 for inverse
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: FFTParams;
@group(0) @binding(1) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let n = params.n;
  if (idx >= n / 2u) { return; }

  let stage = params.stage;
  let halfStep = 1u << stage;
  let step = halfStep << 1u;

  let group = idx / halfStep;
  let pair = idx % halfStep;
  let evenIdx = group * step + pair;
  let oddIdx = evenIdx + halfStep;

  // Twiddle factor: e^{-j·2π·pair/step}
  let angle = params.direction * 2.0 * 3.14159265358979 * f32(pair) / f32(step);
  let twiddle = vec2<f32>(cos(angle), sin(angle));

  let even = input[evenIdx];
  let odd = input[oddIdx];

  // Complex multiply: odd * twiddle
  let t = vec2<f32>(
    odd.x * twiddle.x - odd.y * twiddle.y,
    odd.x * twiddle.y + odd.y * twiddle.x,
  );

  output[evenIdx] = even + t;
  output[oddIdx] = even - t;
}
`;

/**
 * Configuration for WebGPU FFT pipeline.
 * In production, this would create actual GPU resources.
 */
export interface WebGPUFFTPipeline {
  device: unknown; // GPUDevice
  pipeline: unknown; // GPUComputePipeline
  bindGroupLayout: unknown; // GPUBindGroupLayout
  paramsBuffer: unknown; // GPUBuffer
  inputBuffer: unknown; // GPUBuffer
  outputBuffer: unknown; // GPUBuffer
  size: number;
}

/**
 * Placeholder factory for WebGPU FFT module.
 * In production, would check navigator.gpu and create actual GPU pipelines.
 */
export function createWebGPUFFTModule(_config: WebGPUFFTConfig): WebGPUFFTModule {
  return {
    isAvailable() {
      // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
      const nav = (globalThis as Record<string, unknown>).navigator;
      return typeof globalThis !== 'undefined' && typeof nav === 'object' && nav !== null && 'gpu' in nav;
    },
    async initialize() {
      // Would create GPU adapter, device, compile shaders, create pipelines
    },
    async fftForward(_input: Float32Array) {
      throw new Error('WebGPU FFT not implemented — use FallbackFFT');
    },
    async fftInverse(_input: Float32Array) {
      throw new Error('WebGPU FFT not implemented — use FallbackFFT');
    },
    dispose() {
      // Release GPU resources
    },
  };
}
