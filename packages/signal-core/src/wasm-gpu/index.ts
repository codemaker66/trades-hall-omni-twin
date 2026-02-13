// SP-12: WASM/WebGPU DSP
export {
  createFFTModule,
  FallbackFFT,
  type WASMFFTModule,
} from './wasm-fft.js';

export {
  createWebGPUFFTModule,
  STOCKHAM_FFT_WGSL,
  type WebGPUFFTModule,
  type WebGPUFFTPipeline,
} from './webgpu-fft.js';
