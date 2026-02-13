// SP-11: Streaming Architecture
export { SlidingDFT } from './sliding-dft.js';
export { goertzel, StreamingGoertzel, multiGoertzel } from './goertzel.js';
export { RingBuffer } from './ring-buffer.js';
export { StreamProcessor, SpectralSmoother, type SpectralFrame } from './stream-processor.js';
