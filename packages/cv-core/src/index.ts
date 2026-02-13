// ---------------------------------------------------------------------------
// @omni-twin/cv-core — Computer Vision & 3D Reconstruction
// ---------------------------------------------------------------------------
// Barrel re-export for all 12 CV sub-domains (CV-1 through CV-12).
// ---------------------------------------------------------------------------

// Infrastructure: types, vector/matrix utilities, PRNG
export * from './types.js';

// CV-1: Matterport Data Extraction
export * from './matterport/index.js';

// CV-2: Gaussian Splatting
export * from './gaussian-splatting/index.js';

// CV-3: Compositing & Depth Interop
export * from './compositing/index.js';

// CV-4: Photogrammetry & SfM
export * from './photogrammetry/index.js';

// CV-5: NeRF & Volume Rendering
export * from './nerf/index.js';

// CV-6: Mesh Processing
export * from './mesh/index.js';

// CV-7: Rendering & Performance
export * from './rendering/index.js';

// CV-8: Point Cloud Processing
export * from './point-cloud/index.js';

// CV-9: Depth Estimation
export * from './depth/index.js';

// CV-10: Segmentation
export * from './segmentation/index.js';

// CV-11: XR / AR
export * from './xr/index.js';

// CV-12: Asset Pipeline
export * from './asset-pipeline/index.js';
