# TECHNIQUE_11_COMPUTER_VISION_3D_RECONSTRUCTION.md — Computer Vision & 3D for Venue Digital Twins

> ## CRITICAL: NEVER OMIT, SIMPLIFY, OR WEAKEN
>
> Claude Code must NEVER:
> - Omit features, techniques, or subsystems to fit within a single session
> - Simplify implementations to make them "easier" or "more practical"
> - Replace a specified technique with a simpler alternative without explicit approval
> - Skip sections because they seem redundant or low-priority
> - Truncate implementations to keep diffs small — instead, split across commits
>
> If a task is too large for one session: split it into sub-tasks across multiple
> sessions. Document what remains in PROGRESS.md and pick it up next session.
> The project must contain the FULL depth of every technique specified.
> "Good enough" is not good enough. Every feature exists at maximum fidelity.

> **Purpose**: Feed this to Claude Code after the main phases and earlier techniques
> are complete. Implements the full 3D venue pipeline: Matterport data extraction,
> 3D Gaussian Splatting rendering, browser splat+mesh compositing, photogrammetry
> furniture capture, mesh processing, R3F rendering, point cloud processing,
> depth estimation fallback, object segmentation, WebXR, and asset pipeline.
>
> **How to use**: Tell Claude Code: "Read TECHNIQUE_11_COMPUTER_VISION_3D_RECONSTRUCTION.md
> and implement incrementally, starting from CV-1."
>
> **This is 12 sub-domains. Implement ALL of them. Do not skip any.**

---

## Key Papers (Referenced Throughout)

- Kerbl et al. (2023). 3D Gaussian Splatting. SIGGRAPH. arXiv:2308.04079
- Guédon & Lepetit (2024). SuGaR. CVPR. arXiv:2311.12775
- Huang et al. (2024). 2DGS. SIGGRAPH. arXiv:2403.17888
- Yu et al. (2024). GOF Gaussian Opacity Fields. SIGGRAPH Asia. arXiv:2404.10772
- Mildenhall et al. (2020). NeRF. ECCV. arXiv:2003.08934
- Müller et al. (2022). Instant-NGP. SIGGRAPH. arXiv:2201.05989
- Wang et al. (2021). NeuS. NeurIPS. arXiv:2106.10689
- Yang et al. (2024). Depth Anything v2. NeurIPS. arXiv:2406.09414
- Apple (2025). Depth Pro. ICLR. arXiv:2410.02073
- Wang et al. (2024). DUSt3R. CVPR. arXiv:2312.14132
- Leroy et al. (2024). MASt3R. ECCV. arXiv:2406.09756
- Kirillov et al. (2023). SAM. Meta
- Meta (2024). SAM 2. arXiv:2408.00714
- Liu et al. (2023). Grounding DINO. ECCV 2024. arXiv:2303.05499
- Wu et al. (2024). Point Transformer V3. CVPR Oral. arXiv:2312.10035
- Garland & Heckbert (1997). Quadric Error Metric. SIGGRAPH

---

## Architecture Overview

```
apps/
  ml-api/
    src/
      cv/
        matterport/
          api_client.py               — GraphQL Model API v3 integration
          export_handler.py           — MatterPak/E57 export + processing
          floor_plan.py               — Extracted floor plan processing
        gaussian_splatting/
          training.py                 — gsplat/Splatfacto training pipeline
          compression.py             — SPZ/HAC++/SOG compression
          mesh_extraction.py          — SuGaR/2DGS/GOF → mesh
        photogrammetry/
          capture_guide.py            — Optimal capture parameters
          kiri_pipeline.py            — Kiri Engine API integration
          realitycapture.py           — RealityCapture/RealityScan pipeline
          meshroom.py                 — AliceVision open-source pipeline
        nerf/
          nerfstudio.py               — Nerfstudio Nerfacto/Splatfacto
          neus_mesh.py                — NeuS2/BakedSDF mesh extraction
        point_cloud/
          cleaning.py                 — Statistical outlier, voxel downsample
          registration.py             — RANSAC global + colored ICP alignment
          room_boundary.py            — Floor detect → wall slice → alpha shape
          door_window.py              — Architectural feature detection
          semantic_seg.py             — Point Transformer V3, MinkowskiNet
        depth/
          depth_anything.py           — Depth Anything v2 (DINOv2 encoder)
          depth_pro.py                — Depth Pro metric depth (Apple)
          dust3r.py                   — DUSt3R/MASt3R 3D from image pairs
          phone_fallback.py           — iPhone LiDAR + RoomPlan pipeline
        segmentation/
          sam2.py                     — SAM 2 universal segmentation
          grounded_sam.py             — Grounding DINO + SAM (text-prompted)
          yolo.py                     — YOLO11 real-time detection
          mask3d.py                   — 3D instance segmentation
          dimension.py                — Real-world measurement from segmented objects
        mesh/
          decimation.py               — Quadric Error Metric via Blender bpy
          uv_unwrap.py                — Smart UV / xatlas repacking
          baking.py                   — Normal/AO/diffuse map baking
          gltf_optimize.py            — gltf-transform Draco/KTX2/meshopt
          pbr_materials.py            — PBR channel generation
        asset_pipeline/
          quality_gates.py            — Khronos validator + auditor
          metadata_schema.py          — Physical + technical asset metadata
          dvc_manager.py              — DVC + Cloudflare R2 versioning
          cdn_config.py               — R2 CDN with progressive loading
      routes/
        cv.py                         — FastAPI endpoints

packages/
  venue-renderer/                     — Three.js/R3F venue rendering
    src/
      splat/
        SplatScene.tsx                — Spark SplatMesh venue loading
        SplatLoader.ts                — SPZ/PLY/SPLAT format handling
        DepthCompositor.ts            — Splat + mesh depth interop
      furniture/
        FurnitureLoader.tsx           — GLB with Draco/KTX2 decoding
        DragDrop.tsx                  — TransformControls + floor snap
        InstancedFurniture.tsx        — InstancedMesh for repeated items
      scene/
        VenueScene.tsx                — Complete venue composition
        LODManager.tsx                — Progressive LOD switching
        PerformanceMonitor.tsx        — Adaptive quality control
        ShadowStrategy.tsx            — Accumulative/Contact/CSM shadows
      postprocessing/
        Effects.tsx                   — N8AO, Bloom, tone mapping
      xr/
        ARPreview.tsx                 — WebXR AR furniture placement
        VRWalkthrough.tsx             — WebXR VR venue exploration
        ModelViewer.tsx               — Google Model Viewer GLB/USDZ
      types.ts                        — VenueScan, FurnitureAsset types
```

### Python Dependencies

```
open3d>=0.19.0                 # Point cloud processing, ICP, E57
nerfstudio>=1.1.5              # Splatfacto, Nerfacto, ns-train/ns-export
gsplat>=1.5                    # 3DGS training (4× less GPU mem)
trimesh>=4.5                   # Mesh processing
pymeshlab>=2023.12             # Mesh quality checks
ultralytics>=8.3               # YOLO11 detection
SAM-2>=1.0                     # SAM 2 segmentation
torch>=2.4                     # PyTorch
torchvision>=0.19              # Vision transforms
pytorch3d>=0.7.8               # Differentiable 3D operators
dvc[s3]>=3.55                  # Asset versioning with R2 backend
```

### NPM Dependencies

```json
{
  "@sparkjsdev/spark": "^0.1.10",
  "three": "^0.182.0",
  "@react-three/fiber": "^9.5.0",
  "@react-three/drei": "^10.7.7",
  "@react-three/postprocessing": "^3.0.4",
  "@gltf-transform/cli": "^4.3.0",
  "@google/model-viewer": "^4.1.0",
  "n8ao": "^1.9",
  "xatlasjs": "^0.2.0",
  "@three.ez/instanced-mesh": "^0.3.11"
}
```

---

## CV-1: Matterport Pro3 Data Extraction

### What to Build

Interface with Matterport's cloud API, extract all available data
(meshes, point clouds, floor plans, panoramas), and feed into the
3DGS and mesh processing pipelines.

```python
# apps/ml-api/src/cv/matterport/api_client.py

"""
Matterport Pro3 Scanner:
  904nm time-of-flight laser, 100,000 points/second
  Full 360°×295° sweep in <20 seconds
  Per scan: ~1.5M range points + 134.2MP HDR panorama (5 exposures)
  Cortex AI: auto-register, textured mesh, floor plan, dollhouse

Spatial accuracy: ±20mm at 10m
  IEEE 2025 study: 6-floor building (17,567m², 1,099 scans) → RMSE 11.8mm
  vs Leica BLK360 G1 ±4mm, FARO Focus S 150 ±1mm (at ~4× price)

Objects <5cm optimized out of meshes.
Reflective surfaces (glass, mirrors) create holes → use Capture app marking.
Multi-floor: last scan at top of staircase, alignment per-floor.

GraphQL API: https://api.matterport.com/api/models/graph
SDK Bundle v26.1.3 (Jan 22, 2026): self-hosted Showcase embedding,
  direct Three.js scene graph access, DAE/FBX/OBJ/glTF model loading.
Web Component (Beta): modern alternative to iframe Bundle.

Export paths:
  MatterPak ($): OBJ + MTL + JPG textures, XYZ point cloud (~1% of E57), floor plans
  E57 ($149/space): 10× denser point clouds, ASTM E2807 format,
    500MB-1.5GB per space, range up to 100m
  Raw depth maps + per-scan panoramas: LOCKED in Matterport cloud

Post CoStar acquisition (Feb 28, 2025, $1.6B):
  Property Intelligence (auto measurements, room dims, ceiling heights)
  Defurnish feature via Genesis AI
  Marketing Cloud platform
  SDK and API remain active under CoStar.
"""
```

---

## CV-2: 3D Gaussian Splatting Training and Compression

### What to Build

Train 3DGS from venue scans, compress for web delivery, extract meshes
for interaction layer.

```python
# apps/ml-api/src/cv/gaussian_splatting/training.py

"""
3DGS (Kerbl et al., SIGGRAPH 2023, arXiv:2308.04079):
  Millions of anisotropic 3D Gaussians, each with 59 attributes:
  - Position μ ∈ ℝ³
  - Covariance → scaling s ∈ ℝ³ + rotation quaternion q ∈ ℝ⁴
  - Opacity α ∈ ℝ¹
  - Color: degree-3 spherical harmonics (48 coefficients RGB)

  Custom CUDA tile-based rasterizer:
  Project to 2D via EWA splatting → alpha-blend back-to-front.
  Training: COLMAP-posed images → ~35 min on A6000.
  Adaptive density control: clone, split, prune (opacity < 0.005) over 30K iter.

  Mip-NeRF 360 benchmarks: PSNR 27.21, SSIM 0.815, LPIPS 0.214
  at 734MB model size, ≥100 fps at 1080p.
  Matches Mip-NeRF 360 quality at fraction of training time (35min vs ~48hr).

Libraries:
  gsplat: pip install gsplat (Nerfstudio, Apache 2.0, 4× less GPU mem)
  Nerfstudio Splatfacto: ns-train splatfacto --data <path>
  PlayCanvas SuperSplat (superspl.at): browser editor for crop/clean/publish

Anti-aliasing:
  Mip-Splatting (arXiv:2311.16493, CVPR 2024 Best Student Paper):
    Constrains min Gaussian size to training-view frequency.
  Analytic-Splatting (arXiv:2403.11056):
    Integrates Gaussian signal within pixel window.

Editing: GaussianEditor (arXiv:2311.14521, CVPR 2024):
  Semantic tracing + text-driven editing in ~6 min/scene.

Feed-forward: MVSplat (arXiv:2403.14627, ECCV 2024 Oral):
  Regress Gaussians from image pairs at 22 fps, 10× fewer params.
"""

# apps/ml-api/src/cv/gaussian_splatting/compression.py

"""
Compression for web delivery (vanilla: 411-734MB):

| Method         | PSNR       | Compressed Size |
|----------------|-----------|-----------------|
| HAC++          | 27.60-27.82| 8.7-19.4 MB    |
| Compact3D/CompGS| ~27.03   | 13-19 MB        |
| LightGaussian  | 27.28     | 42 MB           |
| gsplat MCMC+SOG| 27.29     | 16 MB           |
| SPZ (Niantic)  | ~equivalent| ~25 MB (90% smaller than PLY) |

SPZ format: fixed-point quantization, smallest-3 quaternion encoding, gzip.
SOG (PlayCanvas): WebP-encoded property images, 15-20MB.
.splat (antimatter15): 32 bytes/splat (SH0 only), ~32MB for 1M splats.

August 2025: Khronos and OGC announced adding 3DGS to glTF ecosystem
with SPZ as compact container format → industry standardization.
"""

# apps/ml-api/src/cv/gaussian_splatting/mesh_extraction.py

"""
3DGS-to-mesh conversion for physics/interaction layer:

SuGaR (Guédon & Lepetit, CVPR 2024, arXiv:2311.12775):
  Regularize Gaussians to align with surfaces → Poisson reconstruction.
  Textured OBJ in ~30 min, configurable density (200K-1M vertices).

2DGS (Huang et al., SIGGRAPH 2024, arXiv:2403.17888):
  Collapse 3D Gaussians to oriented planar disks.
  Perspective-correct splatting + TSDF-based mesh extraction.
  Better surface normals than SuGaR.

GOF (Yu et al., SIGGRAPH Asia 2024, arXiv:2404.10772):
  Continuous opacity field → marching tetrahedra at 0.5 level set.
  Surpasses SuGaR and 2DGS in background reconstruction.

Recommended: 3DGS for rendering + SuGaR/2DGS mesh for interaction.
"""
```

---

## CV-3: Browser Rendering — Splat + Mesh Compositing

### What to Build

Render venue as Gaussian splats via Spark, composite GLB furniture
through depth-buffer interop.

```typescript
// packages/venue-renderer/src/splat/SplatScene.tsx

/**
 * Spark (@sparkjsdev/spark 0.1.10):
 * Three.js-native SplatMesh (Object3D).
 * Formats: .PLY, .SPZ, .SPLAT, .KSPLAT, .SOG
 * Composites naturally with standard Three.js meshes.
 *
 * World Labs successor to GaussianSplats3D.
 */

import { SplatMesh } from "@sparkjsdev/spark";

// Usage:
// const venue = new SplatMesh({ url: "venue.spz" });
// scene.add(venue);
// loader.load('chair.glb', (gltf) => scene.add(gltf.scene));

/**
 * Legacy GaussianSplats3D (@mkkellogg/gaussian-splats-3d 0.4.7):
 * DropInViewer mode adds splats to existing Three.js scene.
 * threeScene parameter: opaque meshes render first (writing depth),
 * then splats with depth test enabled — correctly occluding splats behind furniture.
 *
 * Splat rendering pipeline:
 * 1. Sort Gaussians back-to-front (counting sort in Web Worker, WASM-accelerated)
 * 2. Project each to 2D screen-space ellipse via Jacobian of projection × 3D covariance
 * 3. Rasterize as textured quads with Gaussian alpha falloff
 * 4. Alpha-blend with ONE, ONE_MINUS_SRC_ALPHA
 * Depth test ON, depth write OFF — critical constraint for mesh compositing.
 *
 * Performance:
 * WebGL: 500K-1M splats at 30-60fps desktop (GTX 1060+)
 *   CPU sort at 10-50ms is bottleneck above 1M splats
 * WebGPU: 200+ fps RTX 3090, ~130fps Radeon R9 380 (8yr old!)
 *   Compute shaders eliminate sort bottleneck
 * Mobile: stay under 500K splats
 * Chrome/Chromium: WebGPU supported. Firefox/Safari: experimental.
 *
 * Raycasting against splats (for furniture placement click targets):
 * 1. GaussianSplats3D built-in mesh cursor
 * 2. gsplat.js IntersectionTester
 * 3. Depth buffer readback: render splats with depth write to offscreen target,
 *    readPixels at click position, unproject to 3D
 *
 * RECOMMENDED compositing recipe:
 * - Venue: native splats via Spark (photorealistic appearance)
 * - Collision proxy: SuGaR mesh (raycasting + floor detection)
 * - Furniture placement: snap to proxy mesh floor, physics on proxy
 */
```

---

## CV-4: Photogrammetry Furniture Capture Pipeline

### What to Build

Capture furniture with phone → process → optimize → web-ready GLB.

```python
# apps/ml-api/src/cv/photogrammetry/capture_guide.py

"""
Kiri Engine (kiriengine.app, iOS/Android/Web):
  Cloud processing — phone is capture device only.
  Modes: Photo Scan, NSR (Neural Surface Reconstruction),
  3DGS (video → dense points → Gaussian splat, 7-20min processing),
  LiDAR (iPhone Pro).

  FIRST app to offer 3DGS-to-mesh conversion (co-developed with Chongjie Ye).
  v2.0: predicts normal + depth maps, eliminates holes from shiny/transparent.
  Exports: OBJ, STL, FBX, glTF, GLB, USDZ, PLY, XYZ.
  Pro ($14.99/mo): quad-mesh retopology, AI PBR material generation
    (albedo, roughness, metallic, normal), auto object masking.

Optimal furniture capture:
  70-120 photos, 60-80% overlap, 3-4 orbits at different heights.
  Diffuse lighting (overcast or softboxes).
  Camera: ISO low, f/8-f/11, 35-50mm equivalent.
  Cross-polarized flash for clean PBR.
  Background: some texture (not pure white) for SfM matching.

Desktop alternatives (maximum quality):
  RealityScan 2.0 (Epic): FREE under $1M revenue, $1,250/seat/yr above.
    Speed + quality gold standard.
  Agisoft Metashape 2.3.x: $179 Standard, $3,499 Professional.
    One-time perpetual. Scientific precision.
  Meshroom 2025.1.0 (free, AliceVision 3.3.0):
    AI semantic segmentation, mrGSplat (Gaussian Splatting),
    intrinsic image decomposition. Requires NVIDIA CUDA GPU.

Photogrammetry-to-web compression:
  Raw 2M-tri OBJ (200-500MB) → decimate to 50K tri → bake 2K textures
  → Draco + WebP via gltf-transform → 0.5-5MB GLB
"""

"""
SfM → MVS → Mesh → Texture pipeline stages:
1. Structure from Motion: SIFT features → matching → RANSAC
   → triangulation → bundle adjustment → sparse points + camera poses
2. Multi-View Stereo: per-pixel depth via photometric consistency
   → fusion → dense point cloud
3. Surface Reconstruction: Poisson or Delaunay meshing
4. Texturing: visibility-weighted photo projection onto mesh
"""
```

---

## CV-5: NeRF Decision Matrix

```python
# apps/ml-api/src/cv/nerf/nerfstudio.py

"""
NeRF vs 3DGS decision (early 2026):

3DGS has become the DOMINANT method for real-time applications:
  Training: 23 min vs hours
  Rendering: 100+ fps vs 1-4 fps
  Editing: explicit representation → direct scene editing

NeRF retains advantages in:
  Peak visual quality: Zip-NeRF leads by ~1dB PSNR
  View-dependent effects: continuous specular modeling
  Geometric reconstruction: SDF-based NeuS produces cleaner surfaces
  Compact model size: 10-100MB vs 100-1000MB raw
  Generative 3D: Score Distillation Sampling pipelines

For venue platform:
  3DGS for rendering + SuGaR/2DGS mesh for interaction (RECOMMENDED)
  NeuS2/BakedSDF as alternative when mesh quality paramount.

Nerfstudio v1.1.5: pip install nerfstudio
  Unified framework: Nerfacto, Splatfacto (3DGS via gsplat), Instant-NGP
  CLI: ns-train / ns-export
  Web viewer: viewer.nerf.studio

Key NeRF milestones:
  NeRF (Mildenhall 2020): 8-layer MLP, 1-2 days training, ~30s/frame
  Instant-NGP (Müller 2022): hash tables, 2-layer/64ch, 1000× speedup
  NeuS (Wang 2021): SDF zero-level set, unbiased volume rendering
  NeuS2: hash encoding, 100× speedup over NeuS
  BakedSDF (Yariv 2023): SDF → triangle mesh + spherical Gaussians
  Zip-NeRF (Barron 2023): anti-aliasing + hash grids, 8-77% error reduction
"""
```

---

## CV-6: Mesh Processing — Millions to Thousands of Triangles

### What to Build

Automated mesh decimation, UV unwrap, baking, and GLB compression.

```python
# apps/ml-api/src/cv/mesh/gltf_optimize.py

"""
Decimation via Quadric Error Metric (Garland & Heckbert, SIGGRAPH 1997):
  Per-vertex 4×4 symmetric quadric matrices from adjacent face planes.
  Collapse cheapest edge pair iteratively in O(n log n).
  Photogrammetry meshes: 95% reduction acceptable with normal map baking.
  Target budgets: venue ≤500K tri, furniture ≤50K tri.

UV unwrap after decimation (original UVs destroyed):
  Smart UV Project: angle_limit=66°, island_margin=0.02
  xatlas (xatlasjs 0.2.0): programmatic atlas packing

Baking: Cycles Selected-to-Active mode
  cage_extrusion=0.05, margin=16px
  Maps: normal, diffuse, AO (high-poly → low-poly)
  Texture sizes: 4096² venues, 2048² furniture, 1024² small props

glTF/GLB extensions for compression:
| Extension                    | Purpose                           | Reduction     |
|------------------------------|-----------------------------------|---------------|
| KHR_draco_mesh_compression   | Geometry quant + Edgebreaker      | 5-63× geom    |
| KHR_texture_basisu           | KTX2/Basis Universal GPU textures | 4-10× textures|
| EXT_meshopt_compression      | Vertex/index buffer compression   | 85-92% w/ quant|

gltf-transform v4.3.0:
  gltf-transform optimize input.glb output.glb \
    --compress draco --texture-compress webp --texture-resize 1024

gltfpack (meshoptimizer v1.0):
  Alternative for meshopt compression with native texture support.

Furniture GLB pipeline:
  200-500MB raw OBJ → decimate (15MB) → texture bake (12MB)
  → Draco + WebP (~800KB) → KTX2 + meshopt (~500KB)

Three.js loading:
  DRACOLoader.setDecoderPath('/draco/') (~300KB WASM)
  KTX2Loader.setTranscoderPath('/basis/')
  loader.setMeshoptDecoder(MeshoptDecoder)

PBR materials → MeshStandardMaterial:
  baseColor/map, metalness, roughness, normalMap,
  aoMap (requires UV2 channel)
  MeshPhysicalMaterial for clearcoat, transmission, IOR.
"""
```

---

## CV-7: React Three Fiber Venue Rendering

### What to Build

Declarative R3F scene composition: splat venue + GLB furniture + shadows +
post-processing + drag-and-drop + adaptive performance.

```typescript
// packages/venue-renderer/src/scene/VenueScene.tsx

/**
 * Stack: three@0.182.0, @react-three/fiber@9.5.0 (React 19; v8 for React 18),
 *   @react-three/drei@10.7.7, @react-three/postprocessing@3.0.4
 *
 * useGLTF (drei): auto Draco via CDN, batch preload: useGLTF.preload('/venue.glb')
 * gltfjsx CLI: npx gltfjsx venue.glb -t -s -T --resolution 1024 --draco
 *   → typed JSX components, 70-90% size reduction
 *
 * InstancedMesh for repeated furniture (N draw calls → 1 per type):
 *   @three.ez/instanced-mesh@0.3.11: per-instance frustum culling,
 *   BVH spatial indexing, auto LOD switching.
 *   drei <Merged>: auto instancing from glTF nodes.
 *
 * LOD: drei <Detailed distances={[0, 25, 50, 100]}>
 *   Progressive meshes by camera distance.
 *
 * Shadows for venue interiors:
 *   AccumulativeShadows (drei): temporal progressive soft shadows,
 *     zero runtime cost after accumulation — static presentations.
 *   ContactShadows: cheap planar approx, frames={1} for static.
 *   Cascaded Shadow Maps (three-csm): 4 cascades × 2048px — walkthroughs.
 *
 * Post-processing (@react-three/postprocessing 3.0.4):
 *   N8AO (n8ao): surpasses built-in SSAO
 *   Selective Bloom: luminanceThreshold={1.1}, mipmapBlur
 *   ACES Filmic tone mapping
 *
 * Performance budgets (60fps mid-range):
 *   <100 draw calls, <500K visible tri (desktop) / <200K (mobile)
 *   <256MB texture VRAM, <10MB compressed GLTF payload
 *
 * <PerformanceMonitor> (drei): adaptive quality
 *   Auto-reduce DPR and effects when framerate drops.
 *   flipflops={3} prevents oscillation.
 *   frameloop="demand": re-render only on state changes.
 *
 * Furniture drag-and-drop:
 *   DragControls (drei): axisLock="y" (floor plane), dragLimits (room bounds)
 *   TransformControls: translate/rotate/scale gizmos
 *     translationSnap={0.25} for grid alignment
 *   HTML-to-R3F: invisible floor mesh + onPointerUp → raycast placement
 *
 * Worker offloading:
 *   @react-three/offscreen: entire render loop to Web Worker via OffscreenCanvas
 *   Safari fallback automatic.
 *   Heavy computation (collision, layout optimization) → dedicated workers
 *   via BroadcastChannel.
 */
```

---

## CV-8: Point Cloud Processing for Room Geometry

### What to Build

Extract room boundaries, doors, windows from Matterport E57 point clouds.

```python
# apps/ml-api/src/cv/point_cloud/cleaning.py

"""
Matterport E57: 20-100M colorized points for medium venues (500-2,000m²).
Point spacing: ~5mm at 2m range.
Open3D v0.19.0: o3d.io.read_point_cloud("venue.e57") loads E57 directly.

Cleaning pipeline:
1. Statistical outlier removal: nb_neighbors=20, std_ratio=2.0
   Removes ~1-5% noise points.
2. Voxel grid downsampling: voxel_size=0.02 → ~15× reduction
   Preserves 2cm detail.
3. Normal estimation via PCA: radius=0.1, max_nn=30
   Required for surface reconstruction, plane detection, ICP.
"""

# apps/ml-api/src/cv/point_cloud/registration.py

"""
Multi-scan alignment:
1. RANSAC global registration with FPFH features → initial pose
2. Colored ICP: joint photometric + geometric optimization
   Three scales: 4cm → 2cm → 1cm → sub-centimeter alignment.
"""

# apps/ml-api/src/cv/point_cloud/room_boundary.py

"""
Room boundary extraction pipeline:
1. Detect floor plane: RANSAC (distance_threshold=0.01, ransac_n=3, num_iterations=1000)
2. Slice point cloud at wall height (1.0-1.5m above floor)
3. Project to 2D
4. Compute alpha shape via Shapely → boundary polygon
5. Fit line segments: Douglas-Peucker or Hough transform
6. Regularize to Manhattan-world constraints where applicable

Door/window detection:
  Project wall-plane points → 2D orthographic images.
  Identify rectangular voids (low point density regions).
  Classify: floor-connected >1.8m tall → door; elevated partial-height → window.
"""

# apps/ml-api/src/cv/point_cloud/semantic_seg.py

"""
Point Transformer V3 (Wu et al., CVPR 2024 Oral, arXiv:2312.10035):
  ScanNet v2: ~77.5 mIoU, S3DIS: ~74.7 mIoU
  3× speed, 10× memory improvement over PTv2
  Serialized neighbor mapping via space-filling curves (replaces KNN)

MinkowskiEngine v0.5.4:
  Sparse 3D convolutions on voxelized point clouds.
  MinkowskiNet42: ~73.6 mIoU ScanNet, ~0.2-0.5s inference RTX 3090.
  Requires openblas + CUDA build from source.

Additional libraries:
  PCL 1.14.1 (C++, conda), CloudCompare 2.13.x (GUI),
  PDAL 2.8.4 (ETL), PyTorch3D 0.7.8 (differentiable 3D ops).
"""
```

---

## CV-9: Depth Estimation — Phone-Based Fallback

### What to Build

Enable phone-based venue capture when Matterport isn't available.

```python
# apps/ml-api/src/cv/depth/depth_anything.py

"""
Depth Anything v2 (Yang et al., NeurIPS 2024, arXiv:2406.09414):
  DINOv2 encoders (ViT-S/B/L/G up to 1.3B params).
  Synthetic-supervised teacher + pseudo-labeled real images.
  86.8% accuracy on DA-2K benchmark, 213ms ViT-L on V100.

Depth Pro (Apple, ICLR 2025, arXiv:2410.02073):
  Zero-shot METRIC depth WITH absolute scale WITHOUT camera intrinsics.
  2.25MP depth map in 0.3 seconds. Sharp boundaries.

Metric3D v2 (arXiv:2404.15506):
  #1 on KITTI and NYU. Canonical camera space transformation
  resolves metric ambiguity across camera models.
"""

# apps/ml-api/src/cv/depth/dust3r.py

"""
DUSt3R (Wang et al., CVPR 2024, arXiv:2312.14132):
  Dense 3D from arbitrary image pairs WITHOUT camera calibration or SfM.
  CroCo-pretrained Transformer → per-pixel 3D pointmaps in shared frame.
  >2 images: global alignment via gradient descent.
  Pose accuracy: ~2-3° rotation, ~1-2% translation error.

MASt3R (Leroy et al., ECCV 2024, arXiv:2406.09756):
  Augments DUSt3R with dense local feature matching.
  30% localization improvement. Full SfM replacement via MASt3R-SfM.

Fast3R (Meta, CVPR 2025):
  1000+ images in one forward pass. 300× throughput over DUSt3R.

Phone-based venue capture viability:
  iPhone LiDAR (Pro models): ±50-100mm at short range, real-time mesh.
  RGB photos + DUSt3R/MASt3R from 10-30 images: ~5-10cm accuracy.
  Sufficient for furniture placement visualization, NOT construction-grade.

Apple RoomPlan API (ARKit + LiDAR):
  Detects walls, doors, windows, 16 furniture categories in real time.
  Produces parametric room models → ideal for layout constraints.
"""
```

---

## CV-10: Object Detection and Segmentation

### What to Build

Auto-identify furniture, doors, windows, architectural features in venue
scans and photos. Text-prompted segmentation for any venue element.

```python
# apps/ml-api/src/cv/segmentation/grounded_sam.py

"""
SAM 2 (Meta, 2024, arXiv:2408.00714):
  Streaming memory Transformer. 6× faster than SAM for images.
  3× fewer interactions for video segmentation.
  SAM 2.1 checkpoints (Sep 2024): further improvements.
  pip install SAM-2

Grounding DINO (arXiv:2303.05499, ECCV 2024):
  Open-set detection from arbitrary text prompts.
  52.5 AP on COCO zero-shot.

Grounded SAM (arXiv:2401.14159):
  Text prompts like "chair", "table", "door", "window", "stage"
  → precise segmentation masks for ANY venue element. No training.
  Pipeline: Grounding DINO detects boxes → SAM generates pixel masks
  → project to 3D via depth estimation.

YOLO11 (Ultralytics, Sep 2024):
  C3k2 blocks + C2PSA spatial attention.
  39.5-54.7 mAP (nano→extra), 1.55ms GPU latency (nano).
  YOLO26 (2025): NMS-free end-to-end.
  pip install ultralytics
  COCO pre-trained: chairs, couches, dining tables, beds, etc.
  Fine-tune on custom venue objects (podiums, stages, AV) via Roboflow.

3D instance segmentation:
  Mask3D (arXiv:2210.03105): Transformer queries, SOTA ScanNet (+6.2 mAP)
  SAM3D (arXiv:2306.03908): lift SAM 2D masks into 3D via bidirectional merging
    Training-free, compatible with any SAM checkpoint.

Real-world measurement from segmented objects:
  Depth Pro / Metric3D v2 for metric depth.
  DUSt3R pointmaps: direct distance between any two points.
  iPhone LiDAR: ±2-5cm accuracy at short range.
"""
```

---

## CV-11: WebXR — AR Furniture Preview and VR Walkthroughs

### What to Build

AR overlay of proposed furniture in real venue, VR remote walkthroughs.

```typescript
// packages/venue-renderer/src/xr/ARPreview.tsx

/**
 * WebXR Device API: W3C Candidate Recommendation (Oct 2025)
 *
 * Support:
 *   Chromium: Chrome 79+, Edge 79+, Opera 66+, Samsung Internet 12+,
 *     Meta Quest Browser, Android XR
 *   visionOS Safari: SUPPORTED
 *   iOS Safari: NO WebXR ← CRITICAL GAP (~50% mobile users)
 *   Firefox: NO WebXR
 *
 * Three.js WebXR:
 *   AR: ARButton.createButton(renderer, { requiredFeatures: ['hit-test'] })
 *   VR: VRButton.createButton(renderer)
 *   renderer.xr.enabled = true; alpha: true for camera passthrough.
 *   Hit testing: place furniture on detected real-world surfaces.
 *
 * iOS gap solutions:
 *   8th Wall (Niantic): cross-platform WebAR engine.
 *     World tracking, image targets, VPS.
 *     $700/month per project (reduced 75% in 2025).
 *     Required for: venue manager walking real space with AR furniture overlay.
 *
 *   Google Model Viewer v4.1.0:
 *     <model-viewer src="chair.glb" ios-src="chair.usdz" ar>
 *     WebXR on Android, Scene Viewer fallback Android, AR Quick Look iOS.
 *     Simplest universal AR path.
 *
 * Apple Vision Pro (visionOS):
 *   WebXR in Safari. HTML <model> element (visionOS 26, enabled by default).
 *   Inline stereoscopic 3D models using USDZ.
 *   Development: SwiftUI + RealityKit + ARKit, Reality Composer Pro.
 *
 *   USDZ for AR Quick Look requirements:
 *   PBR metallic-roughness, meter-scale, ≤200K tri, ≤4-8MB,
 *   power-of-two textures. Convert: usd_from_gltf or Reality Converter.
 *
 * RECOMMENDED AR strategy:
 *   WebXR for immersive VR walkthroughs (Quest, desktop VR, visionOS)
 *   Model Viewer with dual GLB/USDZ for universal furniture AR preview
 *   8th Wall for advanced iOS AR (world tracking)
 *   Budget AR layer as PROGRESSIVE ENHANCEMENT, not core dependency.
 *
 * CRITICAL iOS risk:
 *   WebXR reaches ~80% Android, ZERO iOS Safari.
 *   Dual-format strategy (GLB + USDZ via Model Viewer ar-modes cascade).
 *   Apple HTML <model> in visionOS 26 signals convergence,
 *   but mainstream iOS Safari WebXR remains ABSENT.
 */
```

---

## CV-12: Asset Pipeline — Capture to CDN with Quality Gates

### What to Build

Systematic pipeline for 50-100+ furniture models per venue.
Version control, automated quality validation, CDN delivery, progressive loading.

```python
# apps/ml-api/src/cv/asset_pipeline/dvc_manager.py

"""
DVC (pip install dvc[s3]) with Cloudflare R2 backend:
  .dvc pointer files in Git, binary GLB/USDZ blobs on R2.
  Selective dvc pull of individual assets (Git LFS lacks this).

Cloudflare R2 pricing: $0 egress, $0.015/GB/month storage.
  10TB/month: ~$150 on R2 vs ~$1,041 on CloudFront.

CDN: custom domain → R2 (assets.venueplanner.com)
  Cache-Control: public, max-age=31536000, immutable
  Content-addressed asset URLs for cache busting.
"""

# apps/ml-api/src/cv/asset_pipeline/cdn_config.py

"""
Progressive loading (Needle Tools pattern):
  LOD2 placeholder (5-15KB): renders instantly
  LOD1 (100KB): loads within 1-2 seconds
  LOD0 (full quality): streams in background

KTX2/Basis Universal textures:
  Runtime transcode to GPU-native:
  BC7 (desktop), ASTC (mobile), ETC2 (fallback).
  Saves both bandwidth and VRAM.
"""

# apps/ml-api/src/cv/asset_pipeline/quality_gates.py

"""
Automated quality gates in CI/CD:

1. Khronos glTF Validator:
   Schema compliance, accessor bounds, quaternion validity.

2. glTF Asset Auditor (@khronos/gltf-asset-auditor):
   Configurable profiles: max file size, tri count, texture resolution,
   manifold requirement, UV compliance.

3. Custom pymeshlab checks:
   Scale: bounding box diagonal 0.1-5.0m
   Face count: within per-category budget
   Non-manifold edges: must be zero

Quality normalization (photogrammetry vs 3DGS-to-mesh):
  Cleanup (floaters/outliers) → retopology to budget → UV unwrap (xatlas)
  → texture bake from source views → de-lighting (remove baked shadows)
  → PBR channel generation → meter-scale normalization → validation
"""

# apps/ml-api/src/cv/asset_pipeline/metadata_schema.py

"""
Asset metadata schema (physical + technical):

Physical properties:
  width, height, depth (meters)
  weight_capacity_kg
  stackable: bool, max_stack_count: int
  floor_space_m2, clearance_radius_m
  indoor_outdoor_rating: enum
  fire_retardancy_rating: str
  setup_time_minutes: int
  table_pairing: Optional[str]

Technical properties:
  triangle_count: int
  texel_density: float
  psnr_validation_score: float
  audit_profile_result: str
  file_size_bytes: int
  format: str (GLB, USDZ, PLY)
  compression: list[str] (draco, meshopt, ktx2)

Enables: search, filtering, constraint-based layout features.
"""
```

---

## Integration with Other Techniques

- **Signal Processing** (SP-7): Matterport meshes feed acoustic simulation;
  material classification from scan textures assigns absorption coefficients;
  acoustic RIR previews use Web Audio ConvolverNode with mesh-derived room geometry
- **Physics Solvers** (PS-5): SuGaR collision proxy mesh provides physics boundary
  for layout optimization energy terms; furniture placement respects room constraints
  extracted from point cloud processing
- **Graph Neural Networks** (GNN-4): Furniture layouts as graphs with spatial edges
  derived from CV-detected positions; GNN layout quality scoring uses CV-extracted
  room boundaries as constraints
- **HPC** (HPC): WebGPU compute shaders for Gaussian splat sorting; WASM for Draco
  decoding, KTX2 transcoding; Web Workers for offscreen rendering

---

## Session Management

1. **CV-1** (Matterport: API client, E57/MatterPak export, SDK embedding) — 1 session
2. **CV-2** (3DGS: gsplat/Splatfacto training, compression pipelines, SPZ) — 1-2 sessions
3. **CV-3** (Browser rendering: Spark SplatMesh, depth compositing, raycasting) — 1-2 sessions
4. **CV-4** (Photogrammetry: Kiri Engine, RealityCapture, capture guides) — 1 session
5. **CV-5** (NeRF/mesh: Nerfstudio, NeuS2, BakedSDF, decision logic) — 1 session
6. **CV-6** (Mesh processing: QEM decimation, UV, baking, gltf-transform) — 1-2 sessions
7. **CV-7** (R3F rendering: VenueScene, instancing, shadows, post, drag-drop) — 2-3 sessions
8. **CV-8** (Point cloud: Open3D cleaning, ICP, room boundary, semantic seg) — 1-2 sessions
9. **CV-9** (Depth: Depth Anything v2, Depth Pro, DUSt3R/MASt3R, phone fallback) — 1 session
10. **CV-10** (Segmentation: SAM 2, Grounded SAM, YOLO11, Mask3D, dimensions) — 1-2 sessions
11. **CV-11** (WebXR: AR furniture preview, VR walkthrough, Model Viewer, 8th Wall) — 1-2 sessions
12. **CV-12** (Asset pipeline: DVC+R2, quality gates, CDN, progressive loading) — 1-2 sessions

Total: ~14-20 Claude Code sessions.
