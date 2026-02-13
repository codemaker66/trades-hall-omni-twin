# Computer Vision Command Track

Canonical command track: `docs/commands/computer-vision/`

Execution authority for agent commands is `docs/commands/**`.

## Commands

- `CV-1` - Matterport Pro3 Data Extraction (depends_on: None)
- `CV-2` - 3D Gaussian Splatting Training and Compression (depends_on: `CV-1`)
- `CV-3` - Browser Rendering — Splat + Mesh Compositing (depends_on: `CV-2`)
- `CV-4` - Photogrammetry Furniture Capture Pipeline (depends_on: `CV-3`)
- `CV-5` - NeRF Decision Matrix (depends_on: `CV-INT-1`)
- `CV-6` - Mesh Processing — Millions to Thousands of Triangles (depends_on: `CV-5`)
- `CV-7` - React Three Fiber Venue Rendering (depends_on: `CV-6`)
- `CV-8` - Point Cloud Processing for Room Geometry (depends_on: `CV-7`)
- `CV-9` - Depth Estimation — Phone-Based Fallback (depends_on: `CV-INT-2`)
- `CV-10` - Object Detection and Segmentation (depends_on: `CV-9`)
- `CV-11` - WebXR — AR Furniture Preview and VR Walkthroughs (depends_on: `CV-10`)
- `CV-12` - Asset Pipeline — Capture to CDN with Quality Gates (depends_on: `CV-11`)
- `CV-INT-1` - Computer Vision integration checkpoint 1 (depends_on: `CV-1`, `CV-2`, `CV-3`, `CV-4`)
- `CV-INT-2` - Computer Vision integration checkpoint 2 (depends_on: `CV-5`, `CV-6`, `CV-7`, `CV-8`)
- `CV-INT-3` - Computer Vision integration checkpoint 3 (depends_on: `CV-9`, `CV-10`, `CV-11`, `CV-12`)
