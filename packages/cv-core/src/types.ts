// ---------------------------------------------------------------------------
// @omni-twin/cv-core — Computer Vision & 3D Reconstruction Types
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// PRNG (mulberry32, shared pattern across packages)
// ---------------------------------------------------------------------------

/** Seedable PRNG function. */
export type PRNG = () => number;

/** Create a seedable PRNG using the mulberry32 algorithm. */
export function createPRNG(seed: number): PRNG {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller transform: sample from N(mean, std^2) using a PRNG. */
export function normalSample(rng: PRNG, mean: number, std: number): number {
  // Box-Muller requires two uniform samples
  let u1 = rng();
  // Guard against log(0)
  while (u1 === 0) u1 = rng();
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + std * z;
}

// ---------------------------------------------------------------------------
// Vector types
// ---------------------------------------------------------------------------

/** 2D vector. */
export interface Vec2 {
  x: number;
  y: number;
}

/** 3D vector. */
export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

/** 4D vector / homogeneous coordinate. */
export interface Vec4 {
  x: number;
  y: number;
  z: number;
  w: number;
}

// ---------------------------------------------------------------------------
// Vec2 operations
// ---------------------------------------------------------------------------

/** Element-wise addition of two Vec2. */
export function vec2Add(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x + b.x, y: a.y + b.y };
}

/** Element-wise subtraction of two Vec2. */
export function vec2Sub(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x - b.x, y: a.y - b.y };
}

/** Euclidean length of a Vec2. */
export function vec2Length(v: Vec2): number {
  return Math.sqrt(v.x * v.x + v.y * v.y);
}

// ---------------------------------------------------------------------------
// Vector3 operations
// ---------------------------------------------------------------------------

/** Element-wise addition of two Vector3. */
export function vec3Add(a: Vector3, b: Vector3): Vector3 {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}

/** Element-wise subtraction of two Vector3. */
export function vec3Sub(a: Vector3, b: Vector3): Vector3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

/** Scalar multiplication of a Vector3. */
export function vec3Scale(v: Vector3, s: number): Vector3 {
  return { x: v.x * s, y: v.y * s, z: v.z * s };
}

/** Dot product of two Vector3. */
export function vec3Dot(a: Vector3, b: Vector3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

/** Cross product of two Vector3. */
export function vec3Cross(a: Vector3, b: Vector3): Vector3 {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

/** Euclidean length of a Vector3. */
export function vec3Length(v: Vector3): number {
  return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

/** Normalize a Vector3 to unit length. Returns zero vector if length is near zero. */
export function vec3Normalize(v: Vector3): Vector3 {
  const len = vec3Length(v);
  if (len < 1e-15) return { x: 0, y: 0, z: 0 };
  return { x: v.x / len, y: v.y / len, z: v.z / len };
}

// ---------------------------------------------------------------------------
// Matrix4x4 — 16-element Float64Array, column-major (graphics convention)
// ---------------------------------------------------------------------------

/** 4x4 matrix stored as a 16-element Float64Array in column-major order.
 *
 * Layout: [m00, m10, m20, m30, m01, m11, m21, m31, m02, m12, m22, m32, m03, m13, m23, m33]
 * i.e. element at row r, col c is at index c*4 + r.
 */
export type Matrix4x4 = Float64Array;

/** Create a 4x4 identity matrix. */
export function mat4Identity(): Matrix4x4 {
  const m = new Float64Array(16);
  m[0] = 1;   // (0,0)
  m[5] = 1;   // (1,1)
  m[10] = 1;  // (2,2)
  m[15] = 1;  // (3,3)
  return m;
}

/** Get element at (row, col) from a column-major Matrix4x4. */
function mat4Get(m: Matrix4x4, row: number, col: number): number {
  return m[col * 4 + row]!;
}

/** Set element at (row, col) in a column-major Matrix4x4. */
function mat4Set(m: Matrix4x4, row: number, col: number, val: number): void {
  m[col * 4 + row] = val;
}

/** Multiply two 4x4 matrices: C = A * B (column-major). */
export function mat4Multiply(A: Matrix4x4, B: Matrix4x4): Matrix4x4 {
  const C = new Float64Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        sum += A[k * 4 + row]! * B[col * 4 + k]!;
      }
      C[col * 4 + row] = sum;
    }
  }
  return C;
}

/** Transpose a 4x4 matrix. */
export function mat4Transpose(m: Matrix4x4): Matrix4x4 {
  const out = new Float64Array(16);
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      out[col * 4 + row] = m[row * 4 + col]!;
    }
  }
  return out;
}

/** Invert a 4x4 matrix via Gauss-Jordan elimination with partial pivoting. */
export function mat4Inverse(m: Matrix4x4): Matrix4x4 {
  // Build augmented 4x8 matrix in row-major for convenience
  const aug = new Float64Array(4 * 8);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      aug[r * 8 + c] = mat4Get(m, r, c);
    }
    aug[r * 8 + (4 + r)] = 1;
  }

  for (let col = 0; col < 4; col++) {
    // Partial pivoting
    let maxRow = col;
    let maxVal = Math.abs(aug[col * 8 + col]!);
    for (let row = col + 1; row < 4; row++) {
      const val = Math.abs(aug[row * 8 + col]!);
      if (val > maxVal) {
        maxVal = val;
        maxRow = row;
      }
    }
    if (maxRow !== col) {
      for (let j = 0; j < 8; j++) {
        const tmp = aug[col * 8 + j]!;
        aug[col * 8 + j] = aug[maxRow * 8 + j]!;
        aug[maxRow * 8 + j] = tmp;
      }
    }

    const pivot = aug[col * 8 + col]!;
    if (Math.abs(pivot) < 1e-15) {
      throw new Error('Matrix4x4 is singular or near-singular');
    }

    // Scale pivot row
    for (let j = 0; j < 8; j++) {
      aug[col * 8 + j] = aug[col * 8 + j]! / pivot;
    }

    // Eliminate column
    for (let row = 0; row < 4; row++) {
      if (row === col) continue;
      const factor = aug[row * 8 + col]!;
      for (let j = 0; j < 8; j++) {
        aug[row * 8 + j] = aug[row * 8 + j]! - factor * aug[col * 8 + j]!;
      }
    }
  }

  // Extract inverse back into column-major
  const inv = new Float64Array(16);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      mat4Set(inv, r, c, aug[r * 8 + (4 + c)]!);
    }
  }
  return inv;
}

// ---------------------------------------------------------------------------
// Matrix3x3 — 9-element Float64Array, column-major
// ---------------------------------------------------------------------------

/** 3x3 matrix stored as a 9-element Float64Array in column-major order.
 *
 * Element at row r, col c is at index c*3 + r.
 */
export type Matrix3x3 = Float64Array;

/** Create a 3x3 identity matrix. */
export function mat3Identity(): Matrix3x3 {
  const m = new Float64Array(9);
  m[0] = 1;  // (0,0)
  m[4] = 1;  // (1,1)
  m[8] = 1;  // (2,2)
  return m;
}

/** Multiply two 3x3 matrices: C = A * B (column-major). */
export function mat3Multiply(A: Matrix3x3, B: Matrix3x3): Matrix3x3 {
  const C = new Float64Array(9);
  for (let col = 0; col < 3; col++) {
    for (let row = 0; row < 3; row++) {
      let sum = 0;
      for (let k = 0; k < 3; k++) {
        sum += A[k * 3 + row]! * B[col * 3 + k]!;
      }
      C[col * 3 + row] = sum;
    }
  }
  return C;
}

/** Transpose a 3x3 matrix. */
export function mat3Transpose(A: Matrix3x3): Matrix3x3 {
  const out = new Float64Array(9);
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      out[col * 3 + row] = A[row * 3 + col]!;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Quaternion — {x, y, z, w} convention (w is scalar part)
// ---------------------------------------------------------------------------

/** Quaternion with scalar-last convention: q = xi + yj + zk + w. */
export interface Quaternion {
  x: number;
  y: number;
  z: number;
  w: number;
}

/** Multiply two quaternions: result = a * b (Hamilton product). */
export function quatMultiply(a: Quaternion, b: Quaternion): Quaternion {
  return {
    x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
  };
}

/** Normalize a quaternion to unit length. */
export function quatNormalize(q: Quaternion): Quaternion {
  const len = Math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (len < 1e-15) return { x: 0, y: 0, z: 0, w: 1 };
  return { x: q.x / len, y: q.y / len, z: q.z / len, w: q.w / len };
}

/** Convert a unit quaternion to a 3x3 rotation matrix (column-major). */
export function quatToMat3(q: Quaternion): Matrix3x3 {
  const { x, y, z, w } = q;
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;

  const m = new Float64Array(9);
  // Column 0
  m[0] = 1 - (yy + zz);
  m[1] = xy + wz;
  m[2] = xz - wy;
  // Column 1
  m[3] = xy - wz;
  m[4] = 1 - (xx + zz);
  m[5] = yz + wx;
  // Column 2
  m[6] = xz + wy;
  m[7] = yz - wx;
  m[8] = 1 - (xx + yy);
  return m;
}

/** Spherical linear interpolation between two quaternions. */
export function quatSlerp(a: Quaternion, b: Quaternion, t: number): Quaternion {
  let dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;

  // If the dot product is negative, negate one quaternion to take the short path
  let bx = b.x, by = b.y, bz = b.z, bw = b.w;
  if (dot < 0) {
    dot = -dot;
    bx = -bx;
    by = -by;
    bz = -bz;
    bw = -bw;
  }

  // If quaternions are very close, use linear interpolation to avoid division by zero
  if (dot > 0.9995) {
    return quatNormalize({
      x: a.x + t * (bx - a.x),
      y: a.y + t * (by - a.y),
      z: a.z + t * (bz - a.z),
      w: a.w + t * (bw - a.w),
    });
  }

  const theta = Math.acos(dot);
  const sinTheta = Math.sin(theta);
  const wa = Math.sin((1 - t) * theta) / sinTheta;
  const wb = Math.sin(t * theta) / sinTheta;

  return {
    x: wa * a.x + wb * bx,
    y: wa * a.y + wb * by,
    z: wa * a.z + wb * bz,
    w: wa * a.w + wb * bw,
  };
}

// ---------------------------------------------------------------------------
// CV-1: Matterport / Scan Ingestion
// ---------------------------------------------------------------------------

/** A 3D sweep position captured by a Matterport scanner or similar device. */
export interface Sweep {
  /** Unique sweep identifier. */
  id: string;
  /** World-space position of the sweep. */
  position: Vector3;
  /** Orientation of the sweep as a quaternion. */
  rotation: Quaternion;
  /** Optional floor index for multi-storey buildings. */
  floorIndex?: number;
  /** Optional timestamp of capture (ISO 8601). */
  timestamp?: string;
}

/** A room polygon extracted from a floor-plan scan. */
export interface Room {
  /** Room identifier / label. */
  name: string;
  /** Boundary polygon as ordered 2D vertices (in floor-plan coordinates). */
  polygon: Vec2[];
  /** Computed area in square metres. */
  area: number;
  /** Optional room category (e.g. 'office', 'corridor', 'bathroom'). */
  category?: string;
  /** Optional ceiling height in metres. */
  ceilingHeight?: number;
}

/** A 2D floor plan derived from Matterport or similar scans. */
export interface FloorPlan {
  /** Floor index (0-based). */
  floorIndex: number;
  /** Rooms detected on this floor. */
  rooms: Room[];
  /** Outer boundary polygon of the entire floor. */
  boundary: Vec2[];
  /** Width of the floor-plan bounding box in metres. */
  width: number;
  /** Height of the floor-plan bounding box in metres. */
  height: number;
}

/** Top-level Matterport model descriptor. */
export interface MatterportModel {
  /** Model identifier (e.g. Matterport SID). */
  id: string;
  /** Human-readable model name. */
  name: string;
  /** All sweeps in the model. */
  sweeps: Sweep[];
  /** Derived floor plans. */
  floorPlans: FloorPlan[];
  /** Model-to-world coordinate transform. */
  coordinateTransform: CoordinateTransform;
  /** Total scan area in square metres. */
  totalArea: number;
  /** Number of floors. */
  floorCount: number;
}

/** E57 point cloud file header for scan ingestion. */
export interface E57Header {
  /** File GUID. */
  guid: string;
  /** Number of 3D data sections (scans). */
  scanCount: number;
  /** Total point count across all scans. */
  totalPointCount: number;
  /** Coordinate system metadata (e.g. EPSG code or WKT string). */
  coordinateSystem: string;
  /** Creation timestamp (ISO 8601). */
  creationDate: string;
  /** Bounding box minimum corner. */
  boundsMin: Vector3;
  /** Bounding box maximum corner. */
  boundsMax: Vector3;
}

/** Rigid transform for coordinate system conversion. */
export interface CoordinateTransform {
  /** 4x4 transformation matrix (column-major). */
  matrix: Matrix4x4;
  /** Source coordinate system identifier (e.g. 'matterport', 'e57', 'epsg:4326'). */
  sourceSystem: string;
  /** Target coordinate system identifier (e.g. 'world', 'building-local'). */
  targetSystem: string;
  /** Uniform scale factor applied before rotation+translation. */
  scale: number;
}

// ---------------------------------------------------------------------------
// CV-2: Gaussian Splatting
// ---------------------------------------------------------------------------

/** A single 3D Gaussian primitive for Gaussian splatting. */
export interface Gaussian3D {
  /** Centre position in world space. */
  center: Vector3;
  /** 3x3 covariance matrix (column-major Float64Array, 9 elements).
   *  Encodes both the shape (eigenvalues) and orientation (eigenvectors). */
  covariance: Matrix3x3;
  /** Base colour (RGB, each in [0,1]). */
  color: Vector3;
  /** Opacity in [0,1]. */
  opacity: number;
  /** Spherical harmonics coefficients for view-dependent colour.
   *  Length depends on SH degree: degree 0 = 1, degree 1 = 4, degree 2 = 9, degree 3 = 16 bands * 3 channels. */
  sh_coeffs: Float64Array;
}

/** A cloud of 3D Gaussians plus metadata for an entire scene. */
export interface SplatCloud {
  /** All Gaussian primitives. */
  gaussians: Gaussian3D[];
  /** Total primitive count. */
  count: number;
  /** Axis-aligned bounding box minimum. */
  boundsMin: Vector3;
  /** Axis-aligned bounding box maximum. */
  boundsMax: Vector3;
  /** Spherical harmonics degree used (0-3). */
  shDegree: number;
  /** Optional source file path. */
  sourcePath?: string;
}

/** Configuration for GPU radix sort of splats by depth. */
export interface RadixSortConfig {
  /** Number of elements to sort. */
  count: number;
  /** Number of radix-sort passes (typically 4 for 32-bit keys). */
  passes: number;
  /** Workgroup size for the compute shader. */
  workgroupSize: number;
  /** Bits per pass (typically 8). */
  bitsPerPass: number;
}

/** Tile assignment for tile-based rasterisation of splats. */
export interface TileAssignment {
  /** Gaussian index in the SplatCloud. */
  gaussianIndex: number;
  /** Tile X coordinate. */
  tileX: number;
  /** Tile Y coordinate. */
  tileY: number;
  /** Depth key used for sorting within the tile. */
  depthKey: number;
}

/** Header for SPZ (compressed splat) format. */
export interface SPZHeader {
  /** Magic bytes identifier. */
  magic: number;
  /** Format version. */
  version: number;
  /** Number of Gaussians in the file. */
  gaussianCount: number;
  /** SH degree encoded (0-3). */
  shDegree: number;
  /** Whether positions are quantised. */
  quantized: boolean;
  /** Byte offset to the position data section. */
  positionOffset: number;
  /** Byte offset to the covariance data section. */
  covarianceOffset: number;
  /** Byte offset to the colour / SH data section. */
  colorOffset: number;
  /** Byte offset to the opacity data section. */
  opacityOffset: number;
}

/** Configuration for quantisation of Gaussian attributes. */
export interface QuantizationConfig {
  /** Number of bits for position components (e.g. 16). */
  positionBits: number;
  /** Number of bits for covariance scale (e.g. 12). */
  covarianceBits: number;
  /** Number of bits for colour channels (e.g. 8). */
  colorBits: number;
  /** Number of bits for opacity (e.g. 8). */
  opacityBits: number;
  /** Number of bits for SH coefficients (e.g. 8). */
  shBits: number;
  /** Bounding box used for position quantisation. */
  boundsMin: Vector3;
  /** Bounding box used for position quantisation. */
  boundsMax: Vector3;
}

// ---------------------------------------------------------------------------
// CV-3: Compositing (Depth / Splat-Mesh Hybrid)
// ---------------------------------------------------------------------------

/** Depth buffer for depth-based compositing. */
export interface DepthBuffer {
  /** Depth values (width * height, row-major). */
  data: Float64Array;
  /** Buffer width in pixels. */
  width: number;
  /** Buffer height in pixels. */
  height: number;
  /** Near plane distance. */
  near: number;
  /** Far plane distance. */
  far: number;
}

/** Result of a single ray intersection test. */
export interface RaycastHit {
  /** Distance from ray origin to hit point. */
  distance: number;
  /** World-space hit point. */
  point: Vector3;
  /** Surface normal at the hit point. */
  normal: Vector3;
  /** Identifier of the intersected primitive (triangle index, Gaussian index, etc.). */
  primitiveId: number;
}

/** Configuration for compositing splat + mesh render layers. */
export interface CompositeConfig {
  /** Whether to composite by depth test (true) or alpha blend order (false). */
  useDepthComposite: boolean;
  /** Background colour (RGBA, each in [0,1]). */
  backgroundColour: Vec4;
  /** Exposure / tone-mapping multiplier. */
  exposure: number;
  /** Gamma correction exponent. */
  gamma: number;
  /** Whether to apply anti-aliasing on composite edges. */
  antiAlias: boolean;
  /** Blend mode between layers. */
  blendMode: 'alpha-over' | 'additive' | 'multiply';
}

/** Result of ray-ellipsoid intersection for Gaussian splatting compositing. */
export interface EllipsoidIntersection {
  /** Whether the ray hit the ellipsoid. */
  hit: boolean;
  /** Entry distance along the ray (NaN if no hit). */
  tEntry: number;
  /** Exit distance along the ray (NaN if no hit). */
  tExit: number;
  /** Gaussian contribution weight at the intersection centre. */
  weight: number;
  /** Gaussian index that was intersected. */
  gaussianIndex: number;
}

// ---------------------------------------------------------------------------
// CV-4: Photogrammetry
// ---------------------------------------------------------------------------

/** Intrinsic camera parameters (pinhole model). */
export interface CameraIntrinsics {
  /** Focal length in pixels (x-axis). */
  fx: number;
  /** Focal length in pixels (y-axis). */
  fy: number;
  /** Principal point x (pixels). */
  cx: number;
  /** Principal point y (pixels). */
  cy: number;
  /** Image width in pixels. */
  width: number;
  /** Image height in pixels. */
  height: number;
  /** Radial distortion coefficients [k1, k2, k3, ...]. */
  distortion?: Float64Array;
}

/** Extrinsic camera parameters (world-to-camera transform). */
export interface CameraExtrinsics {
  /** 3x3 rotation matrix (column-major). */
  R: Matrix3x3;
  /** Translation vector (3 elements). */
  t: Vector3;
}

/** A 2D image feature with optional descriptor. */
export interface Feature2D {
  /** Feature position x (pixels). */
  x: number;
  /** Feature position y (pixels). */
  y: number;
  /** Feature descriptor vector (e.g. 128-dim SIFT, 256-dim ORB). */
  descriptor: Float64Array;
  /** Feature response / strength. */
  response?: number;
  /** Feature scale (octave). */
  scale?: number;
  /** Feature orientation (radians). */
  angle?: number;
}

/** A match between two features in different images. */
export interface Match {
  /** Index of the feature in the query image. */
  queryIdx: number;
  /** Index of the feature in the train image. */
  trainIdx: number;
  /** Descriptor distance (lower = better). */
  distance: number;
}

/** Result of fundamental / essential matrix estimation. */
export interface FundamentalMatrixResult {
  /** 3x3 fundamental (or essential) matrix (column-major). */
  F: Matrix3x3;
  /** Inlier mask (1 = inlier, 0 = outlier) for each input match. */
  inlierMask: Uint8Array;
  /** Number of inliers. */
  inlierCount: number;
  /** Reprojection error (mean, pixels). */
  meanError: number;
}

/** Configuration for bundle adjustment. */
export interface BundleAdjustmentConfig {
  /** Number of cameras. */
  nCameras: number;
  /** Number of 3D points. */
  nPoints: number;
  /** Number of 2D observations. */
  nObservations: number;
  /** Maximum Levenberg-Marquardt iterations. */
  maxIterations: number;
  /** Convergence tolerance on parameter change. */
  parameterTolerance: number;
  /** Convergence tolerance on cost function change. */
  functionTolerance: number;
  /** Initial Levenberg-Marquardt damping parameter. */
  initialLambda: number;
  /** Whether to optimise intrinsics (true) or hold them fixed (false). */
  optimiseIntrinsics: boolean;
  /** Robust loss function type. */
  lossFunction: 'trivial' | 'huber' | 'cauchy';
  /** Robust loss parameter (e.g. Huber threshold). */
  lossParameter: number;
}

/** Result of bundle adjustment optimisation. */
export interface BundleAdjustmentResult {
  /** Optimised camera extrinsics. */
  cameras: CameraExtrinsics[];
  /** Optimised camera intrinsics (if optimised). */
  intrinsics: CameraIntrinsics[];
  /** Optimised 3D point positions. */
  points: Vector3[];
  /** Final cost (sum of squared reprojection errors). */
  finalCost: number;
  /** Number of iterations performed. */
  iterations: number;
  /** Whether the solver converged. */
  converged: boolean;
  /** Mean reprojection error in pixels. */
  meanReprojError: number;
}

/** A planned camera capture position for photogrammetry. */
export interface CapturePlan {
  /** Ordered list of camera poses to capture. */
  poses: Array<{
    /** Camera position in world space. */
    position: Vector3;
    /** Camera look-at target in world space. */
    lookAt: Vector3;
    /** Camera up vector. */
    up: Vector3;
  }>;
  /** Expected overlap between adjacent images (0-1). */
  overlap: number;
  /** Total number of images in the plan. */
  imageCount: number;
  /** Estimated coverage area in square metres. */
  coverageArea: number;
}

// ---------------------------------------------------------------------------
// CV-5: NeRF / Neural Radiance Fields
// ---------------------------------------------------------------------------

/** A ray for volume rendering. */
export interface Ray {
  /** Ray origin in world space. */
  origin: Vector3;
  /** Ray direction (should be unit-length). */
  direction: Vector3;
}

/** A single sample along a ray for volume rendering. */
export interface VolumeSample {
  /** Parametric distance along the ray. */
  t: number;
  /** Volume density (sigma) at this sample. */
  sigma: number;
  /** Emitted radiance (RGB, each in [0,1]). */
  rgb: Vector3;
}

/** A sample from a signed distance function. */
export interface SDFSample {
  /** Signed distance value (negative = inside, positive = outside). */
  distance: number;
  /** Gradient of the SDF at the sample point (approximates the surface normal). */
  gradient: Vector3;
}

/** Configuration for multi-resolution hash encoding (Instant-NGP style). */
export interface HashEncodingConfig {
  /** Number of resolution levels. */
  nLevels: number;
  /** Number of features per level. */
  nFeaturesPerLevel: number;
  /** Log2 of the hash table size. */
  log2HashTableSize: number;
  /** Finest resolution. */
  baseResolution: number;
  /** Per-level scale factor. */
  perLevelScale: number;
  /** Bounding box minimum for normalisation. */
  boundsMin: Vector3;
  /** Bounding box maximum for normalisation. */
  boundsMax: Vector3;
}

/** Configuration for volume rendering (quadrature along rays). */
export interface VolumeRenderConfig {
  /** Near plane distance. */
  near: number;
  /** Far plane distance. */
  far: number;
  /** Number of coarse samples per ray. */
  nCoarseSamples: number;
  /** Number of fine (importance) samples per ray. */
  nFineSamples: number;
  /** Whether to use hierarchical (coarse + fine) sampling. */
  hierarchical: boolean;
  /** White background flag. */
  whiteBackground: boolean;
  /** Noise standard deviation added to density during training. */
  densityNoise: number;
  /** Chunk size for batched ray processing. */
  chunkSize: number;
}

/** Result of volume rendering a single ray. */
export interface VolumeRenderResult {
  /** Final accumulated colour (RGB). */
  rgb: Vector3;
  /** Accumulated opacity / transmittance remainder. */
  opacity: number;
  /** Expected depth along the ray. */
  depth: number;
  /** Accumulated weights along samples (for diagnostics). */
  weights: Float64Array;
}

// ---------------------------------------------------------------------------
// CV-6: Mesh Processing
// ---------------------------------------------------------------------------

/** A triangle mesh with optional attributes. */
export interface Mesh {
  /** Vertex positions, packed as [x0,y0,z0, x1,y1,z1, ...]. */
  vertices: Float64Array;
  /** Triangle index buffer (3 indices per triangle). */
  indices: Uint32Array;
  /** Vertex normals, packed as [nx0,ny0,nz0, ...] (optional). */
  normals?: Float64Array;
  /** UV texture coordinates, packed as [u0,v0, u1,v1, ...] (optional). */
  uvs?: Float64Array;
  /** Vertex colours, packed as [r0,g0,b0, ...] (optional). */
  colors?: Float64Array;
  /** Number of vertices. */
  vertexCount: number;
  /** Number of triangles. */
  triangleCount: number;
}

/** Configuration for Quadric Error Metric (QEM) mesh decimation. */
export interface QEMConfig {
  /** Target triangle count after decimation. */
  targetTriangles: number;
  /** Maximum allowed geometric error. */
  maxError: number;
  /** Whether to preserve mesh boundary edges. */
  preserveBoundary: boolean;
  /** Whether to lock vertices with high curvature. */
  preserveSharpEdges: boolean;
  /** Sharp edge angle threshold in radians. */
  sharpEdgeAngle: number;
}

/** A single edge collapse operation in mesh decimation. */
export interface EdgeCollapse {
  /** Index of vertex to remove (v0). */
  vertexRemoved: number;
  /** Index of vertex to keep (v1). */
  vertexKept: number;
  /** Optimal placement position for the collapsed vertex. */
  optimalPosition: Vector3;
  /** Quadric error at the optimal position. */
  error: number;
}

/** A UV chart (a connected component of the UV parameterisation). */
export interface UVChart {
  /** Triangle indices belonging to this chart. */
  triangleIndices: Uint32Array;
  /** Bounding rectangle in UV space: [uMin, vMin, uMax, vMax]. */
  boundingRect: Float64Array;
  /** Total area of the chart in UV space. */
  uvArea: number;
  /** Total area of the chart in 3D space (square metres). */
  worldArea: number;
}

/** A complete UV atlas for a mesh. */
export interface UVAtlas {
  /** UV charts. */
  charts: UVChart[];
  /** Atlas resolution (width in pixels). */
  width: number;
  /** Atlas resolution (height in pixels). */
  height: number;
  /** UV coordinates for each vertex (packed [u, v, ...]). */
  uvs: Float64Array;
  /** Packing efficiency (ratio of used to total atlas area). */
  efficiency: number;
  /** Total chart count. */
  chartCount: number;
}

/** Configuration for texture baking. */
export interface BakeConfig {
  /** Output texture resolution (width). */
  outputWidth: number;
  /** Output texture resolution (height). */
  outputHeight: number;
  /** Number of rays per texel for ray-traced baking. */
  samplesPerTexel: number;
  /** Gutter size in pixels to avoid seam bleeding. */
  gutterSize: number;
  /** Bake mode. */
  mode: 'diffuse' | 'normal' | 'ao' | 'lightmap';
  /** AO ray max distance (for mode='ao'). */
  aoMaxDistance?: number;
}

/** glTF accessor descriptor for mesh export. */
export interface GLTFAccessor {
  /** Index of the bufferView this accessor reads from. */
  bufferViewIndex: number;
  /** Byte offset within the bufferView. */
  byteOffset: number;
  /** WebGL component type (e.g. 5126 = FLOAT, 5123 = UNSIGNED_SHORT). */
  componentType: number;
  /** Number of elements. */
  count: number;
  /** Element type ('SCALAR', 'VEC2', 'VEC3', 'VEC4', 'MAT4'). */
  type: 'SCALAR' | 'VEC2' | 'VEC3' | 'VEC4' | 'MAT2' | 'MAT3' | 'MAT4';
  /** Minimum values per component. */
  min?: number[];
  /** Maximum values per component. */
  max?: number[];
}

/** glTF buffer descriptor for mesh export. */
export interface GLTFBuffer {
  /** Total byte length of the buffer. */
  byteLength: number;
  /** URI or data URI of the buffer. */
  uri?: string;
  /** In-memory data (used during construction before serialisation). */
  data?: ArrayBuffer;
}

// ---------------------------------------------------------------------------
// CV-7: Real-time Rendering
// ---------------------------------------------------------------------------

/** A single Level-of-Detail level. */
export interface LODLevel {
  /** The mesh for this LOD. */
  mesh: Mesh;
  /** Screen-space error threshold to transition to this LOD. */
  screenSpaceError: number;
  /** Distance threshold to transition to this LOD (metres). */
  distance: number;
}

/** Per-instance data for instanced rendering. */
export interface InstanceData {
  /** 4x4 model-to-world transform (column-major). */
  transform: Matrix4x4;
  /** Bounding sphere: { centre, radius }. */
  boundingSphere: {
    centre: Vector3;
    radius: number;
  };
  /** Optional instance colour tint (RGBA). */
  tint?: Vec4;
  /** Optional per-instance identifier. */
  instanceId?: number;
}

/** Six frustum planes for culling, each plane as {normal, distance}.
 *  Plane equation: dot(normal, point) + distance >= 0 means inside. */
export interface FrustumPlanes {
  /** Left clipping plane. */
  left: Vec4;
  /** Right clipping plane. */
  right: Vec4;
  /** Bottom clipping plane. */
  bottom: Vec4;
  /** Top clipping plane. */
  top: Vec4;
  /** Near clipping plane. */
  near: Vec4;
  /** Far clipping plane. */
  far: Vec4;
}

/** A single cascade in cascaded shadow mapping. */
export interface ShadowCascade {
  /** Light-space view-projection matrix (column-major). */
  viewProjection: Matrix4x4;
  /** Near split distance from the camera. */
  nearSplit: number;
  /** Far split distance from the camera. */
  farSplit: number;
  /** Shadow map resolution (width = height). */
  resolution: number;
  /** Texel-world-space ratio for filtering. */
  texelSize: number;
}

/** Performance budget for real-time rendering. */
export interface PerfBudget {
  /** Maximum draw calls per frame. */
  maxDrawCalls: number;
  /** Maximum triangles per frame. */
  maxTriangles: number;
  /** Maximum VRAM usage in megabytes. */
  maxVRAM_MB: number;
  /** Target framerate (fps). */
  targetFPS?: number;
  /** Maximum texture memory in megabytes. */
  maxTextureMemory_MB?: number;
  /** Maximum number of active lights. */
  maxLights?: number;
}

// ---------------------------------------------------------------------------
// CV-8: Point Cloud Processing
// ---------------------------------------------------------------------------

/** A 3D point cloud with optional per-point attributes. */
export interface PointCloud {
  /** Point positions, packed as [x0,y0,z0, x1,y1,z1, ...]. */
  positions: Float64Array;
  /** Per-point colours, packed as [r0,g0,b0, ...] (optional, values in [0,1]). */
  colors?: Float64Array;
  /** Per-point normals, packed as [nx0,ny0,nz0, ...] (optional). */
  normals?: Float64Array;
  /** Number of points. */
  count: number;
  /** Axis-aligned bounding box minimum. */
  boundsMin?: Vector3;
  /** Axis-aligned bounding box maximum. */
  boundsMax?: Vector3;
}

/** Voxel grid for downsampling or occupancy queries. */
export interface VoxelGrid {
  /** Voxel side length in metres. */
  voxelSize: number;
  /** Grid origin (minimum corner in world space). */
  origin: Vector3;
  /** Number of voxels along each axis. */
  resolution: { nx: number; ny: number; nz: number };
  /** Occupancy or value data (flattened, row-major: x + y*nx + z*nx*ny). */
  data: Float64Array;
  /** Total number of occupied voxels. */
  occupiedCount: number;
}

/** Configuration for Iterative Closest Point (ICP) registration. */
export interface ICPConfig {
  /** Maximum iterations. */
  maxIterations: number;
  /** Convergence tolerance on transform change. */
  tolerance: number;
  /** Maximum correspondence distance (metres). */
  maxCorrespondenceDistance: number;
  /** ICP variant. */
  method: 'point-to-point' | 'point-to-plane';
  /** Initial guess for the transform (column-major 4x4). */
  initialTransform?: Matrix4x4;
  /** Outlier rejection ratio (0-1, fraction of worst matches to reject). */
  outlierRatio: number;
}

/** Result of ICP registration. */
export interface ICPResult {
  /** Final rigid transform (column-major 4x4). */
  transform: Matrix4x4;
  /** Final mean squared error. */
  mse: number;
  /** Number of iterations performed. */
  iterations: number;
  /** Whether the algorithm converged. */
  converged: boolean;
  /** Number of inlier correspondences in the final iteration. */
  inlierCount: number;
  /** Fitness score (fraction of source points with correspondences). */
  fitness: number;
}

/** Result of RANSAC plane fitting. */
export interface RANSACPlaneResult {
  /** Plane normal (unit vector). */
  normal: Vector3;
  /** Plane distance from origin (dot(normal, p) = distance for points on plane). */
  distance: number;
  /** Inlier indices. */
  inlierIndices: Uint32Array;
  /** Number of inliers. */
  inlierCount: number;
  /** Inlier ratio. */
  inlierRatio: number;
}

/** Configuration for alpha shape / concave hull computation. */
export interface AlphaShapeConfig {
  /** Alpha value (radius of the bounding sphere). Smaller = tighter hull. */
  alpha: number;
  /** Whether to compute the 3D alpha shape (true) or 2D alpha shape on XY plane (false). */
  is3D: boolean;
  /** Whether to return only the boundary edges / triangles. */
  boundaryOnly: boolean;
}

// ---------------------------------------------------------------------------
// CV-9: Depth Estimation
// ---------------------------------------------------------------------------

/** A depth map (per-pixel depth values). */
export interface DepthMap {
  /** Depth values (width * height, row-major). */
  data: Float64Array;
  /** Map width in pixels. */
  width: number;
  /** Map height in pixels. */
  height: number;
  /** Minimum valid depth value. */
  minDepth: number;
  /** Maximum valid depth value. */
  maxDepth: number;
  /** Optional confidence map (same dimensions, values in [0,1]). */
  confidence?: Float64Array;
}

/** Configuration for stereo depth estimation. */
export interface StereoConfig {
  /** Left camera intrinsics. */
  leftIntrinsics: CameraIntrinsics;
  /** Right camera intrinsics. */
  rightIntrinsics: CameraIntrinsics;
  /** Baseline distance between cameras (metres). */
  baseline: number;
  /** Maximum disparity to search (pixels). */
  maxDisparity: number;
  /** Block size for block matching. */
  blockSize: number;
  /** Uniqueness ratio for disparity validation. */
  uniquenessRatio: number;
  /** Speckle filter window size. */
  speckleWindowSize: number;
  /** Speckle filter range. */
  speckleRange: number;
  /** Stereo algorithm. */
  method: 'block-matching' | 'sgbm' | 'neural';
}

/** Truncated signed distance function (TSDF) volume. */
export interface TSDFVolume {
  /** TSDF values (resolution^3, packed x + y*resX + z*resX*resY). */
  data: Float64Array;
  /** Volume resolution per axis. */
  resolution: { x: number; y: number; z: number };
  /** Voxel side length in metres. */
  voxelSize: number;
  /** World-space origin of the volume (corner with minimum coordinates). */
  origin: Vector3;
  /** Truncation distance in metres. */
  truncationDistance: number;
  /** Per-voxel observation weight (same dimensions as data). */
  weights: Float64Array;
}

/** Configuration for bilateral filtering of depth maps. */
export interface BilateralFilterConfig {
  /** Spatial kernel standard deviation (pixels). */
  sigmaSpace: number;
  /** Range (depth-value) kernel standard deviation (metres). */
  sigmaRange: number;
  /** Kernel radius in pixels (window = 2*radius + 1). */
  kernelRadius: number;
  /** Number of iterations. */
  iterations: number;
}

// ---------------------------------------------------------------------------
// CV-10: Segmentation
// ---------------------------------------------------------------------------

/** A 2D binary or labelled mask. */
export interface Mask2D {
  /** Mask data (width * height, row-major). Values are label IDs (0 = background). */
  data: Uint32Array;
  /** Mask width in pixels. */
  width: number;
  /** Mask height in pixels. */
  height: number;
}

/** 2D axis-aligned bounding box. */
export interface BBox2D {
  /** Top-left x (pixels). */
  x: number;
  /** Top-left y (pixels). */
  y: number;
  /** Width (pixels). */
  width: number;
  /** Height (pixels). */
  height: number;
  /** Detection confidence score in [0,1]. */
  confidence: number;
  /** Class label. */
  label: string;
}

/** 3D axis-aligned bounding box. */
export interface BBox3D {
  /** Centre position. */
  center: Vector3;
  /** Half-extents along each axis. */
  halfExtents: Vector3;
  /** Confidence score in [0,1]. */
  confidence: number;
  /** Class label. */
  label: string;
}

/** 3D oriented bounding box. */
export interface OrientedBBox3D {
  /** Centre position. */
  center: Vector3;
  /** Half-extents along local axes. */
  halfExtents: Vector3;
  /** Orientation quaternion. */
  orientation: Quaternion;
  /** Confidence score in [0,1]. */
  confidence: number;
  /** Class label. */
  label: string;
}

/** Configuration for Non-Maximum Suppression. */
export interface NMSConfig {
  /** IoU threshold above which detections are suppressed. */
  iouThreshold: number;
  /** Minimum confidence score to keep. */
  scoreThreshold: number;
  /** Maximum number of detections to keep. */
  maxDetections: number;
  /** Whether to run class-agnostic NMS (true) or per-class (false). */
  classAgnostic: boolean;
}

/** A connected component from instance segmentation. */
export interface ConnectedComponent {
  /** Component / instance ID. */
  id: number;
  /** Pixel count of the component. */
  area: number;
  /** 2D bounding box. */
  bbox: BBox2D;
  /** Centroid x (pixels). */
  centroidX: number;
  /** Centroid y (pixels). */
  centroidY: number;
  /** Class label. */
  label: string;
}

/** Physical dimension estimate derived from segmentation + depth. */
export interface DimensionEstimate {
  /** Estimated width in metres. */
  width: number;
  /** Estimated height in metres. */
  height: number;
  /** Estimated depth in metres. */
  depth: number;
  /** Confidence in the estimate (0-1). */
  confidence: number;
  /** Oriented bounding box in 3D. */
  obb: OrientedBBox3D;
}

// ---------------------------------------------------------------------------
// CV-11: XR / AR Integration
// ---------------------------------------------------------------------------

/** A full 6 degrees-of-freedom pose. */
export interface Pose6DoF {
  /** Position in world space. */
  position: Vector3;
  /** Orientation as a quaternion. */
  orientation: Quaternion;
}

/** Result of an AR hit test against detected surfaces. */
export interface HitTestResult {
  /** Pose of the hit point (position + surface orientation). */
  pose: Pose6DoF;
  /** Distance from the camera to the hit point (metres). */
  distance: number;
  /** Type of surface that was hit. */
  type: 'plane' | 'mesh' | 'point' | 'featurePoint';
  /** Confidence in the hit test result (0-1). */
  confidence: number;
  /** Optional plane classification if type is 'plane'. */
  planeType?: 'horizontal' | 'vertical' | 'ceiling' | 'floor' | 'wall';
}

/** An AR anchor in world space. */
export interface Anchor {
  /** Unique anchor identifier. */
  id: string;
  /** Anchor pose. */
  pose: Pose6DoF;
  /** Tracking confidence (0-1). */
  confidence: number;
  /** Whether the anchor is actively tracked. */
  isTracking: boolean;
  /** Creation timestamp (ms since epoch). */
  createdAt: number;
  /** Last update timestamp (ms since epoch). */
  updatedAt: number;
}

/** Metadata for a USDZ asset (Apple AR Quick Look). */
export interface USDZAssetMeta {
  /** Asset file name. */
  fileName: string;
  /** File size in bytes. */
  fileSize: number;
  /** Mesh count in the USDZ scene. */
  meshCount: number;
  /** Material count. */
  materialCount: number;
  /** Total triangle count. */
  triangleCount: number;
  /** Total texture memory estimate in bytes. */
  textureMemory: number;
  /** Physical dimensions of the bounding box (metres). */
  physicalDimensions: Vector3;
  /** Whether the asset includes animations. */
  hasAnimations: boolean;
  /** USDZ schema version. */
  schemaVersion: string;
}

// ---------------------------------------------------------------------------
// CV-12: Asset Pipeline
// ---------------------------------------------------------------------------

/** Configuration for asset validation checks. */
export interface AssetValidationConfig {
  /** Maximum allowed triangle count. */
  maxTriangles: number;
  /** Maximum allowed texture resolution (max of width, height). */
  maxTextureResolution: number;
  /** Maximum allowed file size in bytes. */
  maxFileSize: number;
  /** Maximum allowed texture file size in bytes. */
  maxTextureFileSize: number;
  /** Whether to require UV coordinates. */
  requireUVs: boolean;
  /** Whether to require normals. */
  requireNormals: boolean;
  /** Whether to check for degenerate triangles (zero area). */
  checkDegenerateTriangles: boolean;
  /** Whether to check for non-manifold edges. */
  checkManifold: boolean;
  /** Minimum allowed edge length (metres, to catch micro-triangles). */
  minEdgeLength: number;
  /** Maximum physical dimension in any axis (metres). */
  maxPhysicalDimension: number;
  /** Allowed mesh file formats. */
  allowedFormats: string[];
}

/** Result of running asset validation. */
export interface AssetValidationResult {
  /** Whether the asset passed all checks. */
  pass: boolean;
  /** List of human-readable violation descriptions. */
  violations: string[];
  /** Number of violations. */
  violationCount: number;
  /** Detailed metrics collected during validation. */
  metrics: AssetMetadata;
  /** Timestamp of the validation run (ISO 8601). */
  validatedAt: string;
}

/** Metadata describing an asset's complexity and resource usage. */
export interface AssetMetadata {
  /** Total triangle count. */
  triangleCount: number;
  /** Total vertex count. */
  vertexCount: number;
  /** Texture resolution (max of width, height) in pixels. */
  textureSize: number;
  /** Total file size in bytes (geometry + textures). */
  fileSize: number;
  /** Physical bounding-box dimensions in metres. */
  physicalDimensions: Vector3;
  /** Number of materials. */
  materialCount: number;
  /** Number of textures. */
  textureCount: number;
  /** Total texture memory in bytes. */
  textureMemoryBytes: number;
  /** Whether the mesh has UV coordinates. */
  hasUVs: boolean;
  /** Whether the mesh has normals. */
  hasNormals: boolean;
  /** Whether the mesh is manifold. */
  isManifold: boolean;
  /** Number of degenerate triangles. */
  degenerateTriangleCount: number;
  /** Number of non-manifold edges. */
  nonManifoldEdgeCount: number;
}

/** Configuration for quality gates in the asset pipeline. */
export interface QualityGateConfig {
  /** Named quality tiers, ordered from highest to lowest quality. */
  tiers: Array<{
    /** Tier name (e.g. 'hero', 'standard', 'background'). */
    name: string;
    /** Validation config for this tier. */
    validation: AssetValidationConfig;
  }>;
  /** Whether to auto-assign the best-matching tier or require explicit tier. */
  autoAssignTier: boolean;
  /** Whether to block pipeline if no tier matches. */
  blockOnFailure: boolean;
}

/** Configuration for progressive LOD generation in the asset pipeline. */
export interface ProgressiveLODConfig {
  /** LOD levels to generate, ordered from highest to lowest detail. */
  levels: Array<{
    /** Target triangle count for this LOD level. */
    targetTriangles: number;
    /** Target texture resolution for this LOD level. */
    targetTextureResolution: number;
    /** Screen-space error threshold for switching to this LOD. */
    screenSpaceError: number;
    /** Distance threshold for switching to this LOD (metres). */
    distance: number;
  }>;
  /** QEM decimation configuration shared across levels. */
  decimationConfig: QEMConfig;
  /** Whether to generate UV atlas for each LOD level. */
  generateUVAtlas: boolean;
  /** Whether to bake textures for simplified LODs. */
  bakeTextures: boolean;
  /** Texture bake config (used when bakeTextures is true). */
  bakeConfig?: BakeConfig;
}
