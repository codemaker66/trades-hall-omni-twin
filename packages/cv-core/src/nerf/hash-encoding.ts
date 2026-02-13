// ---------------------------------------------------------------------------
// CV-5: NeRF — Multi-Resolution Hash Encoding (Instant-NGP style)
// ---------------------------------------------------------------------------

import type { Vector3, HashEncodingConfig } from '../types.js';

// ---------------------------------------------------------------------------
// Spatial hash function
// ---------------------------------------------------------------------------

/**
 * Compute a spatial hash for integer grid coordinates.
 *
 * Uses large co-prime multipliers (following Instant-NGP / NVIDIA's approach)
 * to map 3D integer coordinates to a flat hash table index.
 *
 * @param x         - Integer x coordinate.
 * @param y         - Integer y coordinate.
 * @param z         - Integer z coordinate.
 * @param tableSize - Size of the hash table (should be a power of two).
 * @returns Hash index in [0, tableSize).
 */
export function hashVertex(
  x: number,
  y: number,
  z: number,
  tableSize: number,
): number {
  // Large primes for mixing (same constants used in Instant-NGP paper)
  const p1 = 1;
  const p2 = 2654435761; // golden ratio * 2^32
  const p3 = 805459861;

  // Bitwise XOR mixing — operate in 32-bit integer space
  const hash = ((x * p1) ^ (y * p2) ^ (z * p3)) >>> 0;
  return hash % tableSize;
}

// ---------------------------------------------------------------------------
// Trilinear interpolation
// ---------------------------------------------------------------------------

/**
 * Trilinear interpolation of feature vectors stored on a 3D grid.
 *
 * The grid is a flat Float64Array of size (resolution^3 * featureDim),
 * laid out so that the feature at integer coordinate (ix, iy, iz) starts
 * at offset (ix + iy * resolution + iz * resolution^2) * featureDim.
 *
 * @param grid       - Flat grid of feature vectors.
 * @param x          - Continuous x coordinate in [0, resolution-1].
 * @param y          - Continuous y coordinate in [0, resolution-1].
 * @param z          - Continuous z coordinate in [0, resolution-1].
 * @param resolution - Grid resolution along each axis.
 * @param featureDim - Number of features per grid vertex.
 * @returns Interpolated feature vector of length `featureDim`.
 */
export function trilinearInterpolate(
  grid: Float64Array,
  x: number,
  y: number,
  z: number,
  resolution: number,
  featureDim: number,
): Float64Array {
  // Clamp to valid range
  const maxIdx = resolution - 1;
  const cx = Math.max(0, Math.min(x, maxIdx));
  const cy = Math.max(0, Math.min(y, maxIdx));
  const cz = Math.max(0, Math.min(z, maxIdx));

  const ix0 = Math.min(Math.floor(cx), maxIdx - 1);
  const iy0 = Math.min(Math.floor(cy), maxIdx - 1);
  const iz0 = Math.min(Math.floor(cz), maxIdx - 1);
  const ix1 = ix0 + 1;
  const iy1 = iy0 + 1;
  const iz1 = iz0 + 1;

  const fx = cx - ix0;
  const fy = cy - iy0;
  const fz = cz - iz0;

  const result = new Float64Array(featureDim);

  // Compute the 8 corner offsets into the flat grid
  const stride = featureDim;
  const offsets = [
    (ix0 + iy0 * resolution + iz0 * resolution * resolution) * stride,
    (ix1 + iy0 * resolution + iz0 * resolution * resolution) * stride,
    (ix0 + iy1 * resolution + iz0 * resolution * resolution) * stride,
    (ix1 + iy1 * resolution + iz0 * resolution * resolution) * stride,
    (ix0 + iy0 * resolution + iz1 * resolution * resolution) * stride,
    (ix1 + iy0 * resolution + iz1 * resolution * resolution) * stride,
    (ix0 + iy1 * resolution + iz1 * resolution * resolution) * stride,
    (ix1 + iy1 * resolution + iz1 * resolution * resolution) * stride,
  ];

  // Trilinear weights for the 8 corners
  const weights = [
    (1 - fx) * (1 - fy) * (1 - fz), // (0,0,0)
    fx * (1 - fy) * (1 - fz),        // (1,0,0)
    (1 - fx) * fy * (1 - fz),        // (0,1,0)
    fx * fy * (1 - fz),              // (1,1,0)
    (1 - fx) * (1 - fy) * fz,        // (0,0,1)
    fx * (1 - fy) * fz,              // (1,0,1)
    (1 - fx) * fy * fz,              // (0,1,1)
    fx * fy * fz,                    // (1,1,1)
  ];

  for (let d = 0; d < featureDim; d++) {
    let val = 0;
    for (let c = 0; c < 8; c++) {
      val += weights[c]! * grid[offsets[c]! + d]!;
    }
    result[d] = val;
  }

  return result;
}

// ---------------------------------------------------------------------------
// Multi-resolution hash encoding
// ---------------------------------------------------------------------------

/**
 * Create a multi-resolution hash grid encoding (Instant-NGP style).
 *
 * Returns an object with an `encode` method that maps a 3D point to a
 * concatenated feature vector across all resolution levels.
 *
 * Each level l has resolution:
 *   R_l = floor(baseResolution * perLevelScale^l)
 *
 * The point is normalised into [0, 1]^3 using `boundsMin` / `boundsMax`,
 * then scaled to each level's grid resolution. Features are looked up via
 * the hash table and trilinearly interpolated.
 *
 * @param config - Hash encoding configuration.
 * @returns Object with `encode(point) => Float64Array` method.
 */
export function createHashEncoding(
  config: HashEncodingConfig,
): { encode: (point: Vector3) => Float64Array } {
  const {
    nLevels,
    nFeaturesPerLevel,
    log2HashTableSize,
    baseResolution,
    perLevelScale,
    boundsMin,
    boundsMax,
  } = config;

  const tableSize = 1 << log2HashTableSize;
  const totalFeatures = nLevels * nFeaturesPerLevel;

  // Allocate hash tables (one per level)
  // In a trained model these would hold learned parameters; we initialise
  // to small random values for demonstration.
  const tables: Float64Array[] = [];
  let seed = 12345;
  for (let l = 0; l < nLevels; l++) {
    const table = new Float64Array(tableSize * nFeaturesPerLevel);
    for (let i = 0; i < table.length; i++) {
      // Simple LCG for deterministic initialisation
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      table[i] = (seed / 0x7fffffff - 0.5) * 0.01;
    }
    tables.push(table);
  }

  // Bounding box extents for normalisation
  const extX = boundsMax.x - boundsMin.x;
  const extY = boundsMax.y - boundsMin.y;
  const extZ = boundsMax.z - boundsMin.z;

  function encode(point: Vector3): Float64Array {
    const out = new Float64Array(totalFeatures);

    // Normalise point to [0, 1]
    const nx = extX > 0 ? (point.x - boundsMin.x) / extX : 0;
    const ny = extY > 0 ? (point.y - boundsMin.y) / extY : 0;
    const nz = extZ > 0 ? (point.z - boundsMin.z) / extZ : 0;

    for (let l = 0; l < nLevels; l++) {
      const resolution = Math.floor(baseResolution * Math.pow(perLevelScale, l));
      const table = tables[l]!;

      // Scale normalised coords to grid coordinates
      const gx = nx * (resolution - 1);
      const gy = ny * (resolution - 1);
      const gz = nz * (resolution - 1);

      // Floor indices
      const ix0 = Math.min(Math.floor(gx), resolution - 2);
      const iy0 = Math.min(Math.floor(gy), resolution - 2);
      const iz0 = Math.min(Math.floor(gz), resolution - 2);
      const ix1 = ix0 + 1;
      const iy1 = iy0 + 1;
      const iz1 = iz0 + 1;

      // Fractional parts
      const fx = gx - ix0;
      const fy = gy - iy0;
      const fz = gz - iz0;

      // Hash the 8 corner vertices
      const h000 = hashVertex(ix0, iy0, iz0, tableSize);
      const h100 = hashVertex(ix1, iy0, iz0, tableSize);
      const h010 = hashVertex(ix0, iy1, iz0, tableSize);
      const h110 = hashVertex(ix1, iy1, iz0, tableSize);
      const h001 = hashVertex(ix0, iy0, iz1, tableSize);
      const h101 = hashVertex(ix1, iy0, iz1, tableSize);
      const h011 = hashVertex(ix0, iy1, iz1, tableSize);
      const h111 = hashVertex(ix1, iy1, iz1, tableSize);

      // Trilinear weights
      const w000 = (1 - fx) * (1 - fy) * (1 - fz);
      const w100 = fx * (1 - fy) * (1 - fz);
      const w010 = (1 - fx) * fy * (1 - fz);
      const w110 = fx * fy * (1 - fz);
      const w001 = (1 - fx) * (1 - fy) * fz;
      const w101 = fx * (1 - fy) * fz;
      const w011 = (1 - fx) * fy * fz;
      const w111 = fx * fy * fz;

      const outOffset = l * nFeaturesPerLevel;
      for (let d = 0; d < nFeaturesPerLevel; d++) {
        out[outOffset + d] =
          w000 * table[h000 * nFeaturesPerLevel + d]! +
          w100 * table[h100 * nFeaturesPerLevel + d]! +
          w010 * table[h010 * nFeaturesPerLevel + d]! +
          w110 * table[h110 * nFeaturesPerLevel + d]! +
          w001 * table[h001 * nFeaturesPerLevel + d]! +
          w101 * table[h101 * nFeaturesPerLevel + d]! +
          w011 * table[h011 * nFeaturesPerLevel + d]! +
          w111 * table[h111 * nFeaturesPerLevel + d]!;
      }
    }

    return out;
  }

  return { encode };
}
