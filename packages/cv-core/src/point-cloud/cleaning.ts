// ---------------------------------------------------------------------------
// CV-8: Point Cloud Cleaning — statistical outlier removal and voxel
// downsampling for raw point-cloud data.
// ---------------------------------------------------------------------------

import type { PointCloud } from '../types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Squared Euclidean distance between two 3D points in packed arrays. */
function sqDist(
  a: Float64Array,
  ai: number,
  b: Float64Array,
  bi: number,
): number {
  const dx = a[ai]! - b[bi]!;
  const dy = a[ai + 1]! - b[bi + 1]!;
  const dz = a[ai + 2]! - b[bi + 2]!;
  return dx * dx + dy * dy + dz * dz;
}

// ---------------------------------------------------------------------------
// statisticalOutlierRemoval
// ---------------------------------------------------------------------------

/**
 * Remove statistical outliers from a point cloud.
 *
 * For every point, compute the mean Euclidean distance to its k nearest
 * neighbours.  Then compute the global mean and standard deviation of those
 * per-point mean distances.  Points whose mean k-NN distance exceeds
 * `globalMean + stdRatio * globalStd` are classified as outliers and removed.
 *
 * Uses brute-force k-NN search (O(n^2 k)) which is acceptable for
 * moderate-sized clouds.
 *
 * @param cloud       Input point cloud.
 * @param kNeighbors  Number of nearest neighbours to consider.
 * @param stdRatio    Standard-deviation multiplier for the outlier threshold.
 * @returns           A new point cloud with outliers removed.
 */
export function statisticalOutlierRemoval(
  cloud: PointCloud,
  kNeighbors: number,
  stdRatio: number,
): PointCloud {
  const n = cloud.count;
  if (n <= kNeighbors) return cloud;

  const positions = cloud.positions;
  const k = Math.min(kNeighbors, n - 1);

  // For each point, compute mean distance to k nearest neighbours.
  const meanDists = new Float64Array(n);

  // Temporary array for neighbour distances (reused per point).
  const dists = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const pi = i * 3;

    // Compute distance from point i to every other point.
    for (let j = 0; j < n; j++) {
      dists[j] = j === i ? Infinity : sqDist(positions, pi, positions, j * 3);
    }

    // Partial sort: find the k smallest distances.
    // Use a simple selection approach — copy, sort ascending, take first k.
    const sorted = Float64Array.from(dists).sort();

    let sum = 0;
    for (let ki = 0; ki < k; ki++) {
      sum += Math.sqrt(sorted[ki]!);
    }
    meanDists[i] = sum / k;
  }

  // Compute global mean and std of per-point mean distances.
  let globalSum = 0;
  for (let i = 0; i < n; i++) {
    globalSum += meanDists[i]!;
  }
  const globalMean = globalSum / n;

  let varianceSum = 0;
  for (let i = 0; i < n; i++) {
    const d = meanDists[i]! - globalMean;
    varianceSum += d * d;
  }
  const globalStd = Math.sqrt(varianceSum / n);

  const threshold = globalMean + stdRatio * globalStd;

  // Collect inlier indices.
  const inlierIndices: number[] = [];
  for (let i = 0; i < n; i++) {
    if (meanDists[i]! <= threshold) {
      inlierIndices.push(i);
    }
  }

  return extractSubset(cloud, inlierIndices);
}

// ---------------------------------------------------------------------------
// voxelDownsample
// ---------------------------------------------------------------------------

/**
 * Downsample a point cloud using a voxel grid filter.
 *
 * Each point is assigned to a voxel cell based on its position.  Points within
 * the same cell are averaged (positions, colours, normals) to produce one
 * representative point per occupied voxel.
 *
 * @param cloud     Input point cloud.
 * @param voxelSize Side length of each cubic voxel (metres).
 * @returns         A new, downsampled point cloud.
 */
export function voxelDownsample(
  cloud: PointCloud,
  voxelSize: number,
): PointCloud {
  if (voxelSize <= 0 || cloud.count === 0) return cloud;

  const n = cloud.count;
  const positions = cloud.positions;
  const hasColors = !!cloud.colors;
  const hasNormals = !!cloud.normals;
  const invSize = 1 / voxelSize;

  // Map: voxel key -> accumulated data.
  const voxels = new Map<
    string,
    {
      sx: number;
      sy: number;
      sz: number;
      sr: number;
      sg: number;
      sb: number;
      snx: number;
      sny: number;
      snz: number;
      count: number;
    }
  >();

  for (let i = 0; i < n; i++) {
    const pi = i * 3;
    const px = positions[pi]!;
    const py = positions[pi + 1]!;
    const pz = positions[pi + 2]!;

    const vx = Math.floor(px * invSize);
    const vy = Math.floor(py * invSize);
    const vz = Math.floor(pz * invSize);
    const key = `${vx},${vy},${vz}`;

    let entry = voxels.get(key);
    if (!entry) {
      entry = {
        sx: 0, sy: 0, sz: 0,
        sr: 0, sg: 0, sb: 0,
        snx: 0, sny: 0, snz: 0,
        count: 0,
      };
      voxels.set(key, entry);
    }

    entry.sx += px;
    entry.sy += py;
    entry.sz += pz;
    entry.count += 1;

    if (hasColors) {
      entry.sr += cloud.colors![pi]!;
      entry.sg += cloud.colors![pi + 1]!;
      entry.sb += cloud.colors![pi + 2]!;
    }

    if (hasNormals) {
      entry.snx += cloud.normals![pi]!;
      entry.sny += cloud.normals![pi + 1]!;
      entry.snz += cloud.normals![pi + 2]!;
    }
  }

  const outCount = voxels.size;
  const outPositions = new Float64Array(outCount * 3);
  const outColors = hasColors ? new Float64Array(outCount * 3) : undefined;
  const outNormals = hasNormals ? new Float64Array(outCount * 3) : undefined;

  let idx = 0;
  for (const entry of voxels.values()) {
    const inv = 1 / entry.count;
    const oi = idx * 3;

    outPositions[oi] = entry.sx * inv;
    outPositions[oi + 1] = entry.sy * inv;
    outPositions[oi + 2] = entry.sz * inv;

    if (outColors) {
      outColors[oi] = entry.sr * inv;
      outColors[oi + 1] = entry.sg * inv;
      outColors[oi + 2] = entry.sb * inv;
    }

    if (outNormals) {
      // Re-normalise the averaged normal.
      let nx = entry.snx * inv;
      let ny = entry.sny * inv;
      let nz = entry.snz * inv;
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
      if (len > 1e-15) {
        nx /= len;
        ny /= len;
        nz /= len;
      }
      outNormals[oi] = nx;
      outNormals[oi + 1] = ny;
      outNormals[oi + 2] = nz;
    }

    idx++;
  }

  return {
    positions: outPositions,
    colors: outColors,
    normals: outNormals,
    count: outCount,
  };
}

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

/** Extract a subset of a point cloud by index list. */
function extractSubset(cloud: PointCloud, indices: number[]): PointCloud {
  const m = indices.length;
  const outPositions = new Float64Array(m * 3);
  const outColors = cloud.colors ? new Float64Array(m * 3) : undefined;
  const outNormals = cloud.normals ? new Float64Array(m * 3) : undefined;

  for (let k = 0; k < m; k++) {
    const src = indices[k]! * 3;
    const dst = k * 3;

    outPositions[dst] = cloud.positions[src]!;
    outPositions[dst + 1] = cloud.positions[src + 1]!;
    outPositions[dst + 2] = cloud.positions[src + 2]!;

    if (outColors && cloud.colors) {
      outColors[dst] = cloud.colors[src]!;
      outColors[dst + 1] = cloud.colors[src + 1]!;
      outColors[dst + 2] = cloud.colors[src + 2]!;
    }

    if (outNormals && cloud.normals) {
      outNormals[dst] = cloud.normals[src]!;
      outNormals[dst + 1] = cloud.normals[src + 1]!;
      outNormals[dst + 2] = cloud.normals[src + 2]!;
    }
  }

  return {
    positions: outPositions,
    colors: outColors,
    normals: outNormals,
    count: m,
  };
}
