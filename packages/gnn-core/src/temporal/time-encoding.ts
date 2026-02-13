// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-6: Time Encoding Functions
// Bochner (random Fourier feature) and sinusoidal positional encodings
// for temporal graph networks.
// ---------------------------------------------------------------------------

/**
 * Bochner time encoding using random Fourier features.
 *
 * Phi(t) = sqrt(1/d) * [cos(w_1 t), sin(w_1 t), ..., cos(w_d t), sin(w_d t)]
 *
 * @param timestamps - (T) array of timestamps
 * @param frequencies - (d) array of frequency values
 * @returns (T x 2d) matrix, row-major
 */
export function bochnerTimeEncoding(
  timestamps: Float64Array,
  frequencies: Float64Array,
): Float64Array {
  const T = timestamps.length;
  const d = frequencies.length;
  const outDim = 2 * d;
  const scale = Math.sqrt(1 / d);
  const out = new Float64Array(T * outDim);

  for (let t = 0; t < T; t++) {
    const ts = timestamps[t]!;
    const rowOffset = t * outDim;
    for (let f = 0; f < d; f++) {
      const angle = frequencies[f]! * ts;
      out[rowOffset + 2 * f] = scale * Math.cos(angle);
      out[rowOffset + 2 * f + 1] = scale * Math.sin(angle);
    }
  }

  return out;
}

/**
 * Sinusoidal position encoding (Vaswani et al. 2017).
 *
 * PE(pos, 2i)   = sin(pos / 10000^(2i/dim))
 * PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))
 *
 * @param timestamps - (T) array of position values (timestamps)
 * @param dim - encoding dimension (should be even for clean pairing)
 * @returns (T x dim) matrix, row-major
 */
export function positionEncoding(
  timestamps: Float64Array,
  dim: number,
): Float64Array {
  const T = timestamps.length;
  const out = new Float64Array(T * dim);

  for (let t = 0; t < T; t++) {
    const pos = timestamps[t]!;
    const rowOffset = t * dim;
    for (let i = 0; i < dim; i += 2) {
      const divTerm = Math.pow(10000, i / dim);
      out[rowOffset + i] = Math.sin(pos / divTerm);
      if (i + 1 < dim) {
        out[rowOffset + i + 1] = Math.cos(pos / divTerm);
      }
    }
  }

  return out;
}

/**
 * Relative time encoding: Bochner encoding of |t1 - t2|.
 *
 * @param t1 - first timestamp
 * @param t2 - second timestamp
 * @param frequencies - (d) array of frequency values
 * @returns (2d) vector — a single Bochner encoding of the time delta
 */
export function relativeTimeEncoding(
  t1: number,
  t2: number,
  frequencies: Float64Array,
): Float64Array {
  const dt = Math.abs(t1 - t2);
  const timestamps = new Float64Array(1);
  timestamps[0] = dt;
  // Reuse bochnerTimeEncoding for a single timestamp — returns (1 x 2d)
  return bochnerTimeEncoding(timestamps, frequencies);
}
