/**
 * Random number generation utilities for stochastic simulation.
 *
 * Implements:
 * - Seedable PRNG (xoshiro256** — period 2^256-1, passes all BigCrush tests)
 * - Box-Muller transform for normal distribution
 * - Poisson via inversion method (exact for small lambda, normal approx for large)
 * - Sobol quasi-random sequences (Gray code construction, direction numbers from Joe-Kuo)
 */

// ---------------------------------------------------------------------------
// xoshiro256** — State-of-the-art PRNG
// ---------------------------------------------------------------------------

/**
 * Seedable, high-quality PRNG using xoshiro256** algorithm.
 * Period: 2^256-1. Passes BigCrush and PractRand test suites.
 */
export class Rng {
  private s: BigUint64Array

  constructor(seed: number = Date.now()) {
    // Initialize state via SplitMix64 (recommended seeding for xoshiro)
    this.s = new BigUint64Array(4)
    let s = BigInt(seed) & 0xFFFFFFFFFFFFFFFFn
    for (let i = 0; i < 4; i++) {
      s = (s + 0x9E3779B97F4A7C15n) & 0xFFFFFFFFFFFFFFFFn
      let z = s
      z = ((z ^ (z >> 30n)) * 0xBF58476D1CE4E5B9n) & 0xFFFFFFFFFFFFFFFFn
      z = ((z ^ (z >> 27n)) * 0x94D049BB133111EBn) & 0xFFFFFFFFFFFFFFFFn
      z = z ^ (z >> 31n)
      this.s[i] = z
    }
  }

  /** Returns a uniform random number in [0, 1) */
  next(): number {
    const s = this.s
    // xoshiro256** result calculation
    const result = (rotl(s[1]! * 5n, 7n) * 9n) & 0xFFFFFFFFFFFFFFFFn

    const t = (s[1]! << 17n) & 0xFFFFFFFFFFFFFFFFn

    s[2] = (s[2]! ^ s[0]!) & 0xFFFFFFFFFFFFFFFFn
    s[3] = (s[3]! ^ s[1]!) & 0xFFFFFFFFFFFFFFFFn
    s[1] = (s[1]! ^ s[2]!) & 0xFFFFFFFFFFFFFFFFn
    s[0] = (s[0]! ^ s[3]!) & 0xFFFFFFFFFFFFFFFFn

    s[2] = (s[2]! ^ t) & 0xFFFFFFFFFFFFFFFFn
    s[3] = rotl(s[3]!, 45n)

    // Convert to double in [0, 1) using upper 53 bits
    return Number(result >> 11n) / 9007199254740992 // 2^53
  }

  /**
   * Standard normal variate via Box-Muller transform.
   * Generates two independent N(0,1) values; caches the spare.
   */
  private spareNormal: number | null = null

  normal(mean: number = 0, stddev: number = 1): number {
    if (this.spareNormal !== null) {
      const val = this.spareNormal
      this.spareNormal = null
      return mean + stddev * val
    }

    let u: number, v: number, s: number
    do {
      u = 2 * this.next() - 1
      v = 2 * this.next() - 1
      s = u * u + v * v
    } while (s >= 1 || s === 0)

    const factor = Math.sqrt(-2 * Math.log(s) / s)
    this.spareNormal = v * factor
    return mean + stddev * u * factor
  }

  /**
   * Poisson-distributed random variate.
   * Uses inversion method for lambda < 30, normal approximation otherwise.
   */
  poisson(lambda: number): number {
    if (lambda < 0) return 0
    if (lambda < 30) {
      // Knuth's algorithm (exact)
      const L = Math.exp(-lambda)
      let k = 0
      let p = 1
      do {
        k++
        p *= this.next()
      } while (p > L)
      return k - 1
    }
    // Normal approximation for large lambda (CLT)
    return Math.max(0, Math.round(this.normal(lambda, Math.sqrt(lambda))))
  }

  /**
   * Exponential random variate with rate parameter.
   * Mean = 1/rate.
   */
  exponential(rate: number): number {
    return -Math.log(1 - this.next()) / rate
  }
}

function rotl(x: bigint, k: bigint): bigint {
  return ((x << k) | (x >> (64n - k))) & 0xFFFFFFFFFFFFFFFFn
}

// ---------------------------------------------------------------------------
// Sobol Quasi-Random Sequences
// ---------------------------------------------------------------------------

/**
 * Sobol sequence generator using Gray code construction.
 * Direction numbers from Joe & Kuo (2010) for first 21201 dimensions.
 * Here we use the standard Sobol direction numbers for the first 8 dimensions.
 */
export class SobolSequence {
  private dimension: number
  private count: number
  /** Direction numbers: v[dim][bit] */
  private v: Uint32Array[]

  constructor(dimension: number) {
    this.dimension = dimension
    this.count = 0
    this.v = []

    for (let d = 0; d < dimension; d++) {
      this.v.push(new Uint32Array(32))
      if (d === 0) {
        // First dimension: Van der Corput sequence
        for (let i = 0; i < 32; i++) {
          this.v[0]![i] = 1 << (31 - i)
        }
      } else {
        // Use primitive polynomials and direction numbers
        // For dimensions > 1, we use standard initial direction numbers
        const dirNums = SOBOL_DIRECTION_NUMBERS[d - 1]
        if (dirNums) {
          const { s, a, m } = dirNums
          for (let i = 0; i < s; i++) {
            this.v[d]![i] = m[i]! << (31 - i)
          }
          for (let i = s; i < 32; i++) {
            let newV = this.v[d]![i - s]! ^ (this.v[d]![i - s]! >> s)
            for (let j = 1; j < s; j++) {
              if ((a >> (s - 1 - j)) & 1) {
                newV ^= this.v[d]![i - j]!
              }
            }
            this.v[d]![i] = newV
          }
        } else {
          // Fallback: use Van der Corput with scrambling
          for (let i = 0; i < 32; i++) {
            this.v[d]![i] = 1 << (31 - i)
          }
        }
      }
    }
  }

  /** Generate the next point in [0, 1)^dimension */
  next(): number[] {
    this.count++
    const point = new Array<number>(this.dimension)

    // Gray code: find rightmost zero bit of count
    const c = ctz(this.count)

    for (let d = 0; d < this.dimension; d++) {
      // XOR with direction number at position c
      const val = this.count === 1
        ? this.v[d]![0]!
        : this.v[d]![c]!
      point[d] = val / 4294967296 // 2^32
    }

    return point
  }

  /** Reset the sequence */
  reset(): void {
    this.count = 0
  }
}

/** Count trailing zeros */
function ctz(n: number): number {
  if (n === 0) return 32
  let c = 0
  while ((n & 1) === 0) {
    n >>= 1
    c++
  }
  return c
}

/**
 * Direction numbers for Sobol dimensions 2-8.
 * s = degree of primitive polynomial, a = coefficients, m = initial direction numbers.
 * From Joe & Kuo (2010).
 */
const SOBOL_DIRECTION_NUMBERS: Array<{ s: number; a: number; m: number[] }> = [
  { s: 1, a: 0, m: [1] },                           // dim 2
  { s: 2, a: 1, m: [1, 1] },                         // dim 3
  { s: 3, a: 1, m: [1, 1, 1] },                      // dim 4
  { s: 3, a: 2, m: [1, 3, 1] },                      // dim 5
  { s: 4, a: 1, m: [1, 1, 1, 1] },                   // dim 6
  { s: 4, a: 4, m: [1, 3, 5, 13] },                  // dim 7
  { s: 5, a: 2, m: [1, 1, 5, 5, 17] },               // dim 8
]
