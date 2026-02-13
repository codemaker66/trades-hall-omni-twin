/**
 * PS-6: Restricted Boltzmann Machine
 *
 * RBMs learn the distribution of "good" layouts from training data, then
 * generate new configurations by sampling.
 *
 * Energy: E(v,h) = -a^T v - b^T h - v^T W h
 * P(v,h) = (1/Z) exp(-E(v,h))
 *
 * Training: Contrastive Divergence (Hinton 2002)
 *   dW = lr * (<v0 h0^T>_data - <vk hk^T>_recon)
 *
 * Reference: Hinton (2002), "Training Products of Experts by Minimizing
 * Contrastive Divergence", Neural Computation 14(8).
 */

import type { RBMConfig, PRNG } from './types.js'
import { createPRNG } from './types.js'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sigmoid(x: number): number {
  if (x > 500) return 1.0
  if (x < -500) return 0.0
  return 1.0 / (1.0 + Math.exp(-x))
}

// ---------------------------------------------------------------------------
// RBM Class
// ---------------------------------------------------------------------------

export class RBM {
  readonly nVisible: number
  readonly nHidden: number
  private weights: Float64Array      // [nVisible * nHidden]
  private visibleBias: Float64Array  // [nVisible]
  private hiddenBias: Float64Array   // [nHidden]
  private rng: PRNG

  constructor(nVisible: number, nHidden: number, seed?: number) {
    this.nVisible = nVisible
    this.nHidden = nHidden
    this.rng = createPRNG(seed ?? 42)

    // Xavier initialization: stddev = sqrt(2 / (nVis + nHid))
    const stddev = Math.sqrt(2.0 / (nVisible + nHidden))
    this.weights = new Float64Array(nVisible * nHidden)
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = this.boxMuller() * stddev
    }
    this.visibleBias = new Float64Array(nVisible)
    this.hiddenBias = new Float64Array(nHidden)
  }

  private boxMuller(): number {
    const u1 = this.rng.random() || 1e-10
    const u2 = this.rng.random()
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
  }

  /**
   * P(h_j = 1 | v) = sigmoid(b_j + sum_i W_ij * v_i)
   * Returns hidden probabilities (not sampled).
   */
  hiddenProbabilities(visible: Float64Array): Float64Array {
    const probs = new Float64Array(this.nHidden)
    for (let j = 0; j < this.nHidden; j++) {
      let activation = this.hiddenBias[j]!
      for (let i = 0; i < this.nVisible; i++) {
        activation += (this.weights[i * this.nHidden + j]!) * (visible[i]!)
      }
      probs[j] = sigmoid(activation)
    }
    return probs
  }

  /** Sample binary hidden units given visible */
  sampleHiddenGivenVisible(visible: Float64Array): Float64Array {
    const probs = this.hiddenProbabilities(visible)
    const sampled = new Float64Array(this.nHidden)
    for (let j = 0; j < this.nHidden; j++) {
      sampled[j] = this.rng.random() < probs[j]! ? 1.0 : 0.0
    }
    return sampled
  }

  /**
   * P(v_i = 1 | h) = sigmoid(a_i + sum_j W_ij * h_j)
   * Returns visible probabilities.
   */
  visibleProbabilities(hidden: Float64Array): Float64Array {
    const probs = new Float64Array(this.nVisible)
    for (let i = 0; i < this.nVisible; i++) {
      let activation = this.visibleBias[i]!
      for (let j = 0; j < this.nHidden; j++) {
        activation += (this.weights[i * this.nHidden + j]!) * (hidden[j]!)
      }
      probs[i] = sigmoid(activation)
    }
    return probs
  }

  /** Sample binary visible units given hidden */
  sampleVisibleGivenHidden(hidden: Float64Array): Float64Array {
    const probs = this.visibleProbabilities(hidden)
    const sampled = new Float64Array(this.nVisible)
    for (let i = 0; i < this.nVisible; i++) {
      sampled[i] = this.rng.random() < probs[i]! ? 1.0 : 0.0
    }
    return sampled
  }

  /**
   * Train with CD-k (Contrastive Divergence, k Gibbs steps).
   *
   * data: flat Float64Array of [nSamples * nVisible] training layouts.
   */
  trainCD(data: Float64Array, nSamples: number, config: RBMConfig): void {
    const { cdK, learningRate, epochs, momentum, weightDecay } = config
    const nVis = this.nVisible
    const nHid = this.nHidden

    // Momentum accumulators
    const dW = new Float64Array(nVis * nHid)
    const dVB = new Float64Array(nVis)
    const dHB = new Float64Array(nHid)

    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let s = 0; s < nSamples; s++) {
        // Extract sample
        const v0 = data.slice(s * nVis, (s + 1) * nVis)

        // Positive phase: h0 probabilities from data
        const h0Prob = this.hiddenProbabilities(v0)
        const h0 = new Float64Array(nHid)
        for (let j = 0; j < nHid; j++) {
          h0[j] = this.rng.random() < h0Prob[j]! ? 1.0 : 0.0
        }

        // Negative phase: k steps of Gibbs sampling
        let vk: Float64Array = v0
        let hkProb = h0Prob
        for (let step = 0; step < cdK; step++) {
          vk = new Float64Array(this.sampleVisibleGivenHidden(
            step === 0 ? h0 : this.sampleHiddenGivenVisible(vk),
          ))
          hkProb = this.hiddenProbabilities(vk)
        }

        // Update weights: dW = lr * (v0 * h0^T - vk * hk^T) + momentum * prev - decay * W
        for (let i = 0; i < nVis; i++) {
          for (let j = 0; j < nHid; j++) {
            const idx = i * nHid + j
            const grad = (v0[i]! * h0Prob[j]!) - (vk[i]! * hkProb[j]!)
            dW[idx] = momentum * dW[idx]! + learningRate * (grad - weightDecay * this.weights[idx]!)
            this.weights[idx] = this.weights[idx]! + dW[idx]!
          }
        }

        // Update visible biases
        for (let i = 0; i < nVis; i++) {
          const grad = v0[i]! - vk[i]!
          dVB[i] = momentum * dVB[i]! + learningRate * grad
          this.visibleBias[i] = this.visibleBias[i]! + dVB[i]!
        }

        // Update hidden biases
        for (let j = 0; j < nHid; j++) {
          const grad = h0Prob[j]! - hkProb[j]!
          dHB[j] = momentum * dHB[j]! + learningRate * grad
          this.hiddenBias[j] = this.hiddenBias[j]! + dHB[j]!
        }
      }
    }
  }

  /**
   * Generate a new layout by Gibbs sampling from the learned distribution.
   * Starts from random visible state, runs nGibbsSteps.
   */
  sample(nGibbsSteps: number): Float64Array {
    let visible = new Float64Array(this.nVisible)
    for (let i = 0; i < this.nVisible; i++) {
      visible[i] = this.rng.random() < 0.5 ? 1.0 : 0.0
    }

    for (let step = 0; step < nGibbsSteps; step++) {
      const hidden = this.sampleHiddenGivenVisible(visible)
      visible = new Float64Array(this.sampleVisibleGivenHidden(hidden))
    }
    return visible
  }

  /**
   * Free energy: F(v) = -a^T v - sum_j log(1 + exp(b_j + sum_i W_ij v_i))
   * Low free energy = high probability = layout matches learned distribution.
   */
  freeEnergy(visible: Float64Array): number {
    let energy = 0.0

    // -a^T v
    for (let i = 0; i < this.nVisible; i++) {
      energy -= this.visibleBias[i]! * visible[i]!
    }

    // -sum_j log(1 + exp(b_j + sum_i W_ij v_i))
    for (let j = 0; j < this.nHidden; j++) {
      let activation = this.hiddenBias[j]!
      for (let i = 0; i < this.nVisible; i++) {
        activation += this.weights[i * this.nHidden + j]! * visible[i]!
      }
      energy -= Math.log(1.0 + Math.exp(Math.min(500, activation)))
    }

    return energy
  }

  /** Reconstruct: v -> h_prob -> v_reconstructed (using probabilities, not samples) */
  reconstruct(visible: Float64Array): Float64Array {
    const hProb = this.hiddenProbabilities(visible)
    return this.visibleProbabilities(hProb)
  }

  /** Average MSE between input and reconstruction over all samples */
  getReconstructionError(data: Float64Array, nSamples: number): number {
    let totalError = 0.0
    for (let s = 0; s < nSamples; s++) {
      const v = data.slice(s * this.nVisible, (s + 1) * this.nVisible)
      const recon = this.reconstruct(v)
      for (let i = 0; i < this.nVisible; i++) {
        const diff = v[i]! - recon[i]!
        totalError += diff * diff
      }
    }
    return totalError / (nSamples * this.nVisible)
  }

  /** Export weights for serialization */
  getWeights(): { weights: Float64Array; visibleBias: Float64Array; hiddenBias: Float64Array } {
    return {
      weights: new Float64Array(this.weights),
      visibleBias: new Float64Array(this.visibleBias),
      hiddenBias: new Float64Array(this.hiddenBias),
    }
  }

  /** Import weights */
  setWeights(w: { weights: Float64Array; visibleBias: Float64Array; hiddenBias: Float64Array }): void {
    this.weights.set(w.weights)
    this.visibleBias.set(w.visibleBias)
    this.hiddenBias.set(w.hiddenBias)
  }
}
