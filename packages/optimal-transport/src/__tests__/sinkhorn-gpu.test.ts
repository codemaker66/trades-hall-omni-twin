/**
 * GPU vs CPU output equivalence tests.
 *
 * WebGPU is not available in Node.js/vitest, so these tests verify:
 * 1. The SinkhornCPU wrapper works correctly
 * 2. The SinkhornGPU.create() returns null in non-browser environments
 * 3. The createSolver() auto-selection falls back to CPU
 *
 * For browser-based GPU tests, use the web app's test runner.
 */

import { describe, test, expect } from 'vitest'
import { SinkhornCPU, SinkhornGPU, createSolver } from '../sinkhorn-gpu'
import { sinkhorn } from '../sinkhorn'
import { computeRowSums, computeColSums } from '../utils'

describe('OT-1: GPU Solver (Node.js environment)', () => {
  test('SinkhornGPU.create() returns null without WebGPU', async () => {
    const gpu = await SinkhornGPU.create()
    expect(gpu).toBeNull()
  })

  test('createSolver() falls back to CPU', async () => {
    const solver = await createSolver()
    expect(solver).toBeInstanceOf(SinkhornCPU)
  })

  test('SinkhornCPU produces same results as raw sinkhorn()', () => {
    const a = new Float64Array([0.5, 0.5])
    const b = new Float64Array([0.5, 0.5])
    const C = new Float64Array([0, 1, 1, 0])

    const cpu = new SinkhornCPU()
    const cpuResult = cpu.solve(a, b, C, { epsilon: 0.05 })
    const rawResult = sinkhorn(a, b, C, { epsilon: 0.05 })

    // Should be identical
    expect(cpuResult.cost).toBeCloseTo(rawResult.cost, 10)
    expect(cpuResult.iterations).toBe(rawResult.iterations)
  })

  test('SinkhornCPU handles Float32Array input', () => {
    const a = new Float32Array([0.5, 0.5])
    const b = new Float32Array([0.5, 0.5])
    const C = new Float32Array([0, 1, 1, 0])

    const cpu = new SinkhornCPU()
    const result = cpu.solve(a, b, C, { epsilon: 0.05 })

    expect(result.plan.length).toBe(4)
    expect(result.cost).toBeGreaterThanOrEqual(0)
  })

  test('SinkhornCPU marginals are correct', () => {
    const a = new Float64Array([0.3, 0.3, 0.4])
    const b = new Float64Array([0.2, 0.5, 0.3])
    const C = new Float64Array([
      1, 2, 3,
      4, 1, 2,
      3, 4, 1,
    ])

    const cpu = new SinkhornCPU()
    const result = cpu.solve(a, b, C, { epsilon: 0.1 })

    const rowSums = computeRowSums(result.plan, 3, 3)
    const colSums = computeColSums(result.plan, 3, 3)

    for (let i = 0; i < 3; i++) {
      expect(Math.abs(rowSums[i]! - a[i]!)).toBeLessThan(0.01)
    }
    for (let j = 0; j < 3; j++) {
      expect(Math.abs(colSums[j]! - b[j]!)).toBeLessThan(0.01)
    }
  })
})
