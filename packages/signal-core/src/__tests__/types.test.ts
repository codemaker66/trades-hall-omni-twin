// ---------------------------------------------------------------------------
// SP-0: Types & Matrix Utilities Tests
// ---------------------------------------------------------------------------
import { describe, it, expect } from 'vitest';
import {
  createMatrix,
  matGet,
  matSet,
  matVecMul,
  matMul,
  matTranspose,
  matAdd,
  matSub,
  matScale,
  matIdentity,
  matInvert,
  matDiag,
  arrayToMatrix,
  createPRNG,
} from '../types.js';

describe('Matrix utilities', () => {
  describe('createMatrix', () => {
    it('creates zero matrix by default', () => {
      const m = createMatrix(2, 3);
      expect(m.rows).toBe(2);
      expect(m.cols).toBe(3);
      expect(m.data.length).toBe(6);
      for (let i = 0; i < 6; i++) expect(m.data[i]).toBe(0);
    });

    it('creates matrix from data using arrayToMatrix', () => {
      const m = arrayToMatrix(new Float64Array([1, 2, 3, 4]), 2, 2);
      expect(matGet(m, 0, 0)).toBe(1);
      expect(matGet(m, 0, 1)).toBe(2);
      expect(matGet(m, 1, 0)).toBe(3);
      expect(matGet(m, 1, 1)).toBe(4);
    });
  });

  describe('matGet / matSet', () => {
    it('gets and sets values', () => {
      const m = createMatrix(3, 3);
      matSet(m, 1, 2, 7);
      expect(matGet(m, 1, 2)).toBe(7);
    });
  });

  describe('matIdentity', () => {
    it('creates identity matrix', () => {
      const I = matIdentity(3);
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          expect(matGet(I, i, j)).toBe(i === j ? 1 : 0);
        }
      }
    });
  });

  describe('matVecMul', () => {
    it('multiplies matrix by vector', () => {
      const A = arrayToMatrix(new Float64Array([1, 2, 3, 4, 5, 6]), 2, 3);
      const x = new Float64Array([1, 1, 1]);
      const result = matVecMul(A, x);
      expect(result[0]).toBe(6);  // 1+2+3
      expect(result[1]).toBe(15); // 4+5+6
    });
  });

  describe('matMul', () => {
    it('multiplies two matrices', () => {
      const A = arrayToMatrix(new Float64Array([1, 2, 3, 4]), 2, 2);
      const B = arrayToMatrix(new Float64Array([5, 6, 7, 8]), 2, 2);
      const C = matMul(A, B);
      expect(matGet(C, 0, 0)).toBe(19);  // 1*5 + 2*7
      expect(matGet(C, 0, 1)).toBe(22);  // 1*6 + 2*8
      expect(matGet(C, 1, 0)).toBe(43);  // 3*5 + 4*7
      expect(matGet(C, 1, 1)).toBe(50);  // 3*6 + 4*8
    });
  });

  describe('matTranspose', () => {
    it('transposes matrix', () => {
      const A = arrayToMatrix(new Float64Array([1, 2, 3, 4, 5, 6]), 2, 3);
      const AT = matTranspose(A);
      expect(AT.rows).toBe(3);
      expect(AT.cols).toBe(2);
      expect(matGet(AT, 0, 0)).toBe(1);
      expect(matGet(AT, 0, 1)).toBe(4);
      expect(matGet(AT, 2, 0)).toBe(3);
      expect(matGet(AT, 2, 1)).toBe(6);
    });
  });

  describe('matAdd / matSub', () => {
    it('adds two matrices', () => {
      const A = arrayToMatrix(new Float64Array([1, 2, 3, 4]), 2, 2);
      const B = arrayToMatrix(new Float64Array([5, 6, 7, 8]), 2, 2);
      const C = matAdd(A, B);
      expect(matGet(C, 0, 0)).toBe(6);
      expect(matGet(C, 1, 1)).toBe(12);
    });

    it('subtracts two matrices', () => {
      const A = arrayToMatrix(new Float64Array([5, 6, 7, 8]), 2, 2);
      const B = arrayToMatrix(new Float64Array([1, 2, 3, 4]), 2, 2);
      const C = matSub(A, B);
      expect(matGet(C, 0, 0)).toBe(4);
      expect(matGet(C, 1, 1)).toBe(4);
    });
  });

  describe('matScale', () => {
    it('scales matrix by scalar', () => {
      const A = arrayToMatrix(new Float64Array([1, 2, 3, 4]), 2, 2);
      const B = matScale(A, 3);
      expect(matGet(B, 0, 0)).toBe(3);
      expect(matGet(B, 1, 1)).toBe(12);
    });
  });

  describe('matDiag', () => {
    it('creates diagonal matrix', () => {
      const D = matDiag(new Float64Array([2, 3, 4]));
      expect(D.rows).toBe(3);
      expect(D.cols).toBe(3);
      expect(matGet(D, 0, 0)).toBe(2);
      expect(matGet(D, 1, 1)).toBe(3);
      expect(matGet(D, 2, 2)).toBe(4);
      expect(matGet(D, 0, 1)).toBe(0);
    });
  });

  describe('matInvert', () => {
    it('inverts identity', () => {
      const I = matIdentity(3);
      const inv = matInvert(I);
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          expect(matGet(inv, i, j)).toBeCloseTo(i === j ? 1 : 0, 10);
        }
      }
    });

    it('inverts 2x2 matrix', () => {
      const A = arrayToMatrix(new Float64Array([4, 7, 2, 6]), 2, 2);
      const inv = matInvert(A);
      // A * A^-1 = I
      const prod = matMul(A, inv);
      expect(matGet(prod, 0, 0)).toBeCloseTo(1, 8);
      expect(matGet(prod, 0, 1)).toBeCloseTo(0, 8);
      expect(matGet(prod, 1, 0)).toBeCloseTo(0, 8);
      expect(matGet(prod, 1, 1)).toBeCloseTo(1, 8);
    });

    it('throws for singular matrix', () => {
      const A = arrayToMatrix(new Float64Array([1, 2, 2, 4]), 2, 2);
      expect(() => matInvert(A)).toThrow();
    });
  });
});

describe('PRNG', () => {
  it('produces values in [0, 1)', () => {
    const rng = createPRNG(42);
    for (let i = 0; i < 100; i++) {
      const v = rng();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('is deterministic with same seed', () => {
    const rng1 = createPRNG(123);
    const rng2 = createPRNG(123);
    for (let i = 0; i < 50; i++) {
      expect(rng1()).toBe(rng2());
    }
  });

  it('different seeds produce different sequences', () => {
    const rng1 = createPRNG(1);
    const rng2 = createPRNG(2);
    let same = 0;
    for (let i = 0; i < 50; i++) {
      if (rng1() === rng2()) same++;
    }
    expect(same).toBeLessThan(5); // extremely unlikely to be the same
  });
});
