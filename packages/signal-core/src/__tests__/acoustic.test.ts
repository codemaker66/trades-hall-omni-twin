// ---------------------------------------------------------------------------
// SP-7: Acoustic Simulation Tests
// ---------------------------------------------------------------------------
import { describe, it, expect } from 'vitest';
import { getMaterial, weightedAbsorption, OCTAVE_BANDS, MATERIAL_DB } from '../acoustic/material-database.js';
import { calculateRT60, furnitureAcousticImpact } from '../acoustic/rt60.js';
import { estimateSTI, estimateSTIFromRT60 } from '../acoustic/sti.js';
import { generateShoeboxRIR, convolveWithRIR } from '../acoustic/impulse-response.js';
import type { RoomGeometry } from '../types.js';

// Helper: build a simple shoebox RoomGeometry from dimensions
function makeRoom(
  length: number,
  width: number,
  height: number,
  floorMaterial: string,
  ceilingMaterial: string,
  wallMaterial: string,
): RoomGeometry {
  return {
    dimensions: [length, width, height],
    volume: length * width * height,
    surfaces: [
      { area: length * width, material: getMaterial(floorMaterial) },   // floor
      { area: length * width, material: getMaterial(ceilingMaterial) },  // ceiling
      { area: 2 * (length + width) * height, material: getMaterial(wallMaterial) }, // walls
    ],
  };
}

describe('SP-7: Acoustic Simulation', () => {
  describe('Material Database', () => {
    it('contains known materials', () => {
      const concrete = getMaterial('concrete');
      expect(concrete).toBeDefined();
      expect(concrete.absorption.length).toBe(6);
    });

    it('returns concrete fallback for unknown material', () => {
      // getMaterial falls back to concrete for unknown keys
      const result = getMaterial('unobtanium');
      expect(result).toBeDefined();
      expect(result.name).toBe('Concrete (unpainted)');
    });

    it('computes weighted absorption for a single band', () => {
      const surfaces = [
        { material: getMaterial('concrete'), area: 100 },
        { material: getMaterial('carpet-thin'), area: 50 },
      ];
      // weightedAbsorption returns a single number for a specific band index
      for (let band = 0; band < 6; band++) {
        const result = weightedAbsorption(surfaces, band);
        expect(result).toBeGreaterThan(0);
        expect(result).toBeLessThan(1);
      }
    });

    it('has 6 octave bands', () => {
      expect(OCTAVE_BANDS.length).toBe(6);
    });

    it('database has at least 10 materials', () => {
      expect(Object.keys(MATERIAL_DB).length).toBeGreaterThanOrEqual(10);
    });
  });

  describe('RT60 Calculation', () => {
    it('Sabine RT60 for typical room', () => {
      const room = makeRoom(10, 8, 3, 'concrete', 'concrete', 'plaster');
      const result = calculateRT60(room, 'sabine');
      expect(result.rt60.length).toBe(6);
      expect(result.formula).toBe('sabine');
      for (const rt of result.rt60) {
        expect(rt).toBeGreaterThan(0);
        expect(rt).toBeLessThan(20); // Reasonable RT60 range
      }
    });

    it('Eyring RT60 <= Sabine RT60', () => {
      const room = makeRoom(10, 8, 3, 'concrete', 'concrete', 'plaster');
      const sabine = calculateRT60(room, 'sabine');
      const eyring = calculateRT60(room, 'eyring');
      for (let i = 0; i < 6; i++) {
        expect(eyring.rt60[i]!).toBeLessThanOrEqual(sabine.rt60[i]! + 0.01);
      }
    });

    it('Fitzroy RT60 computed', () => {
      const room = makeRoom(10, 8, 3, 'concrete', 'carpet-thin', 'plaster');
      const result = calculateRT60(room, 'fitzroy');
      expect(result.rt60.length).toBe(6);
      expect(result.formula).toBe('fitzroy');
      for (const rt of result.rt60) {
        expect(rt).toBeGreaterThan(0);
      }
    });

    it('provides mid-frequency RT60 average', () => {
      const room = makeRoom(10, 8, 3, 'concrete', 'concrete', 'plaster');
      const result = calculateRT60(room);
      // rt60Mid is the average of 500Hz and 1kHz bands (indices 2 and 3)
      const expected = (result.rt60[2] + result.rt60[3]) / 2;
      expect(result.rt60Mid).toBeCloseTo(expected, 10);
    });
  });

  describe('Furniture acoustic impact', () => {
    it('reduces RT60 with absorptive furniture', () => {
      const room = makeRoom(10, 8, 3, 'concrete', 'concrete', 'plaster');
      const furnitureSurfaces: RoomGeometry['surfaces'] = [
        { area: 25, material: getMaterial('occupied-fabric-seat') },  // 50 chairs ~0.5m^2 each
        { area: 10, material: getMaterial('table-wood') },            // tables
        { area: 20, material: getMaterial('curtain-heavy') },         // curtains
      ];
      const impact = furnitureAcousticImpact(room, furnitureSurfaces);
      // Adding absorptive furniture should reduce RT60 (negative delta)
      // The impact.rt60 contains delta values: new - base
      for (let i = 0; i < 6; i++) {
        expect(impact.rt60[i]).toBeLessThanOrEqual(0.01);
      }
    });
  });

  describe('STI', () => {
    it('estimates STI from RT60 mid-frequency value', () => {
      // estimateSTIFromRT60 takes a single rt60Mid number
      const result = estimateSTIFromRT60(1.0);
      expect(result.sti).toBeGreaterThan(0);
      expect(result.sti).toBeLessThanOrEqual(1);
      expect(result.rating).toBeDefined();
      expect(result.mtf).toBeDefined();
    });

    it('lower RT60 gives better STI', () => {
      const longReverb = estimateSTIFromRT60(3.0);
      const shortReverb = estimateSTIFromRT60(0.5);
      expect(shortReverb.sti).toBeGreaterThan(longReverb.sti);
    });

    it('assigns correct STI ratings', () => {
      // Very short RT60 should yield excellent STI
      const excellent = estimateSTIFromRT60(0.1);
      expect(excellent.rating).toBe('excellent');

      // Very long RT60 should yield poor/bad STI
      const poor = estimateSTIFromRT60(5.0);
      expect(['poor', 'bad']).toContain(poor.rating);
    });

    it('estimates STI from impulse response', () => {
      const sr = 8000;
      const n = sr; // 1 second
      const samples = new Float64Array(n);
      samples[0] = 1;
      for (let i = 1; i < n; i++) samples[i] = Math.exp(-5 * i / sr) * 0.005;
      const result = estimateSTI({ samples, sampleRate: sr, duration: n / sr });
      expect(result.sti).toBeGreaterThan(0);
      expect(result.sti).toBeLessThanOrEqual(1);
      expect(result.rating).toBeDefined();
      expect(result.mtf).toBeDefined();
    });
  });

  describe('Shoebox RIR', () => {
    it('generates impulse response', () => {
      const room = makeRoom(6, 4, 3, 'plaster', 'plaster', 'plaster');
      const rir = generateShoeboxRIR(
        [6, 4, 3],          // dimensions
        [2, 2, 1.5],        // source
        [4, 2, 1.5],        // receiver
        room.surfaces,       // surfaces
        8000,                // sampleRate
        2,                   // maxOrder
      );
      expect(rir.samples.length).toBeGreaterThan(0);
      expect(rir.sampleRate).toBe(8000);
      expect(rir.duration).toBeGreaterThan(0);
      // Direct path should have highest energy
      let maxIdx = 0;
      let maxVal = 0;
      for (let i = 0; i < rir.samples.length; i++) {
        if (Math.abs(rir.samples[i]!) > maxVal) {
          maxVal = Math.abs(rir.samples[i]!);
          maxIdx = i;
        }
      }
      expect(maxIdx).toBeLessThan(rir.samples.length / 2); // Direct path is early
    });

    it('convolves audio with RIR', () => {
      const room = makeRoom(6, 4, 3, 'plaster', 'plaster', 'plaster');
      const rir = generateShoeboxRIR(
        [6, 4, 3],
        [2, 2, 1.5],
        [4, 2, 1.5],
        room.surfaces,
        8000,
        1,
      );
      const input = new Float64Array(1000);
      for (let i = 0; i < 1000; i++) input[i] = Math.sin(2 * Math.PI * 440 * i / 8000);
      const output = convolveWithRIR(input, rir);
      expect(output.length).toBeGreaterThanOrEqual(input.length);
    });
  });
});
