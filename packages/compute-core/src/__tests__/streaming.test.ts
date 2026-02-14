import { describe, it, expect } from 'vitest';

import {
  // Bloom filter
  createBloomFilter,
  bloomAdd,
  bloomTest,
  bloomFalsePositiveRate,
  bloomMerge,
  bloomReset,
  // Count-Min Sketch
  createCountMinSketch,
  cmsAdd,
  cmsQuery,
  cmsMerge,
  cmsReset,
  cmsEstimateError,
  // HyperLogLog
  createHyperLogLog,
  hllAdd,
  hllCount,
  hllMerge,
  hllReset,
  hllError,
  // T-Digest
  createTDigest,
  tdigestAdd,
  tdigestQuantile,
  tdigestCDF,
  tdigestMerge,
  tdigestMean,
  tdigestMin,
  tdigestMax,
  tdigestCount,
} from '../streaming/index.js';

// ===================================================================
// Bloom Filter
// ===================================================================

describe('Bloom filter', () => {
  it('returns true for items that were added', () => {
    const bf = createBloomFilter({ expectedItems: 100, falsePositiveRate: 0.01 });
    bloomAdd(bf, 'hello');
    bloomAdd(bf, 'world');
    expect(bloomTest(bf, 'hello')).toBe(true);
    expect(bloomTest(bf, 'world')).toBe(true);
  });

  it('returns false for items that were never added (no false negatives)', () => {
    const bf = createBloomFilter({ expectedItems: 1000, falsePositiveRate: 0.001 });
    for (let i = 0; i < 100; i++) {
      bloomAdd(bf, `item-${i}`);
    }
    // Test 500 items that were never added. With FP rate 0.001 and a well-sized
    // filter, false positives among 500 tests should be very rare.
    let falsePositives = 0;
    for (let i = 1000; i < 1500; i++) {
      if (bloomTest(bf, `item-${i}`)) falsePositives++;
    }
    // Allow up to 5% false positive rate among the test set (very generous)
    expect(falsePositives).toBeLessThan(25);
  });

  it('has lower FP rate with a larger filter', () => {
    const small = createBloomFilter({ expectedItems: 100, falsePositiveRate: 0.1 });
    const large = createBloomFilter({ expectedItems: 100, falsePositiveRate: 0.001 });
    // Larger filter should have more bits
    expect(large.size).toBeGreaterThan(small.size);
  });

  it('merge combines two filters so both sets test positive', () => {
    const config = { expectedItems: 100, falsePositiveRate: 0.01 };
    const a = createBloomFilter(config);
    const b = createBloomFilter(config);

    bloomAdd(a, 'alpha');
    bloomAdd(b, 'beta');

    const merged = bloomMerge(a, b);
    expect(bloomTest(merged, 'alpha')).toBe(true);
    expect(bloomTest(merged, 'beta')).toBe(true);
  });

  it('reset clears the filter so nothing tests positive', () => {
    const bf = createBloomFilter({ expectedItems: 100, falsePositiveRate: 0.01 });
    bloomAdd(bf, 'test');
    expect(bloomTest(bf, 'test')).toBe(true);

    bloomReset(bf);
    expect(bloomTest(bf, 'test')).toBe(false);
    expect(bf.count).toBe(0);
  });

  it('bloomFalsePositiveRate increases as items are added', () => {
    const bf = createBloomFilter({ expectedItems: 100, falsePositiveRate: 0.01 });
    const rateEmpty = bloomFalsePositiveRate(bf);

    for (let i = 0; i < 50; i++) bloomAdd(bf, `item-${i}`);
    const rateHalf = bloomFalsePositiveRate(bf);

    for (let i = 50; i < 100; i++) bloomAdd(bf, `item-${i}`);
    const rateFull = bloomFalsePositiveRate(bf);

    expect(rateEmpty).toBe(0);
    expect(rateHalf).toBeGreaterThan(rateEmpty);
    expect(rateFull).toBeGreaterThan(rateHalf);
  });

  it('merge throws when filter configurations differ', () => {
    const a = createBloomFilter({ expectedItems: 100, falsePositiveRate: 0.01 });
    const b = createBloomFilter({ expectedItems: 1000, falsePositiveRate: 0.01 });
    expect(() => bloomMerge(a, b)).toThrow();
  });

  it('tracks count as items are added', () => {
    const bf = createBloomFilter({ expectedItems: 100, falsePositiveRate: 0.01 });
    expect(bf.count).toBe(0);
    bloomAdd(bf, 'a');
    bloomAdd(bf, 'b');
    bloomAdd(bf, 'c');
    expect(bf.count).toBe(3);
  });
});

// ===================================================================
// Count-Min Sketch
// ===================================================================

describe('Count-Min Sketch', () => {
  it('returns at least the true count for added items', () => {
    const cms = createCountMinSketch({ width: 1000, depth: 5 });
    cmsAdd(cms, 'apple', 10);
    cmsAdd(cms, 'banana', 3);

    expect(cmsQuery(cms, 'apple')).toBeGreaterThanOrEqual(10);
    expect(cmsQuery(cms, 'banana')).toBeGreaterThanOrEqual(3);
  });

  it('returns the minimum estimate property (upper bound)', () => {
    const cms = createCountMinSketch({ width: 100, depth: 3 });
    cmsAdd(cms, 'x', 5);
    // The estimate is always >= true count
    expect(cmsQuery(cms, 'x')).toBeGreaterThanOrEqual(5);
  });

  it('returns 0 for items never added (or very small overcount)', () => {
    const cms = createCountMinSketch({ width: 10000, depth: 7 });
    cmsAdd(cms, 'exists', 42);
    // An item never added should have count close to 0
    const estimate = cmsQuery(cms, 'never_added');
    expect(estimate).toBeGreaterThanOrEqual(0);
    // With width=10000 and only 42 total insertions, collision should be rare
    expect(estimate).toBeLessThan(10);
  });

  it('error bound matches totalCount / width', () => {
    const cms = createCountMinSketch({ width: 500, depth: 5 });
    cmsAdd(cms, 'a', 100);
    cmsAdd(cms, 'b', 200);
    // totalCount = 300, error = 300 / 500 = 0.6
    expect(cmsEstimateError(cms)).toBeCloseTo(0.6, 5);
  });

  it('merge sums counts from both sketches', () => {
    const config = { width: 1000, depth: 5 };
    const a = createCountMinSketch(config);
    const b = createCountMinSketch(config);

    cmsAdd(a, 'item', 10);
    cmsAdd(b, 'item', 20);

    const merged = cmsMerge(a, b);
    expect(cmsQuery(merged, 'item')).toBeGreaterThanOrEqual(30);
    expect(merged.totalCount).toBe(30);
  });

  it('reset clears all counters', () => {
    const cms = createCountMinSketch({ width: 100, depth: 3 });
    cmsAdd(cms, 'data', 50);
    expect(cmsQuery(cms, 'data')).toBeGreaterThanOrEqual(50);

    cmsReset(cms);
    expect(cmsQuery(cms, 'data')).toBe(0);
    expect(cms.totalCount).toBe(0);
  });

  it('handles multiple distinct items without excessive collision', () => {
    const cms = createCountMinSketch({ width: 10000, depth: 5 });
    for (let i = 0; i < 100; i++) {
      cmsAdd(cms, `item-${i}`, 1);
    }
    // Each item has true count 1. With wide sketch, estimates should be close.
    let maxEstimate = 0;
    for (let i = 0; i < 100; i++) {
      const est = cmsQuery(cms, `item-${i}`);
      if (est > maxEstimate) maxEstimate = est;
    }
    // Even with collisions, no item should have an estimate > 10 with width 10000
    expect(maxEstimate).toBeLessThanOrEqual(10);
  });

  it('merge throws for mismatched dimensions', () => {
    const a = createCountMinSketch({ width: 100, depth: 3 });
    const b = createCountMinSketch({ width: 200, depth: 3 });
    expect(() => cmsMerge(a, b)).toThrow();
  });
});

// ===================================================================
// HyperLogLog
// ===================================================================

describe('HyperLogLog', () => {
  it('count increases as distinct items are added', () => {
    const hll = createHyperLogLog(10);
    hllAdd(hll, 'a');
    const count1 = hllCount(hll);
    hllAdd(hll, 'b');
    const count2 = hllCount(hll);
    hllAdd(hll, 'c');
    const count3 = hllCount(hll);
    expect(count2).toBeGreaterThanOrEqual(count1);
    expect(count3).toBeGreaterThanOrEqual(count2);
  });

  it('estimates approximately correct cardinality for 1000 items (within 20%)', () => {
    const hll = createHyperLogLog(14);
    for (let i = 0; i < 1000; i++) {
      hllAdd(hll, `unique-item-${i}`);
    }
    const count = hllCount(hll);
    expect(count).toBeGreaterThan(800);
    expect(count).toBeLessThan(1200);
  });

  it('adding duplicate items does not increase cardinality significantly', () => {
    const hll = createHyperLogLog(14);
    for (let i = 0; i < 100; i++) {
      hllAdd(hll, `item-${i}`);
    }
    const countBefore = hllCount(hll);

    // Add duplicates
    for (let i = 0; i < 100; i++) {
      hllAdd(hll, `item-${i}`);
    }
    const countAfter = hllCount(hll);

    // Cardinality should not change (or change very little)
    expect(countAfter).toBeCloseTo(countBefore, -1);
  });

  it('merge gives union cardinality', () => {
    const a = createHyperLogLog(14);
    const b = createHyperLogLog(14);

    // Add 500 unique items to each, with 200 overlap
    for (let i = 0; i < 500; i++) hllAdd(a, `a-item-${i}`);
    for (let i = 0; i < 200; i++) hllAdd(b, `a-item-${i}`); // overlap
    for (let i = 0; i < 300; i++) hllAdd(b, `b-only-${i}`);

    const merged = hllMerge(a, b);
    const count = hllCount(merged);
    // Union should be ~800 (500 + 300 unique from b)
    expect(count).toBeGreaterThan(600);
    expect(count).toBeLessThan(1000);
  });

  it('error formula is 1.04 / sqrt(m)', () => {
    const hll = createHyperLogLog(10); // m = 1024
    const expectedError = 1.04 / Math.sqrt(1024);
    expect(hllError(hll)).toBeCloseTo(expectedError, 5);
  });

  it('reset zeros the cardinality', () => {
    const hll = createHyperLogLog(10);
    for (let i = 0; i < 50; i++) hllAdd(hll, `item-${i}`);
    expect(hllCount(hll)).toBeGreaterThan(0);

    hllReset(hll);
    expect(hllCount(hll)).toBe(0);
  });

  it('higher precision gives lower error', () => {
    const lowP = createHyperLogLog(4);
    const highP = createHyperLogLog(14);
    expect(hllError(highP)).toBeLessThan(hllError(lowP));
  });

  it('merge throws for different precisions', () => {
    const a = createHyperLogLog(10);
    const b = createHyperLogLog(14);
    expect(() => hllMerge(a, b)).toThrow();
  });
});

// ===================================================================
// T-Digest
// ===================================================================

describe('T-Digest', () => {
  /**
   * Populate a t-digest with uniformly distributed values in [0, 1).
   * Uses a simple deterministic sequence.
   */
  function buildUniformDigest(n: number) {
    const td = createTDigest({ compression: 100 });
    for (let i = 0; i < n; i++) {
      tdigestAdd(td, i / n);
    }
    return td;
  }

  it('median of uniform data is approximately 0.5', () => {
    const td = buildUniformDigest(10000);
    const median = tdigestQuantile(td, 0.5);
    expect(median).toBeGreaterThan(0.4);
    expect(median).toBeLessThan(0.6);
  });

  it('p95 of uniform data is approximately 0.95', () => {
    const td = buildUniformDigest(10000);
    const p95 = tdigestQuantile(td, 0.95);
    expect(p95).toBeGreaterThan(0.9);
    expect(p95).toBeLessThan(1.0);
  });

  it('mean is approximately correct for uniform data', () => {
    const td = buildUniformDigest(10000);
    const mean = tdigestMean(td);
    // Mean of [0, 1/n, 2/n, ..., (n-1)/n] is approximately 0.5
    expect(mean).toBeGreaterThan(0.45);
    expect(mean).toBeLessThan(0.55);
  });

  it('tracks min and max correctly', () => {
    const td = createTDigest();
    tdigestAdd(td, 5);
    tdigestAdd(td, 1);
    tdigestAdd(td, 10);
    tdigestAdd(td, 3);
    expect(tdigestMin(td)).toBe(1);
    expect(tdigestMax(td)).toBe(10);
  });

  it('tdigestCount tracks total values added', () => {
    const td = createTDigest();
    tdigestAdd(td, 1.0);
    tdigestAdd(td, 2.0);
    tdigestAdd(td, 3.0, 5);
    expect(tdigestCount(td)).toBe(7);
  });

  it('merge combines two digests', () => {
    const a = createTDigest({ compression: 100 });
    const b = createTDigest({ compression: 100 });

    // a: values 0-499, b: values 500-999
    for (let i = 0; i < 500; i++) tdigestAdd(a, i);
    for (let i = 500; i < 1000; i++) tdigestAdd(b, i);

    const merged = tdigestMerge(a, b);
    expect(tdigestCount(merged)).toBe(1000);
    expect(tdigestMin(merged)).toBe(0);
    expect(tdigestMax(merged)).toBe(999);

    // Median should be around 500
    const median = tdigestQuantile(merged, 0.5);
    expect(median).toBeGreaterThan(400);
    expect(median).toBeLessThan(600);
  });

  it('CDF at the median is approximately 0.5', () => {
    const td = buildUniformDigest(5000);
    const median = tdigestQuantile(td, 0.5);
    const cdfAtMedian = tdigestCDF(td, median);
    expect(cdfAtMedian).toBeGreaterThan(0.4);
    expect(cdfAtMedian).toBeLessThan(0.6);
  });

  it('CDF returns 0 at min and 1 at max', () => {
    const td = createTDigest();
    for (let i = 0; i < 100; i++) tdigestAdd(td, i);
    expect(tdigestCDF(td, tdigestMin(td))).toBe(0);
    expect(tdigestCDF(td, tdigestMax(td))).toBe(1);
  });

  it('quantile returns NaN for empty digest', () => {
    const td = createTDigest();
    expect(tdigestQuantile(td, 0.5)).toBeNaN();
  });

  it('quantile at 0 returns min and at 1 returns max', () => {
    const td = createTDigest();
    tdigestAdd(td, 10);
    tdigestAdd(td, 20);
    tdigestAdd(td, 30);
    expect(tdigestQuantile(td, 0)).toBe(tdigestMin(td));
    expect(tdigestQuantile(td, 1)).toBe(tdigestMax(td));
  });
});
