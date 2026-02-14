import { describe, it, expect } from 'vitest';
import {
  createTimer,
  timerStart,
  timerStop,
  timerLap,
  timerReset,
  timerGetLaps,
  computeProfileStatistics,
  computeMean,
  computeMedian,
  computePercentile,
  computeStdDev,
  isOutlier,
  runningStatistics,
  finalizeRunningStats,
  detectAntipatterns,
  checkExcessiveReadbacks,
  checkSmallDispatches,
  checkPerFramePipelineCreation,
  checkHotLoopAllocation,
  checkFrequentPostMessage,
  checkLargeClones,
  severityScore,
} from '../profiling/index.js';

import type { ProfileSample, AntipatternReport } from '../types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeSample(
  overrides?: Partial<ProfileSample>,
): ProfileSample {
  return {
    name: 'test',
    startMs: 0,
    durationMs: 10,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Timer
// ---------------------------------------------------------------------------

describe('Timer', () => {
  it('createTimer returns a stopped timer with the given name', () => {
    const t = createTimer('my-timer');
    expect(t.name).toBe('my-timer');
    expect(t.running).toBe(false);
    expect(t.laps).toEqual([]);
    expect(t.startMs).toBe(0);
  });

  it('start/stop measures a non-negative duration', () => {
    const t = createTimer('duration');
    timerStart(t);
    expect(t.running).toBe(true);
    const elapsed = timerStop(t);
    expect(elapsed).toBeGreaterThanOrEqual(0);
    expect(t.running).toBe(false);
  });

  it('stop on a non-running timer returns 0', () => {
    const t = createTimer('idle');
    expect(timerStop(t)).toBe(0);
  });

  it('laps are recorded while the timer is running', () => {
    const t = createTimer('laps');
    timerStart(t);
    const lap1 = timerLap(t, 'first');
    const lap2 = timerLap(t, 'second');
    timerStop(t);

    expect(lap1).toBeGreaterThanOrEqual(0);
    expect(lap2).toBeGreaterThanOrEqual(lap1);

    const laps = timerGetLaps(t);
    expect(laps).toHaveLength(2);
    expect(laps[0]!.name).toBe('first');
    expect(laps[1]!.name).toBe('second');
  });

  it('lap on a stopped timer returns 0 without recording', () => {
    const t = createTimer('stopped');
    const result = timerLap(t, 'nope');
    expect(result).toBe(0);
    expect(timerGetLaps(t)).toHaveLength(0);
  });

  it('reset clears laps and stops the timer', () => {
    const t = createTimer('reset');
    timerStart(t);
    timerLap(t, 'a');
    timerReset(t);
    expect(t.running).toBe(false);
    expect(t.startMs).toBe(0);
    expect(timerGetLaps(t)).toHaveLength(0);
  });

  it('timerGetLaps returns a copy, not a reference', () => {
    const t = createTimer('copy');
    timerStart(t);
    timerLap(t, 'x');
    const laps = timerGetLaps(t);
    laps.push({ name: 'fake', ms: 999 });
    expect(timerGetLaps(t)).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

describe('Statistics', () => {
  it('computeMean of [1,2,3,4,5] equals 3', () => {
    const arr = new Float64Array([1, 2, 3, 4, 5]);
    expect(computeMean(arr)).toBe(3);
  });

  it('computeMean of empty array returns 0', () => {
    expect(computeMean(new Float64Array(0))).toBe(0);
  });

  it('computeMedian of [1,2,3] equals 2', () => {
    const arr = new Float64Array([1, 2, 3]);
    expect(computeMedian(arr)).toBe(2);
  });

  it('computeMedian of even-length array averages middle two', () => {
    const arr = new Float64Array([1, 2, 3, 4]);
    expect(computeMedian(arr)).toBe(2.5);
  });

  it('computePercentile p95 returns correct value', () => {
    // 20 values: 1..20
    const values = new Float64Array(20);
    for (let i = 0; i < 20; i++) {
      values[i] = i + 1;
    }
    const p95 = computePercentile(values, 0.95);
    // nearest-rank: ceil(0.95 * 20) = 19, index 18 -> value 19
    expect(p95).toBe(19);
  });

  it('computeStdDev of identical values returns 0', () => {
    const arr = new Float64Array([5, 5, 5, 5]);
    expect(computeStdDev(arr)).toBe(0);
  });

  it('computeStdDev of varying values is positive', () => {
    const arr = new Float64Array([1, 2, 3, 4, 5]);
    const sd = computeStdDev(arr);
    expect(sd).toBeGreaterThan(0);
    // Population std dev of [1,2,3,4,5] = sqrt(2) ~ 1.414
    expect(sd).toBeCloseTo(Math.sqrt(2), 10);
  });

  it('isOutlier detects values far from the mean', () => {
    const mean = 10;
    const stdDev = 2;
    // 3-sigma: threshold = 6, so value 17 (distance 7) is an outlier
    expect(isOutlier(17, mean, stdDev, 3)).toBe(true);
    // value 12 (distance 2) is not
    expect(isOutlier(12, mean, stdDev, 3)).toBe(false);
  });

  it('isOutlier with stdDev=0 returns true for any value != mean', () => {
    expect(isOutlier(5, 5, 0)).toBe(false);
    expect(isOutlier(5.001, 5, 0)).toBe(true);
  });

  it('runningStatistics + finalizeRunningStats matches batch computation', () => {
    const values = [4, 7, 13, 16];
    let acc = { count: 0, mean: 0, m2: 0 };
    for (const v of values) {
      acc = runningStatistics(acc, v);
    }
    const result = finalizeRunningStats(acc);
    const arr = new Float64Array(values);
    expect(result.mean).toBe(computeMean(arr));
    expect(result.stdDev).toBeCloseTo(computeStdDev(arr), 10);
  });

  it('computeProfileStatistics returns all expected fields', () => {
    const samples: ProfileSample[] = [
      makeSample({ name: 'op', durationMs: 5 }),
      makeSample({ name: 'op', durationMs: 10 }),
      makeSample({ name: 'op', durationMs: 15 }),
      makeSample({ name: 'op', durationMs: 20 }),
      makeSample({ name: 'op', durationMs: 25 }),
    ];
    const stats = computeProfileStatistics(samples);
    expect(stats.name).toBe('op');
    expect(stats.count).toBe(5);
    expect(stats.meanMs).toBe(15);
    expect(stats.medianMs).toBe(15);
    expect(stats.minMs).toBe(5);
    expect(stats.maxMs).toBe(25);
    expect(stats.stdDevMs).toBeGreaterThan(0);
    expect(stats.p95Ms).toBeGreaterThanOrEqual(stats.medianMs);
    expect(stats.p99Ms).toBeGreaterThanOrEqual(stats.p95Ms);
  });

  it('computeProfileStatistics of empty array returns zeros', () => {
    const stats = computeProfileStatistics([]);
    expect(stats.name).toBe('');
    expect(stats.count).toBe(0);
    expect(stats.meanMs).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Antipatterns
// ---------------------------------------------------------------------------

describe('Antipatterns', () => {
  it('checkExcessiveReadbacks detects >2 readbacks in a 16ms window', () => {
    const samples: ProfileSample[] = [
      makeSample({ name: 'gpu_readback', startMs: 0, durationMs: 1 }),
      makeSample({ name: 'gpu_readback', startMs: 5, durationMs: 1 }),
      makeSample({ name: 'gpu_readback', startMs: 10, durationMs: 1 }),
    ];
    const report = checkExcessiveReadbacks(samples);
    expect(report).not.toBeNull();
    expect(report!.name).toBe('excessive_gpu_readback');
    expect(report!.severity).toBe('critical');
  });

  it('checkExcessiveReadbacks returns null when count <= threshold', () => {
    const samples: ProfileSample[] = [
      makeSample({ name: 'gpu_readback', startMs: 0 }),
      makeSample({ name: 'gpu_readback', startMs: 100 }),
    ];
    expect(checkExcessiveReadbacks(samples)).toBeNull();
  });

  it('checkSmallDispatches flags dispatches with few elements', () => {
    const samples: ProfileSample[] = [
      makeSample({ name: 'dispatch_compute', metadata: { elements: 100 } }),
      makeSample({ name: 'dispatch_compute', metadata: { elements: 500 } }),
    ];
    const report = checkSmallDispatches(samples);
    expect(report).not.toBeNull();
    expect(report!.name).toBe('small_dispatch');
    expect(report!.metric).toBe(2);
  });

  it('checkSmallDispatches returns null when all dispatches are large enough', () => {
    const samples: ProfileSample[] = [
      makeSample({ name: 'dispatch_compute', metadata: { elements: 2048 } }),
    ];
    expect(checkSmallDispatches(samples)).toBeNull();
  });

  it('checkPerFramePipelineCreation detects rapid pipeline creation', () => {
    const samples: ProfileSample[] = [
      makeSample({ name: 'createPipeline', startMs: 0 }),
      makeSample({ name: 'createPipeline', startMs: 16 }),
      makeSample({ name: 'createPipeline', startMs: 32 }),
    ];
    const report = checkPerFramePipelineCreation(samples);
    expect(report).not.toBeNull();
    expect(report!.name).toBe('per_frame_pipeline');
    expect(report!.severity).toBe('critical');
  });

  it('checkHotLoopAllocation detects rapid allocations', () => {
    const samples: ProfileSample[] = [];
    // 5 allocations 1ms apart -> burst of 5 >= 3
    for (let i = 0; i < 5; i++) {
      samples.push(makeSample({ name: 'alloc_buffer', startMs: i }));
    }
    const report = checkHotLoopAllocation(samples);
    expect(report).not.toBeNull();
    expect(report!.name).toBe('hot_loop_allocation');
    expect(report!.severity).toBe('warning');
  });

  it('checkFrequentPostMessage detects > 60 messages per second', () => {
    const samples: ProfileSample[] = [];
    for (let i = 0; i < 100; i++) {
      samples.push(makeSample({ name: 'postMessage', startMs: i * 5 }));
    }
    const report = checkFrequentPostMessage(samples);
    expect(report).not.toBeNull();
    expect(report!.name).toBe('frequent_postmessage');
  });

  it('checkLargeClones detects large structured clones', () => {
    const samples: ProfileSample[] = [
      makeSample({
        name: 'structuredClone',
        metadata: { bytes: 2 * 1024 * 1024 },
      }),
    ];
    const report = checkLargeClones(samples);
    expect(report).not.toBeNull();
    expect(report!.name).toBe('structured_clone_large');
    expect(report!.severity).toBe('warning');
  });

  it('detectAntipatterns returns an array of detected issues', () => {
    const samples: ProfileSample[] = [
      makeSample({ name: 'gpu_readback', startMs: 0, durationMs: 1 }),
      makeSample({ name: 'gpu_readback', startMs: 5, durationMs: 1 }),
      makeSample({ name: 'gpu_readback', startMs: 10, durationMs: 1 }),
      makeSample({ name: 'dispatch_compute', metadata: { elements: 50 } }),
    ];
    const reports = detectAntipatterns(samples);
    expect(Array.isArray(reports)).toBe(true);
    expect(reports.length).toBeGreaterThanOrEqual(2);
    const names = reports.map((r) => r.name);
    expect(names).toContain('excessive_gpu_readback');
    expect(names).toContain('small_dispatch');
  });

  it('severityScore: critical > warning > info', () => {
    const criticalReports: AntipatternReport[] = [
      { name: 'x', severity: 'critical', description: '', metric: 0, threshold: 0 },
    ];
    const warningReports: AntipatternReport[] = [
      { name: 'x', severity: 'warning', description: '', metric: 0, threshold: 0 },
    ];
    const infoReports: AntipatternReport[] = [
      { name: 'x', severity: 'info', description: '', metric: 0, threshold: 0 },
    ];
    const critScore = severityScore(criticalReports);
    const warnScore = severityScore(warningReports);
    const infoScore = severityScore(infoReports);
    expect(critScore).toBeGreaterThan(warnScore);
    expect(warnScore).toBeGreaterThan(infoScore);
    expect(infoScore).toBeGreaterThan(0);
  });

  it('severityScore clamps to 100', () => {
    const reports: AntipatternReport[] = Array.from({ length: 10 }, () => ({
      name: 'x',
      severity: 'critical' as const,
      description: '',
      metric: 0,
      threshold: 0,
    }));
    // 10 critical * 50 = 500, clamped to 100
    expect(severityScore(reports)).toBe(100);
  });

  it('severityScore of empty array is 0', () => {
    expect(severityScore([])).toBe(0);
  });
});
