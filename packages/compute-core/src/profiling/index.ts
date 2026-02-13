// ---------------------------------------------------------------------------
// @omni-twin/compute-core — HPC-10: Profiling Toolkit (barrel)
// ---------------------------------------------------------------------------

export {
  createTimer,
  timerStart,
  timerStop,
  timerLap,
  timerReset,
  timerGetLaps,
} from './timer.js';
export type { TimerState } from './timer.js';

export {
  computeProfileStatistics,
  computeMean,
  computeMedian,
  computePercentile,
  computeStdDev,
  isOutlier,
  runningStatistics,
  finalizeRunningStats,
} from './statistics.js';

export {
  detectAntipatterns,
  checkExcessiveReadbacks,
  checkSmallDispatches,
  checkPerFramePipelineCreation,
  checkHotLoopAllocation,
  checkFrequentPostMessage,
  checkLargeClones,
  severityScore,
} from './antipatterns.js';
