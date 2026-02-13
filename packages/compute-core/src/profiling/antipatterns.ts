// ---------------------------------------------------------------------------
// HPC-10: Profiling — Antipattern detection for HPC code
// ---------------------------------------------------------------------------
// Analyses profiling samples to detect common GPU, WASM, and worker
// antipatterns. Each check function returns a single AntipatternReport or
// null when no issue is detected. The aggregate severityScore gives a 0-100
// summary for dashboards.
// ---------------------------------------------------------------------------

import type { AntipatternReport, ProfileSample } from '../types.js';

// ---------------------------------------------------------------------------
// Constants / thresholds
// ---------------------------------------------------------------------------

const READBACK_THRESHOLD = 2; // max readbacks per frame (16 ms window)
const SMALL_DISPATCH_THRESHOLD = 1024; // min elements per dispatch
const PIPELINE_CREATION_INTERVAL_MS = 100; // if two pipeline creations are closer than this
const HOT_LOOP_ALLOC_INTERVAL_MS = 2; // allocation samples closer than this are suspicious
const POST_MESSAGE_PER_SECOND = 60; // max messages per second
const LARGE_CLONE_BYTES = 1024 * 1024; // 1 MB

// ---------------------------------------------------------------------------
// checkExcessiveReadbacks
// ---------------------------------------------------------------------------

/**
 * Detect excessive GPU -> CPU readbacks.
 *
 * Looks for samples whose name contains "readback" or "mapAsync" and checks
 * whether more than 2 occur within any 16 ms window.
 */
export function checkExcessiveReadbacks(
  samples: ProfileSample[],
): AntipatternReport | null {
  const readbacks: ProfileSample[] = [];
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i]!;
    const lower = s.name.toLowerCase();
    if (lower.includes('readback') || lower.includes('mapasync')) {
      readbacks.push(s);
    }
  }

  if (readbacks.length <= READBACK_THRESHOLD) return null;

  // Sort by startMs
  readbacks.sort((a, b) => a.startMs - b.startMs);

  // Sliding window of 16 ms
  let maxInWindow = 0;
  let windowStart = 0;
  for (let i = 0; i < readbacks.length; i++) {
    const cur = readbacks[i]!;
    while (windowStart < i && cur.startMs - readbacks[windowStart]!.startMs > 16) {
      windowStart++;
    }
    const count = i - windowStart + 1;
    if (count > maxInWindow) maxInWindow = count;
  }

  if (maxInWindow > READBACK_THRESHOLD) {
    return {
      name: 'excessive_gpu_readback',
      severity: 'critical',
      description: `Detected ${maxInWindow} GPU readbacks within a 16 ms window (limit: ${READBACK_THRESHOLD})`,
      metric: maxInWindow,
      threshold: READBACK_THRESHOLD,
    };
  }

  return null;
}

// ---------------------------------------------------------------------------
// checkSmallDispatches
// ---------------------------------------------------------------------------

/**
 * Detect GPU dispatches with too few elements.
 *
 * Looks for samples with name containing "dispatch" whose metadata.elements
 * is below 1024. Small dispatches underutilise GPU parallelism.
 */
export function checkSmallDispatches(
  samples: ProfileSample[],
): AntipatternReport | null {
  let smallCount = 0;
  let smallestElements = Infinity;

  for (let i = 0; i < samples.length; i++) {
    const s = samples[i]!;
    if (!s.name.toLowerCase().includes('dispatch')) continue;
    const elements = s.metadata?.['elements'];
    if (elements !== undefined && elements < SMALL_DISPATCH_THRESHOLD) {
      smallCount++;
      if (elements < smallestElements) smallestElements = elements;
    }
  }

  if (smallCount === 0) return null;

  return {
    name: 'small_dispatch',
    severity: smallCount > 5 ? 'warning' : 'info',
    description: `Found ${smallCount} GPU dispatch(es) with fewer than ${SMALL_DISPATCH_THRESHOLD} elements (smallest: ${smallestElements})`,
    metric: smallCount,
    threshold: SMALL_DISPATCH_THRESHOLD,
  };
}

// ---------------------------------------------------------------------------
// checkPerFramePipelineCreation
// ---------------------------------------------------------------------------

/**
 * Detect pipeline creation happening inside the render loop.
 *
 * If two or more pipeline creation samples occur within 100 ms of each other,
 * the pipeline is likely being recreated every frame instead of cached.
 */
export function checkPerFramePipelineCreation(
  samples: ProfileSample[],
): AntipatternReport | null {
  const pipelineSamples: ProfileSample[] = [];

  for (let i = 0; i < samples.length; i++) {
    const s = samples[i]!;
    const lower = s.name.toLowerCase();
    if (
      lower.includes('createpipeline') ||
      lower.includes('create_pipeline') ||
      lower.includes('pipeline_creation') ||
      lower.includes('pipelinecreation')
    ) {
      pipelineSamples.push(s);
    }
  }

  if (pipelineSamples.length < 2) return null;

  pipelineSamples.sort((a, b) => a.startMs - b.startMs);

  let closeCount = 0;
  for (let i = 1; i < pipelineSamples.length; i++) {
    const gap = pipelineSamples[i]!.startMs - pipelineSamples[i - 1]!.startMs;
    if (gap < PIPELINE_CREATION_INTERVAL_MS) {
      closeCount++;
    }
  }

  if (closeCount === 0) return null;

  return {
    name: 'per_frame_pipeline',
    severity: 'critical',
    description: `Pipeline created ${closeCount + 1} times within ${PIPELINE_CREATION_INTERVAL_MS} ms intervals — cache pipelines instead`,
    metric: closeCount + 1,
    threshold: 1,
  };
}

// ---------------------------------------------------------------------------
// checkHotLoopAllocation
// ---------------------------------------------------------------------------

/**
 * Detect allocation inside tight loops.
 *
 * Looks for samples with name containing "alloc", "new", or "create" that
 * occur in rapid succession (< 2 ms apart). Such patterns cause GC pressure
 * in hot paths.
 */
export function checkHotLoopAllocation(
  samples: ProfileSample[],
): AntipatternReport | null {
  const allocSamples: ProfileSample[] = [];

  for (let i = 0; i < samples.length; i++) {
    const s = samples[i]!;
    const lower = s.name.toLowerCase();
    if (
      lower.includes('alloc') ||
      lower.includes('new_') ||
      lower.includes('create_buffer') ||
      lower.includes('createbuffer')
    ) {
      allocSamples.push(s);
    }
  }

  if (allocSamples.length < 3) return null;

  allocSamples.sort((a, b) => a.startMs - b.startMs);

  let burstCount = 0;
  let currentBurst = 1;
  for (let i = 1; i < allocSamples.length; i++) {
    const gap = allocSamples[i]!.startMs - allocSamples[i - 1]!.startMs;
    if (gap < HOT_LOOP_ALLOC_INTERVAL_MS) {
      currentBurst++;
    } else {
      if (currentBurst >= 3) burstCount += currentBurst;
      currentBurst = 1;
    }
  }
  // Final burst
  if (currentBurst >= 3) burstCount += currentBurst;

  if (burstCount === 0) return null;

  return {
    name: 'hot_loop_allocation',
    severity: 'warning',
    description: `Detected ${burstCount} rapid allocations (< ${HOT_LOOP_ALLOC_INTERVAL_MS} ms apart) — pre-allocate buffers`,
    metric: burstCount,
    threshold: 3,
  };
}

// ---------------------------------------------------------------------------
// checkFrequentPostMessage
// ---------------------------------------------------------------------------

/**
 * Detect too many worker postMessage calls per second.
 *
 * Groups samples with name containing "postmessage" or "post_message" into
 * 1-second buckets and checks for any bucket exceeding 60 messages/s.
 */
export function checkFrequentPostMessage(
  samples: ProfileSample[],
): AntipatternReport | null {
  const msgSamples: ProfileSample[] = [];

  for (let i = 0; i < samples.length; i++) {
    const s = samples[i]!;
    const lower = s.name.toLowerCase();
    if (lower.includes('postmessage') || lower.includes('post_message')) {
      msgSamples.push(s);
    }
  }

  if (msgSamples.length === 0) return null;

  msgSamples.sort((a, b) => a.startMs - b.startMs);

  // Bucket into 1-second windows
  const buckets = new Map<number, number>();
  for (let i = 0; i < msgSamples.length; i++) {
    const bucket = Math.floor(msgSamples[i]!.startMs / 1000);
    buckets.set(bucket, (buckets.get(bucket) ?? 0) + 1);
  }

  let maxPerSecond = 0;
  for (const count of buckets.values()) {
    if (count > maxPerSecond) maxPerSecond = count;
  }

  if (maxPerSecond <= POST_MESSAGE_PER_SECOND) return null;

  return {
    name: 'frequent_postmessage',
    severity: maxPerSecond > 200 ? 'critical' : 'warning',
    description: `Peak ${maxPerSecond} postMessage calls/second (limit: ${POST_MESSAGE_PER_SECOND}) — batch messages`,
    metric: maxPerSecond,
    threshold: POST_MESSAGE_PER_SECOND,
  };
}

// ---------------------------------------------------------------------------
// checkLargeClones
// ---------------------------------------------------------------------------

/**
 * Detect structured clone operations transferring more than 1 MB.
 *
 * Looks for samples with name containing "clone" or "structuredclone" whose
 * metadata.bytes exceeds the threshold.
 */
export function checkLargeClones(
  samples: ProfileSample[],
): AntipatternReport | null {
  let largeCount = 0;
  let largestBytes = 0;

  for (let i = 0; i < samples.length; i++) {
    const s = samples[i]!;
    const lower = s.name.toLowerCase();
    if (!lower.includes('clone') && !lower.includes('structuredclone')) {
      continue;
    }
    const bytes = s.metadata?.['bytes'];
    if (bytes !== undefined && bytes > LARGE_CLONE_BYTES) {
      largeCount++;
      if (bytes > largestBytes) largestBytes = bytes;
    }
  }

  if (largeCount === 0) return null;

  return {
    name: 'structured_clone_large',
    severity: largestBytes > 10 * LARGE_CLONE_BYTES ? 'critical' : 'warning',
    description: `Found ${largeCount} structured clone(s) exceeding 1 MB (largest: ${(largestBytes / (1024 * 1024)).toFixed(1)} MB) — use Transferable`,
    metric: largestBytes,
    threshold: LARGE_CLONE_BYTES,
  };
}

// ---------------------------------------------------------------------------
// detectAntipatterns
// ---------------------------------------------------------------------------

/**
 * Run all antipattern checks against the given samples and return a list of
 * detected issues. Empty array means no antipatterns found.
 */
export function detectAntipatterns(
  samples: ProfileSample[],
): AntipatternReport[] {
  const reports: AntipatternReport[] = [];

  const checks = [
    checkExcessiveReadbacks,
    checkSmallDispatches,
    checkPerFramePipelineCreation,
    checkHotLoopAllocation,
    checkFrequentPostMessage,
    checkLargeClones,
  ] as const;

  for (let i = 0; i < checks.length; i++) {
    const result = checks[i]!(samples);
    if (result !== null) {
      reports.push(result);
    }
  }

  return reports;
}

// ---------------------------------------------------------------------------
// severityScore
// ---------------------------------------------------------------------------

/** Severity weights for computing aggregate score. */
const SEVERITY_WEIGHTS: Record<AntipatternReport['severity'], number> = {
  info: 10,
  warning: 30,
  critical: 50,
};

/**
 * Compute an aggregate severity score (0-100) from a list of reports.
 *
 * Each report contributes its severity weight; the total is clamped to 100.
 * An empty array yields 0.
 */
export function severityScore(reports: AntipatternReport[]): number {
  let score = 0;
  for (let i = 0; i < reports.length; i++) {
    score += SEVERITY_WEIGHTS[reports[i]!.severity];
  }
  return Math.min(100, score);
}
