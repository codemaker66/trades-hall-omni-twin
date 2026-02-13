// ---------------------------------------------------------------------------
// HPC-8: Scheduling — Job Scheduler with Retry & Exponential Backoff
// ---------------------------------------------------------------------------
// Manages a queue of ScheduledJobs with priority ordering, lifecycle tracking,
// and configurable retry policies with exponential backoff.
// ---------------------------------------------------------------------------

import type { ScheduledJob, JobStatus, RetryPolicy } from '../types.js';
import {
  createPriorityQueue,
  pqPush,
  pqPop,
  pqSize,
  pqIsEmpty,
} from './priority-queue.js';
import type { PriorityQueue } from './priority-queue.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Internal record tracking a job's lifecycle. */
type JobRecord = {
  job: ScheduledJob;
  status: JobStatus;
  attempt: number;
  lastAttemptAt: number;
  nextRetryAt: number;
};

/** Mutable state for the job scheduler. */
export type JobSchedulerState = {
  queue: PriorityQueue<ScheduledJob>;
  jobs: Map<string, JobRecord>;
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Create a new empty job scheduler.
 */
export function createJobScheduler(): JobSchedulerState {
  return {
    queue: createPriorityQueue<ScheduledJob>(),
    jobs: new Map(),
  };
}

/**
 * Submit a new job to the scheduler.
 * The job is placed in the priority queue ordered by its `priority` field
 * (lower number = higher priority).
 */
export function submitJob(
  scheduler: JobSchedulerState,
  job: ScheduledJob,
): void {
  const record: JobRecord = {
    job,
    status: 'pending',
    attempt: 0,
    lastAttemptAt: 0,
    nextRetryAt: 0,
  };
  scheduler.jobs.set(job.id, record);
  pqPush(scheduler.queue, job, job.priority);
}

/**
 * Get the next job that is ready to run at the given time.
 * A job is ready if:
 *   - Its `scheduledAt` is <= now
 *   - Its status is 'pending' or 'retrying' with `nextRetryAt` <= now
 *
 * Returns null if no job is ready. The returned job's status is set to 'running'.
 */
export function getNextJob(
  scheduler: JobSchedulerState,
  now: number,
): ScheduledJob | null {
  // We need to scan through the queue to find a ready job.
  // Pop items that are not ready and re-queue them.
  const deferred: Array<{ job: ScheduledJob; priority: number }> = [];

  let result: ScheduledJob | null = null;

  while (!pqIsEmpty(scheduler.queue)) {
    const candidate = pqPop(scheduler.queue);
    if (!candidate) break;

    const record = scheduler.jobs.get(candidate.id);
    if (!record) continue;

    // Skip cancelled, completed, or failed jobs
    if (
      record.status === 'cancelled' ||
      record.status === 'completed' ||
      record.status === 'failed'
    ) {
      continue;
    }

    // Check if the job is ready
    const isScheduleReady = candidate.scheduledAt <= now;
    const isRetryReady =
      record.status === 'retrying' && record.nextRetryAt <= now;
    const isPendingReady = record.status === 'pending' && isScheduleReady;

    if (isPendingReady || isRetryReady) {
      record.status = 'running';
      record.attempt++;
      record.lastAttemptAt = now;
      result = candidate;
      break;
    }

    // Not ready yet — defer for re-insertion
    deferred.push({ job: candidate, priority: candidate.priority });
  }

  // Re-insert deferred jobs
  for (const d of deferred) {
    pqPush(scheduler.queue, d.job, d.priority);
  }

  return result;
}

/**
 * Mark a running job as completed or failed.
 */
export function completeJob(
  scheduler: JobSchedulerState,
  jobId: string,
  status: 'completed' | 'failed',
): void {
  const record = scheduler.jobs.get(jobId);
  if (!record) return;
  record.status = status;
}

/**
 * Schedule a retry for a failed job using exponential backoff.
 * Returns false if the job is not found, not in a retryable state,
 * or has exceeded max retries.
 */
export function retryJob(
  scheduler: JobSchedulerState,
  jobId: string,
  now: number,
): boolean {
  const record = scheduler.jobs.get(jobId);
  if (!record) return false;

  // Only retry failed or running jobs that haven't exceeded max retries
  if (record.status !== 'failed' && record.status !== 'running') return false;
  if (record.attempt >= record.job.maxRetries) return false;

  // Compute retry delay using exponential backoff
  // Default retry policy if none specified
  const delay = computeRetryDelay(
    {
      maxAttempts: record.job.maxRetries,
      backoffMs: 1000,
      backoffMultiplier: 2,
      maxBackoffMs: 60000,
    },
    record.attempt,
  );

  record.status = 'retrying';
  record.nextRetryAt = now + delay;

  // Re-insert into queue
  pqPush(scheduler.queue, record.job, record.job.priority);

  return true;
}

/**
 * Cancel a pending or retrying job.
 * Returns false if the job is not found or is already completed/failed/cancelled.
 */
export function cancelJob(
  scheduler: JobSchedulerState,
  jobId: string,
): boolean {
  const record = scheduler.jobs.get(jobId);
  if (!record) return false;

  if (
    record.status === 'completed' ||
    record.status === 'failed' ||
    record.status === 'cancelled'
  ) {
    return false;
  }

  record.status = 'cancelled';
  return true;
}

/**
 * Get the current status of a job, or null if not found.
 */
export function getJobStatus(
  scheduler: JobSchedulerState,
  jobId: string,
): JobStatus | null {
  const record = scheduler.jobs.get(jobId);
  if (!record) return null;
  return record.status;
}

/**
 * Compute the retry delay using exponential backoff.
 *
 *   delay = min(backoffMs * backoffMultiplier^attempt, maxBackoffMs)
 */
export function computeRetryDelay(
  policy: RetryPolicy,
  attempt: number,
): number {
  const raw = policy.backoffMs * Math.pow(policy.backoffMultiplier, attempt);
  return Math.min(raw, policy.maxBackoffMs);
}

/**
 * Return the number of jobs currently in the priority queue.
 * Note: this includes jobs that may not be ready yet.
 */
export function getQueueDepth(scheduler: JobSchedulerState): number {
  return pqSize(scheduler.queue);
}

/**
 * Return all jobs with 'pending' or 'retrying' status.
 */
export function getPendingJobs(scheduler: JobSchedulerState): ScheduledJob[] {
  const result: ScheduledJob[] = [];
  for (const [, record] of scheduler.jobs) {
    if (record.status === 'pending' || record.status === 'retrying') {
      result.push(record.job);
    }
  }
  return result;
}
