// ---------------------------------------------------------------------------
// HPC-8: Scheduling — Barrel Export
// ---------------------------------------------------------------------------

export {
  createPriorityQueue,
  pqPush,
  pqPop,
  pqPeek,
  pqSize,
  pqIsEmpty,
  pqUpdatePriority,
} from './priority-queue.js';
export type { PriorityQueue } from './priority-queue.js';

export {
  createJobScheduler,
  submitJob,
  getNextJob,
  completeJob,
  retryJob,
  cancelJob,
  getJobStatus,
  computeRetryDelay,
  getQueueDepth,
  getPendingJobs,
} from './job-scheduler.js';
export type { JobSchedulerState } from './job-scheduler.js';

export {
  createWorkflow,
  getReadySteps,
  completeStep,
  failStep,
  isWorkflowComplete,
  topologicalSort,
  validateWorkflow,
  estimateWorkflowDurationMs,
} from './workflow.js';
