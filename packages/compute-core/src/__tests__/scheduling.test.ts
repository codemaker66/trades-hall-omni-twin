import { describe, it, expect } from 'vitest';
import {
  createPriorityQueue,
  pqPush,
  pqPop,
  pqPeek,
  pqSize,
  pqIsEmpty,
  pqUpdatePriority,
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
  createWorkflow,
  getReadySteps,
  completeStep,
  isWorkflowComplete,
  topologicalSort,
  validateWorkflow,
  estimateWorkflowDurationMs,
} from '../scheduling/index.js';
import type { ScheduledJob, WorkflowStep, WorkflowDefinition } from '../types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeJob(overrides: Partial<ScheduledJob> = {}): ScheduledJob {
  return {
    id: overrides.id ?? 'job-1',
    type: overrides.type ?? 'compute',
    priority: overrides.priority ?? 1,
    createdAt: overrides.createdAt ?? 0,
    scheduledAt: overrides.scheduledAt ?? 0,
    timeoutMs: overrides.timeoutMs ?? 5000,
    retries: overrides.retries ?? 0,
    maxRetries: overrides.maxRetries ?? 3,
    payload: overrides.payload ?? null,
  };
}

function makeStep(
  name: string,
  dependsOn: string[] = [],
  timeoutMs = 1000,
): WorkflowStep {
  return {
    name,
    action: `run_${name}`,
    dependsOn,
    timeoutMs,
    retryPolicy: {
      maxAttempts: 3,
      backoffMs: 1000,
      backoffMultiplier: 2,
      maxBackoffMs: 30000,
    },
  };
}

// ---------------------------------------------------------------------------
// Priority Queue
// ---------------------------------------------------------------------------

describe('PriorityQueue', () => {
  it('pops items in priority order (lowest number first)', () => {
    const pq = createPriorityQueue<string>();
    pqPush(pq, 'low', 10);
    pqPush(pq, 'high', 1);
    pqPush(pq, 'mid', 5);

    expect(pqPop(pq)).toBe('high');
    expect(pqPop(pq)).toBe('mid');
    expect(pqPop(pq)).toBe('low');
  });

  it('peek returns highest-priority item without removing it', () => {
    const pq = createPriorityQueue<string>();
    pqPush(pq, 'first', 1);
    pqPush(pq, 'second', 2);

    expect(pqPeek(pq)).toBe('first');
    expect(pqSize(pq)).toBe(2); // still 2 items
  });

  it('size tracks the number of items', () => {
    const pq = createPriorityQueue<number>();
    expect(pqSize(pq)).toBe(0);
    pqPush(pq, 10, 1);
    pqPush(pq, 20, 2);
    expect(pqSize(pq)).toBe(2);
    pqPop(pq);
    expect(pqSize(pq)).toBe(1);
  });

  it('isEmpty returns true for empty queue and false otherwise', () => {
    const pq = createPriorityQueue<string>();
    expect(pqIsEmpty(pq)).toBe(true);
    pqPush(pq, 'x', 1);
    expect(pqIsEmpty(pq)).toBe(false);
    pqPop(pq);
    expect(pqIsEmpty(pq)).toBe(true);
  });

  it('updatePriority changes the dequeue order', () => {
    const pq = createPriorityQueue<string>();
    const itemA = 'a';
    const itemB = 'b';
    pqPush(pq, itemA, 10);
    pqPush(pq, itemB, 5);

    // b has higher priority (5). Now promote a to 1.
    const updated = pqUpdatePriority(pq, itemA, 1);
    expect(updated).toBe(true);
    expect(pqPop(pq)).toBe('a'); // a should now come first
  });

  it('updatePriority returns false for item not in queue', () => {
    const pq = createPriorityQueue<string>();
    pqPush(pq, 'a', 1);
    expect(pqUpdatePriority(pq, 'not-in-queue', 0)).toBe(false);
  });

  it('pop returns null on empty queue', () => {
    const pq = createPriorityQueue<string>();
    expect(pqPop(pq)).toBeNull();
  });

  it('maintains FIFO order for equal priorities', () => {
    const pq = createPriorityQueue<string>();
    pqPush(pq, 'first', 1);
    pqPush(pq, 'second', 1);
    pqPush(pq, 'third', 1);

    expect(pqPop(pq)).toBe('first');
    expect(pqPop(pq)).toBe('second');
    expect(pqPop(pq)).toBe('third');
  });
});

// ---------------------------------------------------------------------------
// Job Scheduler
// ---------------------------------------------------------------------------

describe('JobScheduler', () => {
  it('submit and getNext returns the submitted job', () => {
    const sched = createJobScheduler();
    const job = makeJob({ id: 'j1', scheduledAt: 0 });
    submitJob(sched, job);

    const next = getNextJob(sched, 1000);
    expect(next).not.toBeNull();
    expect(next!.id).toBe('j1');
  });

  it('getNext returns null when no jobs are ready', () => {
    const sched = createJobScheduler();
    const job = makeJob({ id: 'j1', scheduledAt: 99999 });
    submitJob(sched, job);

    expect(getNextJob(sched, 0)).toBeNull();
  });

  it('completeJob sets status to completed', () => {
    const sched = createJobScheduler();
    const job = makeJob({ id: 'j1' });
    submitJob(sched, job);
    getNextJob(sched, 1000); // sets to running

    completeJob(sched, 'j1', 'completed');
    expect(getJobStatus(sched, 'j1')).toBe('completed');
  });

  it('retryJob increments attempt and re-queues', () => {
    const sched = createJobScheduler();
    const job = makeJob({ id: 'j1', maxRetries: 5 });
    submitJob(sched, job);
    getNextJob(sched, 1000); // attempt 1, status = running

    // Mark as failed then retry
    completeJob(sched, 'j1', 'failed');
    const retried = retryJob(sched, 'j1', 1000);
    expect(retried).toBe(true);
    expect(getJobStatus(sched, 'j1')).toBe('retrying');
  });

  it('cancelJob sets status to cancelled', () => {
    const sched = createJobScheduler();
    const job = makeJob({ id: 'j1' });
    submitJob(sched, job);

    const cancelled = cancelJob(sched, 'j1');
    expect(cancelled).toBe(true);
    expect(getJobStatus(sched, 'j1')).toBe('cancelled');
  });

  it('cancelJob returns false for already-completed job', () => {
    const sched = createJobScheduler();
    const job = makeJob({ id: 'j1' });
    submitJob(sched, job);
    getNextJob(sched, 1000);
    completeJob(sched, 'j1', 'completed');

    expect(cancelJob(sched, 'j1')).toBe(false);
  });

  it('computeRetryDelay uses exponential backoff capped at maxBackoffMs', () => {
    const policy = {
      maxAttempts: 5,
      backoffMs: 1000,
      backoffMultiplier: 2,
      maxBackoffMs: 10000,
    };

    // attempt 0: 1000 * 2^0 = 1000
    expect(computeRetryDelay(policy, 0)).toBe(1000);
    // attempt 1: 1000 * 2^1 = 2000
    expect(computeRetryDelay(policy, 1)).toBe(2000);
    // attempt 2: 1000 * 2^2 = 4000
    expect(computeRetryDelay(policy, 2)).toBe(4000);
    // attempt 3: 1000 * 2^3 = 8000
    expect(computeRetryDelay(policy, 3)).toBe(8000);
    // attempt 4: 1000 * 2^4 = 16000, capped at 10000
    expect(computeRetryDelay(policy, 4)).toBe(10000);
  });

  it('getQueueDepth reflects number of items in the priority queue', () => {
    const sched = createJobScheduler();
    submitJob(sched, makeJob({ id: 'j1' }));
    submitJob(sched, makeJob({ id: 'j2' }));
    submitJob(sched, makeJob({ id: 'j3' }));

    expect(getQueueDepth(sched)).toBe(3);
  });

  it('getPendingJobs returns only pending and retrying jobs', () => {
    const sched = createJobScheduler();
    submitJob(sched, makeJob({ id: 'j1' }));
    submitJob(sched, makeJob({ id: 'j2' }));
    submitJob(sched, makeJob({ id: 'j3' }));

    // Move j1 to running then completed
    getNextJob(sched, 1000);
    completeJob(sched, 'j1', 'completed');

    const pending = getPendingJobs(sched);
    const ids = pending.map((j) => j.id);
    expect(ids).toContain('j2');
    expect(ids).toContain('j3');
    expect(ids).not.toContain('j1');
  });
});

// ---------------------------------------------------------------------------
// Workflow
// ---------------------------------------------------------------------------

describe('Workflow', () => {
  const linearDef: WorkflowDefinition = {
    name: 'linear',
    steps: [
      makeStep('A', []),
      makeStep('B', ['A']),
      makeStep('C', ['B']),
    ],
  };

  const diamondDef: WorkflowDefinition = {
    name: 'diamond',
    steps: [
      makeStep('start', [], 100),
      makeStep('left', ['start'], 200),
      makeStep('right', ['start'], 300),
      makeStep('end', ['left', 'right'], 100),
    ],
  };

  it('getReadySteps returns steps with all deps satisfied', () => {
    const exec = createWorkflow(linearDef);
    const ready = getReadySteps(exec, linearDef);
    expect(ready.length).toBe(1);
    expect(ready[0]!.name).toBe('A');
  });

  it('getReadySteps advances after completing a step', () => {
    const exec = createWorkflow(linearDef);
    completeStep(exec, 'A');
    const ready = getReadySteps(exec, linearDef);
    expect(ready.length).toBe(1);
    expect(ready[0]!.name).toBe('B');
  });

  it('completeStep marks step as done', () => {
    const exec = createWorkflow(linearDef);
    completeStep(exec, 'A');
    expect(exec.completedSteps).toContain('A');
  });

  it('topologicalSort returns valid execution order', () => {
    const sorted = topologicalSort(diamondDef.steps);
    const startIdx = sorted.indexOf('start');
    const leftIdx = sorted.indexOf('left');
    const rightIdx = sorted.indexOf('right');
    const endIdx = sorted.indexOf('end');

    expect(startIdx).toBeLessThan(leftIdx);
    expect(startIdx).toBeLessThan(rightIdx);
    expect(leftIdx).toBeLessThan(endIdx);
    expect(rightIdx).toBeLessThan(endIdx);
  });

  it('topologicalSort throws on circular dependencies', () => {
    const cyclicSteps: WorkflowStep[] = [
      makeStep('A', ['C']),
      makeStep('B', ['A']),
      makeStep('C', ['B']),
    ];
    expect(() => topologicalSort(cyclicSteps)).toThrow('cycle');
  });

  it('validateWorkflow detects circular dependencies', () => {
    const cyclicDef: WorkflowDefinition = {
      name: 'cyclic',
      steps: [
        makeStep('A', ['B']),
        makeStep('B', ['A']),
      ],
    };
    const errors = validateWorkflow(cyclicDef);
    expect(errors.length).toBeGreaterThan(0);
    expect(errors.some((e) => e.toLowerCase().includes('cycle'))).toBe(true);
  });

  it('isWorkflowComplete returns true when all steps are done', () => {
    const exec = createWorkflow(linearDef);
    completeStep(exec, 'A');
    completeStep(exec, 'B');
    completeStep(exec, 'C');
    expect(isWorkflowComplete(exec, linearDef)).toBe(true);
  });

  it('isWorkflowComplete returns false when steps remain', () => {
    const exec = createWorkflow(linearDef);
    completeStep(exec, 'A');
    expect(isWorkflowComplete(exec, linearDef)).toBe(false);
  });

  it('estimateWorkflowDurationMs computes critical path length', () => {
    // diamond: start(100) -> left(200) or right(300) -> end(100)
    // Critical path: start(100) + right(300) + end(100) = 500
    const duration = estimateWorkflowDurationMs(diamondDef);
    expect(duration).toBe(500);
  });

  it('diamond workflow has two ready steps after completing start', () => {
    const exec = createWorkflow(diamondDef);
    completeStep(exec, 'start');
    const ready = getReadySteps(exec, diamondDef);
    expect(ready.length).toBe(2);
    const names = ready.map((s) => s.name);
    expect(names).toContain('left');
    expect(names).toContain('right');
  });
});
