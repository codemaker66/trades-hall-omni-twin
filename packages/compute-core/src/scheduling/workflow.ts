// ---------------------------------------------------------------------------
// HPC-8: Scheduling — Workflow Execution Engine
// ---------------------------------------------------------------------------
// Manages execution of multi-step workflow DAGs. Each workflow is defined as
// a set of named steps with dependency edges. The engine determines which
// steps are ready to execute, tracks completion/failure, and provides
// topological ordering, cycle detection, and critical-path estimation.
// ---------------------------------------------------------------------------

import type {
  WorkflowStep,
  WorkflowDefinition,
  WorkflowExecution,
} from '../types.js';

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Create a new workflow execution from a workflow definition.
 * All steps start in an unexecuted state.
 */
export function createWorkflow(definition: WorkflowDefinition): WorkflowExecution {
  return {
    workflowId: definition.name,
    status: 'running',
    completedSteps: [],
    currentStep: null,
    startedAt: Date.now(),
  };
}

/**
 * Get all steps whose dependencies have been fully satisfied
 * (i.e., all items in `dependsOn` are in `completedSteps`).
 * Only returns steps that are not yet completed themselves.
 */
export function getReadySteps(
  workflow: WorkflowExecution,
  definition: WorkflowDefinition,
): WorkflowStep[] {
  const completedSet = new Set(workflow.completedSteps);
  const ready: WorkflowStep[] = [];

  for (const step of definition.steps) {
    // Skip already-completed steps
    if (completedSet.has(step.name)) continue;

    // Check if all dependencies are satisfied
    const allDepsMet = step.dependsOn.every((dep) => completedSet.has(dep));
    if (allDepsMet) {
      ready.push(step);
    }
  }

  return ready;
}

/**
 * Mark a step as completed in the workflow execution.
 * Returns a new WorkflowExecution with the step added to completedSteps.
 */
export function completeStep(
  execution: WorkflowExecution,
  stepName: string,
): void {
  // Use type assertion to write to readonly property — we own the state
  if (!execution.completedSteps.includes(stepName)) {
    (execution as unknown as { completedSteps: string[] }).completedSteps = [
      ...execution.completedSteps,
      stepName,
    ];
  }
  (execution as { currentStep: string | null }).currentStep = null;
}

/**
 * Mark a step as failed, which fails the entire workflow.
 */
export function failStep(
  execution: WorkflowExecution,
  stepName: string,
): void {
  (execution as { status: WorkflowExecution['status'] }).status = 'failed';
  (execution as { currentStep: string | null }).currentStep = stepName;
}

/**
 * Check whether the workflow is fully complete: all defined steps
 * have been completed.
 */
export function isWorkflowComplete(
  execution: WorkflowExecution,
  definition: WorkflowDefinition,
): boolean {
  if (execution.status === 'failed') return false;
  const completedSet = new Set(execution.completedSteps);
  return definition.steps.every((step) => completedSet.has(step.name));
}

/**
 * Topological sort of workflow steps using Kahn's algorithm.
 * Returns step names in a valid execution order.
 * Throws if the graph contains a cycle.
 */
export function topologicalSort(steps: readonly WorkflowStep[]): string[] {
  // Build adjacency list and in-degree map
  const inDegree = new Map<string, number>();
  const adjacency = new Map<string, string[]>();

  for (const step of steps) {
    if (!inDegree.has(step.name)) {
      inDegree.set(step.name, 0);
    }
    if (!adjacency.has(step.name)) {
      adjacency.set(step.name, []);
    }
    for (const dep of step.dependsOn) {
      if (!adjacency.has(dep)) {
        adjacency.set(dep, []);
      }
      adjacency.get(dep)!.push(step.name);
      inDegree.set(step.name, (inDegree.get(step.name) ?? 0) + 1);
      if (!inDegree.has(dep)) {
        inDegree.set(dep, 0);
      }
    }
  }

  // Kahn's algorithm: start with nodes having in-degree 0
  const queue: string[] = [];
  for (const [name, deg] of inDegree) {
    if (deg === 0) queue.push(name);
  }

  const sorted: string[] = [];
  while (queue.length > 0) {
    const node = queue.shift()!;
    sorted.push(node);

    const neighbors = adjacency.get(node);
    if (neighbors) {
      for (const neighbor of neighbors) {
        const newDeg = (inDegree.get(neighbor) ?? 1) - 1;
        inDegree.set(neighbor, newDeg);
        if (newDeg === 0) {
          queue.push(neighbor);
        }
      }
    }
  }

  // If sorted doesn't contain all nodes, there is a cycle
  if (sorted.length !== inDegree.size) {
    throw new Error('Workflow contains a cycle');
  }

  return sorted;
}

/**
 * Validate a workflow definition.
 * Returns an array of error strings (empty if valid).
 *
 * Checks performed:
 *   1. Duplicate step names
 *   2. References to undefined dependencies
 *   3. Cycle detection
 *   4. Self-dependencies
 */
export function validateWorkflow(definition: WorkflowDefinition): string[] {
  const errors: string[] = [];
  const stepNames = new Set<string>();

  // 1. Check for duplicate step names
  for (const step of definition.steps) {
    if (stepNames.has(step.name)) {
      errors.push(`Duplicate step name: "${step.name}"`);
    }
    stepNames.add(step.name);
  }

  // 2. Check for undefined dependencies and self-dependencies
  for (const step of definition.steps) {
    for (const dep of step.dependsOn) {
      if (dep === step.name) {
        errors.push(`Step "${step.name}" depends on itself`);
      } else if (!stepNames.has(dep)) {
        errors.push(
          `Step "${step.name}" depends on undefined step "${dep}"`,
        );
      }
    }
  }

  // 3. Cycle detection via topological sort
  try {
    topologicalSort(definition.steps);
  } catch {
    errors.push('Workflow contains a cycle');
  }

  return errors;
}

/**
 * Estimate the total workflow duration in milliseconds by computing
 * the critical path (longest path through the DAG by step timeouts).
 *
 * Uses dynamic programming on a topological ordering.
 */
export function estimateWorkflowDurationMs(
  definition: WorkflowDefinition,
): number {
  // Build a map of step name -> step for quick lookup
  const stepMap = new Map<string, WorkflowStep>();
  for (const step of definition.steps) {
    stepMap.set(step.name, step);
  }

  // Topological order
  let order: string[];
  try {
    order = topologicalSort(definition.steps);
  } catch {
    // If there's a cycle, return Infinity
    return Infinity;
  }

  // dp[stepName] = longest path from any source to this step (inclusive)
  const dp = new Map<string, number>();

  for (const name of order) {
    const step = stepMap.get(name);
    if (!step) continue;

    // Find the max completion time of all dependencies
    let maxDepTime = 0;
    for (const dep of step.dependsOn) {
      const depTime = dp.get(dep) ?? 0;
      if (depTime > maxDepTime) maxDepTime = depTime;
    }

    dp.set(name, maxDepTime + step.timeoutMs);
  }

  // The critical path duration is the maximum across all steps
  let maxDuration = 0;
  for (const [, duration] of dp) {
    if (duration > maxDuration) maxDuration = duration;
  }

  return maxDuration;
}
