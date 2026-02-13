// ---------------------------------------------------------------------------
// TreeSHAP-inspired Feature Importance
// ---------------------------------------------------------------------------

import type { TreeNode, TreeLeaf, SHAPValue } from '../types.js';
import { isLeaf } from './decision-tree.js';

/**
 * Approximate SHAP values for a single tree and a single sample x.
 *
 * Uses the tree path decomposition approach inspired by Lundberg & Lee (2017).
 * For each feature, the SHAP value is the weighted average change in
 * prediction when that feature's value directs the sample to the left
 * or right child at each internal node that splits on that feature.
 *
 * This is an approximation of the true TreeSHAP algorithm — it computes
 * contributions by tracking the difference in expected predictions along
 * the path that x follows vs. the unconditional expectation, weighted by
 * the fraction of training data at each node.
 *
 * @param tree  Root of the decision tree
 * @param x     Feature vector for the sample
 * @param featureNames  Human-readable feature names
 * @returns Array of SHAP values, one per feature
 */
export function computeTreeSHAP(
  tree: TreeNode | TreeLeaf,
  x: number[],
  featureNames: string[],
): SHAPValue[] {
  const nFeatures = featureNames.length;
  const contributions = new Array<number>(nFeatures).fill(0);

  // Compute the expected value (mean) of the whole tree
  const baseValue = nodeExpectedValue(tree);

  // Walk the decision path and accumulate contributions
  accumulateContributions(tree, x, contributions, baseValue);

  // Convert to SHAPValue objects
  const shapValues: SHAPValue[] = [];
  for (let f = 0; f < nFeatures; f++) {
    const val = contributions[f] ?? 0;
    shapValues.push({
      feature: featureNames[f] ?? `feature_${f}`,
      value: val,
      direction: val >= 0 ? 'increases' : 'decreases',
    });
  }

  return shapValues;
}

/**
 * Aggregate SHAP values from multiple trees by computing the mean
 * absolute SHAP value per feature, preserving the direction based on
 * the mean signed value.
 *
 * @param shapPerTree  Array of SHAP value arrays (one per tree)
 * @param featureNames Feature names for the output
 * @returns Aggregated SHAP values, one per feature
 */
export function aggregateSHAP(
  shapPerTree: SHAPValue[][],
  featureNames: string[],
): SHAPValue[] {
  const nFeatures = featureNames.length;
  const nTrees = shapPerTree.length;

  if (nTrees === 0) {
    return featureNames.map((name) => ({
      feature: name,
      value: 0,
      direction: 'increases' as const,
    }));
  }

  // Sum signed values across trees
  const sumValues = new Array<number>(nFeatures).fill(0);

  for (const treeShap of shapPerTree) {
    for (let f = 0; f < nFeatures; f++) {
      const sv = treeShap[f];
      if (sv) {
        sumValues[f] = (sumValues[f] ?? 0) + sv.value;
      }
    }
  }

  // Mean across trees
  const result: SHAPValue[] = [];
  for (let f = 0; f < nFeatures; f++) {
    const meanVal = (sumValues[f] ?? 0) / nTrees;
    result.push({
      feature: featureNames[f] ?? `feature_${f}`,
      value: meanVal,
      direction: meanVal >= 0 ? 'increases' : 'decreases',
    });
  }

  return result;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Compute the expected prediction (weighted mean of leaf values)
 * for a node's entire subtree. The weight of each leaf is proportional
 * to how many training samples it contains.
 */
function nodeExpectedValue(node: TreeNode | TreeLeaf): number {
  if (isLeaf(node)) {
    return leafMean(node.values);
  }
  const leftCount = nodeCount(node.left);
  const rightCount = nodeCount(node.right);
  const total = leftCount + rightCount;
  if (total === 0) return 0;

  const leftE = nodeExpectedValue(node.left);
  const rightE = nodeExpectedValue(node.right);

  return (leftCount * leftE + rightCount * rightE) / total;
}

/** Count total training samples under a node (sum of all leaf sizes). */
function nodeCount(node: TreeNode | TreeLeaf): number {
  if (isLeaf(node)) {
    return node.values.length;
  }
  return nodeCount(node.left) + nodeCount(node.right);
}

/**
 * Walk the path that x follows through the tree. At each internal node
 * that splits on feature f, the contribution of f is:
 *
 *   contribution[f] += E[child x goes to] - E[parent]
 *
 * where E[node] is the expected value of that subtree weighted by
 * training sample counts. This decomposes the final prediction as
 *
 *   prediction = baseValue + sum(contributions)
 *
 * which satisfies the local accuracy property of SHAP.
 */
function accumulateContributions(
  node: TreeNode | TreeLeaf,
  x: number[],
  contributions: number[],
  parentExpected: number,
): void {
  if (isLeaf(node)) {
    // Leaf reached — nothing more to attribute
    return;
  }

  const featureVal = x[node.featureIndex] ?? 0;
  const goLeft = featureVal <= node.threshold;

  // Expected values of left and right subtrees
  const leftCount = nodeCount(node.left);
  const rightCount = nodeCount(node.right);
  const total = leftCount + rightCount;

  const leftE = nodeExpectedValue(node.left);
  const rightE = nodeExpectedValue(node.right);

  // The unconditional expected value at this node
  const nodeE =
    total > 0 ? (leftCount * leftE + rightCount * rightE) / total : 0;

  // Child that x goes to
  const childNode = goLeft ? node.left : node.right;
  const childE = goLeft ? leftE : rightE;

  // The contribution of this feature at this split: difference between
  // the child's expected value and the node's unconditional expected value
  const featureContribution = childE - nodeE;

  if (
    node.featureIndex >= 0 &&
    node.featureIndex < contributions.length
  ) {
    contributions[node.featureIndex] =
      (contributions[node.featureIndex] ?? 0) + featureContribution;
  }

  // Continue down the tree
  accumulateContributions(childNode, x, contributions, childE);
}

function leafMean(values: number[]): number {
  if (values.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i] ?? 0;
  }
  return sum / values.length;
}
