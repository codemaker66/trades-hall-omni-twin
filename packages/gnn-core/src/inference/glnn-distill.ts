// ---------------------------------------------------------------------------
// GNN-9 Scalable Inference — GLNN Knowledge Distillation (GNN -> MLP)
// ---------------------------------------------------------------------------
//
// Implements GNN-to-MLP knowledge distillation, where a trained GNN (teacher)
// produces soft labels that an MLP (student) learns to match. At inference
// time, the MLP runs without graph structure, enabling O(1) per-node latency.
//
// The loss combines:
//   L = lambda * KL(student/T, teacher/T) + (1 - lambda) * CE(student, hard)
//
// References:
//   Zhang et al., "Graph-less Neural Networks: Teaching Old MLPs New Tricks
//   Via Distillation" (ICLR 2022) — GLNN
// ---------------------------------------------------------------------------

import type {
  Graph,
  PRNG,
  GNNForwardFn,
  MLPWeights,
  GLNNConfig,
  GLNNResult,
} from '../types.js';
import { xavierInit, matVecMul, relu, softmax, crossEntropyLoss, argmax } from '../tensor.js';

// ---------------------------------------------------------------------------
// 1. distillGNNToMLP — Train MLP student from GNN teacher
// ---------------------------------------------------------------------------

/**
 * Distill a GNN teacher into an MLP student network.
 *
 * Algorithm:
 * 1. Run the teacher GNN forward pass to get soft logits for all nodes.
 * 2. Derive hard labels from the teacher (argmax of teacher logits).
 * 3. Initialize MLP weights (Xavier) with dimensions specified in config.
 * 4. For each training epoch:
 *    a. Forward pass: for each node, run MLP to get student logits.
 *    b. Compute combined loss:
 *       - KL divergence between softmax(student/T) and softmax(teacher/T)
 *       - Cross-entropy between student logits and hard labels
 *       - L = lambda * KL_loss + (1 - lambda) * CE_loss
 *    c. Compute gradients via finite differences (simplified) or analytical.
 *    d. SGD update of MLP weights.
 * 5. Compute final accuracy on the training set.
 *
 * @param teacherFn - GNN forward function: (graph, features) -> logits.
 * @param graph - Input CSR Graph (used only for teacher inference).
 * @param features - Node feature matrix (numNodes * featureDim), row-major.
 * @param config - Distillation configuration (hiddenDims, lambda, temperature, etc.).
 * @param rng - Deterministic PRNG function.
 * @returns GLNNResult with trained MLP weights, loss history, and accuracy.
 */
export function distillGNNToMLP(
  teacherFn: GNNForwardFn,
  graph: Graph,
  features: Float64Array,
  config: GLNNConfig,
  rng: PRNG,
): GLNNResult {
  const numNodes = graph.numNodes;
  const featureDim = graph.featureDim;

  // --- Step 1: Get teacher soft labels ---
  const teacherLogits = teacherFn(graph, features);

  // Infer output dimension: teacherLogits is numNodes * outDim
  const outDim = teacherLogits.length / numNodes;

  // --- Step 2: Derive hard labels from teacher ---
  const hardLabels = new Uint32Array(numNodes);
  for (let i = 0; i < numNodes; i++) {
    const nodeLogits = teacherLogits.subarray(i * outDim, (i + 1) * outDim);
    hardLabels[i] = argmax(nodeLogits);
  }

  // Compute teacher soft probabilities at temperature T
  const T = config.temperature;
  const teacherScaled = new Float64Array(teacherLogits.length);
  for (let i = 0; i < teacherLogits.length; i++) {
    teacherScaled[i] = teacherLogits[i]! / T;
  }
  const teacherSoft = softmax(teacherScaled, outDim);

  // --- Step 3: Initialize MLP weights ---
  const hiddenDims = config.hiddenDims;
  const layerDims: number[] = [featureDim, ...hiddenDims, outDim];
  const layers: { W: Float64Array; bias: Float64Array; inDim: number; outDim: number }[] = [];

  for (let l = 0; l < layerDims.length - 1; l++) {
    const inD = layerDims[l]!;
    const outD = layerDims[l + 1]!;
    layers.push({
      W: xavierInit(inD, outD, rng),
      bias: new Float64Array(outD),
      inDim: inD,
      outDim: outD,
    });
  }

  const mlpWeights: MLPWeights = { layers };

  // --- Step 4: Training loop ---
  const lr = config.learningRate;
  const lambda = config.lambda;
  const losses: number[] = [];

  for (let epoch = 0; epoch < config.epochs; epoch++) {
    let epochLoss = 0;

    for (let node = 0; node < numNodes; node++) {
      // Forward pass through MLP
      const nodeFeatures = features.subarray(
        node * featureDim,
        (node + 1) * featureDim,
      );

      // Store activations for backprop
      const activations: Float64Array[] = [new Float64Array(nodeFeatures)];
      let current = new Float64Array(nodeFeatures);

      for (let l = 0; l < layers.length; l++) {
        const layer = layers[l]!;
        // out = W^T * in + bias (W is inDim x outDim, stored row-major)
        const preAct = matVecMul(
          transposeWeight(layer.W, layer.inDim, layer.outDim),
          current,
          layer.outDim,
          layer.inDim,
        );
        // Add bias
        for (let j = 0; j < layer.outDim; j++) {
          preAct[j] = preAct[j]! + layer.bias[j]!;
        }

        // Activation: ReLU for hidden layers, none for output
        if (l < layers.length - 1) {
          current = relu(preAct) as Float64Array<ArrayBuffer>;
        } else {
          current = preAct as Float64Array<ArrayBuffer>;
        }
        activations.push(new Float64Array(current));
      }

      const studentLogits = current;

      // Compute KL divergence loss (student/T vs teacher/T)
      const studentScaled = new Float64Array(outDim);
      for (let j = 0; j < outDim; j++) {
        studentScaled[j] = studentLogits[j]! / T;
      }
      const studentSoft = softmax(studentScaled, outDim);

      const teacherSoftNode = teacherSoft.subarray(
        node * outDim,
        (node + 1) * outDim,
      );

      // KL(teacher || student) = sum(teacher * log(teacher / student))
      let klLoss = 0;
      for (let j = 0; j < outDim; j++) {
        const tProb = teacherSoftNode[j]!;
        const sProb = studentSoft[j]!;
        if (tProb > 1e-15) {
          klLoss += tProb * Math.log(tProb / Math.max(sProb, 1e-15));
        }
      }
      // Scale KL by T^2 as per Hinton et al.
      klLoss *= T * T;

      // Cross-entropy loss with hard labels
      const ceLoss = crossEntropyLoss(studentLogits, hardLabels[node]!, outDim);

      const totalLoss = lambda * klLoss + (1 - lambda) * ceLoss;
      epochLoss += totalLoss;

      // --- Backpropagation ---
      // Gradient of loss w.r.t. student logits (combined)
      const studentProbs = softmax(new Float64Array(studentLogits), outDim);

      // dL/d(logits) for CE part: softmax(logits) - one_hot(label)
      const gradLogitsCE = new Float64Array(outDim);
      for (let j = 0; j < outDim; j++) {
        gradLogitsCE[j] = studentProbs[j]!;
      }
      gradLogitsCE[hardLabels[node]!] = gradLogitsCE[hardLabels[node]!]! - 1;

      // dL/d(logits) for KL part: (softmax(logits/T) - teacher_soft) / T * T^2
      //   = T * (studentSoft - teacherSoft)
      const gradLogitsKL = new Float64Array(outDim);
      for (let j = 0; j < outDim; j++) {
        gradLogitsKL[j] = T * (studentSoft[j]! - teacherSoftNode[j]!);
      }

      // Combined gradient
      let gradOutput = new Float64Array(outDim);
      for (let j = 0; j < outDim; j++) {
        gradOutput[j] = lambda * gradLogitsKL[j]! + (1 - lambda) * gradLogitsCE[j]!;
      }

      // Backprop through layers (reverse order)
      for (let l = layers.length - 1; l >= 0; l--) {
        const layer = layers[l]!;
        const input = activations[l]!;

        // Gradient w.r.t. bias
        const gradBias = gradOutput;

        // Gradient w.r.t. W: outer product of input and gradOutput
        // W is inDim x outDim (row-major), gradient is input (col) x gradOutput (row)
        const gradW = new Float64Array(layer.inDim * layer.outDim);
        for (let ii = 0; ii < layer.inDim; ii++) {
          for (let jj = 0; jj < layer.outDim; jj++) {
            gradW[ii * layer.outDim + jj] = input[ii]! * gradOutput[jj]!;
          }
        }

        // Gradient w.r.t. input: W * gradOutput
        const gradInput = new Float64Array(layer.inDim);
        for (let ii = 0; ii < layer.inDim; ii++) {
          let s = 0;
          for (let jj = 0; jj < layer.outDim; jj++) {
            s += layer.W[ii * layer.outDim + jj]! * gradOutput[jj]!;
          }
          gradInput[ii] = s;
        }

        // Apply ReLU derivative if this is a hidden layer
        if (l > 0) {
          const prevAct = activations[l]!;
          for (let ii = 0; ii < layer.inDim; ii++) {
            if (prevAct[ii]! <= 0) {
              gradInput[ii] = 0;
            }
          }
        }

        // SGD update
        for (let ii = 0; ii < layer.W.length; ii++) {
          layer.W[ii] = layer.W[ii]! - lr * gradW[ii]!;
        }
        for (let jj = 0; jj < layer.bias.length; jj++) {
          layer.bias[jj] = layer.bias[jj]! - lr * gradBias[jj]!;
        }

        gradOutput = gradInput;
      }
    }

    losses.push(epochLoss / numNodes);
  }

  // --- Step 5: Compute accuracy ---
  let correct = 0;
  for (let node = 0; node < numNodes; node++) {
    const nodeFeatures = features.subarray(
      node * featureDim,
      (node + 1) * featureDim,
    );
    const output = mlpForwardSingle(nodeFeatures, mlpWeights);
    const predicted = argmax(output);
    if (predicted === hardLabels[node]!) {
      correct++;
    }
  }

  const accuracy = numNodes > 0 ? correct / numNodes : 0;

  return { mlpWeights, losses, accuracy };
}

// ---------------------------------------------------------------------------
// 2. mlpInference — Pure MLP forward pass (no graph)
// ---------------------------------------------------------------------------

/**
 * Run MLP forward pass for all nodes. This is the inference function used
 * after distillation -- it does not need the graph at all.
 *
 * Algorithm:
 * For each node:
 *   1. Extract the node's feature vector.
 *   2. For each layer: out = activation(W^T * in + bias).
 *   3. ReLU for hidden layers, no activation for the output layer.
 *
 * @param features - Node feature matrix (numNodes * featureDim), row-major.
 * @param mlpWeights - Trained MLP weights from distillation.
 * @param numNodes - Number of nodes.
 * @param featureDim - Dimension of each node's feature vector.
 * @returns Output matrix (numNodes * outDim), row-major.
 */
export function mlpInference(
  features: Float64Array,
  mlpWeights: MLPWeights,
  numNodes: number,
  featureDim: number,
): Float64Array {
  const numLayers = mlpWeights.layers.length;
  if (numLayers === 0) return new Float64Array(0);

  const outDim = mlpWeights.layers[numLayers - 1]!.outDim;
  const result = new Float64Array(numNodes * outDim);

  for (let node = 0; node < numNodes; node++) {
    const nodeFeatures = features.subarray(
      node * featureDim,
      (node + 1) * featureDim,
    );

    const output = mlpForwardSingle(nodeFeatures, mlpWeights);

    // Copy output into result matrix
    for (let j = 0; j < outDim; j++) {
      result[node * outDim + j] = output[j]!;
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Run MLP forward pass for a single input vector.
 *
 * @param input - Input feature vector.
 * @param weights - MLP weights.
 * @returns Output vector.
 */
function mlpForwardSingle(
  input: Float64Array,
  weights: MLPWeights,
): Float64Array {
  let current = new Float64Array(input);

  for (let l = 0; l < weights.layers.length; l++) {
    const layer = weights.layers[l]!;
    // out = W^T * in + bias
    const preAct = matVecMul(
      transposeWeight(layer.W, layer.inDim, layer.outDim),
      current,
      layer.outDim,
      layer.inDim,
    );

    for (let j = 0; j < layer.outDim; j++) {
      preAct[j] = preAct[j]! + layer.bias[j]!;
    }

    // ReLU for hidden layers, no activation for output
    if (l < weights.layers.length - 1) {
      current = relu(preAct) as Float64Array<ArrayBuffer>;
    } else {
      current = preAct as Float64Array<ArrayBuffer>;
    }
  }

  return current;
}

/**
 * Transpose a weight matrix from (inDim x outDim) to (outDim x inDim).
 * Both layouts are row-major.
 */
function transposeWeight(
  W: Float64Array,
  inDim: number,
  outDim: number,
): Float64Array {
  const Wt = new Float64Array(outDim * inDim);
  for (let i = 0; i < inDim; i++) {
    for (let j = 0; j < outDim; j++) {
      Wt[j * inDim + i] = W[i * outDim + j]!;
    }
  }
  return Wt;
}
