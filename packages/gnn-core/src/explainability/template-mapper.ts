// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — Template Mapper
// Map explanation subgraphs (edge masks) to natural language descriptions
// using configurable templates. Bridges GNN explanations to human-readable text.
// ---------------------------------------------------------------------------

import type {
  Graph,
  ExplanationTemplate,
  NaturalLanguageExplanation,
} from '../types.js';
import { getEdgeIndex } from '../graph.js';

// ---- Helpers ----

/**
 * Infer the "type" of a node from its feature vector.
 *
 * For graphs with one-hot type encoding in the first few feature dimensions,
 * the type index is the argmax of those dimensions. Falls back to "unknown"
 * if the feature vector is empty or all zeros.
 *
 * @param graph - Input graph.
 * @param nodeIdx - Node index.
 * @param numTypes - Number of possible node types encoded in the features.
 * @returns Type index (0-based) or -1 if indeterminate.
 */
function inferNodeType(
  graph: Graph,
  nodeIdx: number,
  numTypes: number,
): number {
  if (graph.featureDim === 0 || numTypes <= 0) return -1;

  const typeSlots = Math.min(numTypes, graph.featureDim);
  const offset = nodeIdx * graph.featureDim;

  let bestIdx = -1;
  let bestVal = -Infinity;
  for (let t = 0; t < typeSlots; t++) {
    const val = graph.nodeFeatures[offset + t]!;
    if (val > bestVal) {
      bestVal = val;
      bestIdx = t;
    }
  }

  // Only return a type if there is a clear positive signal
  return bestVal > 0 ? bestIdx : -1;
}

/**
 * Fill a template string with specific node/edge information.
 *
 * Supported placeholders:
 *   {src}       - Source node index
 *   {dst}       - Destination node index
 *   {srcType}   - Source node type name
 *   {dstType}   - Destination node type name
 *   {edgeType}  - Edge type name
 *   {weight}    - Edge weight / importance score
 *
 * @param template - Template string with placeholders.
 * @param src - Source node index.
 * @param dst - Destination node index.
 * @param edgeType - Edge type name.
 * @param srcType - Source node type name.
 * @param dstType - Destination node type name.
 * @param weight - Edge importance weight.
 * @returns Filled template string.
 */
function fillTemplate(
  template: string,
  src: number,
  dst: number,
  edgeType: string,
  srcType: string,
  dstType: string,
  weight: number,
): string {
  return template
    .replace(/\{src\}/g, String(src))
    .replace(/\{dst\}/g, String(dst))
    .replace(/\{srcType\}/g, srcType)
    .replace(/\{dstType\}/g, dstType)
    .replace(/\{edgeType\}/g, edgeType)
    .replace(/\{weight\}/g, weight.toFixed(3));
}

// ---- Main Exports ----

/**
 * Map an edge mask to natural-language explanations using templates.
 *
 * Algorithm:
 * 1. Extract edges with mask values above the threshold.
 * 2. For each high-importance edge:
 *    a. Determine source and destination node types (if available).
 *    b. Find a matching template by edge type and/or node types.
 *    c. Fill the template with concrete node/edge information.
 * 3. Sort explanations by importance score (descending).
 * 4. Return the list of explanation strings and their scores.
 *
 * @param edgeMask - Float64Array of edge importance values (one per edge in CSR order).
 * @param graph - Input CSR graph.
 * @param templates - Array of explanation templates to match against.
 * @param threshold - Minimum mask value to consider an edge important (default 0.5).
 * @param nodeTypeNames - Optional mapping from node type index to human-readable name.
 * @param edgeTypeNames - Optional mapping from edge index to edge type name.
 * @param numNodeTypes - Number of node types encoded in node features (default 0).
 * @returns NaturalLanguageExplanation with sorted explanations and importance scores.
 */
export function edgeMaskToExplanation(
  edgeMask: Float64Array,
  graph: Graph,
  templates: ExplanationTemplate[],
  threshold = 0.5,
  nodeTypeNames?: Map<number, string>,
  edgeTypeNames?: Map<number, string>,
  numNodeTypes = 0,
): NaturalLanguageExplanation {
  const [srcArray, dstArray] = getEdgeIndex(graph);

  // Collect important edges with their scores
  const importantEdges: {
    edgeIdx: number;
    src: number;
    dst: number;
    score: number;
  }[] = [];

  for (let e = 0; e < graph.numEdges; e++) {
    if (edgeMask[e]! > threshold) {
      importantEdges.push({
        edgeIdx: e,
        src: srcArray[e]!,
        dst: dstArray[e]!,
        score: edgeMask[e]!,
      });
    }
  }

  // Sort by importance score descending
  importantEdges.sort((a, b) => b.score - a.score);

  const explanations: string[] = [];
  const importanceScores: number[] = [];

  for (const edge of importantEdges) {
    // Determine node types
    const srcTypeIdx = inferNodeType(graph, edge.src, numNodeTypes);
    const dstTypeIdx = inferNodeType(graph, edge.dst, numNodeTypes);

    const srcTypeName = nodeTypeNames?.get(srcTypeIdx) ?? `node_type_${srcTypeIdx}`;
    const dstTypeName = nodeTypeNames?.get(dstTypeIdx) ?? `node_type_${dstTypeIdx}`;
    const edgeTypeName = edgeTypeNames?.get(edge.edgeIdx) ?? 'default';

    // Find a matching template
    let matchedTemplate: ExplanationTemplate | undefined;

    // Priority 1: exact match on all three fields
    for (const tpl of templates) {
      if (
        tpl.edgeType === edgeTypeName &&
        tpl.sourceNodeType === srcTypeName &&
        tpl.targetNodeType === dstTypeName
      ) {
        matchedTemplate = tpl;
        break;
      }
    }

    // Priority 2: match on edge type only
    if (!matchedTemplate) {
      for (const tpl of templates) {
        if (tpl.edgeType === edgeTypeName) {
          matchedTemplate = tpl;
          break;
        }
      }
    }

    // Priority 3: match on source/target node types with wildcard edge type
    if (!matchedTemplate) {
      for (const tpl of templates) {
        if (
          tpl.edgeType === '*' &&
          (tpl.sourceNodeType === srcTypeName || tpl.sourceNodeType === '*') &&
          (tpl.targetNodeType === dstTypeName || tpl.targetNodeType === '*')
        ) {
          matchedTemplate = tpl;
          break;
        }
      }
    }

    // Priority 4: fallback to any wildcard template
    if (!matchedTemplate) {
      for (const tpl of templates) {
        if (tpl.edgeType === '*' && tpl.sourceNodeType === '*' && tpl.targetNodeType === '*') {
          matchedTemplate = tpl;
          break;
        }
      }
    }

    // Generate explanation text
    if (matchedTemplate) {
      const text = fillTemplate(
        matchedTemplate.template,
        edge.src,
        edge.dst,
        edgeTypeName,
        srcTypeName,
        dstTypeName,
        edge.score,
      );
      explanations.push(text);
    } else {
      // Default fallback explanation
      explanations.push(
        `Edge (${edge.src} -> ${edge.dst}) is important for the prediction (score: ${edge.score.toFixed(3)}).`,
      );
    }

    importanceScores.push(edge.score);
  }

  return { explanations, importanceScores };
}

/**
 * Return pre-defined explanation templates for the venue recommendation domain.
 *
 * These templates cover common relationship types in venue/event knowledge graphs:
 * - venue <-> event_type relationships
 * - venue <-> location proximity
 * - planner <-> venue preference history
 * - venue <-> amenity associations
 * - venue <-> capacity constraints
 * - temporal/seasonal patterns
 *
 * Each template uses {src}, {dst}, {srcType}, {dstType}, and {weight} placeholders
 * that are filled at explanation time.
 *
 * @returns Array of ExplanationTemplate objects for the venue domain.
 */
export function defaultVenueTemplates(): ExplanationTemplate[] {
  return [
    {
      edgeType: 'hosted_event_type',
      sourceNodeType: 'venue',
      targetNodeType: 'event_type',
      template:
        'Venue {src} has previously hosted events of type "{dstType}" (importance: {weight}), making it a strong match for this event.',
    },
    {
      edgeType: 'similar_event_type',
      sourceNodeType: 'venue',
      targetNodeType: 'event_type',
      template:
        'Venue {src} is suited for event types similar to "{dstType}" (importance: {weight}).',
    },
    {
      edgeType: 'nearby',
      sourceNodeType: 'venue',
      targetNodeType: 'venue',
      template:
        'Nearby venue {dst} was successful for similar events, suggesting venue {src} is also a good choice (importance: {weight}).',
    },
    {
      edgeType: 'preferred_by',
      sourceNodeType: 'planner',
      targetNodeType: 'venue',
      template:
        'Planners with similar preferences to planner {src} also chose venue {dst} (importance: {weight}).',
    },
    {
      edgeType: 'booked_by',
      sourceNodeType: 'venue',
      targetNodeType: 'planner',
      template:
        'Venue {src} was previously booked by planner {dst} with a positive outcome (importance: {weight}).',
    },
    {
      edgeType: 'has_amenity',
      sourceNodeType: 'venue',
      targetNodeType: 'amenity',
      template:
        'Venue {src} offers amenity "{dstType}" which is relevant for this event (importance: {weight}).',
    },
    {
      edgeType: 'capacity_match',
      sourceNodeType: 'venue',
      targetNodeType: 'event',
      template:
        'Venue {src} has capacity matching the expected attendance for event {dst} (importance: {weight}).',
    },
    {
      edgeType: 'seasonal_fit',
      sourceNodeType: 'venue',
      targetNodeType: 'time_slot',
      template:
        'Venue {src} performs well during the requested time period (importance: {weight}).',
    },
    {
      edgeType: 'price_range',
      sourceNodeType: 'venue',
      targetNodeType: 'budget',
      template:
        'Venue {src} fits within the specified budget range (importance: {weight}).',
    },
    {
      edgeType: 'style_match',
      sourceNodeType: 'venue',
      targetNodeType: 'style',
      template:
        'Venue {src} matches the desired style or aesthetic for this event (importance: {weight}).',
    },
    // Wildcard fallback template
    {
      edgeType: '*',
      sourceNodeType: '*',
      targetNodeType: '*',
      template:
        'The connection between {srcType} (node {src}) and {dstType} (node {dst}) is relevant to the recommendation (importance: {weight}).',
    },
  ];
}
