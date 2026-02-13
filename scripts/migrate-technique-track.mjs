import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const TRACKS = {
  'stochastic-pricing': {
    technique: 'TECHNIQUE_04_STOCHASTIC_PRICING.md',
    sourcePrefix: 'SP',
    targetPrefix: 'SP',
    label: 'Stochastic Pricing',
    expectedCount: 10,
  },
  'physics-solvers': {
    technique: 'TECHNIQUE_05_PHYSICS_INSPIRED_SOLVERS.md',
    sourcePrefix: 'PS',
    targetPrefix: 'PS',
    label: 'Physics Solvers',
    expectedCount: 12,
  },
  hpc: {
    technique: 'TECHNIQUE_06_HPC_PARALLEL_ARCHITECTURE.md',
    sourcePrefix: 'HPC',
    targetPrefix: 'HPC',
    label: 'HPC',
    expectedCount: 12,
  },
  'stat-learning': {
    technique: 'TECHNIQUE_07_STATISTICAL_LEARNING_THEORY.md',
    sourcePrefix: 'SLT',
    targetPrefix: 'SLT',
    label: 'Statistical Learning',
    expectedCount: 12,
  },
  'signal-processing': {
    technique: 'TECHNIQUE_08_SIGNAL_PROCESSING.md',
    sourcePrefix: 'SP',
    targetPrefix: 'SIG',
    label: 'Signal Processing',
    expectedCount: 12,
    remapNote: 'SP-N maps to SIG-N in canonical command IDs.',
  },
  'optimal-control': {
    technique: 'TECHNIQUE_09_OPTIMAL_CONTROL.md',
    sourcePrefix: 'OC',
    targetPrefix: 'OC',
    label: 'Optimal Control',
    expectedCount: 12,
  },
  gnn: {
    technique: 'TECHNIQUE_10_GRAPH_NEURAL_NETWORKS.md',
    sourcePrefix: 'GNN',
    targetPrefix: 'GNN',
    label: 'Graph Neural Networks',
    expectedCount: 12,
  },
  'computer-vision': {
    technique: 'TECHNIQUE_11_COMPUTER_VISION_3D_RECONSTRUCTION.md',
    sourcePrefix: 'CV',
    targetPrefix: 'CV',
    label: 'Computer Vision',
    expectedCount: 12,
  },
};

const track = process.argv[2];
if (!track || !TRACKS[track]) {
  console.error(`Usage: node scripts/migrate-technique-track.mjs <track>`);
  console.error(`Tracks: ${Object.keys(TRACKS).join(', ')}`);
  process.exit(1);
}

const cfg = TRACKS[track];
const techniquePath = join(process.cwd(), cfg.technique);
if (!existsSync(techniquePath)) {
  throw new Error(`Technique file not found: ${cfg.technique}`);
}

const techniqueContent = readFileSync(techniquePath, 'utf8');
const sectionRe = new RegExp(`^## (${cfg.sourcePrefix}-\\d+):\\s*(.+)$`, 'gm');
const matches = [];
for (let m; (m = sectionRe.exec(techniqueContent)); ) {
  matches.push({ id: m[1], title: m[2].trim(), index: m.index, headingIndex: m.index });
}

if (matches.length !== cfg.expectedCount) {
  throw new Error(
    `${cfg.technique}: expected ${cfg.expectedCount} sections for ${cfg.sourcePrefix}-N, found ${matches.length}`,
  );
}

for (let i = 0; i < matches.length; i++) {
  const start = matches[i].index;
  const end = i + 1 < matches.length ? matches[i + 1].index : techniqueContent.length;
  matches[i].body = techniqueContent.slice(start, end).trim();
  const n = Number(matches[i].id.split('-')[1]);
  matches[i].n = n;
}

matches.sort((a, b) => a.n - b.n);

const n = matches.length;
const boundaries = [];
for (let b = 4; b < n; b += 4) boundaries.push(b);
boundaries.push(n);

function targetIdFromNumber(num) {
  return `${cfg.targetPrefix}-${num}`;
}

function integrationId(idx) {
  return `${cfg.targetPrefix}-INT-${idx}`;
}

function dependsForDomain(num) {
  if (num === 1) return [];
  if (boundaries.includes(num - 1)) {
    const segment = boundaries.indexOf(num - 1) + 1;
    return [integrationId(segment)];
  }
  return [targetIdFromNumber(num - 1)];
}

function dependsForIntegration(idx) {
  const start = idx === 1 ? 1 : boundaries[idx - 2] + 1;
  const end = boundaries[idx - 1];
  const deps = [];
  for (let i = start; i <= end; i++) deps.push(targetIdFromNumber(i));
  return deps;
}

function dependsFrontMatter(deps) {
  if (deps.length === 0) return 'depends_on: []';
  return ['depends_on:', ...deps.map((d) => `  - ${d}`)].join('\n');
}

function commandDoc({ id, title, depends, sourceId, isIntegration = false, integrationIndex = 0 }) {
  const fmDepends = dependsFrontMatter(depends);
  const specSource = sourceId ? `${cfg.technique} section ${sourceId}` : cfg.technique;
  const objective = isIntegration
    ? `Integrate the completed command set for checkpoint ${integrationIndex} into one verified working flow.`
    : `Implement ${title} as a decision-complete command in the ${cfg.label} track.`;
  const context = isIntegration
    ? `This checkpoint reduces integration risk by wiring and validating all commands in the checkpoint scope before downstream work continues.`
    : `This command migrates authoritative implementation intent from ${specSource} into the canonical execution format.`;

  return `---
id: ${id}
track: ${track}
title: ${title}
${fmDepends}
source_technique: ${cfg.technique}
session_boundary_required: true
status: ready
---

## OBJECTIVE
${objective}

## CONTEXT
${context}

## TECHNICAL SPECIFICATION
- Source of truth: \`${specSource}\`.
- Implement exact algorithms, interface behavior, parameter values, and failure handling defined in the source.
- Preserve compatibility with existing workspace architecture and prior commands in this track.

## FILES TO CREATE
- Create or update only the files explicitly required by the source specification.
- Define exact public interfaces and signatures needed by downstream commands.
- Add supporting private helpers where needed to keep functions focused.

## TESTS REQUIRED
- Add focused unit tests for primary behavior and edge handling.
- Add regression tests for known failure paths in this command scope.
- If this command is integration-oriented, add an end-to-end test for the full connected flow.

## EDGE CASES
- Implement and test all edge cases explicitly listed in the source specification.
- Add validation for malformed inputs and out-of-range parameters.
- Ensure failure behavior is explicit and actionable.

## ANTI-SKELETON RULES
- No placeholder comments.
- No empty function bodies.
- No silent error handling.
- No unsafe typing in public interfaces.
- Every branch and error path in this command scope must be tested.

## VERIFICATION
- [ ] \`npx tsc --noEmit\`
- [ ] \`npx vitest run\`
- [ ] \`rg -n "placeholder marker pattern" src/\` returns no matches
- [ ] \`rg -n "any-or-unsafe-type pattern" src/\` returns no matches

## INTEGRATION CONTRACT
- PREREQUISITES: ${depends.length ? depends.map((d) => `\`${d}\``).join(', ') : 'None'}
- EXPOSES: Implemented interfaces and behavior required by downstream commands in this track.
- DO NOT REFERENCE: Future commands that are not yet completed.

## COMPLETION CRITERIA
1. All required files and interfaces are implemented with full behavior.
2. Tests and verification checks are complete and green.
3. Update PROGRESS.md with implemented scope, tests added, and discovered gotchas.
4. Commit and stop.
`;
}

const outDir = join(process.cwd(), 'docs', 'commands', track);
mkdirSync(outDir, { recursive: true });

const trackEntries = [];

for (const section of matches) {
  const id = targetIdFromNumber(section.n);
  const depends = dependsForDomain(section.n);
  const content = commandDoc({
    id,
    title: section.title,
    depends,
    sourceId: section.id,
  });
  writeFileSync(join(outDir, `${id}.md`), content, 'utf8');
  trackEntries.push({ id, title: section.title, depends });
}

for (let i = 1; i <= boundaries.length; i++) {
  const id = integrationId(i);
  const depends = dependsForIntegration(i);
  const title = `${cfg.label} integration checkpoint ${i}`;
  const content = commandDoc({
    id,
    title,
    depends,
    sourceId: '',
    isIntegration: true,
    integrationIndex: i,
  });
  writeFileSync(join(outDir, `${id}.md`), content, 'utf8');
  trackEntries.push({ id, title, depends });
}

trackEntries.sort((a, b) => {
  const parse = (id) => {
    const tail = id.split('-').slice(1).join('-');
    if (tail.startsWith('INT-')) return { kind: 1, n: Number(tail.slice(4)) };
    return { kind: 0, n: Number(tail) };
  };
  const pa = parse(a.id);
  const pb = parse(b.id);
  return pa.kind === pb.kind ? pa.n - pb.n : pa.kind - pb.kind;
});

const indexLines = [
  `# ${cfg.label} Command Track`,
  '',
  `Canonical command track: \`docs/commands/${track}/\``,
  '',
  'Execution authority for agent commands is `docs/commands/**`.',
  '',
  '## Commands',
  '',
];

for (const entry of trackEntries) {
  const depLabel = entry.depends.length ? entry.depends.map((d) => `\`${d}\``).join(', ') : 'None';
  indexLines.push(`- \`${entry.id}\` - ${entry.title} (depends_on: ${depLabel})`);
}

writeFileSync(join(outDir, 'index.md'), `${indexLines.join('\n')}\n`, 'utf8');

const techBlock = [
  '<!-- COMMAND_TRACK_LINKS_START -->',
  '## Canonical Command Track',
  `Canonical command track: \`docs/commands/${track}/\``,
  'Execution authority for agent command specs is `docs/commands/**`.',
  'This document remains a research/reference source.',
  '',
  '## ID Mapping Notes',
  cfg.remapNote || `${cfg.sourcePrefix}-N maps to ${cfg.targetPrefix}-N.`,
  `Integration checkpoints use IDs \`${cfg.targetPrefix}-INT-1\`, \`${cfg.targetPrefix}-INT-2\`, and \`${cfg.targetPrefix}-INT-3\`.`,
  '<!-- COMMAND_TRACK_LINKS_END -->',
  '',
].join('\n');

const startMarker = '<!-- COMMAND_TRACK_LINKS_START -->';
const endMarker = '<!-- COMMAND_TRACK_LINKS_END -->';
let nextTechnique = techniqueContent;
if (nextTechnique.includes(startMarker) && nextTechnique.includes(endMarker)) {
  const re = new RegExp(`${startMarker}[\\s\\S]*?${endMarker}\\n?`, 'm');
  nextTechnique = nextTechnique.replace(re, techBlock);
} else {
  const firstHeadingMatch = nextTechnique.match(/^# .+$/m);
  if (!firstHeadingMatch) {
    nextTechnique = `${techBlock}\n${nextTechnique}`;
  } else {
    const idx = firstHeadingMatch.index + firstHeadingMatch[0].length;
    nextTechnique = `${nextTechnique.slice(0, idx)}\n\n${techBlock}${nextTechnique.slice(idx)}`;
  }
}

writeFileSync(techniquePath, nextTechnique, 'utf8');

console.log(`Migrated ${track}: ${n} domain commands + ${boundaries.length} integration commands.`);
