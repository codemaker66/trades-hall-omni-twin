import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

const ROOT = process.cwd();
const COMMAND_ROOT = join(ROOT, 'docs', 'commands');

const TECHNIQUE_MAP = [
  {
    technique: 'TECHNIQUE_04_STOCHASTIC_PRICING.md',
    track: 'stochastic-pricing',
    sourcePrefix: 'SP',
    targetPrefix: 'SP',
  },
  {
    technique: 'TECHNIQUE_05_PHYSICS_INSPIRED_SOLVERS.md',
    track: 'physics-solvers',
    sourcePrefix: 'PS',
    targetPrefix: 'PS',
  },
  {
    technique: 'TECHNIQUE_06_HPC_PARALLEL_ARCHITECTURE.md',
    track: 'hpc',
    sourcePrefix: 'HPC',
    targetPrefix: 'HPC',
  },
  {
    technique: 'TECHNIQUE_07_STATISTICAL_LEARNING_THEORY.md',
    track: 'stat-learning',
    sourcePrefix: 'SLT',
    targetPrefix: 'SLT',
  },
  {
    technique: 'TECHNIQUE_08_SIGNAL_PROCESSING.md',
    track: 'signal-processing',
    sourcePrefix: 'SP',
    targetPrefix: 'SIG',
  },
  {
    technique: 'TECHNIQUE_09_OPTIMAL_CONTROL.md',
    track: 'optimal-control',
    sourcePrefix: 'OC',
    targetPrefix: 'OC',
  },
  {
    technique: 'TECHNIQUE_10_GRAPH_NEURAL_NETWORKS.md',
    track: 'gnn',
    sourcePrefix: 'GNN',
    targetPrefix: 'GNN',
  },
  {
    technique: 'TECHNIQUE_11_COMPUTER_VISION_3D_RECONSTRUCTION.md',
    track: 'computer-vision',
    sourcePrefix: 'CV',
    targetPrefix: 'CV',
  },
];

function walk(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const st = statSync(full);
    if (st.isDirectory()) {
      if (entry === '_templates') continue;
      out.push(...walk(full));
    } else if (st.isFile() && entry.endsWith('.md') && entry !== 'index.md') {
      out.push(full);
    }
  }
  return out;
}

function parseFrontMatter(content) {
  const m = content.match(/^---\n([\s\S]*?)\n---\n?/);
  if (!m) return null;
  const data = {};
  let currentArrayKey = null;

  for (const rawLine of m[1].split('\n')) {
    const line = rawLine.trimEnd();
    if (!line.trim()) continue;

    if (/^\s*-\s+/.test(line)) {
      if (!currentArrayKey) continue;
      data[currentArrayKey].push(line.replace(/^\s*-\s+/, '').trim());
      continue;
    }

    const idx = line.indexOf(':');
    if (idx === -1) continue;

    const key = line.slice(0, idx).trim();
    const value = line.slice(idx + 1).trim();
    if (value === '') {
      data[key] = [];
      currentArrayKey = key;
    } else {
      currentArrayKey = null;
      data[key] = value;
    }
  }

  return data;
}

const commandFiles = walk(COMMAND_ROOT);
const commandMeta = commandFiles.map((file) => {
  const content = readFileSync(file, 'utf8');
  const fm = parseFrontMatter(content) || {};
  return {
    file,
    id: String(fm.id || ''),
    track: String(fm.track || ''),
    sourceTechnique: String(fm.source_technique || ''),
  };
});

const errors = [];

for (const mapping of TECHNIQUE_MAP) {
  const techniquePath = join(ROOT, mapping.technique);
  if (!existsSync(techniquePath)) {
    errors.push(`${mapping.technique}: technique file not found`);
    continue;
  }

  const techniqueContent = readFileSync(techniquePath, 'utf8');
  const trackPointer = `Canonical command track: \`docs/commands/${mapping.track}/\``;
  if (!techniqueContent.includes(trackPointer)) {
    errors.push(`${mapping.technique}: missing canonical track pointer '${trackPointer}'`);
  }
  if (!/## ID Mapping Notes/.test(techniqueContent)) {
    errors.push(`${mapping.technique}: missing '## ID Mapping Notes' section`);
  }

  const sectionRe = /^## ([A-Z]+-\d+): /gm;
  const sectionIds = [];
  for (let m; (m = sectionRe.exec(techniqueContent)); ) {
    sectionIds.push(m[1]);
  }

  const filtered = sectionIds.filter((id) => id.startsWith(`${mapping.sourcePrefix}-`));
  for (const id of filtered) {
    const suffix = id.split('-')[1];
    const mappedId = `${mapping.targetPrefix}-${suffix}`;
    const commandPath = join(COMMAND_ROOT, mapping.track, `${mappedId}.md`);
    if (!existsSync(commandPath)) {
      errors.push(`${mapping.technique}: missing mapped command '${mapping.track}/${mappedId}.md'`);
    }

    const meta = commandMeta.find((m) => m.id === mappedId && m.track === mapping.track);
    if (!meta) {
      errors.push(`${mapping.technique}: command metadata not found for id '${mappedId}' in track '${mapping.track}'`);
    } else if (meta.sourceTechnique !== mapping.technique) {
      errors.push(`${meta.file}: source_technique must be '${mapping.technique}'`);
    }
  }
}

for (const meta of commandMeta) {
  if (meta.id.startsWith('SP-') && meta.track !== 'stochastic-pricing') {
    errors.push(`${meta.file}: SP-* id is only allowed in stochastic-pricing track`);
  }
  if (meta.id.startsWith('SIG-') && meta.track !== 'signal-processing') {
    errors.push(`${meta.file}: SIG-* id is only allowed in signal-processing track`);
  }
}

if (errors.length > 0) {
  console.error('Technique/track link validation failed:\n');
  for (const e of errors) console.error(`- ${e}`);
  process.exit(1);
}

console.log('Technique/track link validation passed.');
