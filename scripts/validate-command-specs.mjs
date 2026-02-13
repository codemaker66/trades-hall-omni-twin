import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const ROOT = process.cwd();
const COMMAND_ROOT = join(ROOT, 'docs', 'commands');
const TRACKS = new Set([
  'stochastic-pricing',
  'physics-solvers',
  'hpc',
  'stat-learning',
  'signal-processing',
  'optimal-control',
  'gnn',
  'computer-vision',
]);
const ID_RE = /^(SP|PS|HPC|SLT|SIG|OC|GNN|CV)-([0-9]+|INT-[0-9]+)$/;
const STATUS = new Set(['draft', 'ready', 'done']);
const COMMAND_FILE_RE = /^(SP|PS|HPC|SLT|SIG|OC|GNN|CV)-(?:[0-9]+|INT-[0-9]+)\.md$/;
const REQUIRED_HEADINGS = [
  '## OBJECTIVE',
  '## CONTEXT',
  '## TECHNICAL SPECIFICATION',
  '## FILES TO CREATE',
  '## TESTS REQUIRED',
  '## EDGE CASES',
  '## ANTI-SKELETON RULES',
  '## VERIFICATION',
  '## INTEGRATION CONTRACT',
  '## COMPLETION CRITERIA',
];

function walk(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const st = statSync(full);
    if (st.isDirectory()) {
      if (entry === '_templates') continue;
      out.push(...walk(full));
    } else if (st.isFile() && COMMAND_FILE_RE.test(entry)) {
      out.push(full);
    }
  }
  return out;
}

function parseFrontMatter(content, file) {
  const m = content.match(/^---\n([\s\S]*?)\n---\n?/);
  if (!m) {
    throw new Error(`${file}: missing YAML front matter`);
  }
  const fm = m[1];
  const data = {};
  let currentArrayKey = null;

  for (const rawLine of fm.split('\n')) {
    const line = rawLine.trimEnd();
    if (!line.trim()) continue;

    if (/^\s*-\s+/.test(line)) {
      if (!currentArrayKey) {
        throw new Error(`${file}: array item found without key`);
      }
      data[currentArrayKey].push(line.replace(/^\s*-\s+/, '').trim());
      continue;
    }

    const idx = line.indexOf(':');
    if (idx === -1) {
      throw new Error(`${file}: invalid front matter line '${line}'`);
    }
    const key = line.slice(0, idx).trim();
    const value = line.slice(idx + 1).trim();

    if (value === '[]') {
      data[key] = [];
      currentArrayKey = null;
    } else if (value === '') {
      data[key] = [];
      currentArrayKey = key;
    } else {
      currentArrayKey = null;
      if (value === 'true') {
        data[key] = true;
      } else if (value === 'false') {
        data[key] = false;
      } else {
        data[key] = value;
      }
    }
  }

  return data;
}

function getSection(content, heading) {
  const escaped = heading.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(`${escaped}\\n([\\s\\S]*?)(?:\\n## |$)`);
  const m = content.match(re);
  return m ? m[1] : '';
}

const files = walk(COMMAND_ROOT);
const errors = [];

for (const file of files) {
  const rel = relative(ROOT, file).replace(/\\/g, '/');
  const content = readFileSync(file, 'utf8');
  let fm;

  try {
    fm = parseFrontMatter(content, rel);
  } catch (err) {
    errors.push(String(err.message || err));
    continue;
  }

  if (!ID_RE.test(String(fm.id || ''))) {
    errors.push(`${rel}: invalid id '${fm.id ?? ''}'`);
  }
  if (!TRACKS.has(String(fm.track || ''))) {
    errors.push(`${rel}: invalid track '${fm.track ?? ''}'`);
  }
  if (!String(fm.title || '').trim()) {
    errors.push(`${rel}: title must be non-empty`);
  }
  if (!Array.isArray(fm.depends_on)) {
    errors.push(`${rel}: depends_on must be an array`);
  }
  if (!String(fm.source_technique || '').trim()) {
    errors.push(`${rel}: source_technique must be set`);
  }
  if (fm.session_boundary_required !== true) {
    errors.push(`${rel}: session_boundary_required must be true`);
  }
  if (!STATUS.has(String(fm.status || ''))) {
    errors.push(`${rel}: status must be one of draft|ready|done`);
  }

  for (const heading of REQUIRED_HEADINGS) {
    if (!content.includes(heading)) {
      errors.push(`${rel}: missing required heading '${heading}'`);
    }
  }

  const verification = getSection(content, '## VERIFICATION');
  if (!/(tsc|typecheck|noemit)/i.test(verification)) {
    errors.push(`${rel}: VERIFICATION is missing a typecheck command`);
  }
  if (!/(vitest|jest|pytest|npm run test|pnpm test|turbo run test)/i.test(verification)) {
    errors.push(`${rel}: VERIFICATION is missing a test command`);
  }
  if (!/placeholder/i.test(verification)) {
    errors.push(`${rel}: VERIFICATION is missing anti-placeholder grep check`);
  }
  if (!/\bany\b|unsafe/i.test(verification)) {
    errors.push(`${rel}: VERIFICATION is missing anti-any check`);
  }

  if (!/Commit and stop\./.test(content)) {
    errors.push(`${rel}: required session boundary phrase 'Commit and stop.' not found`);
  }
  if (!/PROGRESS\.md/.test(content)) {
    errors.push(`${rel}: required PROGRESS.md update language not found`);
  }

  if (/\b(TODO|FIXME|HACK|TBD)\b/i.test(content)) {
    errors.push(`${rel}: contains forbidden placeholder marker`);
  }
  if (/^\s*pass\s*$/m.test(content)) {
    errors.push(`${rel}: contains forbidden standalone 'pass' placeholder`);
  }
}

if (errors.length > 0) {
  console.error('Command spec validation failed:\n');
  for (const e of errors) {
    console.error(`- ${e}`);
  }
  process.exit(1);
}

console.log(`Command spec validation passed for ${files.length} files.`);
