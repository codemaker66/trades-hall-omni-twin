import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const ROOT = process.cwd();
const COMMAND_ROOT = join(ROOT, 'docs', 'commands');
const ID_RE = /^(SP|PS|HPC|SLT|SIG|OC|GNN|CV)-([0-9]+|INT-[0-9]+)$/;
const COMMAND_FILE_RE = /^(SP|PS|HPC|SLT|SIG|OC|GNN|CV)-(?:[0-9]+|INT-[0-9]+)\.md$/;

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
  if (!m) throw new Error(`${file}: missing YAML front matter`);
  const lines = m[1].split('\n');
  const data = {};
  let currentArrayKey = null;

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    if (!line.trim()) continue;

    if (/^\s*-\s+/.test(line)) {
      if (!currentArrayKey) throw new Error(`${file}: array item without key`);
      data[currentArrayKey].push(line.replace(/^\s*-\s+/, '').trim());
      continue;
    }

    const idx = line.indexOf(':');
    if (idx === -1) throw new Error(`${file}: invalid front matter line '${line}'`);

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
      data[key] = value;
    }
  }

  return data;
}

function parseId(id) {
  const m = id.match(ID_RE);
  if (!m) return null;
  const prefix = m[1];
  const tail = m[2];
  if (tail.startsWith('INT-')) {
    return { prefix, kind: 'int', n: Number(tail.slice(4)) };
  }
  return { prefix, kind: 'domain', n: Number(tail) };
}

function buildExpectedOrder(domainCount) {
  const boundaries = [];
  for (let b = 4; b < domainCount; b += 4) {
    boundaries.push(b);
  }
  boundaries.push(domainCount);

  const order = [];
  let start = 1;
  for (let i = 0; i < boundaries.length; i++) {
    const end = boundaries[i];
    for (let n = start; n <= end; n++) {
      order.push({ kind: 'domain', n });
    }
    order.push({ kind: 'int', n: i + 1 });
    start = end + 1;
  }

  return order;
}

function expectedDepends(domainCount, kind, n, prefix) {
  const boundaries = [];
  for (let b = 4; b < domainCount; b += 4) {
    boundaries.push(b);
  }
  boundaries.push(domainCount);

  if (kind === 'domain') {
    if (n === 1) return [];
    if (boundaries.includes(n - 1)) {
      const segment = boundaries.indexOf(n - 1) + 1;
      return [`${prefix}-INT-${segment}`];
    }
    return [`${prefix}-${n - 1}`];
  }

  const segmentIdx = n - 1;
  const start = segmentIdx === 0 ? 1 : boundaries[segmentIdx - 1] + 1;
  const end = boundaries[segmentIdx];
  if (!Number.isFinite(start) || !Number.isFinite(end)) return [];
  const deps = [];
  for (let i = start; i <= end; i++) {
    deps.push(`${prefix}-${i}`);
  }
  return deps;
}

const files = walk(COMMAND_ROOT);
const errors = [];
const nodes = new Map();
const byTrack = new Map();

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

  const id = String(fm.id || '');
  if (!ID_RE.test(id)) {
    errors.push(`${rel}: invalid id '${id}'`);
    continue;
  }

  if (nodes.has(id)) {
    errors.push(`${rel}: duplicate id '${id}' also seen in ${nodes.get(id).file}`);
    continue;
  }

  const track = String(fm.track || '');
  const depends = Array.isArray(fm.depends_on) ? fm.depends_on.map(String) : [];
  nodes.set(id, { id, track, depends, file: rel });

  const arr = byTrack.get(track) ?? [];
  arr.push(id);
  byTrack.set(track, arr);
}

for (const node of nodes.values()) {
  for (const dep of node.depends) {
    if (!nodes.has(dep)) {
      errors.push(`${node.file}: dependency '${dep}' does not exist`);
    }
  }
}

for (const [track, ids] of byTrack.entries()) {
  const parsed = ids.map((id) => ({ id, parsed: parseId(id) })).filter((x) => x.parsed);
  if (parsed.length === 0) continue;

  const prefix = parsed[0].parsed.prefix;
  const domain = parsed.filter((x) => x.parsed.kind === 'domain').sort((a, b) => a.parsed.n - b.parsed.n);
  const ints = parsed.filter((x) => x.parsed.kind === 'int').sort((a, b) => a.parsed.n - b.parsed.n);

  const n = domain.length;
  if (!(n === 10 || n === 12)) {
    errors.push(`track '${track}': expected 10 or 12 domain commands, found ${n}`);
    continue;
  }
  if (ints.length !== 3) {
    errors.push(`track '${track}': expected 3 integration commands, found ${ints.length}`);
    continue;
  }

  for (let i = 1; i <= n; i++) {
    const expected = `${prefix}-${i}`;
    if (!domain.find((d) => d.id === expected)) {
      errors.push(`track '${track}': missing domain command '${expected}'`);
    }
  }
  for (let i = 1; i <= 3; i++) {
    const expected = `${prefix}-INT-${i}`;
    if (!ints.find((d) => d.id === expected)) {
      errors.push(`track '${track}': missing integration command '${expected}'`);
    }
  }

  const orderItems = buildExpectedOrder(n);
  const order = orderItems.map((item) =>
    item.kind === 'domain' ? `${prefix}-${item.n}` : `${prefix}-INT-${item.n}`,
  );
  const position = new Map(order.map((id, idx) => [id, idx]));

  for (const id of order) {
    const node = nodes.get(id);
    if (!node) continue;

    const p = parseId(id);
    const expected = expectedDepends(n, p.kind, p.n, prefix).sort();
    const actual = [...node.depends].sort();
    if (expected.join('|') !== actual.join('|')) {
      errors.push(
        `${node.file}: depends_on mismatch for ${id}; expected [${expected.join(', ')}], got [${actual.join(', ')}]`,
      );
    }

    const currentPos = position.get(id);
    for (const dep of node.depends) {
      const depPos = position.get(dep);
      if (depPos === undefined) continue;
      if (depPos >= currentPos) {
        errors.push(`${node.file}: forward dependency '${dep}' is not allowed`);
      }
    }
  }
}

const visiting = new Set();
const visited = new Set();
function dfs(id, stack) {
  if (visiting.has(id)) {
    errors.push(`cyclic dependency detected: ${[...stack, id].join(' -> ')}`);
    return;
  }
  if (visited.has(id)) return;

  visiting.add(id);
  const node = nodes.get(id);
  if (node) {
    for (const dep of node.depends) {
      if (nodes.has(dep)) dfs(dep, [...stack, id]);
    }
  }
  visiting.delete(id);
  visited.add(id);
}

for (const id of nodes.keys()) {
  dfs(id, []);
}

if (errors.length > 0) {
  console.error('Command dependency validation failed:\n');
  for (const e of errors) console.error(`- ${e}`);
  process.exit(1);
}

console.log(`Command dependency validation passed for ${nodes.size} commands.`);
