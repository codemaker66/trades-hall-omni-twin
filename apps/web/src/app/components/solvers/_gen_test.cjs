const fs = require('fs');
const dir = 'c:/Users/blake/Documents/GitHub/trades-hall-omni-twin/apps/web/src/app/components/solvers';

const BT = String.fromCharCode(96);
const SQ = String.fromCharCode(39);
const DQ = String.fromCharCode(34);
const DS = String.fromCharCode(36);
const NL = String.fromCharCode(10);
const BS = String.fromCharCode(92);

const w = [];
function a(s) { w.push(s === void 0 ? '' : s); }

// ---- Begin EnergyLandscape.tsx content ----
a(SQ + 'use client' + SQ);
a('');
a('console.log("test");');

const content = w.join(NL);
fs.writeFileSync(dir + '/EnergyLandscape.tsx', content);
console.log('Written: ' + content.length + ' bytes');