const fs = require("fs");
const SQ = String.fromCharCode(39);
const BT = String.fromCharCode(96);
const DS = String.fromCharCode(36);
const NL = String.fromCharCode(10);

const L = [];
function a(s) { L.push(s); }
function e() { L.push(""); }

// Write lines array to file
const outPath = "c:/Users/blake/Documents/GitHub/trades-hall-omni-twin/packages/physics-solvers/src/orchestrator.ts";

// The content will be built by another script and read from stdin
const data = fs.readFileSync(0, "utf-8");
fs.writeFileSync(outPath, data);
console.log("Written " + fs.statSync(outPath).size + " bytes");
