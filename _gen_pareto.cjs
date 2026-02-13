const fs = require("fs");
const outPath = "c:/Users/blake/Documents/GitHub/trades-hall-omni-twin/apps/web/src/app/components/solvers/ParetoDashboard.tsx";

// Character constants for template literals in output
const BT = String.fromCharCode(96);  // backtick
const DS = String.fromCharCode(36);  // dollar sign

// tl: wrap content in backticks to form template literal
function tl(s) { return BT + s + BT; }
// te: template expression ${...}
function te(e) { return DS + "{" + e + "}"; }

const c = [];

c.push("'use client'");
c.push("");
c.push("import { useMemo, useState, useCallback, type MouseEvent } from 'react'");
c.push("");
c.push("// ---------------------------------------------------------------------------");
c.push("// Types");
c.push("// ---------------------------------------------------------------------------");
c.push("");
