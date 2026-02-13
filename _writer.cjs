// Auto-generated file writer - creates ScheduleGantt.tsx and ConstraintPanel.tsx
var fs = require("fs");
var Q = String.fromCharCode(39);
var BT = String.fromCharCode(96);
var DL = String.fromCharCode(36);

var ganttPath = "c:/Users/blake/Documents/GitHub/trades-hall-omni-twin/apps/web/src/app/components/solvers/ScheduleGantt.tsx";
var panelPath = "c:/Users/blake/Documents/GitHub/trades-hall-omni-twin/apps/web/src/app/components/solvers/ConstraintPanel.tsx";

// Read content templates from adjacent .template files
var ganttContent = fs.readFileSync("c:/Users/blake/Documents/GitHub/trades-hall-omni-twin/_gantt.template", "utf8");
var panelContent = fs.readFileSync("c:/Users/blake/Documents/GitHub/trades-hall-omni-twin/_panel.template", "utf8");

fs.writeFileSync(ganttPath, ganttContent);
fs.writeFileSync(panelPath, panelContent);
console.log("ScheduleGantt.tsx: " + ganttContent.length + " bytes");
console.log("ConstraintPanel.tsx: " + panelContent.length + " bytes");