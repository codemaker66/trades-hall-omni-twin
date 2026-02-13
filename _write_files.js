const fs = require("fs");
const path1 = "c:/Users/blake/Documents/GitHub/trades-hall-omni-twin/apps/web/src/app/components/solvers/ScheduleGantt.tsx";
const path2 = "c:/Users/blake/Documents/GitHub/trades-hall-omni-twin/apps/web/src/app/components/solvers/ConstraintPanel.tsx";
const b641 = fs.readFileSync("c:/Users/blake/Documents/GitHub/trades-hall-omni-twin/_gantt.b64", "utf8");
const b642 = fs.readFileSync("c:/Users/blake/Documents/GitHub/trades-hall-omni-twin/_panel.b64", "utf8");
fs.writeFileSync(path1, Buffer.from(b641, "base64").toString("utf8"));
fs.writeFileSync(path2, Buffer.from(b642, "base64").toString("utf8"));
console.log("Both files written successfully");