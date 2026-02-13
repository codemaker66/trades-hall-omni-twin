const fs = require("fs");
const dir = "c:/Users/blake/Documents/GitHub/trades-hall-omni-twin/apps/web/src/app/components/solvers";
const b64 = fs.readFileSync(dir + "/_energy.b64", "utf8");
const content = Buffer.from(b64, "base64").toString("utf8");
fs.writeFileSync(dir + "/EnergyLandscape.tsx", content);
console.log("EnergyLandscape.tsx written:", content.length, "bytes");

const b64b = fs.readFileSync(dir + "/_temp.b64", "utf8");
const contentB = Buffer.from(b64b, "base64").toString("utf8");
fs.writeFileSync(dir + "/TemperatureViz.tsx", contentB);
console.log("TemperatureViz.tsx written:", contentB.length, "bytes");