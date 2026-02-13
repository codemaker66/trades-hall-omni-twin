const fs = require("fs");
const dir = "c:/Users/blake/Documents/GitHub/trades-hall-omni-twin/apps/web/src/app/components/solvers";

// Read content from stdin or from a separate data file
const args = process.argv.slice(2);
const file = args[0];
const dataFile = args[1];
const content = fs.readFileSync(dataFile, "utf8");
fs.writeFileSync(dir + "/" + file, content);
console.log(file + " written: " + content.length + " bytes");