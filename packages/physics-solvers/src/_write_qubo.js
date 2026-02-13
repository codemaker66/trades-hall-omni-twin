const fs = require("fs");
const path = process.argv[2];
const content = fs.readFileSync(0, "utf8");
fs.writeFileSync(path, content);
console.log("Written", content.length, "bytes");
