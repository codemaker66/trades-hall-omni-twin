const fs = require("fs");
let data = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", chunk => data += chunk);
process.stdin.on("end", () => {
  const outPath = process.argv[2];
  fs.writeFileSync(outPath, data);
  console.log(outPath + ": " + data.length + " bytes written");
});