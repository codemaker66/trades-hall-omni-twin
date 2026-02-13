const fs = require('fs');
const dir = 'c:/Users/blake/Documents/GitHub/trades-hall-omni-twin/apps/web/src/app/components/solvers';
const b64 = process.argv[2];
const content = Buffer.from(b64, 'base64').toString('utf8');
const outFile = dir + '/' + process.argv[3];
fs.writeFileSync(outFile, content);
console.log(process.argv[3] + ': ' + content.length + ' bytes');