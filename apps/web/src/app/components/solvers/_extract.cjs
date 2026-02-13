// Self-extracting script: content after __DATA__ marker is the file content
const fs = require('fs');
const dir = 'c:/Users/blake/Documents/GitHub/trades-hall-omni-twin/apps/web/src/app/components/solvers';
const selfContent = fs.readFileSync(__filename, 'utf8');
const marker = '__DATA__';
const idx = selfContent.indexOf(marker);
if (idx === -1) { console.error('No __DATA__ marker found'); process.exit(1); }
const data = selfContent.substring(idx + marker.length + 1);
const parts = data.split('
===FILE_BOUNDARY===
');
if (parts.length >= 1) {
  fs.writeFileSync(dir + '/EnergyLandscape.tsx', parts[0]);
  console.log('EnergyLandscape.tsx: ' + parts[0].length + ' bytes');
}
if (parts.length >= 2) {
  fs.writeFileSync(dir + '/TemperatureViz.tsx', parts[1]);
  console.log('TemperatureViz.tsx: ' + parts[1].length + ' bytes');
}
// Clean up temp files
try { fs.unlinkSync(dir + '/_test.txt'); } catch(e) {}
try { fs.unlinkSync(dir + '/_test2.txt'); } catch(e) {}
try { fs.unlinkSync(dir + '/_write_files.js'); } catch(e) {}
try { fs.unlinkSync(dir + '/_gen.js'); } catch(e) {}
try { fs.unlinkSync(dir + '/_gen_test.cjs'); } catch(e) {}
try { fs.unlinkSync(dir + '/_write1.ps1'); } catch(e) {}
try { fs.unlinkSync(dir + '/_b64writer.cjs'); } catch(e) {}
try { fs.unlinkSync(dir + '/_gen_energy2.cjs'); } catch(e) {}
try { fs.unlinkSync(dir + '/_stdin_writer.cjs'); } catch(e) {}
try { fs.unlinkSync(dir + '/_energy.b64'); } catch(e) {}
try { fs.unlinkSync(dir + '/_temp.b64'); } catch(e) {}
try { fs.unlinkSync(__filename); } catch(e) {}
console.log('Done. Temp files cleaned up.');

// Everything below this line is file content
process.exit(0);
__DATA__