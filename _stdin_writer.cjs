const fs = require('fs');
const chunks = [];
process.stdin.on('data', d => chunks.push(d));
process.stdin.on('end', () => {
  const content = Buffer.concat(chunks).toString('utf8');
  const [gantt, panel] = content.split('===SPLIT===');
  fs.writeFileSync('c:/Users/blake/Documents/GitHub/trades-hall-omni-twin/apps/web/src/app/components/solvers/ScheduleGantt.tsx', gantt.trim());
  fs.writeFileSync('c:/Users/blake/Documents/GitHub/trades-hall-omni-twin/apps/web/src/app/components/solvers/ConstraintPanel.tsx', panel.trim());
  console.log('ScheduleGantt.tsx: ' + gantt.trim().length + ' bytes');
  console.log('ConstraintPanel.tsx: ' + panel.trim().length + ' bytes');
});
