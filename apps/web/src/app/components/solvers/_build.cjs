const fs=require("fs");const p="c:/Users/blake/Documents/GitHub/trades-hall-omni-twin/apps/web/src/app/components/solvers";const c=[];function a(l){c.push(l);}function w(f){fs.writeFileSync(p+"/"+f,c.join("
"));console.log(f+": "+c.length+" lines, "+c.join("
").length+" bytes");c.length=0;}
