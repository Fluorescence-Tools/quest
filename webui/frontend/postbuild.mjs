import fs from "node:fs";
import path from "node:path";

const nextDir = path.resolve(".next");
const standaloneRoot = path.join(nextDir, "standalone");

function copyRecursive(src, dest) {
  if (!fs.existsSync(src)) return;
  const stat = fs.statSync(src);
  if (stat.isDirectory()) {
    fs.mkdirSync(dest, { recursive: true });
    for (const child of fs.readdirSync(src)) {
      copyRecursive(path.join(src, child), path.join(dest, child));
    }
    return;
  }
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.copyFileSync(src, dest);
}

copyRecursive(path.join(nextDir, "static"), path.join(standaloneRoot, ".next", "static"));
copyRecursive(path.resolve("public"), path.join(standaloneRoot, "public"));
console.log("Next standalone assets copied.");
