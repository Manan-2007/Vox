/**
 * Run every `*.check.ts` under src/.
 *
 * These files existed before and were never wired to anything, which meant the
 * invariants they assert — bone lengths, joint limits, spring continuity — were
 * only ever verified by whoever last remembered the incantation in their
 * docstring. A check nobody runs is a comment.
 *
 * Each check is bundled with rolldown (the app's own bundler, so imports resolve
 * exactly as they do in the browser) and run in node. A non-zero exit from any
 * one of them fails the whole run.
 *
 *     npm run check
 *     npm run check -- math          only files matching "math"
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, readdirSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const FRONTEND = dirname(dirname(fileURLToPath(import.meta.url)));
const SRC = join(FRONTEND, "src");
const filter = process.argv[2] ?? "";

function findChecks(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    if (name === "node_modules") continue;
    const path = join(dir, name);
    if (statSync(path).isDirectory()) out.push(...findChecks(path));
    else if (name.endsWith(".check.ts")) out.push(path);
  }
  return out.sort();
}

const checks = findChecks(SRC).filter((path) => path.includes(filter));
if (checks.length === 0) {
  console.error(`No checks matched ${filter ? `"${filter}"` : "src/"}`);
  process.exit(1);
}

const work = mkdtempSync(join(tmpdir(), "vox-check-"));
let failed = 0;

for (const check of checks) {
  const label = relative(FRONTEND, check);
  const bundle = join(work, `${label.replace(/[\\/]/g, "_")}.mjs`);
  console.log(`\n── ${label} ${"─".repeat(Math.max(0, 60 - label.length))}`);
  try {
    execFileSync(
      "npx",
      ["rolldown", check, "-f", "esm", "-o", bundle, "--platform", "node"],
      { cwd: FRONTEND, stdio: ["ignore", "ignore", "inherit"] },
    );
  } catch {
    console.error(`  BUILD FAILED  ${label}`);
    failed += 1;
    continue;
  }
  try {
    // Checks read fixtures relative to the frontend directory (public/signs).
    execFileSync("node", [bundle], { cwd: FRONTEND, stdio: "inherit" });
  } catch {
    failed += 1;
  }
}

rmSync(work, { recursive: true, force: true });

console.log(
  `\n${failed === 0 ? "✓ all checks passed" : `✗ ${failed} of ${checks.length} check files failed`}`,
);
process.exit(failed === 0 ? 0 : 1);
