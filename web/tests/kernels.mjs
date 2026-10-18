// Every model's fused kernel on two devices, in headless Chromium (Playwright).
//
//   node web/tests/kernels.mjs [model-filter]
//
// For every model of the library (syn.models()), runs the same specification on WebGPU (the WGSL
// on SwiftShader) and on the library's software GPU (the same WGSL, executed by the library in
// WebAssembly across workers), and compares every number of the two results. Both run the same
// kernels with the same random numbers in 32-bit floats, so they must agree to rounding: the
// tolerance is relative, 2·10⁻³ (rounding through a whole simulation), plus an absolute 2·10⁻³
// for values near zero.
// Environment: PLAYWRIGHT_MODULE (path to playwright's index.mjs), CHROMIUM_PATH.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const filter = process.argv[2];
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright').catch(() =>
  import('/opt/node22/lib/node_modules/playwright/index.mjs'));

const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.wasm': 'application/wasm' };
const server = http.createServer((req, res) => {
  const url = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const file = path.join(root, url === '/' ? 'index.html' : url);
  if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'content-type': types[path.extname(file)] || 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const localChromium = ['/opt/pw-browsers/chromium', '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'].find((p) => fs.existsSync(p) && fs.statSync(p).isFile());
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || localChromium,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--disable-vulkan-surface'],
});
const page = await browser.newPage();
const logs = [];
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') logs.push(m.text()); });
await page.goto(`${base}/kernels.html`);

// Small specifications keep the run fast. `flips`: an absolute allowance for models whose agents
// take threshold decisions, where 32-bit rounding can flip one agent's path (1/agents per flip).
// `relative`: a wider relative tolerance for ratios of small variances (sensitivity's variance
// reduction divides by the variance of common-random-number differences, which amplifies rounding).
const SPECS = {
  default: { spec: { sim: { learners: 300, seed: 11 } } },
  equilibration: { spec: { sim: { seed: 11 } }, flips: 0.02 },
  // Cleaning up a binding under crosstalk is an argmax among near-ties.
  fgaAlgebra: { spec: { sim: { seed: 11 } }, flips: 0.01 },
  sensitivity: { spec: { sim: { learners: 300, seed: 11 } }, relative: 0.03 },
  // A student's path of lessons turns on argmax choices among near-ties (exp is approximate on GPUs).
  readingStudent: { spec: { sim: { learners: 300, seed: 11 } }, relative: 0.05 },
};

const report = await page.evaluate(async ({ specs, filter }) => {
  const { run } = await import('/js/syn-client.js');
  const { runOnGpu, runOnSoftwareGpu } = await import('/js/engine/run.js');
  const models = await run('models', {});
  const out = [];
  const leaves = (v, p, acc) => {
    if (typeof v === 'number') acc.push([p, v]);
    else if (ArrayBuffer.isView(v) || Array.isArray(v)) Array.from(v).forEach((x, i) => leaves(x, `${p}[${i}]`, acc));
    else if (v && typeof v === 'object') for (const k of Object.keys(v)) if (!['engineMs', 'adapter'].includes(k)) leaves(v[k], `${p}.${k}`, acc);
    return acc;
  };
  for (const { name } of models) {
    if (filter && !name.includes(filter)) continue;
    const { spec, flips = 0, relative = 2e-3 } = specs[name] || specs.default;
    const row = { name };
    try {
      const gpu = await runOnGpu(name, spec);
      const soft = await runOnSoftwareGpu(name, spec);
      row.gpuMs = Math.round(gpu.engineMs);
      row.softMs = Math.round(soft.engineMs);
      const G = leaves(gpu, '', []), S = leaves(soft, '', []);
      row.values = S.length;
      if (G.length !== S.length) {
        row.error = `WebGPU: ${G.length} values vs ${S.length}`;
      } else {
        let worst = { excess: 0 };
        G.forEach(([p, x], i) => {
          const y = S[i][1];
          if (!Number.isFinite(x) || !Number.isFinite(y)) { if (Number.isFinite(x) !== Number.isFinite(y)) worst = { excess: Infinity, path: p, x, y }; return; }
          const excess = Math.abs(x - y) / (2e-3 + flips + relative * Math.abs(y));
          if (excess > worst.excess) worst = { excess, path: p, x, y };
        });
        row.worst = worst;
      }
    } catch (e) {
      row.error = String(e.message || e);
    }
    out.push(row);
  }
  return out;
}, { specs: SPECS, filter });

let failed = 0;
for (const r of report) {
  const bad = r.error || !r.worst || r.worst.excess > 1;
  if (bad) failed++;
  const w = r.worst;
  const fmt = (x) => (x && x.excess > 0 ? `${x.excess.toFixed(2)} at ${x.path} (${x.x} vs ${x.y})` : 'exact');
  console.log(`${bad ? 'FAIL' : 'ok  '} ${r.name.padEnd(20)} ${r.error ? r.error : `${String(r.values).padStart(6)} values  ${fmt(w)}  (WebGPU ${r.gpuMs} ms, software GPU ${r.softMs} ms)`}`);
}
if (logs.length) console.log(logs.slice(0, 10).join('\n'));
await browser.close();
server.close();
if (failed) { console.log(`\n${failed} model(s) failed`); process.exit(1); }
console.log('\nall models agree on WebGPU and the software GPU');
