// Cross-engine check of every model's fused kernel in headless Chromium (Playwright).
//
//   node web/tests/kernels.mjs [model-filter]
//
// For every model of the library (syn.models()), runs the same specification on WebGPU (WGSL on
// SwiftShader), on the CPU kernels (32-bit floats in workers) and on WebAssembly (64-bit floats),
// and compares every number of the three results. The engines share their random numbers, so
// they must agree to rounding: the tolerance is relative, 2·10⁻³ (32- vs 64-bit floats through
// a whole simulation), plus an absolute 2·10⁻³ for values near zero.
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

// Small specifications keep the run fast.
const SPECS = {
  default: { sim: { learners: 300, seed: 11 } },
};

const report = await page.evaluate(async ({ specs, filter }) => {
  const { run } = await import('/js/syn-client.js');
  const { runOnGpu, runOnCpuKernels, runOnWasm } = await import('/js/engine/run.js');
  const models = await run('models', {});
  const out = [];
  const leaves = (v, p, acc) => {
    if (typeof v === 'number') acc.push([p, v]);
    else if (ArrayBuffer.isView(v) || Array.isArray(v)) Array.from(v).forEach((x, i) => leaves(x, `${p}[${i}]`, acc));
    else if (v && typeof v === 'object') for (const k of Object.keys(v)) if (!['engineMs', 'adapter', 'plans', 'jobs', 'workers', 'elapsedMs'].includes(k)) leaves(v[k], `${p}.${k}`, acc);
    return acc;
  };
  for (const { name } of models) {
    if (filter && !name.includes(filter)) continue;
    const spec = specs[name] || specs.default;
    const row = { name };
    try {
      const t0 = performance.now();
      const gpu = await runOnGpu(name, spec);
      const t1 = performance.now();
      const cpu = await runOnCpuKernels(name, spec);
      const wasm = await runOnWasm(name, spec);
      row.gpuMs = Math.round(t1 - t0);
      const L = { gpu: leaves(gpu, '', []), cpu: leaves(cpu, '', []), wasm: leaves(wasm, '', []) };
      row.values = L.wasm.length;
      row.worst = {};
      for (const e of ['gpu', 'cpu']) {
        if (L[e].length !== L.wasm.length) { row.error = `${e}: ${L[e].length} values vs ${L.wasm.length}`; break; }
        let worst = { excess: 0 };
        L[e].forEach(([p, x], i) => {
          const y = L.wasm[i][1];
          if (!Number.isFinite(x) || !Number.isFinite(y)) { if (Number.isFinite(x) !== Number.isFinite(y)) worst = { excess: Infinity, path: p, x, y }; return; }
          const excess = Math.abs(x - y) / (2e-3 + 2e-3 * Math.abs(y));
          if (excess > worst.excess) worst = { excess, path: p, x, y };
        });
        row.worst[e] = worst;
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
  const bad = r.error || Object.values(r.worst || {}).some((w) => w.excess > 1);
  if (bad) failed++;
  const w = r.worst || {};
  const fmt = (x) => (x && x.excess > 0 ? `${x.excess.toFixed(2)} at ${x.path} (${x.x} vs ${x.y})` : 'exact');
  console.log(`${bad ? 'FAIL' : 'ok  '} ${r.name.padEnd(20)} ${r.error ? r.error : `${String(r.values).padStart(6)} values  gpu ${fmt(w.gpu)}  cpu ${fmt(w.cpu)}  (${r.gpuMs} ms)`}`);
}
if (logs.length) console.log(logs.slice(0, 10).join('\n'));
await browser.close();
server.close();
if (failed) { console.log(`\n${failed} model(s) failed`); process.exit(1); }
console.log('\nall models agree across engines');
