// End-to-end test of the web front end in headless Chromium (Playwright).
//
//   node web/tests/e2e.mjs [page-filter] [--root DIR] [--file]
//
// Serves web/ (or --root DIR, e.g. dist/standalone) on a local port, or with --file opens the
// pages straight from disk as file:// URLs (the copy-and-open deployment of the standalone
// build, which must then make no network request at all). Opens every page, waits for its
// automatic run and checks that it finished without errors and rendered charts. Every model
// runs as a fused WGSL kernel: on WebGPU when there is an adapter, else on the library's software
// GPU (the same WGSL in WebAssembly across workers). Both must agree to rounding.
// WebGPU runs on SwiftShader in headless mode, so this works without a GPU.
// Environment: PLAYWRIGHT_MODULE (path to playwright's index.mjs), CHROMIUM_PATH.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const argv = process.argv.slice(2);
const option = (name) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : undefined; };
const fileMode = argv.includes('--file');
const root = path.resolve(option('--root') || path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..'));
const filter = argv.find((a, i) => !a.startsWith('--') && argv[i - 1] !== '--root');
const { chromium, devices } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright').catch(() =>
  import('/opt/node22/lib/node_modules/playwright/index.mjs'));

const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.wasm': 'application/wasm', '.png': 'image/png', '.json': 'application/json' };
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

// Small learner counts keep the run fast; the pages pick the spec up from localStorage.
const testSpec = { sim: { learners: 200, seed: 11 } };
const pairSpec = { sim: { learners: 3000, seed: 11 } };
const pages = ['index', 'core', 'behaviorism', 'humanism', 'cognitivism', 'constructivism', 'social', 'development', 'andragogy', 'motivation', 'audience', 'outcomes', 'assessment', 'strategies',
  'classroom', 'theories', 'allocation', 'sensitivity'];
const cases = [
  ...pages.map((name) => ({ name, file: name, spec: testSpec, expectEngine: 'WebGPU' })),
  // A first visit (nothing stored): the defaults, on Auto.
  { name: 'classroom-first-visit', file: 'classroom', expectEngine: 'WebGPU' },
  // Adults with the andragogy design, chosen through the stored spec.
  { name: 'classroom-adults', file: 'classroom', spec: { ...testSpec, population: { preset: 'adults', age: 35 }, designPreset: 'andragogy' }, expectEngine: 'WebGPU' },
  // The longest course a learner's units hold (16 sessions): still one kernel on WebGPU.
  { name: 'classroom-long-course', file: 'classroom', spec: { ...testSpec, course: { sessions: 16, sessionGap: 3, difficulty: 0.3, difficultyStep: 0.05, delayedTest: 30, abstraction: 0.6, atRiskBelow: 0.3 } }, expectEngine: 'WebGPU' },
  { name: 'gpu', file: 'gpu', spec: testSpec },
  // The same spec on WebGPU and on the software GPU (no adapter): headline numbers must agree to rounding.
  ...['classroom', 'theories', 'allocation'].flatMap((name) => [
    { name: `${name}-gpu-pair`, file: name, spec: pairSpec, expectEngine: 'WebGPU', pair: name, side: 'gpu' },
    { name: `${name}-software-pair`, file: name, spec: pairSpec, expectEngine: 'software GPU', gpuStub: 'no-adapter', pair: name, side: 'software' },
  ]),
  // No adapter at all (WebGPU blocked): the same kernels run on the software GPU.
  { name: 'classroom-no-adapter', file: 'classroom', spec: testSpec, expectEngine: 'software GPU', gpuStub: 'no-adapter', expectReason: 'WebGPU unavailable', device: 'Pixel 7' },
  { name: 'sensitivity-no-adapter', file: 'sensitivity', spec: testSpec, expectEngine: 'software GPU', gpuStub: 'no-adapter' },
  { name: 'assessment-no-adapter', file: 'assessment', spec: testSpec, expectEngine: 'software GPU', gpuStub: 'no-adapter' },
  { name: 'cognitivism-no-adapter', file: 'cognitivism', spec: testSpec, expectEngine: 'software GPU', gpuStub: 'no-adapter' },
  { name: 'strategies-no-adapter', file: 'strategies', spec: testSpec, expectEngine: 'software GPU', gpuStub: 'no-adapter' },
  { name: 'gpu-no-adapter', file: 'gpu', spec: testSpec, gpuStub: 'no-adapter' },
  // Only a compatibility-mode adapter (WebGPU on OpenGL ES): the kernels stay within its limits.
  { name: 'classroom-compat-only', file: 'classroom', spec: testSpec, expectEngine: 'WebGPU', gpuStub: 'compat-only', expectReason: 'compatibility mode' },
  // Phones: the collapsed menu must leave the page content in view, with no sideways scroll.
  { name: 'classroom-mobile', file: 'classroom', spec: testSpec, expectEngine: 'WebGPU', device: 'iPhone 14' },
  { name: 'theories-mobile', file: 'theories', spec: testSpec, device: 'Pixel 7' },
];
const pairs = {};
let failures = 0;
console.log(`testing ${root} ${fileMode ? 'from file:// (no server)' : `over ${base}`}`);

for (const { name, file, spec, expectEngine, expectReason, pair, side, device, gpuStub } of cases.filter((c) => !filter || c.name.includes(filter))) {
  const context = await browser.newContext(device ? { ...devices[device] } : { viewport: { width: 1400, height: 1000 } });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  const network = [];
  page.on('request', (r) => { if (!/^(file|data|blob):/.test(r.url())) network.push(r.url()); });
  if (gpuStub) await page.addInitScript((stub) => {
    const gpu = navigator.gpu;
    if (!gpu) return;
    const original = gpu.requestAdapter.bind(gpu);
    gpu.requestAdapter = (options = {}) =>
      stub === 'no-adapter' || (stub === 'compat-only' && !('featureLevel' in options)) ? Promise.resolve(null) : original(options);
  }, gpuStub);
  await page.addInitScript(({ stored }) => {
    if (!sessionStorage.getItem('seeded')) {
      if (stored) localStorage.setItem('syn-spec-v1', JSON.stringify(stored)); else localStorage.removeItem('syn-spec-v1');
      sessionStorage.setItem('seeded', '1');
    }
  }, { stored: spec || null });
  const t0 = Date.now();
  await page.goto(fileMode ? pathToFileURL(path.join(root, `${file}.html`)).href : `${base}/${file}.html`);
  let status = '';
  try {
    await page.waitForFunction(() => {
      const s = document.querySelector('#run-status');
      return s && /Done|Error/.test(s.textContent);
    }, null, { timeout: 180000 });
    status = await page.textContent('#run-status');
  } catch (e) {
    status = 'timeout';
  }
  const charts = await page.$$eval('.chart canvas', (cs) => cs.filter((c) => c.width > 0).length);
  const wasm = await page.textContent('nav .env').catch(() => '');
  const problems = [...errors];
  if (/Error|timeout/.test(status)) problems.push('status: ' + status);
  if (charts === 0) problems.push('no charts rendered');
  if (!/WASM: synaptic \d/.test(wasm)) problems.push(`library version not shown: ${wasm}`);
  if (fileMode && network.length) problems.push(`network requests from a file:// page: ${network.slice(0, 3).join(', ')}`);
  if (expectEngine && !status.includes(`· ${expectEngine}`)) problems.push(`expected the ${expectEngine} engine: ${status}`);
  if (expectReason && !status.includes(expectReason)) problems.push(`expected the reason "${expectReason}": ${status}`);
  // The status line is not enough: the fused kernels must really have been dispatched on WebGPU
  // (and never when another engine ran).
  const gpuRuns = await page.evaluate(() => globalThis.__synGpuRuns || 0);
  const softRuns = await page.evaluate(() => globalThis.__synSoftwareGpuRuns || 0);
  if (expectEngine === 'WebGPU' && gpuRuns === 0) problems.push('no fused pipeline was dispatched on WebGPU');
  if (expectEngine === 'software GPU' && softRuns === 0) problems.push('the software GPU did not run');
  if (expectEngine && expectEngine !== 'WebGPU' && gpuRuns > 0) problems.push(`${gpuRuns} WebGPU dispatches while the ${expectEngine} engine ran`);
  if (pair) pairs[pair] = { ...(pairs[pair] || {}), [side]: await page.evaluate(() => window.__synEngineRun) };
  if (file === 'gpu' && !problems.length) {
    const r = await page.evaluate(() => window.__synKernelCheck);
    if (!r) problems.push('no validation result');
    else {
      if (r.passed !== r.models) problems.push(`${r.models - r.passed} model(s) disagree across engines: ${r.rows.filter((x) => x.error || x.cpu > 1 || x.gpu > 1).map((x) => x.name).join(', ')}`);
      if (!gpuStub && !r.adapter) problems.push('the GPU page found no WebGPU adapter');
      if (gpuStub && r.adapter) problems.push(`without an adapter the GPU page must run the software GPU only: ${r.adapter}`);
      console.log(`       ${r.passed}/${r.models} models ${r.adapter ? `agree on WebGPU (${r.adapter}) and the software GPU` : 'run on the software GPU'}`);
    }
  }
  if (device) {
    const layout = await page.evaluate(() => ({
      menuHidden: document.getElementById('nav-links').offsetParent === null,
      titleTop: document.querySelector('main h1').getBoundingClientRect().top,
      scrollW: document.documentElement.scrollWidth, width: document.documentElement.clientWidth,
    }));
    if (!layout.menuHidden) problems.push('mobile menu is not collapsed');
    if (layout.titleTop > 200) problems.push(`page title starts ${Math.round(layout.titleTop)}px down: content hidden below the menu`);
    if (layout.scrollW > layout.width + 1) problems.push(`horizontal scroll on mobile (${layout.scrollW} > ${layout.width})`);
    await page.click('.menu-toggle');
    if (await page.evaluate(() => document.getElementById('nav-links').offsetParent === null)) problems.push('menu button does not open the menu');
  }
  if (name === 'classroom-adults') {
    const shown = await page.evaluate(() => document.querySelector('details.spec select').value);
    if (shown !== 'adults') problems.push(`population selector shows ${shown}`);
  }
  if (process.env.SCREENSHOTS) await page.screenshot({ path: path.join(process.env.SCREENSHOTS, `${name}.png`), fullPage: true });
  console.log(`${problems.length ? 'FAIL' : 'ok  '} ${name.padEnd(24)} ${String(Date.now() - t0).padStart(6)} ms  charts=${charts}  gpu=${gpuRuns}  ${status.trim().slice(0, 70)}`);
  for (const p of problems) console.log('       ' + p);
  failures += problems.length ? 1 : 0;
  await context.close();
}

// The same spec on WebGPU and on the software GPU: the same WGSL with the same random numbers in
// 32-bit floats on two devices.
const metrics = {
  classroom: [['delayed score', (r) => r.delayedScore], ['end score', (r) => r.endScore]],
  theories: [['blended (secondary)', (r) => r.blended], ['banking (secondary)', (r) => r.banking]],
  allocation: [['total gain', (r) => r.totalGain]],
};
for (const [name, r] of Object.entries(pairs)) {
  if (!r.gpu || !r.software) continue;
  for (const [label, get] of metrics[name]) {
    const g = get(r.gpu), w = get(r.software);
    const diff = Math.abs(g - w);
    const tol = 2e-3 + 2e-3 * Math.abs(w);
    const ok = diff < tol;
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${`${name} WebGPU/software`.padEnd(26)} ${label}: ${g.toFixed(5)} vs ${w.toFixed(5)} (diff ${diff.toExponential(1)}, tol ${tol.toFixed(4)})`);
    if (!ok) failures++;
  }
}

await browser.close();
server.close();
console.log(failures ? `\n${failures} page(s) failed` : '\nall pages passed');
process.exit(failures ? 1 : 0);
