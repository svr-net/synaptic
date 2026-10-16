// End-to-end test of the web front end in headless Chromium (Playwright).
//
//   node web/tests/e2e.mjs [page-filter] [--root DIR] [--file]
//
// Serves web/ (or --root DIR, e.g. dist/standalone) on a local port, or with --file opens the
// pages straight from disk as file:// URLs (the copy-and-open deployment of the standalone
// build, which must then make no network request at all). Opens every page, waits for its
// automatic run and checks that it finished without errors and rendered charts.
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
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || localChromium });

// Small learner counts keep the run fast; the pages pick the spec up from localStorage.
const testSpec = { sim: { learners: 200, seed: 11 } };
const pages = ['index', 'core', 'behaviorism', 'humanism', 'cognitivism', 'constructivism', 'social', 'development', 'andragogy', 'motivation',
  'classroom', 'theories', 'allocation', 'sensitivity'];
const cases = [
  ...pages.map((name) => ({ name, file: name, spec: testSpec })),
  // A first visit (nothing stored): the defaults must work on their own.
  { name: 'classroom-first-visit', file: 'classroom' },
  // Adults with the andragogy design, chosen through the stored spec.
  { name: 'classroom-adults', file: 'classroom', spec: { ...testSpec, population: { preset: 'adults', age: 35 }, designPreset: 'andragogy' } },
  // A phone: the collapsed menu must leave the page content in view, with no sideways scroll.
  { name: 'classroom-mobile', file: 'classroom', spec: testSpec, device: 'iPhone 14' },
  { name: 'theories-mobile', file: 'theories', spec: testSpec, device: 'Pixel 7' },
];
let failures = 0;
console.log(`testing ${root} ${fileMode ? 'from file:// (no server)' : `over ${base}`}`);

for (const { name, file, spec, device } of cases.filter((c) => !filter || c.name.includes(filter))) {
  const context = await browser.newContext(device ? { ...devices[device] } : { viewport: { width: 1400, height: 1000 } });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  const network = [];
  page.on('request', (r) => { if (!/^(file|data|blob):/.test(r.url())) network.push(r.url()); });
  await page.addInitScript((stored) => {
    if (!sessionStorage.getItem('seeded')) {
      if (stored) localStorage.setItem('syn-spec-v1', JSON.stringify(stored)); else localStorage.removeItem('syn-spec-v1');
      sessionStorage.setItem('seeded', '1');
    }
  }, spec || null);
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
  console.log(`${problems.length ? 'FAIL' : 'ok  '} ${name.padEnd(22)} ${String(Date.now() - t0).padStart(6)} ms  charts=${charts}  ${status.trim().slice(0, 60)}`);
  for (const p of problems) console.log('       ' + p);
  failures += problems.length ? 1 : 0;
  await context.close();
}

await browser.close();
server.close();
console.log(failures ? `\n${failures} page(s) failed` : '\nall pages passed');
process.exit(failures ? 1 : 0);
