// The QA golden test on WebGPU: the same readingQA kernels, in headless Chromium (Playwright),
// on the golden questions of DATA_DIR; scores the answers against the golden ones (token F1) and
// checks that they are the answers the software GPU gave (DATA_DIR/answers.tsv from qa_golden).
//
//   node tools/qa_golden/webgpu.mjs DATA_DIR
// Environment: PLAYWRIGHT_MODULE (path to playwright's index.mjs), CHROMIUM_PATH.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = process.argv[2];
if (!dir) { console.error('usage: node tools/qa_golden/webgpu.mjs DATA_DIR'); process.exit(2); }
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../web');
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright').catch(() =>
  import('/opt/node22/lib/node_modules/playwright/index.mjs'));

const tsv = (f) => fs.readFileSync(path.join(dir, f), 'utf8').split('\n').filter(Boolean).map((l) => l.split('\t'));
const qa = tsv('qa.tsv');
const corpus = tsv('corpus.tsv').map((r) => r[1]);
const passages = [], questions = [], passageOf = new Map();
for (const [doc, question, , passage] of qa) {
  if (!passageOf.has(doc)) { passageOf.set(doc, passages.length); passages.push(passage); }
  questions.push({ question, passage: passageOf.get(doc) });
}
// Every question names its passage's capsule; the capsule is ingested and recalled on WebGPU too.
questions.forEach((q) => { q.capsule = q.passage; });
const spec = { sim: { seed: 2026 }, readingQA: { passages, questions, corpus } };

// SQuAD token F1, as in qa_golden.cpp.
const norm = (s) => s.toLowerCase().replace(/[^a-z0-9\s]/g, '').split(/\s+/).filter((w) => w && !['a', 'an', 'the'].includes(w));
function f1(p, g) {
  const P = norm(p), G = norm(g);
  if (!P.length || !G.length) return P.length === G.length ? 1 : 0;
  const count = new Map();
  for (const w of G) count.set(w, (count.get(w) || 0) + 1);
  let common = 0;
  for (const w of P) if ((count.get(w) || 0) > 0) { common++; count.set(w, count.get(w) - 1); }
  if (!common) return 0;
  const pr = common / P.length, rc = common / G.length;
  return (2 * pr * rc) / (pr + rc);
}

const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.wasm': 'application/wasm' };
const server = http.createServer((req, res) => {
  const url = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const file = path.join(root, url === '/' ? 'index.html' : url);
  if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'content-type': types[path.extname(file)] || 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const localChromium = ['/opt/pw-browsers/chromium', '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'].find((p) => fs.existsSync(p) && fs.statSync(p).isFile());
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || localChromium,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--disable-vulkan-surface'],
});
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${server.address().port}/kernels.html`);
const result = await page.evaluate(async (s) => {
  const { runOnGpu } = await import('/js/engine/run.js');
  const t0 = performance.now();
  const ing = await runOnGpu('capsuleIngest', { sim: s.sim, capsuleIngest: { passages: s.readingQA.passages } });
  const rec = await runOnGpu('capsuleRecall', { sim: s.sim, capsuleRecall: { capsule: ing.capsule } });
  const r = await runOnGpu('readingQA', { ...s, readingQA: { ...s.readingQA, recalled: rec.recalled } });
  return { answers: r.answers, ms: performance.now() - t0, adapter: r.adapter };
}, spec);
await browser.close();
server.close();

const native = tsv('answers.tsv').slice(1);
let readingF1 = 0, openF1 = 0, rememberedF1 = 0, constructedF1 = 0, same = 0;
const clean = (s) => (s || '').replace(/\t/g, ' ');
result.answers.forEach((a, i) => {
  readingF1 += f1(a.reading, qa[i][2]);
  openF1 += f1(a.openBook, qa[i][2]);
  rememberedF1 += f1(a.remembered || '', qa[i][2]);
  constructedF1 += f1(a.constructed || '', qa[i][2]);
  const n = native[i];
  if (n && n[2] === clean(a.reading) && n[4] === clean(a.openBook) && (n[7] ?? '') === clean(a.remembered) && (n[9] ?? '') === clean(a.constructed)) same++;
});
const n = result.answers.length;
console.log(`WebGPU (${result.adapter || 'adapter'}): ${n} golden questions in ${Math.round(result.ms)} ms`);
console.log(`token F1: reader ${(readingF1 / n).toFixed(3)}, open book ${(openF1 / n).toFixed(3)}, remembering ${(rememberedF1 / n).toFixed(3)}, constructed from its capsule ${(constructedF1 / n).toFixed(3)}`);
console.log(`same answers as the software GPU: ${same}/${n}`);
if (same !== n) process.exit(1);
