import { run } from '../syn-client.js';
import { barChart, fmt } from '../charts.js';
import { gpuEngine, runOnCpuKernels, runOnGpu, runOnWasm } from '../engine/run.js';
import { probeAdapters } from '../engine/gpu.js';
import { card, el, grid, initPage, passFail, runButton, specEditor, table, tiles, toolbarNumber } from '../ui.js';

const page = initPage({
  id: 'gpu',
  title: 'WebGPU fused kernels',
  context: 'syn/kernel: agentKernel · reduceKernel · quantileKernel · runTiles · analyseOutputs — every model of the library',
  description: 'Every model of the library is one fused compute kernel: one invocation per agent (a learner, a class, an instructor, a rater, a ' +
    'grid point) runs the model\'s agent program in the space Cl(4, 1) and adds its statistics; the 64 agents of a workgroup reduce them ' +
    'together, and the second grid dimension is the job (designs, conditions, seeds), so a whole analysis is one dispatch. The agent program ' +
    'is written once and compiled both as WGSL and as C++, so the same kernel also runs on the CPU (32-bit floats, across Web Workers) and in ' +
    'WebAssembly (64-bit floats). The engines draw the same random numbers, so they must agree to rounding, not merely within Monte Carlo ' +
    'error: this page runs every model on all three and checks it.',
});
specEditor(page, ['population', 'course', 'design', 'simulation']);
const learners = toolbarNumber(page, 'classroom learners', 4000, { step: 500, width: 80 });

// Every value of two results, in order (timings and engine names left out).
function leaves(v, acc = []) {
  if (typeof v === 'number') acc.push(v);
  else if (ArrayBuffer.isView(v) || Array.isArray(v)) for (const x of v) leaves(x, acc);
  else if (v && typeof v === 'object') for (const k of Object.keys(v)) if (!['engineMs', 'adapter', 'kernelPlans', 'kernelJobs', 'workers'].includes(k)) leaves(v[k], acc);
  return acc;
}
// Models whose agents take threshold decisions, where 32-bit rounding can flip one agent's path;
// and sensitivity, whose variance reduction is a ratio of small variances that amplifies rounding.
const ALLOW = { equilibration: { flips: 0.02 }, sensitivity: { relative: 0.03 } };
function worst(a, b, { flips = 0, relative = 2e-3 } = {}) {
  const x = leaves(a), y = leaves(b);
  if (x.length !== y.length) return Infinity;
  let w = 0;
  x.forEach((v, i) => { if (Number.isFinite(v) && Number.isFinite(y[i])) w = Math.max(w, Math.abs(v - y[i]) / (2e-3 + flips + relative * Math.abs(y[i]))); });
  return w;
}

runButton(page, 'Validate every kernel', async (spec) => {
  const probe = await probeAdapters();
  window.__synWebGpuProbe = probe;
  let adapter = null;
  try { adapter = (await gpuEngine()).name; } catch (e) { adapter = null; window.__synGpuError = e.message; }
  const models = await run('models', {});
  const small = { ...spec, sim: { ...spec.sim, learners: Math.min(300, spec.sim.learners) } };
  const rows = [];
  for (const m of models) {
    const row = { name: m.name, title: m.title };
    try {
      const w = await runOnWasm(m.name, small);
      const c = await runOnCpuKernels(m.name, small);
      row.wasmMs = w.engineMs;
      row.cpuMs = c.engineMs;
      row.values = leaves(w).length;
      row.cpu = worst(c, w, ALLOW[m.name]);
      if (adapter) {
        const g = await runOnGpu(m.name, small);
        row.gpuMs = g.engineMs;
        row.gpu = worst(g, w, ALLOW[m.name]);
      }
    } catch (e) {
      row.error = e.message;
    }
    rows.push(row);
  }
  const passed = rows.filter((r) => !r.error && r.cpu <= 1 && (!adapter || r.gpu <= 1)).length;
  window.__synKernelCheck = { adapter, models: rows.length, passed, rows };

  // A large classroom: the GPU against WebAssembly.
  const big = { ...spec, sim: { ...spec.sim, learners: Math.max(64, Math.round(learners())) } };
  const tw = await runOnWasm('classroom', big);
  const tg = adapter ? await runOnGpu('classroom', big) : await runOnCpuKernels('classroom', big);
  tiles(page.content, [
    { label: 'Kernels agreeing on every engine', value: `${passed} / ${rows.length}`, hint: adapter ? `WebGPU: ${adapter}` : 'no WebGPU adapter: CPU kernels and WebAssembly only' },
    { label: 'Values compared', value: fmt.num(rows.reduce((s, r) => s + (r.values || 0), 0)), hint: 'every number of every result' },
    { label: `Classroom, ${fmt.num(big.sim.learners)} learners`, value: `${fmt.num(tg.engineMs, 0)} ms`, hint: `${adapter ? 'WebGPU' : 'CPU kernels'} vs ${fmt.num(tw.engineMs, 0)} ms in WebAssembly` },
    { label: 'Delayed score', value: `${fmt.num(tg.delayedScore, 4)} / ${fmt.num(tw.delayedScore, 4)}`, hint: '32-bit kernel / 64-bit WebAssembly' },
  ]);
  const g = grid(page.content, true);
  const fmtExcess = (x) => (x === undefined ? '—' : x === Infinity ? 'shape differs' : `${(100 * x).toFixed(1)}% of tolerance`);
  table(card(g, 'Every model on every engine', 'Largest difference from WebAssembly over every value of the result, as a share of the tolerance (2·10⁻³ relative + 2·10⁻³).'),
    ['model', 'values', 'WebGPU', 'CPU kernels', 'ms (GPU / CPU / WASM)', 'result'],
    rows.map((r) => [r.name, fmt.num(r.values || 0), fmtExcess(r.gpu), fmtExcess(r.cpu),
      `${r.gpuMs ? fmt.num(r.gpuMs, 0) : '—'} / ${fmt.num(r.cpuMs || 0, 0)} / ${fmt.num(r.wasmMs || 0, 0)}`,
      r.error ? passFail(false, r.error) : passFail(r.cpu <= 1 && (!adapter || r.gpu <= 1))]));
  const g2 = grid(page.content);
  barChart(card(g2, 'Time per model', `${fmt.num(small.sim.learners)} agents per population; milliseconds per run.`), {
    labels: rows.map((r) => r.name),
    series: [...(adapter ? [{ name: 'WebGPU', values: rows.map((r) => r.gpuMs || 0) }] : []),
      { name: 'CPU kernels', values: rows.map((r) => r.cpuMs || 0), colorIndex: 1 }, { name: 'WebAssembly', values: rows.map((r) => r.wasmMs || 0), colorIndex: 2 }],
    yFormat: (v) => `${v.toFixed(0)} ms`,
  });
  const diag = card(g2, 'WebGPU diagnostics', 'Every adapter request and its answer.');
  diag.append(el('dl', { class: 'diagnostics' },
    el('dt', { text: 'navigator.gpu' }), el('dd', { text: probe.api ? 'available' : 'not available' }),
    el('dt', { text: 'secure context' }), el('dd', { text: String(probe.secureContext) }),
    ...probe.adapters.flatMap((a) => [el('dt', { text: `requestAdapter(${a.request})` }),
      el('dd', { text: a.error ? `error: ${a.error}` : a.adapter ? `${a.adapter} · storage buffers/stage ${a.maxStorageBuffersPerShaderStage} · max binding ${fmt.num(a.maxStorageBufferBindingSize / 2 ** 20)} MiB` : 'no adapter' })])));
  const sources = await run('kernelSources', { model: 'classroom' });
  table(card(g2, 'Kernels', 'WGSL generated by the C++ library: the space, the counter-based random numbers, the model\'s agent and the driver.'), ['kernel', 'lines', 'workgroup'], [
    ['agent (classroom)', fmt.num(sources.agent.split('\n').length), '64 agents × 1 job'],
    ['reduce', fmt.num(sources.reduce.split('\n').length), '64 (job, statistic) sums'],
    ['quantiles', fmt.num(sources.quantile.split('\n').length), '128, one (value, job) each'],
  ]);
  return `Done · ${passed}/${rows.length} kernels agree · ${adapter ? `WebGPU (${adapter})` : 'no WebGPU adapter'}`;
});
