import { run } from '../syn-client.js';
import { barChart, fmt } from '../charts.js';
import { gpuEngine, runOnGpu, runOnSoftwareGpu } from '../engine/run.js';
import { probeAdapters } from '../engine/gpu.js';
import { card, el, grid, initPage, passFail, runButton, specEditor, table, tiles, toolbarNumber } from '../ui.js';

const page = initPage({
  id: 'gpu',
  title: 'WebGPU fused kernels',
  context: 'syn/kernel: agentKernel · reduceKernel · quantileKernel   syn/gpu: the software GPU — every model of the library',
  description: 'Every model of the library is one fused compute kernel, written once in WGSL: one invocation per agent (a learner, a class, an ' +
    'instructor, a rater, a grid point) runs the model\'s agent program in the space Cl(4, 1) and adds its statistics; the 64 agents of a ' +
    'workgroup reduce them together, and the second grid dimension is the job (designs, conditions, seeds), so a whole analysis is one dispatch. ' +
    'The kernels run on WebGPU. Where a browser has no WebGPU adapter they run on the library\'s <b>software GPU</b>: the same WGSL, compiled to ' +
    'bytecode in WebAssembly and executed as a GPU does — workgroups of lanes in lockstep, barriers, workgroup memory, 32-bit arithmetic — ' +
    'across Web Workers. There is no other implementation. This page runs every model on both and compares every number.',
});
specEditor(page, ['population', 'course', 'design', 'simulation']);
const learners = toolbarNumber(page, 'classroom learners', 4000, { step: 500, width: 80 });

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
      const s = await runOnSoftwareGpu(m.name, small);
      row.softMs = s.engineMs;
      if (adapter) {
        const g = await runOnGpu(m.name, small);
        row.gpuMs = g.engineMs;
        // number by number, within the rounding two devices may differ by (syn::kernel::compareResults)
        const c = await run('compareResults', { model: m.name, a: g, b: s });
        row.values = c.values;
        row.diff = c.excess;
      }
    } catch (e) {
      row.error = e.message;
    }
    rows.push(row);
  }
  const ok = (r) => !r.error && (!adapter || r.diff <= 1);
  const passed = rows.filter(ok).length;
  window.__synKernelCheck = { adapter, models: rows.length, passed, rows };

  // A large classroom on both devices.
  const big = { ...spec, sim: { ...spec.sim, learners: Math.max(64, Math.round(learners())) } };
  const ts = await runOnSoftwareGpu('classroom', big);
  const tg = adapter ? await runOnGpu('classroom', big) : null;
  tiles(page.content, [
    { label: adapter ? 'Kernels agreeing on WebGPU and the software GPU' : 'Kernels run on the software GPU', value: `${passed} / ${rows.length}`,
      hint: adapter ? `WebGPU: ${adapter}` : 'no WebGPU adapter in this browser' },
    { label: 'Values compared', value: adapter ? fmt.num(rows.reduce((s, r) => s + (r.values || 0), 0)) : '—', hint: 'every number of every result' },
    { label: `Classroom, ${fmt.num(big.sim.learners)} learners`, value: `${fmt.num((tg || ts).engineMs, 0)} ms`,
      hint: tg ? `WebGPU; ${fmt.num(ts.engineMs, 0)} ms on the software GPU (${ts.adapter})` : `software GPU (${ts.adapter})` },
    { label: 'Delayed score', value: tg ? `${fmt.num(tg.delayedScore, 4)} / ${fmt.num(ts.delayedScore, 4)}` : fmt.num(ts.delayedScore, 4),
      hint: tg ? 'WebGPU / software GPU' : 'software GPU' },
  ]);
  const g = grid(page.content, true);
  const fmtDiff = (x) => (x === undefined ? '—' : x === Infinity ? 'shape differs' : `${(100 * x).toFixed(1)}% of tolerance`);
  table(card(g, 'Every model on both devices', 'Largest difference between WebGPU and the software GPU over every value of the result, as a share of the ' +
    'tolerance (2·10⁻³ relative + 2·10⁻³): the same WGSL in 32-bit floats on two devices.'),
    ['model', 'values', 'difference', 'ms (WebGPU / software GPU)', 'result'],
    rows.map((r) => [r.name, fmt.num(r.values || 0), fmtDiff(r.diff),
      `${r.gpuMs ? fmt.num(r.gpuMs, 0) : '—'} / ${fmt.num(r.softMs || 0, 0)}`,
      r.error ? passFail(false, r.error) : passFail(ok(r))]));
  const g2 = grid(page.content);
  barChart(card(g2, 'Time per model', `${fmt.num(small.sim.learners)} agents per population; milliseconds per run.`), {
    labels: rows.map((r) => r.name),
    series: [...(adapter ? [{ name: 'WebGPU', values: rows.map((r) => r.gpuMs || 0) }] : []),
      { name: 'software GPU', values: rows.map((r) => r.softMs || 0), colorIndex: 1 }],
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
  return `Done · ${passed}/${rows.length} kernels ${adapter ? 'agree' : 'run'} · ${adapter ? `WebGPU (${adapter}) and the software GPU` : 'software GPU only (no WebGPU adapter)'}`;
});
