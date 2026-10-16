import { run } from '../syn-client.js';
import { barChart, fmt, lineChart } from '../charts.js';
import { gpuEngine, runOnCpuKernels, runOnGpu } from '../gpu/backend.js';
import { probeAdapters } from '../gpu/engine.js';
import { card, el, grid, initPage, passFail, runButton, specEditor, table, tiles, toolbarNumber } from '../ui.js';

const page = initPage({
  id: 'gpu',
  title: 'WebGPU fused kernels',
  context: 'syn/gpu: compile · classroomFusedKernel · reducePartialsKernel · scoreQuantilesKernel · summarise',
  description: 'The whole classroom simulation as one fused compute kernel: one invocation per learner samples the learner, runs every session ' +
    '(needs, affect, motivation, the ZPD in pure elliptic CGA, encoding and the spaced/retrieval review slots, mindset) and evaluates the test ' +
    'score on every reporting day; the 64 learners of a workgroup reduce their statistics together. The second grid dimension is the job, ' +
    'so a whole Shapley allocation or sensitivity analysis is one dispatch. C++ compiles the buffers and owns the WGSL; this page checks the GPU ' +
    'against the WebAssembly engine (same model, different random numbers: they must agree within Monte Carlo error) and times both.',
});
specEditor(page, ['population', 'course', 'design', 'simulation']);
const learners = toolbarNumber(page, 'learners', 4000, { step: 500, width: 80 });

runButton(page, 'Validate and benchmark', async (spec) => {
  const n = Math.max(64, Math.round(learners()));
  const vspec = { ...spec, sim: { ...spec.sim, learners: n } };
  const probe = await probeAdapters();
  window.__synWebGpuProbe = probe;

  let label = 'WebGPU', fusedRun;
  try {
    const gpu = await gpuEngine();
    fusedRun = await runOnGpu('classroom', vspec, { warmUp: true });
    label = `WebGPU (${gpu.name})`;
  } catch (e) {
    // No adapter: validate the same kernels on the CPU instead.
    fusedRun = await runOnCpuKernels('classroom', vspec);
    label = `CPU fused kernels (${fusedRun.result.adapter}); WebGPU: ${e.message}`;
  }
  const g = fusedRun.result;
  const t0 = performance.now();
  const w = await run('classroom', vspec);
  const wasmMs = performance.now() - t0;

  // Agreement within Monte Carlo error: 4 standard errors of the mean score.
  const scoreSd = Math.max(0.02, (w.score.q90[w.score.q90.length - 1] - w.score.q10[w.score.q10.length - 1]) / 2.56);
  const tol = 4 * scoreSd / Math.sqrt(n) + 0.003;
  const checks = {
    delayed: Math.abs(g.delayedScore - w.delayedScore),
    end: Math.abs(g.endScore - w.endScore),
    median: Math.abs(g.score.q50[g.score.q50.length - 1] - w.score.q50[w.score.q50.length - 1]),
    effort: Math.abs(g.sessions.effort[g.sessions.effort.length - 1] - w.sessions.effort[w.sessions.effort.length - 1]),
    tol,
  };
  window.__synLastRun = { checks, engine: label };
  tiles(page.content, [
    { label: 'Delayed score: fused vs WASM', value: `${fmt.num(g.delayedScore, 4)} / ${fmt.num(w.delayedScore, 4)}`, hint: `difference ${fmt.num(checks.delayed, 4)}, tolerance ${fmt.num(tol, 4)}` },
    { label: 'Fused kernels', value: `${fmt.num(g.gpuMs, 1)} ms`, hint: label },
    { label: 'WebAssembly', value: `${fmt.num(wasmMs, 1)} ms`, hint: 'ClassroomEngine in a worker' },
    { label: 'Speed-up', value: `×${fmt.num(wasmMs / Math.max(0.01, g.gpuMs), 1)}`, hint: `${fmt.num(n)} learners × ${spec.course.sessions} sessions` },
  ]);
  const gr = grid(page.content);
  const days = Array.from(w.score.days);
  lineChart(card(gr, 'Score profile: fused kernels vs WebAssembly', 'Different random numbers, same model: the curves agree within Monte Carlo error.'), {
    x: days, xLabel: 'day', yLabel: 'score',
    series: [{ name: 'WebAssembly mean', y: Array.from(w.score.mean), band: { lower: Array.from(w.score.q10), upper: Array.from(w.score.q90) } },
      { name: 'fused mean', y: Array.from(g.score.mean), colorIndex: 1, dash: true },
      { name: 'fused 10% / 90%', y: Array.from(g.score.q10), colorIndex: 2, dash: true }, { name: '', y: Array.from(g.score.q90), colorIndex: 2, dash: true, hideLegend: true }],
  });
  table(card(gr, 'Validation', ''), ['check', 'difference', 'result'], [
    ['delayed test score', fmt.num(checks.delayed, 5), passFail(checks.delayed < tol)],
    ['end-of-course score', fmt.num(checks.end, 5), passFail(checks.end < 1.5 * tol)],
    ['median delayed score (histogram)', fmt.num(checks.median, 5), passFail(checks.median < 2 * tol + 0.002)],
    ['effort in the last session', fmt.num(checks.effort, 5), passFail(checks.effort < 0.02)],
  ]);

  // Benchmark: a batch of 64 courses (an allocation-sized job list) on both engines.
  const bspec = { ...spec, sim: { ...spec.sim, learners: Math.min(n, 1000) }, allocation: { permutations: 4 } };
  const tb = performance.now();
  const wa = await run('allocation', bspec);
  const wasmBatchMs = performance.now() - tb;
  let fusedBatch;
  try {
    fusedBatch = (await runOnGpu('allocation', bspec)).result;
  } catch (_) {
    fusedBatch = (await runOnCpuKernels('allocation', bspec)).result;
  }
  barChart(card(gr, `Batch of ${wa.evaluations} courses (a Shapley allocation)`, `${fmt.num(bspec.sim.learners)} learners each; the fused kernels run them as one dispatch.`), {
    labels: ['WebAssembly', 'fused kernels'], series: [{ name: 'milliseconds', values: [wasmBatchMs, fusedBatch.gpuMs] }], yFormat: (v) => `${v.toFixed(0)} ms`,
  });
  table(card(gr, 'Batch agreement', 'Shapley contributions add up exactly on both engines.'), ['quantity', 'WebAssembly', 'fused'], [
    ['total gain', fmt.num(wa.totalGain, 4), fmt.num(fusedBatch.totalGain, 4)],
    ['Σ Shapley', fmt.num(wa.shapleySum, 4), fmt.num(fusedBatch.shapleySum, 4)],
    ['courses', fmt.num(wa.evaluations), fmt.num(fusedBatch.evaluations)],
  ]);

  const diag = card(gr, 'WebGPU diagnostics', 'Every adapter request and its answer.');
  diag.append(el('dl', { class: 'diagnostics' },
    el('dt', { text: 'navigator.gpu' }), el('dd', { text: probe.api ? 'available' : 'not available' }),
    el('dt', { text: 'secure context' }), el('dd', { text: String(probe.secureContext) }),
    ...probe.adapters.flatMap((a) => [el('dt', { text: `requestAdapter(${a.request})` }),
      el('dd', { text: a.error ? `error: ${a.error}` : a.adapter ? `${a.adapter} · storage buffers/stage ${a.maxStorageBuffersPerShaderStage} · max binding ${fmt.num(a.maxStorageBufferBindingSize / 2 ** 20)} MiB` : 'null' })])));
  const kernels = await run('gpuKernels', {});
  table(card(gr, 'Kernels', 'WGSL sources owned by the C++ library.'), ['kernel', 'lines', 'workgroup'], [
    ['classroom-fused', fmt.num(kernels.classroomFused.split('\n').length), '64 learners × 1 job'],
    ['reduce-partials', fmt.num(kernels.reducePartials.split('\n').length), '64 statistics'],
    ['score-quantiles', fmt.num(kernels.scoreQuantiles.split('\n').length), '128, one (day, job) each'],
  ]);
  return `Done · ${label}`;
});
