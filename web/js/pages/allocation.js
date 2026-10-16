import { barChart, fmt } from '../charts.js';
import { engineNote, engineSelector, runAnalysis } from '../gpu/backend.js';
import { THEORY_PRESETS } from '../spec.js';
import { card, grid, initPage, passFail, runButton, specEditor, table, tiles, toolbarNumber, toolbarSelect } from '../ui.js';

const page = initPage({
  id: 'allocation',
  title: 'Allocating the learning gain to teaching practices',
  context: 'syn/classroom: allocateLearningGain (Shapley · incremental · standalone)',
  description: 'A design improves the delayed test score over a baseline lesson. Teaching practices interact (spacing needs retrieval, retrieval needs ' +
    'feedback, everything needs a learner who is not too anxious to think), so the gain has to be split. The <b>Shapley</b> contribution averages ' +
    "each practice's marginal effect over random orders of adding them, and adds up exactly to the total. The <b>incremental</b> effect is " +
    'what removing the practice from the design would cost; the <b>standalone</b> effect is what it does alone. Neither adds up.',
});
specEditor(page, ['population', 'course', 'design', 'simulation']);

const baseline = toolbarSelect(page, 'baseline', Object.keys(THEORY_PRESETS), 'banking lecture');
const permutations = toolbarNumber(page, 'orders', 12, { step: 1, width: 60 });

const button = runButton(page, 'Allocate', async (spec) => {
  const t0 = performance.now();
  const runResult = await runAnalysis('allocation', { ...spec, allocation: { baseline: THEORY_PRESETS[baseline()], permutations: permutations() } });
  const r = runResult.result;
  window.__synEngineRun = { page: 'allocation', engine: runResult.engine, totalGain: r.totalGain, shapleySum: r.shapleySum };
  const sum = (k) => r.levers.reduce((s, l) => s + l[k], 0);
  tiles(page.content, [
    { label: 'Design score', value: fmt.num(r.designScore, 3), hint: `baseline (${baseline()}): ${fmt.num(r.baselineScore, 3)}` },
    { label: 'Gain to allocate', value: fmt.num(r.totalGain, 3) },
    { label: 'Σ Shapley', value: fmt.num(r.shapleySum, 3), hint: 'adds up exactly' },
    { label: 'Σ standalone / Σ incremental', value: `${fmt.num(sum('standalone'), 3)} / ${fmt.num(sum('incremental'), 3)}`, hint: 'interactions: do not add up' },
    { label: 'Courses run', value: fmt.num(r.evaluations), hint: runResult.fused ? `one batch, ${fmt.num(r.gpuMs)} ms on the ${runResult.label}` : `${r.permutations} orders, ${fmt.num(r.elapsedMs)} ms in WebAssembly` },
  ]);
  const g = grid(page.content, true);
  const sorted = [...r.levers].sort((a, b) => b.shapley - a.shapley);
  barChart(card(g, 'Contribution of each practice', 'Same learners and random numbers in every run (common random numbers).'), {
    labels: sorted.map((l) => l.lever),
    series: [{ name: 'Shapley', values: sorted.map((l) => l.shapley) }, { name: 'incremental', values: sorted.map((l) => l.incremental) }, { name: 'standalone', values: sorted.map((l) => l.standalone) }],
    yFormat: (v) => v.toFixed(3),
  });
  table(card(g, 'Allocation table', ''), ['practice', 'theory', 'baseline → design', 'Shapley', 'share', 'incremental', 'standalone'], [
    ...sorted.map((l) => [l.lever, l.theory, `${fmt.num(l.baseline, 2)} → ${fmt.num(l.design, 2)}`, fmt.num(l.shapley, 4), r.totalGain ? fmt.pct(l.shapley / r.totalGain, 1) : '–', fmt.num(l.incremental, 4), fmt.num(l.standalone, 4)]),
    ['total', '', '', fmt.num(r.shapleySum, 4), passFail(Math.abs(r.shapleySum - r.totalGain) < 1e-9, 'adds up'), fmt.num(sum('incremental'), 4), fmt.num(sum('standalone'), 4)],
  ]);
  return engineNote(runResult, performance.now() - t0);
});
engineSelector(page, () => button.click());
