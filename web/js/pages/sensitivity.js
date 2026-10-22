import { barChart, fmt } from '../charts.js';
import { engineNote, runModel } from '../engine/run.js';
import { card, grid, initPage, runButton, specEditor, table, tiles, toolbarNumber } from '../ui.js';

const page = initPage({
  id: 'sensitivity',
  title: 'Lever sensitivities: where would more help most?',
  context: 'models: sensitivity · fused kernels in the space Cl(4, 1)',
  description: 'The slope of the delayed test score in each teaching practice, estimated by nudging the lever up and down and rerunning the course. ' +
    'With <b>common random numbers</b> both runs use the same learners and the same luck, so the difference measures the practice. With ' +
    '<b>independent seeds</b> the Monte Carlo noise of each run is divided by the small bump and swamps the effect. Repeating the estimate ' +
    'over several seeds shows the standard error of each method.' +
    ' <i>In the space:</i> the classroom agent with each lever bumped up and down, on common random numbers and on independent seeds.',
});
specEditor(page, ['population', 'course', 'design', 'simulation']);

const bump = toolbarNumber(page, 'bump', 0.1, { step: 0.05, width: 60 });
const reps = toolbarNumber(page, 'replications', 6, { step: 1, width: 60 });

runButton(page, 'Compute sensitivities', async (spec) => {
  const t0 = performance.now();
  // Every course of the analysis is one job of a single dispatch.
  const runResult = await runModel('sensitivity', { ...spec, sensitivity: { bump: bump(), replications: reps(), maxLearners: 300 } });
  const r = runResult.result;
  window.__synEngineRun = { page: 'sensitivity', engine: runResult.label, varianceReduction: r.varianceReduction, score: r.score };
  const sorted = Array.from(r.ranked, (i) => r.levers[i]);
  tiles(page.content, [
    { label: 'Delayed score', value: fmt.num(r.score, 3) },
    { label: 'Most valuable next step', value: sorted[0].lever, hint: `+${fmt.num(sorted[0].crn * 0.1, 4)} per +0.1` },
    { label: 'Least valuable', value: sorted[sorted.length - 1].lever },
    { label: 'Variance reduction (CRN)', value: `×${fmt.num(r.varianceReduction, 0)}`, hint: 'mean (se independent / se CRN)²' },
    { label: 'Courses run', value: fmt.num(1 + 3 * r.replications * r.levers.length), hint: `${r.learners} learners each, ` + `one batch, ${fmt.num(r.engineMs)} ms on ${runResult.label}` },
  ]);
  const g = grid(page.content, true);
  barChart(card(g, '∂ score / ∂ lever', 'Common random numbers against independent seeds, at the current design.'), {
    labels: sorted.map((l) => l.lever), series: [{ name: 'common random numbers', values: sorted.map((l) => l.crn) }, { name: 'independent seeds', values: sorted.map((l) => l.independent) }],
    yFormat: (v) => v.toFixed(3),
  });
  table(card(g, 'Sensitivity table', ''), ['practice', 'theory', 'level', 'CRN', '± se', 'independent', '± se', 'se ratio'],
    sorted.map((l) => [l.lever, l.theory, fmt.num(l.value, 2), fmt.num(l.crn, 4), fmt.num(l.crnSe, 4), fmt.num(l.independent, 4), fmt.num(l.independentSe, 4), l.crnSe > 0 ? `×${fmt.num(l.independentSe / l.crnSe, 0)}` : '–']));
  return engineNote(runResult, performance.now() - t0);
});
