import { barChart, fmt, lineChart } from '../charts.js';
import { engineNote, runModel } from '../engine/run.js';
import { LEVERS } from '../spec.js';
import { callout, card, grid, initPage, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'theories',
  title: 'Theories compared (Activity 3.3)',
  context: 'models: theories · fused kernels in the space Cl(4, 1)',
  description: "The chapter's Activity 3.3: teach a class to brainstorm keywords and combine them with AND, OR and NOT, designing the lesson from " +
    'the perspective of one theory. Each theory becomes a set of teaching practices (levers), and every design is run on the same learners. ' +
    'The chapter\'s conclusion, that the theories are not mutually exclusive and are best combined, is the <i>blended</i> design.' +
    ' <i>In the space:</i> the classroom agent under the eight theory designs, on three populations.',
});
specEditor(page, ['course', 'simulation']);

runButton(page, 'Compare', async (spec) => {
  const t0 = performance.now();
  const runResult = await runModel('theories', spec);
  const r = runResult.result;
  window.__synEngineRun = { page: 'theories', engine: runResult.label, blended: r.populations[1].results[7].delayedScore, banking: r.populations[1].results[0].delayedScore };
  const [kids, teens, adults] = r.populations;
  const best = (p) => p.best, single = (p) => p.bestSingle;
  tiles(page.content, [
    { label: 'Best design (all populations)', value: r.theories[best(teens)] },
    { label: 'Best single theory, children', value: r.theories[single(kids)], hint: fmt.num(kids.results[single(kids)].delayedScore, 3) },
    { label: 'Best single theory, secondary', value: r.theories[single(teens)], hint: fmt.num(teens.results[single(teens)].delayedScore, 3) },
    { label: 'Best single theory, adults', value: r.theories[single(adults)], hint: fmt.num(adults.results[single(adults)].delayedScore, 3) },
  ]);
  const g = grid(page.content);
  barChart(card(g, 'Delayed test score by theory and population', 'Mean score a month after the course.'), {
    labels: r.theories, series: r.populations.map((p, i) => ({ name: p.population, values: p.results.map((x) => x.delayedScore), colorIndex: i })), yFormat: (v) => v.toFixed(3),
  });
  barChart(card(g, 'Gain over the banking lecture, relative', 'Score ÷ banking-lecture score: andragogy pays most with adults, rewards with children.'), {
    labels: r.theories.slice(1), series: r.populations.map((p, i) => ({ name: p.population, values: p.results.slice(1).map((x) => x.delayedScore / p.results[0].delayedScore), colorIndex: i })),
    yFormat: (v) => `×${v.toFixed(2)}`,
  });
  lineChart(card(g, 'Score profiles, secondary school', ''), {
    x: Array.from(teens.results[0].days), xLabel: 'day', yLabel: 'mean score',
    series: teens.results.map((x, i) => ({ name: r.theories[i], y: Array.from(x.mean), colorIndex: i % 8, dash: i === 0 })),
  });
  barChart(card(g, 'Equity: culture gap at the delayed test', 'High minus low cultural capital (secondary school).'), {
    labels: r.theories, series: [{ name: 'gap', values: teens.results.map((x) => x.equityGap), colorIndex: 7 }], yFormat: (v) => v.toFixed(3),
  });
  table(card(g, 'Results (secondary school)', ''), ['theory', 'end of course', 'delayed', 'retention', 'at risk', 'culture gap', 'Δθ'],
    teens.results.map((x, i) => [r.theories[i], fmt.num(x.endScore, 3), fmt.num(x.delayedScore, 3), fmt.pct(x.retention, 0), fmt.pct(x.atRiskDelayed, 0), fmt.num(x.equityGap, 3), fmt.num(x.abilityGain, 2)]));
  table(card(g, 'The lesson each theory designs', 'Lever settings (0–1).'), ['lever', ...r.theories], LEVERS.map((l) => [l.name, ...r.designs.map((d) => fmt.num(d[l.key], 1))]));
  callout(page.content, 'Load any of these designs into the shared specification with the <i>start from the lesson of</i> selector on the <a href="classroom.html">classroom page</a>, ' +
    'then see where its gain comes from on <a href="allocation.html">Gain allocation</a>.');
  return engineNote(runResult, performance.now() - t0);
});
