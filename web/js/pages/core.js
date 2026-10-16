import { run } from '../syn-client.js';
import { fmt, histogram, lineChart } from '../charts.js';
import { card, grid, initPage, passFail, runButton, table, tiles } from '../ui.js';

const page = initPage({
  id: 'core',
  title: 'Core numerics',
  context: 'syn/core: Rng · summarise · quantile · logistic · SessionGrid',
  description: 'Every model in the library is a Monte Carlo simulation of learners. Each learner draws from its own reproducible random stream ' +
    '(xoshiro256** seeded through splitmix64), so two teaching designs can be run on the same learners and the same luck ' +
    '(common random numbers). Abilities and difficulties live on a logit scale and are turned into probabilities by the logistic function. ' +
    'Course calendars are session grids: uniform for weekly lessons, expanding for spaced review.',
});

runButton(page, 'Run', async (spec) => {
  const r = await run('coreDemo', { ...spec, core: { samples: 20000, gap: spec.course.sessionGap, sessions: spec.course.sessions } });
  tiles(page.content, [
    { label: 'Mean of normals', value: r.summary.mean.toFixed(4), hint: 'expected 0' },
    { label: 'Standard deviation', value: r.summary.sd.toFixed(4), hint: 'expected 1' },
    { label: '10% / 90% quantile', value: `${r.summary.q10.toFixed(3)} / ${r.summary.q90.toFixed(3)}`, hint: 'expected ∓1.2816' },
    { label: 'Streams', value: r.streamsReproducible && r.seedsIndependent ? 'ok' : 'broken', hint: 'same seed → same draws' },
  ]);
  const g = grid(page.content);
  histogram(card(g, 'Standard normal draws', '20,000 Box–Muller normals from one stream.'), Array.from(r.normals), { bins: 50, xFormat: (v) => v.toFixed(1) });
  lineChart(card(g, 'Logistic link σ(θ − b)', 'Probability of success of a learner of ability θ on a task of difficulty b.'), {
    x: Array.from(r.logisticX), xLabel: 'θ − b (logits)', yLabel: 'P(success)', series: [{ name: 'σ(x)', y: Array.from(r.logisticY) }],
  });
  const u = Array.from(r.uniformGrid), e = Array.from(r.expandingGrid);
  lineChart(card(g, 'Session grids', 'Weekly lessons against an expanding review calendar (gaps 1, 2, 4, 8… days).'), {
    x: u.map((_, i) => i + 1), xLabel: 'session', yLabel: 'day',
    series: [{ name: 'uniform (weekly)', y: u, markers: true }, { name: 'expanding', y: e, markers: true, colorIndex: 1 }],
  });
  table(card(g, 'Checks', ''), ['check', 'result'], [
    ['two streams of the same seed agree', passFail(r.streamsReproducible)],
    ['different seeds give different numbers', passFail(r.seedsIndependent)],
    ['normal mean within 0.02', passFail(Math.abs(r.summary.mean) < 0.02)],
    ['normal sd within 0.02', passFail(Math.abs(r.summary.sd - 1) < 0.02)],
    ['time (ms)', fmt.num(r.elapsedMs, 1)],
  ]);
});
