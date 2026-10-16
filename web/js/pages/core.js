import { barChart, fmt, lineChart } from '../charts.js';
import { modelButton, runModels } from '../engine/run.js';
import { card, grid, initPage, passFail, table, tiles } from '../ui.js';

const page = initPage({
  id: 'core',
  title: 'The space',
  context: 'syn/space: Cl(4, 1) · atLogit · chance · shift · mixture · rotate · cap · sampleLevel   syn/kernel: fused kernels',
  description: 'Every model of the library lives in one space: the <b>pure elliptic conformal geometric algebra</b> Cl(4, 1), the conformal model ' +
    'of the 3-sphere, with directions N (the learner), A (cognitive), B (affective), C (social) and e with e² = −1. Points are null vectors x + e. ' +
    '<b>Scalars are the miniature basis</b>: a probability is not stored but read off a point, chance(X) = 1 + ½·X·L = cos²(φ/2) against the ' +
    'learner L = N + e, so the point of logit x has chance σ(x). Effects that add on the logit scale are <b>dilation versors</b> about the learner; ' +
    'learning-rate updates are <b>mixtures</b> (the reading is linear in them); rotors move content between channels; thresholds and zones are ' +
    '<b>caps</b>; random quantities are random points. Every model runs as a <b>fused kernel</b> — WebGPU, the same kernel on the CPU in 32-bit ' +
    'floats, or WebAssembly in 64-bit floats — with the same random numbers, so the engines agree to rounding.',
});

modelButton(page, 'Run', async (spec) => {
  const { results: [r], note } = await runModels([['space', { ...spec, space: { samples: 20000 } }]], page);
  const c = r.checks;
  tiles(page.content, [
    { label: 'chance(atLogit(x)) − σ(x)', value: c.chanceIsLogistic.toExponential(1), hint: 'the reading of a point is the logistic of its logit' },
    { label: 'Dilation adds logits', value: c.dilationAddsLogits.toExponential(1), hint: `shift by ${r.params.shift}` },
    { label: 'Random points: logit mean / sd', value: `${fmt.num(r.random.meanLogit, 3)} / ${fmt.num(r.random.sdLogit, 3)}`, hint: `${fmt.num(r.random.samples)} points, expected 0 / 1` },
    { label: 'Points are null', value: c.pointsAreNull.toExponential(1), hint: 'max |X·X|' },
  ]);
  const g = grid(page.content);
  const x = Array.from(r.x);
  lineChart(card(g, 'The miniature basis', 'The chance of the point of logit x, read off the algebra, against the logistic σ(x); the same point dilated by d, against σ(x + d).'), {
    x, xLabel: 'logit x', yLabel: 'chance',
    series: [{ name: 'chance(atLogit(x))', y: Array.from(r.chance) }, { name: 'σ(x)', y: Array.from(r.logistic), dash: true, colorIndex: 1 },
      { name: `chance(shift(X, ${r.params.shift}))`, y: Array.from(r.shifted), colorIndex: 2 }, { name: `σ(x + ${r.params.shift})`, y: Array.from(r.logisticShifted), dash: true, colorIndex: 3 }],
  });
  lineChart(card(g, 'Mixtures and caps', `A mixture towards the learner by ${r.params.mixture} moves the chance linearly; the cap of chance ${r.params.cap} is a threshold.`), {
    x, xLabel: 'logit x', yLabel: '',
    series: [{ name: 'chance', y: Array.from(r.chance) }, { name: `mixture towards L (${r.params.mixture})`, y: Array.from(r.mixed), colorIndex: 1 },
      { name: `inside the cap of chance ${r.params.cap}`, y: Array.from(r.insideCap), colorIndex: 2, dash: true }],
  });
  lineChart(card(g, 'Angle from the learner', 'φ = 2·atan(e^{−x/2}): the stereographic coordinate of the chance.'), {
    x, xLabel: 'logit x', yLabel: 'φ (radians)', series: [{ name: 'φ', y: Array.from(r.angle) }, { name: 'similarity to the same level in another channel', y: Array.from(r.crossChannel), colorIndex: 1 }],
  });
  barChart(card(g, 'Random points', 'Points sampled uniformly in chance: their logits are logistic, the space\'s own distribution (scaled to sd 1).'), {
    labels: Array.from(r.random.binCentres, (v) => v.toFixed(1)), series: [{ name: 'share of points', values: Array.from(r.random.binShare) }], yFormat: (v) => fmt.pct(v, 1),
  });
  table(card(g, 'Checks', 'Every closed-form versor is also tested against the geometric product of Cl(4, 1) in the C++ tests.'), ['check', 'value', 'result'], [
    ['chance of a point = logistic of its logit', c.chanceIsLogistic.toExponential(1), passFail(c.chanceIsLogistic < 1e-4)],
    ['dilation by d adds d to the logit', c.dilationAddsLogits.toExponential(1), passFail(c.dilationAddsLogits < 1e-4)],
    ['mixture moves the chance linearly', c.mixtureIsLinear.toExponential(1), passFail(c.mixtureIsLinear < 1e-4)],
    ['points are null vectors', c.pointsAreNull.toExponential(1), passFail(c.pointsAreNull < 1e-4)],
    ['the logit read back', c.logitReadBack.toExponential(1), passFail(c.logitReadBack < 1e-2)],
    ['rotors between channels keep the chance', c.rotorKeepsChance.toExponential(1), passFail(c.rotorKeepsChance < 1e-4)],
    ['random logits: mean 0, sd 1', `${fmt.num(r.random.meanLogit, 3)}, ${fmt.num(r.random.sdLogit, 3)}`, passFail(Math.abs(r.random.meanLogit) < 0.05 && Math.abs(r.random.sdLogit - 1) < 0.05)],
  ]);
  return note;
});
