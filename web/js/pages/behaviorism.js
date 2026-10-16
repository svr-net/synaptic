import { barChart, fmt, lineChart } from '../charts.js';
import { modelButton, runModels } from '../engine/run.js';
import { callout, card, grid, initPage, table, tiles, toolbarNumber } from '../ui.js';

const page = initPage({
  id: 'behaviorism',
  title: 'Behaviourism: conditioning and reinforcement',
  context: 'models: conditioning · operant · tokenEconomy · fused kernels in the space Cl(4, 1)',
  description: 'Learning as a change in behaviour brought about by the environment. <b>Classical conditioning</b> (Pavlov, Watson) uses the ' +
    'Rescorla–Wagner rule ΔV = αβ(λ − ΣV): a bell paired with food comes to predict it. <b>Operant conditioning</b> (Skinner) strengthens ' +
    'behaviour that is reinforced, on continuous or intermittent schedules, and weakens it when reinforcement stops. A classroom ' +
    '<b>point system</b> treats motivation as external: engagement rises while points are given and falls back when they are withdrawn.' +
    ' <i>In the space:</i> associative strengths and response probabilities are points; Rescorla–Wagner trials and reinforcement are mixtures towards the learner; every random event is a cap test; fear generalises by the similarity of stimulus points.',
});

const salience = toolbarNumber(page, 'CS salience α', 0.3, { step: 0.05 });
const ratio = toolbarNumber(page, 'schedule ratio / interval', 5, { step: 1 });

modelButton(page, 'Run experiments', async (spec) => {
  const { results: [c, o, t], note } = await runModels([
    ['conditioning', { ...spec, conditioning: { salience: salience() } }],
    ['operant', { ...spec, operant: { ratio: ratio(), learners: 200 } }],
    ['tokenEconomy', spec],
  ], page);
  const crf = o.runs[0], vr = o.runs[2];
  tiles(page.content, [
    { label: 'Bell after acquisition', value: fmt.num(c.pavlov[c.acquisitionTrials - 1], 3), hint: 'associative strength V' },
    { label: 'Blocked tone', value: fmt.num(c.blocking.blockedTone, 3), hint: `vs ${fmt.num(c.blocking.controlToneFinal, 3)} without pre-training` },
    { label: 'Extinction: CRF vs VR', value: `${fmt.num(crf.resistance)} / ${fmt.num(vr.resistance)}`, hint: 'opportunities to halve responding' },
    { label: 'Point system', value: `${fmt.pct(t.before, 0)} → ${fmt.pct(t.during, 0)} → ${fmt.pct(t.after, 0)}`, hint: 'engagement before / during / after' },
  ]);
  const g = grid(page.content);
  const trials = Array.from(c.pavlov, (_, i) => i + 1);
  lineChart(card(g, "Pavlov's dog", 'Bell paired with food, then the bell alone. Dashed: the closed form λ(1 − (1 − αβ)ⁿ).'), {
    x: trials, xLabel: 'trial', yLabel: 'associative strength',
    series: [{ name: 'bell', y: Array.from(c.pavlov) }, { name: 'closed form', y: Array.from(c.closedForm), dash: true, colorIndex: 1 }],
  });
  const bt = Array.from(c.blocking.light, (_, i) => i + 1);
  const control = new Array(c.blocking.compoundStart).fill(NaN).concat(Array.from(c.blocking.controlTone));
  lineChart(card(g, 'Kamin blocking', 'The light is learned first; when the tone is added the food is already predicted, so the tone learns almost nothing.'), {
    x: bt, xLabel: 'trial', series: [{ name: 'light', y: Array.from(c.blocking.light) }, { name: 'tone (blocked)', y: Array.from(c.blocking.tone), colorIndex: 1 },
      { name: 'tone (no pre-training)', y: control, dash: true, colorIndex: 2 }],
  });
  barChart(card(g, "Watson's experiment: generalised fear", 'A white rat paired with a loud noise; fear spreads to things that resemble the rat.'), {
    labels: c.fear.map((p) => p.stimulus), series: [{ name: 'fear response', values: c.fear.map((p) => p.fear) }], yFormat: (v) => v.toFixed(2),
  });
  const steps = Array.from(o.steps);
  lineChart(card(g, 'Response probability by schedule', `Training for ${o.training} opportunities, then extinction (no reinforcement).`), {
    x: steps, xLabel: 'opportunity', yLabel: 'P(response)', series: o.runs.map((r, i) => ({ name: r.schedule, y: Array.from(r.probability), colorIndex: i })),
  });
  lineChart(card(g, "Cumulative record (Skinner's recorder)", 'Slope = response rate. Intermittently reinforced behaviour keeps going long after reinforcement stops.'), {
    x: steps, xLabel: 'opportunity', yLabel: 'responses', series: o.runs.map((r, i) => ({ name: r.schedule, y: Array.from(r.cumulativeResponses), colorIndex: i })),
  });
  const days = Array.from(t.engagement, (_, i) => i);
  lineChart(card(g, 'Classroom point system', `Points are given from day ${t.rewardStart} to day ${t.rewardEnd}. Students differ in intrinsic interest.`), {
    x: days, xLabel: 'day', yLabel: 'P(engage)',
    series: [{ name: 'all students', y: Array.from(t.engagement) }, { name: 'low intrinsic interest', y: Array.from(t.low), colorIndex: 1, dash: true },
      { name: 'high intrinsic interest', y: Array.from(t.high), colorIndex: 2, dash: true }],
  });
  table(card(g, 'Schedules', 'Partial reinforcement: slower to acquire, far slower to extinguish.'), ['schedule', 'acquisition', 'resistance to extinction', 'reinforcers / response'],
    o.runs.map((r) => [r.schedule, fmt.num(r.acquisitionSteps), fmt.num(r.resistance), fmt.num(r.reinforcersPerResponse, 2)]));
  callout(page.content, 'In the classroom engine the <i>reinforcement</i> lever is the point system: it adds external value (more for younger learners), builds habit, ' +
    'and slightly erodes intrinsic interest. See <a href="classroom.html">the classroom simulation</a>.');
  return note;
});
