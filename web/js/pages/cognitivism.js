import { barChart, fmt, lineChart } from '../charts.js';
import { runModels } from '../engine/run.js';
import { callout, card, grid, initPage, runButton, specEditor, table, tiles, toolbarNumber } from '../ui.js';

const page = initPage({
  id: 'cognitivism',
  title: 'Cognitivism: memory, retrieval and the nine events',
  context: 'models: memory · interleaving · schema · gagne · fused kernels in the space Cl(4, 1)',
  description: 'The mind as an information processor: knowledge is encoded into long-term memory as connected <b>schemas</b> and must be ' +
    'retrieved to be used. Rereading and drilling do little; <b>retrieval practice</b>, <b>spaced practice</b> and <b>interleaving</b> make ' +
    "knowledge stick (Brown, Roediger & McDaniel 2014). A memory trace's stability grows most when it is recalled just as it was becoming hard " +
    "to recall. <b>Gagné's nine events</b> set the external conditions of learning; the last card removes each event in turn from a full lesson." +
    " <i>In the space:</i> a memory trace is a point that forgetting turns away from the learner (a rotor in the N–A plane); retrieval succeeds inside the cap of a uniform draw; execution, discrimination and spreading activation are mixtures.",
});
specEditor(page, ['population', 'course', 'simulation']);

const testDay = toolbarNumber(page, 'test on day', 30, { step: 1 });
const gap = toolbarNumber(page, 'spacing gap (days)', 3, { step: 1 });

runButton(page, 'Run', async (spec) => {
  const { results: [memory, interleaving, schema, gagne], note } = await runModels([
    ['memory', { ...spec, memory: { testDay: testDay(), gap: gap(), horizon: Math.max(45, testDay() + 10) } }],
    ['interleaving', spec],
    ['schema', spec],
    ['gagne', { ...spec, sim: { ...spec.sim, learners: Math.min(spec.sim.learners, 400) } }],
  ], page);
  const m = { ...memory, interleaving, ...schema };
  const by = (name) => m.plans.find((p) => p.name.startsWith(name));
  // Lasting recall: at the end of the horizon, well after the test (when the cram has faded).
  const lasting = (p) => p.recall[p.recall.length - 1];
  const horizon = m.plans[0].days[m.plans[0].days.length - 1];
  const massed = m.plans[0], best = m.plans[m.bestLasting];
  tiles(page.content, [
    { label: `Recall on day ${m.testDay}: massed rereading`, value: fmt.pct(massed.recallAtTest, 0) },
    { label: `Best lasting recall (day ${horizon})`, value: fmt.pct(lasting(best), 0), hint: best.name },
    { label: 'Interleaved vs blocked test', value: `${fmt.pct(m.interleaving.interleaved.test, 0)} vs ${fmt.pct(m.interleaving.blocked.test, 0)}`,
      hint: `practice: ${fmt.pct(m.interleaving.interleaved.practiceMean, 0)} vs ${fmt.pct(m.interleaving.blocked.practiceMean, 0)}` },
    { label: 'Full nine-event lesson', value: fmt.num(gagne.full, 3), hint: `presentation only: ${fmt.num(gagne.presentOnly, 3)}` },
  ]);
  const g = grid(page.content);
  lineChart(card(g, 'Forgetting curves under six study plans', `Mean recall over learners. All plans start with one encoding on day 0 and add the same number of study events.`), {
    x: Array.from(m.plans[0].days), xLabel: 'day', yLabel: 'P(recall)',
    series: m.plans.map((p, i) => ({ name: p.name, y: Array.from(p.recall), colorIndex: i })),
  });
  const sp = by('spaced retrieval');
  lineChart(card(g, 'Spaced retrieval: spread across learners', 'Band: 10% to 90% of learners.'), {
    x: Array.from(sp.days), xLabel: 'day', yLabel: 'P(recall)', series: [{ name: sp.name, y: Array.from(sp.recall), band: { lower: Array.from(sp.q10), upper: Array.from(sp.q90) } }],
  });
  lineChart(card(g, 'Desirable difficulty', 'Stability multiplier of one study event against how retrievable the item was at the time.'), {
    x: Array.from(m.gainR), xLabel: 'retrievability at review', yLabel: 'stability × ',
    series: [{ name: 'retrieval (success)', y: Array.from(m.gainRetrieval) }, { name: 'restudy', y: Array.from(m.gainRestudy), colorIndex: 1 }],
  });
  const il = m.interleaving;
  lineChart(card(g, 'Blocked vs interleaved practice', 'Accuracy on each practice problem (AND, OR, NOT searches). Blocked practice looks better while practising.'), {
    x: Array.from(il.blocked.practice, (_, i) => i + 1), xLabel: 'practice problem', yLabel: 'accuracy',
    series: [{ name: 'blocked', y: Array.from(il.blocked.practice) }, { name: 'interleaved', y: Array.from(il.interleaved.practice), colorIndex: 1 }],
  });
  barChart(card(g, 'Mixed test after practice', 'Choosing the right procedure is learned only when practice forces a choice.'), {
    labels: ['test accuracy', 'chooses right procedure', 'executes procedure'],
    series: [{ name: 'blocked', values: [il.blocked.test, il.blocked.choice, il.blocked.execution] }, { name: 'interleaved', values: [il.interleaved.test, il.interleaved.choice, il.interleaved.execution] }],
    yFormat: (v) => fmt.pct(v, 0),
  });
  const nodes = m.schemaPrimed.nodes;
  barChart(card(g, 'Schemas: recalling "AND" from each cue', `AND encoded while recalling prior knowledge has ${m.schemaPrimed.degree} links, without it ${m.schemaPlain.degree}.`), {
    labels: nodes.slice(0, -1), series: [{ name: 'prior knowledge stimulated', values: Array.from(m.schemaPrimed.recallFrom).slice(0, -1) },
      { name: 'not stimulated', values: Array.from(m.schemaPlain.recallFrom).slice(0, -1) }], yFormat: (v) => fmt.pct(v, 0),
  });
  barChart(card(g, "Gagné's nine events: the cost of leaving one out", 'Delayed test score of the full lesson against the lesson without each event, on the shared population.'), {
    labels: ['attention', 'objectives', 'prior recall', 'stimulus', 'guidance', 'performance', 'feedback', 'assessment', 'retention'], series: [{ name: 'drop in score', values: gagne.events.map((e) => gagne.full - e.withoutIt) }], yFormat: (v) => v.toFixed(3),
  });
  table(card(g, 'Study plans', 'Cramming wins the test the next day and loses afterwards.'), ['plan', `recall day ${m.testDay}`, `recall day ${horizon}`, 'stability (days)', 'retrieval success'],
    m.plans.map((p) => [p.name, fmt.pct(p.recallAtTest, 1), fmt.pct(lasting(p), 1), fmt.num(p.stability, 1), fmt.pct(p.successRate, 0)]));
  table(card(g, "Gagné's nine events", ''), ['event', 'score without it', 'drop'],
    gagne.events.map((e) => [e.event, fmt.num(e.withoutIt, 3), fmt.num(gagne.full - e.withoutIt, 3)]));
  callout(page.content, 'Gagné does not address emotion: the nine-event lesson leaves the humanist levers at zero. Compare it with the blended design on ' +
    '<a href="theories.html">Theories compared</a>.');
  return note;
});
