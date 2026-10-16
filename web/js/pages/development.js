import { barChart, fmt, lineChart } from '../charts.js';
import { modelButton, runModels } from '../engine/run.js';
import { card, el, grid, initPage, table, tiles, toolbarNumber } from '../ui.js';

const page = initPage({
  id: 'development',
  title: 'Developmental stages: Piaget and Perry',
  context: 'models: development · fused kernels in the space Cl(4, 1)',
  description: "<b>Piaget's four stages</b> (Table 3.1): sensorimotor, preoperational, concrete operational and formal operational. Children " +
    'reach them at roughly, not exactly, the nominal ages, so the fit of an activity to a class is a probability. <b>Perry\'s scheme</b>: ' +
    'dualism → multiplicity → relativism → commitment. The position shapes how a student judges sources: by authority, by personal ' +
    'resonance (all views equal) or by evidence weighed through the methods of a discipline.' +
    ' <i>In the space:</i> reaching a stage is the chance of a point at logit (age − onset)/spread; Perry transitions are cap tests; a judgement of a source dilates the equator by weighted logits.',
});

const onsetSd = toolbarNumber(page, 'stage onset spread (years)', 0.9, { step: 0.1 });

modelButton(page, 'Run', async (spec) => {
  const { results: [r], note } = await runModels([['development', { ...spec, development: { onsetSd: onsetSd() } }]], page);
  const ages = Array.from(r.ages);
  const at = (age) => ages.findIndex((a) => Math.abs(a - age) < 1e-9);
  const acc = Array.from(r.sourceAccuracy);
  tiles(page.content, [
    { label: 'Age 12 in formal operations', value: fmt.pct(r.stageDistribution[3][at(12)], 0) },
    { label: '"Anticipate consequences" fits age 9', value: fmt.pct(r.activities[4].fit[at(9)], 0), hint: `age 15: ${fmt.pct(r.activities[4].fit[at(15)], 0)}` },
    { label: 'Source judgement: multiplicity', value: fmt.pct(acc[1], 0), hint: 'pairs ordered by true evidence quality' },
    { label: 'Source judgement: relativism', value: fmt.pct(acc[2], 0) },
  ]);
  const g = grid(page.content);
  lineChart(card(g, "Piaget's stages by age", 'Probability that a child of that age is in each stage.'), {
    x: ages, xLabel: 'age (years)', yLabel: 'share', series: r.stages.map((s, k) => ({ name: `${s.name} (${s.ages})`, y: Array.from(r.stageDistribution[k]), colorIndex: k })),
  });
  lineChart(card(g, 'The climate-change lesson by age', 'Share of children for whom each activity is developmentally appropriate.'), {
    x: ages, xLabel: 'age (years)', yLabel: 'fit', series: r.activities.map((a, i) => ({ name: a.name, y: Array.from(a.fit), colorIndex: i })),
  });
  const terms = Array.from(r.perryPlain[0], (_, i) => i);
  lineChart(card(g, "Perry's scheme over college terms", 'Solid: ordinary teaching. Dashed: teaching that challenges students to compare and evaluate sources.'), {
    x: terms, xLabel: 'term', yLabel: 'share',
    series: [...r.perryNames.map((n, k) => ({ name: n, y: Array.from(r.perryPlain[k]), colorIndex: k })),
      ...r.perryNames.map((n, k) => ({ name: `${n} (challenged)`, y: Array.from(r.perryChallenged[k]), colorIndex: k, dash: true, hideLegend: true }))],
  });
  barChart(card(g, 'Judging sources by position', 'Share of source pairs ordered as their true evidence quality (0.5 = chance).'), {
    labels: r.perryNames, series: [{ name: 'accuracy', values: acc }], yFormat: (v) => fmt.pct(v, 0),
  });
  const box = card(g, 'Activities and what they demand', '');
  table(box, ['activity', 'needs', 'age 5', 'age 9', 'age 15'], r.activities.map((a) => [a.name, a.demands.join(', '),
    fmt.pct(a.fit[at(5)], 0), fmt.pct(a.fit[at(9)], 0), fmt.pct(a.fit[at(15)], 0)]));
  box.append(el('p', { class: 'note', text: 'The classroom engine uses the stage distribution too: abstract content (Boolean logic) needs formal operations, which scaffolding partly substitutes for.' }));
  return note;
});
