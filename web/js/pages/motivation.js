import { barChart, fmt, lineChart } from '../charts.js';
import { modelButton, runModels } from '../engine/run.js';
import { card, grid, initPage, table, tiles } from '../ui.js';

const page = initPage({
  id: 'motivation',
  title: 'Motivation and mindset',
  context: 'models: motivation · fused kernels in the space Cl(4, 1)',
  description: "<b>Svinicki</b>: motivation balances the value of the learning against the expectation of success (self-efficacy), M = V·E. " +
    'Five levers raise them: relevance, appropriate challenge, choice and control, setting learners up for success, and guided self-assessment. ' +
    "<b>Dweck</b>: learners with a growth mindset meet failure with more effort and new strategies; those with a fixed mindset withdraw. " +
    'Praising process and saying "not yet" moves learners towards a growth mindset; praising ability moves them away.' +
    ' <i>In the space:</i> value, self-efficacy and mindset are points; success is a cap test of a dilated attempt; efficacy follows experience by a mixture.',
});

modelButton(page, 'Run', async (spec) => {
  const { results: [r], note } = await runModels([['motivation', spec]], page);
  const none = r.terms[0], all = r.terms[r.terms.length - 1];
  const [ability, , process] = r.mindset;
  tiles(page.content, [
    { label: 'Ability gain over a term', value: `${fmt.num(none.gain, 2)} → ${fmt.num(all.gain, 2)}`, hint: 'no levers → all five' },
    { label: 'Motivation at term end', value: `${fmt.num(none.finalMotivation, 2)} → ${fmt.num(all.finalMotivation, 2)}` },
    { label: 'Still trying after 30 weeks', value: `${fmt.pct(ability.persist, 0)} → ${fmt.pct(process.persist, 0)}`, hint: 'praise ability → praise process' },
    { label: 'Mindset at the end', value: `${fmt.num(ability.mindset[ability.mindset.length - 1], 2)} → ${fmt.num(process.mindset[process.mindset.length - 1], 2)}`, hint: '0 fixed, 1 growth' },
  ]);
  const g = grid(page.content);
  const weeks = Array.from(none.motivation, (_, i) => i + 1);
  lineChart(card(g, 'Motivation M = V × E over a term', 'Each lever alone, none, and all five together.'), {
    x: weeks, xLabel: 'week', yLabel: 'motivation', series: r.terms.map((t, i) => ({ name: t.label, y: Array.from(t.motivation), colorIndex: i % 8, dash: i === 0 })),
  });
  barChart(card(g, 'Ability gain by lever', ''), {
    labels: r.terms.map((t) => t.label), series: [{ name: 'ability gain', values: r.terms.map((t) => t.gain) }], yFormat: (v) => v.toFixed(2),
  });
  lineChart(card(g, 'Appropriate challenge', 'Interest in a task against its difficulty relative to the learner: too easy bores, too hard overwhelms.'), {
    x: Array.from(r.offset), xLabel: 'difficulty − ability', yLabel: 'value added', series: [{ name: 'interest', y: Array.from(r.interest) }],
  });
  const mw = Array.from(process.persisting, (_, i) => i + 1);
  lineChart(card(g, 'Persistence by feedback style', 'Share of learners still engaged with the subject.'), {
    x: mw, xLabel: 'week', yLabel: 'share', series: r.mindset.map((m, i) => ({ name: m.feedback, y: Array.from(m.persisting), colorIndex: i })),
  });
  lineChart(card(g, 'Skill by starting mindset', 'Solid: learners who start with a growth mindset; dashed: fixed.'), {
    x: mw, xLabel: 'week', yLabel: 'skill gained',
    series: r.mindset.flatMap((m, i) => [{ name: `${m.feedback}: growth`, y: Array.from(m.skillGrowth), colorIndex: i }, { name: `${m.feedback}: fixed`, y: Array.from(m.skillFixed), colorIndex: i, dash: true }]),
  });
  table(card(g, 'Feedback styles', ''), ['feedback', 'skill gain', 'persisting', 'final mindset'],
    r.mindset.map((m) => [m.feedback, fmt.num(m.gain, 2), fmt.pct(m.persist, 0), fmt.num(m.mindset[m.mindset.length - 1], 2)]));
  return note;
});
