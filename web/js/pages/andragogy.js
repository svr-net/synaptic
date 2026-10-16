import { barChart, fmt, lineChart } from '../charts.js';
import { modelButton, runModels } from '../engine/run.js';
import { card, el, grid, initPage, table, tiles } from '../ui.js';

const page = initPage({
  id: 'andragogy',
  title: 'Andragogy: how adults learn',
  context: 'models: andragogy · fused kernels in the space Cl(4, 1)',
  description: "Knowles' six assumptions about adult learners (self-direction, experience, readiness tied to social roles, problem-centred " +
    'orientation, internal motivation, the need to know why) each pair with a design feature. Engagement rises when the design matches how ' +
    "strongly the learner holds each assumption. <b>Bartle's seven steps</b> make it a process: climate, mechanisms for input, diagnosis of " +
    'needs, objectives, design, execution, evaluation and rediagnosis.' +
    ' <i>In the space:</i> assumptions and design features are points; engagement is the equator dilated by every main effect and interaction.',
});

modelButton(page, 'Run', async (spec) => {
  const { results: [r], note } = await runModels([['andragogy', spec]], page);
  const c = r.curves.map((x) => Array.from(x));
  const last = c[0].length - 1;
  tiles(page.content, [
    { label: 'Child: pedagogical → andragogical', value: `${fmt.pct(c[0][0], 0)} → ${fmt.pct(c[0][last], 0)}` },
    { label: 'Adolescent', value: `${fmt.pct(c[1][0], 0)} → ${fmt.pct(c[1][last], 0)}` },
    { label: 'Adult', value: `${fmt.pct(c[2][0], 0)} → ${fmt.pct(c[2][last], 0)}` },
    { label: 'Needs diagnosis hit rate', value: `${fmt.pct(r.cycles[0].hitRate[0], 0)} → ${fmt.pct(r.cycles[2].hitRate[0], 0)}`, hint: 'without → with learner input' },
  ]);
  const g = grid(page.content);
  lineChart(card(g, 'Engagement as the lesson moves from pedagogical to andragogical', '0: teacher-directed, subject-centred, rewards. 1: self-directed, experience-based, role-relevant, transparent.'), {
    x: Array.from(r.mix), xLabel: 'andragogical share of the design', yLabel: 'P(engaged)',
    series: ['child', 'adolescent', 'adult'].map((n, i) => ({ name: n, y: c[i], colorIndex: i })),
  });
  barChart(card(g, 'Why: change in the engagement logit, by assumption', 'Pedagogical → andragogical design. Positive for adults on every assumption; mostly negative for children.'), {
    labels: r.assumptions, series: ['child', 'adolescent', 'adult'].map((n, i) => ({ name: n, values: Array.from(r.termShift[i]), colorIndex: i })), yFormat: (v) => v.toFixed(2),
  });
  const cyc = Array.from(r.cycles[0].gap, (_, i) => i);
  lineChart(card(g, "Bartle's cycle: diagnose, teach, rediagnose", 'Mean competency gap after each cycle, by how much learner input and a cooperative climate sharpen the diagnosis.'), {
    x: cyc, xLabel: 'cycle', yLabel: 'mean gap', series: r.cycles.map((x, i) => ({ name: `input & climate ${x.input}`, y: Array.from(x.gap), colorIndex: i, markers: true })),
  });
  const box = card(g, 'Assumptions and the design features that serve them', '');
  table(box, ['assumption about adult learners', 'design feature'], r.assumptions.map((a, k) => [a, r.features[k]]));
  box.append(el('h2', { text: 'Seven steps (Bartle 2019)', style: 'margin-top:12px' }), el('ol', { class: 'plain' }, r.steps.map((s) => el('li', { text: s }))));
  return note;
});
