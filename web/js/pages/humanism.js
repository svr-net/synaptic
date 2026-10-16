import { run } from '../syn-client.js';
import { barChart, fmt, lineChart } from '../charts.js';
import { card, grid, initPage, runButton, table, tiles, toolbarNumber } from '../ui.js';

const page = initPage({
  id: 'humanism',
  title: 'Humanism: needs, emotions and self-direction',
  context: 'syn/humanism: needActivation · learningReadiness · simulateInformationSearch',
  description: "Learners are whole people. <b>Maslow's hierarchy</b>: higher needs (esteem, self-actualisation) motivate only once lower ones " +
    '(safety, belonging) are met, and an empathetic teacher can meet some of them. <b>Library anxiety</b> (Mellon) and the feelings of ' +
    "<b>Kuhlthau's Information Search Process</b> (uncertainty, optimism, confusion, clarity, confidence) interfere with the cognitive work " +
    'of research; acknowledging them as normal and offering support lowers the interference.',
});

const anxiety = toolbarNumber(page, 'library anxiety (0–1)', 0.5, { step: 0.05 });
const deadline = toolbarNumber(page, 'deadline (days)', 42, { step: 1 });

runButton(page, 'Run', async (spec) => {
  const r = await run('humanism', { ...spec, humanism: { anxiety: anxiety(), deadline: deadline() } });
  const [none, half, full] = r.isp;
  tiles(page.content, [
    { label: 'Dominant need (student)', value: r.dominant, hint: `with support: ${r.dominantSupported}` },
    { label: 'On time, no support', value: fmt.pct(none.completedOnTime, 0), hint: `${fmt.num(none.daysInStage[2], 1)} days exploring` },
    { label: 'On time, full support', value: fmt.pct(full.completedOnTime, 0), hint: `${fmt.num(full.daysInStage[2], 1)} days exploring` },
    { label: 'Satisfied with the result', value: `${fmt.pct(none.satisfied, 0)} → ${fmt.pct(full.satisfied, 0)}`, hint: 'focus formulated early enough' },
  ]);
  const g = grid(page.content);
  lineChart(card(g, 'Readiness for learning', 'Gate product through belonging, as safety and belonging rise together; dashed: with an empathetic, inclusive climate.'), {
    x: Array.from(r.needGrid), xLabel: 'safety and belonging met', yLabel: 'readiness',
    series: [{ name: 'readiness', y: Array.from(r.readiness) }, { name: 'with supportive climate', y: Array.from(r.readinessWithClimate), dash: true, colorIndex: 2 }],
  });
  barChart(card(g, "A student's needs", 'Satisfaction of each level, and how much each level can currently motivate (gated by the levels below).'), {
    labels: r.needs, series: [{ name: 'satisfied', values: Array.from(r.student) }, { name: 'satisfied with support', values: Array.from(r.supported) },
      { name: 'can motivate', values: Array.from(r.activation) }, { name: 'can motivate with support', values: Array.from(r.activationSupported) }],
    yFormat: (v) => v.toFixed(2),
  });
  lineChart(card(g, 'Felt uncertainty during a research assignment', 'Mean over students still working. The peak is the exploration stage.'), {
    x: Array.from(none.days), xLabel: 'day', yLabel: 'uncertainty',
    series: r.isp.map((x, i) => ({ name: `support ${x.support}`, y: Array.from(x.uncertainty), colorIndex: i })),
  });
  lineChart(card(g, 'Where students are (no support)', `Share of students in each ISP stage; the deadline is day ${r.deadline}.`), {
    x: Array.from(none.days), xLabel: 'day', yLabel: 'share',
    series: [...r.stages.map((s, k) => ({ name: s.name, y: Array.from(none.stageShare[k]), colorIndex: k })), { name: 'done', y: Array.from(none.stageShare[6]), colorIndex: 7, dash: true }],
  });
  table(card(g, 'Kuhlthau stages', 'Typical feelings and the days students spend in each stage.'), ['stage', 'feeling', 'uncertainty', 'days (none)', 'days (half)', 'days (full)'],
    r.stages.map((s, k) => [s.name, s.feeling, fmt.num(s.uncertainty, 2), fmt.num(none.daysInStage[k], 1), fmt.num(half.daysInStage[k], 1), fmt.num(full.daysInStage[k], 1)]));
});
