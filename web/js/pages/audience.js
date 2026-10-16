import { run } from '../syn-client.js';
import { barChart, fmt, lineChart } from '../charts.js';
import { callout, card, grid, initPage, runButton, table, tiles, toolbarNumber } from '../ui.js';

const page = initPage({
  id: 'audience',
  title: 'Identifying audience needs',
  context: 'syn/audience: simulateOneShot · PriorSource · PreAssessment · simulateCommunity · IspParams (zones of intervention)',
  description: 'Chapter 7: before a one-shot session the instructor knows the class only through <b>audience research</b> (Project Information ' +
    'Literacy, Pew), <b>community data</b> (census, campus and community offices, the curriculum) and, if there is time, a <b>pre-assessment</b>. ' +
    'Each source is a belief about the class, and each pre-assessment instrument is a measurement with a cost in class time and analysis time and ' +
    'its own biases (recall is not understanding; public answers flatter). The instructor combines them by precision weighting, pitches the ' +
    'session at the class\'s ZPD and may group learners by score. Demographics describe groups, not individuals, and must be reported in aggregate. ' +
    'Kuhlthau\'s zones of intervention ask when in the research process a session helps.',
});

const classSize = toolbarNumber(page, 'class size', 25, { step: 1, width: 60 });
const minutes = toolbarNumber(page, 'session (min)', 50, { step: 5, width: 60 });

runButton(page, 'Run', async (spec) => {
  const r = await run('audience', { ...spec, audience: { classSize: classSize(), sessionMinutes: minutes() } });
  const P = r.priors.length;
  const assumption = 0, research = 1;
  const best = r.instruments.reduce((a, b) => (b.gain[research] > a.gain[research] ? b : a));
  const c = r.community;
  const bestZone = r.zones.reduce((a, z) => (Math.max(...z.onTime) > Math.max(...a.onTime) ? z : a));
  tiles(page.content, [
    { label: 'Best instrument (research prior)', value: best.instrument, hint: `learning gain ${fmt.num(best.gain[research], 4)} vs ${fmt.num(r.instruments[0].gain[research], 4)} without` },
    { label: 'Individual needs explained', value: `${fmt.pct(c.intersectionR2, 0)} vs ${fmt.pct(c.preAssessmentR2, 0)}`, hint: 'demographic intersections vs a pre-assessment' },
    { label: 'Best session timing', value: `${bestZone.stage}`, hint: `on time ${fmt.pct(Math.max(...bestZone.onTime), 0)} vs ${fmt.pct(r.noSession, 0)} with no session` },
    { label: 'Suppressed at k = 10', value: fmt.pct(c.suppressedCells[2], 0), hint: `of ${c.cells} intersection cells (${fmt.pct(c.suppressedLearners[2], 1)} of learners)` },
  ]);
  const g = grid(page.content);
  barChart(card(g, 'Value of a pre-assessment, by what the instructor already knows', 'Gain in learning from each instrument over pitching the session on the prior alone (same classes, same learners).'), {
    labels: r.instruments.slice(1).map((x) => x.instrument),
    series: r.priors.map((name, k) => ({ name, values: r.instruments.slice(1).map((x) => x.value[k]), colorIndex: k })),
    yFormat: (v) => v.toFixed(3),
  });
  barChart(card(g, 'How well the class is known', 'Error of the estimated class level (logits). Pre-assessments shrink it; recall-only and public answers keep a bias.'), {
    labels: r.instruments.map((x) => x.instrument),
    series: [r.priors[assumption], r.priors[research]].map((name, i) => ({ name, values: r.instruments.map((x) => x.rmse[i === 0 ? assumption : research]), colorIndex: i })),
    yFormat: (v) => v.toFixed(2),
  });
  lineChart(card(g, 'Analysis time limits open-ended instruments', 'Share of responses read within a 3-minute analysis budget, by class size.'), {
    x: Array.from(r.sizes), xLabel: 'class size', yLabel: 'share read', series: r.bySize.map((x, i) => ({ name: x.instrument, y: Array.from(x.read), colorIndex: i, markers: true })),
  });
  lineChart(card(g, 'Differentiation: groups by pre-assessment score', 'Learning gain by number of groups. Grouping pays in mixed classes, not homogeneous ones.'), {
    x: [1, 2, 3, 4], xLabel: 'groups', yLabel: 'learning gain', series: r.grouping.map((x, i) => ({ name: `within-class sd ${x.spread}`, y: Array.from(x.gain), colorIndex: i, markers: true })),
  });
  barChart(card(g, 'Demographics describe groups, not individuals', 'Variance of individual prior knowledge explained (R²) by each source. People are intersectional.'), {
    labels: [...c.attributes, 'all intersections', 'pre-assessment', 'both'],
    series: [{ name: 'R²', values: [...Array.from(c.singleR2), c.intersectionR2, c.preAssessmentR2, c.combinedR2] }], yFormat: (v) => fmt.pct(v, 1),
  });
  lineChart(card(g, 'Privacy: aggregate data only', 'Intersection cells (and learners in them) suppressed at a minimum cell size k.'), {
    x: Array.from(c.minCellSizes), xLabel: 'minimum cell size k', yLabel: 'share suppressed',
    series: [{ name: 'cells', y: Array.from(c.suppressedCells), markers: true }, { name: 'learners', y: Array.from(c.suppressedLearners), colorIndex: 1, markers: true }],
  });
  lineChart(card(g, 'Zones of intervention', 'Share of students finishing a research paper on time, by the day of a one-shot session and the stage it targets. Dashed: no session.'), {
    x: Array.from(r.sessionDays), xLabel: 'day of the session', yLabel: 'on time',
    series: [...r.zones.map((z, i) => ({ name: `session on ${z.stage}`, y: Array.from(z.onTime), colorIndex: i, markers: true })),
      { name: 'no session', y: r.sessionDays.map(() => r.noSession), dash: true, colorIndex: 7 }],
  });
  table(card(g, 'Pre-assessment instruments', 'Properties of each instrument and its learning gain from each prior.'),
    ['instrument', 'class min', 'read min/learner', 'noise', 'bias', 'responds', 'individual', ...r.priors],
    r.instruments.map((x) => [x.instrument, fmt.num(x.classMinutes), fmt.num(x.analysisPerLearner, 2), fmt.num(x.noise, 1), fmt.num(x.bias, 1),
      fmt.pct(x.participation, 0), x.individual ? 'yes' : 'no', ...Array.from({ length: P }, (_, k) => fmt.num(x.gain[k], 4))]));
  callout(page.content, 'The pitch uses the ZPD of the <a href="social.html">social constructivism page</a> (pure elliptic CGA); the stages are those of the ' +
    '<a href="humanism.html">humanism page</a>, now with Kuhlthau\'s seventh stage, assessment.');
});
