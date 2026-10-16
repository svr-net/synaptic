import { barChart, fmt, lineChart } from '../charts.js';
import { modelButton, runModels } from '../engine/run.js';
import { callout, card, el, grid, initPage, table, tiles, toolbarNumber } from '../ui.js';

const page = initPage({
  id: 'outcomes',
  title: 'Learning goals and outcomes',
  context: 'models: outcomes · fused kernels in the space Cl(4, 1)',
  description: 'Chapter 8: <b>Backward Design</b> plans a session from its results: stage 1 writes the learning outcomes, stage 2 chooses the ' +
    'evidence that they were met, stage 3 the activities. It avoids the <b>twin sins</b> of coverage-based instruction (everything, by lecture) and ' +
    'activity-based instruction (practice without the when and why). Outcomes start with an observable action verb from <b>Bloom\'s taxonomy</b>; ' +
    '"understand" and "know" cannot be measured. Content is prioritised into big ideas, the basic ideas they rest on, and what is only worth ' +
    'being familiar with; about three to four outcomes fit an hour. Shared outcomes, essential questions and affective framing make the relevance explicit.' +
    ' <i>In the space:</i> Bloom\'s levels are points; mastery climbs the hierarchy by mixtures; content mastery is a point raised by the minutes; the outcome level fits the audience\'s ZPD.',
});

const minutes = toolbarNumber(page, 'session (min)', 50, { step: 5, width: 60 });
const count = toolbarNumber(page, 'outcomes', 4, { step: 1, width: 50 });
const box = el('textarea', { rows: 4, style: 'width:100%;font:inherit;box-sizing:border-box', spellcheck: 'false' });
box.value = ['Understand Boolean operators', 'Learners will be able to apply Boolean operators to broaden and narrow a search',
  'Be familiar with the library catalogue', 'Evaluate sources for authority and currency using the CRAAP criteria'].join('\n');
page.main.insertBefore(el('section', { class: 'card' }, el('h2', { text: 'Your outcomes' }), el('p', { class: 'note', text: 'One per line, with or without "learners will be able to"; analysed on Run.' }), box), page.content);

const verdict = (a) => (a.strong ? '✓ strong' : a.measurable ? '~ measurable, too broad' : '✗ not measurable');

modelButton(page, 'Run', async (spec) => {
  const texts = box.value.split('\n').map((s) => s.trim()).filter(Boolean);
  const { results: [r], note } = await runModels([['outcomes', { ...spec, outcomes: { minutes: minutes(), outcomes: count(), texts } }]], page);
  const [coverage, activity, backward] = r.plans;
  const hour = r.sweep.find((s) => s.minutes === 60);
  const bestK = r.outcomeCounts[hour.enduring.indexOf(Math.max(...hour.enduring))];
  tiles(page.content, [
    { label: 'Behaviour change', value: `${fmt.pct(backward.behaviourChange, 0)}`, hint: `backward design vs ${fmt.pct(coverage.behaviourChange, 0)} coverage, ${fmt.pct(activity.behaviourChange, 0)} activity` },
    { label: 'Outcomes in an hour', value: `${bestK}`, hint: 'most enduring learning in a 60-minute session' },
    { label: 'Coverage touches', value: `${coverage.itemsTouched} topics`, hint: `outcomes met ${fmt.pct(coverage.outcomesMet, 0)}` },
    { label: 'Recall evidence for an apply outcome', value: `+${fmt.pct(r.evidenceBias[0], 0)}`, hint: 'overstates achievement' },
  ]);
  const w = grid(page.content, true);
  table(card(w, 'Your outcomes', 'Verb, Bloom level and issues for each line of the box above.'), ['outcome', 'verb', 'level', 'verdict', 'issues'],
    r.analysed.map((a) => [a.text, a.verb || '—', a.levelName, verdict(a), a.issues.join('; ') || '—']));
  table(card(w, 'Table 8.3: poorly and strongly worded', 'The analyser applied to the chapter\'s examples.'), ['poorly worded', 'verdict', 'strongly worded', 'level', 'verdict'],
    r.table83.map((x) => [x.poor.text, verdict(x.poor), x.strong.text, x.strong.levelName, verdict(x.strong)]));
  const g = grid(page.content);
  barChart(card(g, 'The twin sins', `${minutes()}-minute session, same learners. Mean mastery of basic ideas, understanding of big ideas, and transfer.`), {
    labels: r.plans.map((p) => p.plan),
    series: [['basic ideas', 'basics'], ['big ideas', 'bigIdeas'], ['nice to know', 'niceToKnow'], ['transfer', 'transfer'], ['behaviour change', 'behaviourChange']]
      .map(([name, key], i) => ({ name, values: r.plans.map((p) => p[key]), colorIndex: i })),
    yFormat: (v) => fmt.pct(v, 0),
  });
  lineChart(card(g, 'Three to four outcomes per hour', 'Enduring learning (basic ideas + 2 × big ideas) by the number of outcomes, for three session lengths.'), {
    x: Array.from(r.outcomeCounts), xLabel: 'outcomes', yLabel: 'enduring learning',
    series: r.sweep.map((s, i) => ({ name: `${s.minutes} min`, y: Array.from(s.enduring), colorIndex: i, markers: true })),
  });
  lineChart(card(g, 'Outcomes met', 'Share of (learner, outcome) pairs reaching mastery 0.6.'), {
    x: Array.from(r.outcomeCounts), xLabel: 'outcomes', yLabel: 'met',
    series: r.sweep.map((s, i) => ({ name: `${s.minutes} min`, y: Array.from(s.met), colorIndex: i, markers: true })),
  });
  barChart(card(g, 'Stage 2: evidence must match the outcome', 'An "apply" outcome assessed at each Bloom level: bias of the measured achievement, and accuracy of the met / not met judgement.'), {
    labels: Array.from(r.levels),
    series: [{ name: 'bias', values: Array.from(r.evidenceBias) }, { name: 'accuracy', values: Array.from(r.evidenceAccuracy), colorIndex: 1 }],
    yFormat: (v) => v.toFixed(2),
  });
  barChart(card(g, 'Shared outcomes, essential questions, caring', 'Backward design with each framing added in turn.'), {
    labels: r.framing.map((f) => f.name),
    series: [{ name: 'behaviour change', values: r.framing.map((f) => f.behaviourChange) }, { name: 'self-assessment accuracy', values: r.framing.map((f) => f.selfAssessment), colorIndex: 1 }],
    yFormat: (v) => fmt.pct(v, 0),
  });
  lineChart(card(g, 'Outcomes appropriate to the audience', 'Chance of success with the session\'s scaffold at each Bloom level, by audience. The legend names the highest level inside each audience\'s ZPD.'), {
    x: [0, 1, 2, 3, 4, 5], xLabel: 'Bloom level (remember … create)', yLabel: 'success with help',
    series: r.levelFit.map((f, i) => ({ name: `audience ${f.audience >= 0 ? '+' : ''}${f.audience} → ${r.levels[f.recommended]}`, y: Array.from(f.assisted), colorIndex: i, markers: true })),
  });
  const w2 = grid(page.content, true);
  table(card(w2, 'Figure 8.3: what to teach', 'Big ideas, the basic ideas they rest on, and what is only worth being familiar with; minutes each plan gives them.'),
    ['item', 'kind', 'builds', 'Bloom level', 'needs (min)', 'coverage', 'activity', 'backward design'],
    r.content.map((c, i) => [c.name, c.kind, c.bigIdea || '—', c.level, fmt.num(c.minutes), ...r.plans.map((p) => (p.minutes[i] > 0 ? `${fmt.num(p.minutes[i], 1)} → ${fmt.pct(p.mastery[i], 0)}` : '—'))]));
  table(card(w2, 'Bloom\'s revised taxonomy', 'Action verbs by level; remember, understand and apply are lower order.'), ['level', 'verbs'],
    r.levels.map((l, k) => [l, r.verbs[k]]));
  callout(page.content, 'The audience fit uses the ZPD of the <a href="social.html">social constructivism page</a> (pure elliptic CGA); knowing the audience is the ' +
    '<a href="audience.html">previous chapter</a>.');
  return note;
});
