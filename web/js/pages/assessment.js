import { run } from '../syn-client.js';
import { barChart, fmt, lineChart } from '../charts.js';
import { callout, card, grid, initPage, runButton, table, tiles, toolbarNumber, toolbarSelect } from '../ui.js';

const page = initPage({
  id: 'assessment',
  title: 'Assessing learning',
  context: 'syn/assessment: scoreActivities · simulateSignals · simulateAssessedSession · simulateAssessmentCycle · simulateRubric · simulateSelfAssessment',
  description: 'Chapter 9: <b>assessment</b> measures progress toward the outcomes to improve teaching and learning; it is not grading (attendance, ' +
    'effort, conduct) and not evaluation (satisfaction). Activities are <b>formative</b> or <b>summative</b>, <b>direct</b> or <b>indirect</b>, formal or ' +
    'informal, traditional or authentic, and are chosen first for alignment with the outcome, then for the audience and the time. The ' +
    '<b>assessment cycle</b> turns the data into better teaching, offering after offering. <b>Rubrics</b> make "good" explicit, so raters agree and ' +
    'learners know what to aim for. Self-assessments flatter the least skilled; feedback on them calibrates learners and helps them learn.',
});

const AUDIENCES = ['undergraduates', 'high school students', 'older adults (public library)', 'English-language learners'];
const minutes = toolbarNumber(page, 'session (min)', 45, { step: 5, width: 60 });
const audience = toolbarSelect(page, 'audience', AUDIENCES, AUDIENCES[0]);

runButton(page, 'Run', async (spec) => {
  const r = await run('assessment', { ...spec, assessment: { minutes: minutes() } });
  const au = r.audiences[Math.max(0, AUDIENCES.indexOf(audience()))];
  const [none, , informal, , formal] = r.sessions;
  const direct = r.cycle[3];
  const precise = r.rubrics[3].private, noRubric = r.rubrics[0].private;
  tiles(page.content, [
    { label: 'Formative checks', value: `+${fmt.pct(informal.outcomeMet - none.outcomeMet, 0)}`, hint: `learners meeting the outcome (${fmt.pct(none.outcomeMet, 0)} → ${fmt.pct(informal.outcomeMet, 0)})` },
    { label: 'Assessment cycle', value: `${fmt.pct(direct.met[0], 0)} → ${fmt.pct(direct.met[direct.met.length - 1], 0)}`, hint: `meeting the outcome after ${direct.met.length} offerings with direct evidence` },
    { label: 'Rater agreement (κ)', value: `${fmt.num(noRubric.kappa, 2)} → ${fmt.num(precise.kappa, 2)}`, hint: 'no rubric → precise rubric' },
    { label: 'Signal of learning', value: `r = ${fmt.num(r.signals.direct, 2)}`, hint: `direct assessment; grades ${fmt.num(r.signals.grade, 2)}, satisfaction ${fmt.num(r.signals.satisfaction, 2)}` },
  ]);
  const w = grid(page.content, true);
  const top = (vals) => Array.from(vals, (v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 3).map(([v, i]) => `${r.activities[i].name} (${fmt.num(v, 2)})`).join(' · ');
  table(card(w, `Selecting an assessment activity: ${au.name}, ${minutes()}-minute session`, 'The three best activities for an outcome at each Bloom level: alignment × information × fit to the audience, less the share of the session it takes.'),
    ['outcome level', 'best activities (value)'], r.levels.map((l, k) => [l, top(au.value[k])]));
  const g = grid(page.content);
  barChart(card(g, 'Assessment is not grading or evaluation', 'Correlation with what learners actually learned.'), {
    labels: ['direct assessment', 'grade', 'self-report', 'satisfaction'],
    series: [{ name: 'correlation with learning', values: [r.signals.direct, r.signals.grade, r.signals.selfReport, r.signals.satisfaction] }],
    yFormat: (v) => v.toFixed(2),
  });
  barChart(card(g, 'Formative and summative', 'Each design: learners meeting the outcome, gaps the checks missed (per class) and the error of the instructor\'s belief about the outcome.'), {
    labels: r.sessions.map((s) => s.name.replace(' check', '').replace(' + summative', ' + summ.')),
    series: [['met the outcome', 'outcomeMet'], ['missed gaps', 'missedGaps'], ['estimate error', 'estimateError']]
      .map(([name, key], i) => ({ name, values: r.sessions.map((s) => s[key]), colorIndex: i })),
    yFormat: (v) => v.toFixed(2),
  });
  lineChart(card(g, 'Chunks build on each other', 'Mean mastery of each chunk of the lesson. Formative checks catch an unclear chunk before the next builds on it.'), {
    x: none.chunkMastery.map((_, j) => j + 1), xLabel: 'chunk', yLabel: 'mastery',
    series: [none, informal, formal].map((s, i) => ({ name: s.name, y: Array.from(s.chunkMastery), colorIndex: i, markers: true })),
  });
  lineChart(card(g, 'The assessment cycle (Figure 9.1)', 'Learners meeting the outcome, offering after offering, by what the instructor learns. One part of the lesson is unclear.'), {
    x: direct.met.map((_, k) => k + 1), xLabel: 'offering', yLabel: 'met the outcome',
    series: r.cycle.map((c, i) => ({ name: c.evidence, y: Array.from(c.met), colorIndex: i, markers: true })),
  });
  barChart(card(g, 'Rubrics: agreement and validity', 'Cohen\'s κ between raters, and the correlation of one rater\'s levels with the work\'s true quality.'), {
    labels: r.rubrics.map((x) => x.kind),
    series: [{ name: 'κ', values: r.rubrics.map((x) => x.private.kappa) }, { name: 'validity', values: r.rubrics.map((x) => x.private.validity), colorIndex: 1 }],
    yFormat: (v) => v.toFixed(2),
  });
  barChart(card(g, 'Sharing the rubric with learners', 'Change in the work when learners see the rubric first: quality (sd) and scholarly references per paper. Counting references buys quantity, not quality.'), {
    labels: r.rubrics.map((x) => x.kind),
    series: [{ name: 'quality gained', values: r.rubrics.map((x) => x.shared.meanQuality - x.private.meanQuality) },
      { name: 'references added', values: r.rubrics.map((x) => x.shared.meanCount - x.private.meanCount), colorIndex: 1 }],
    yFormat: (v) => v.toFixed(2),
  });
  lineChart(card(g, 'How many levels?', 'A precise rubric with 2–6 levels: exact agreement falls as levels are added; most instructors use three or four.'), {
    x: Array.from(r.rubricLevels.levels), xLabel: 'levels', yLabel: '',
    series: [['exact agreement', 'exactAgreement'], ['κ', 'kappa'], ['validity', 'validity']].map(([name, key], i) => ({ name, y: Array.from(r.rubricLevels[key]), colorIndex: i, markers: true })),
  });
  const [alone, fb] = r.selfAssessment;
  lineChart(card(g, 'Self-assessment by quartile of skill', 'Actual and self-assessed skill. The least skilled overestimate most; feedback on self-assessments closes the gap.'), {
    x: [1, 2, 3, 4], xLabel: 'quartile of actual skill', yLabel: 'skill',
    series: [
      { name: 'actual', y: Array.from(alone.actualFirst), colorIndex: 0, markers: true },
      { name: 'self-assessed', y: Array.from(alone.perceivedFirst), colorIndex: 1, markers: true },
      { name: 'actual, after feedback', y: Array.from(fb.actualLast), colorIndex: 0, dash: true },
      { name: 'self-assessed, after feedback', y: Array.from(fb.perceivedLast), colorIndex: 1, dash: true },
    ],
  });
  lineChart(card(g, 'Feedback on self-assessment', 'Mean |self-assessed − actual| and mean skill, by round of reflection.'), {
    x: alone.calibrationError.map((_, k) => k + 1), xLabel: 'round', yLabel: '',
    series: [{ name: 'calibration error, reflection only', y: Array.from(alone.calibrationError), colorIndex: 0, markers: true },
      { name: 'calibration error, with feedback', y: Array.from(fb.calibrationError), colorIndex: 1, markers: true },
      { name: 'skill, reflection only', y: Array.from(alone.meanSkill), colorIndex: 0, dash: true },
      { name: 'skill, with feedback', y: Array.from(fb.meanSkill), colorIndex: 1, dash: true }],
  });
  const w2 = grid(page.content, true);
  table(card(w2, 'Assessment activities', 'The chapter\'s activities and their properties in the model.'),
    ['activity', 'class min', 'direct', 'formal', 'authentic', 'Bloom level', 'noise', 'reach'],
    r.activities.map((a) => [a.name, a.afterSession ? 'after class' : fmt.num(a.minutes), a.direct ? 'yes' : 'no', a.formal ? 'yes' : 'no',
      a.authentic ? 'yes' : 'no', a.measures, fmt.num(a.noise, 2), fmt.pct(a.reach, 0)]));
  callout(page.content, 'Learners master each chunk through the ZPD of the <a href="social.html">social constructivism page</a> (pure elliptic CGA); the evidence ' +
    'levels are those of the <a href="outcomes.html">outcomes page</a>.');
});
