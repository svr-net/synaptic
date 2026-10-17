import { barChart, fmt, lineChart } from '../charts.js';
import { runModels } from '../engine/run.js';
import { callout, card, grid, initPage, runButton, table, tiles, toolbarNumber } from '../ui.js';

const page = initPage({
  id: 'reflective',
  title: 'Reflective teaching',
  context: 'models: reflection · studentFeedback · criticalReflection · professionalGrowth · fused kernels in the space Cl(4, 1)',
  description: 'Chapter 14: good instructors assess themselves as well as their learners. <b>Reflective practice</b> is intentional and analytic — ' +
    'the four steps of York-Barr et al. (what? why? so what? now what?) in a journal, soon after the session, ending in a change of practice; ' +
    '<b>evidence</b> beside it — a video, a peer observation, Brookfield\'s critical incident questionnaire — shows what the instructor cannot see ' +
    'from inside. Change one or two things at a time; be honest but kind to yourself. <b>Student feedback</b> is read for patterns, not single ' +
    'comments; ratings say little about learning and are biased. <b>Critical reflection</b> asks for whom a "good" practice does not hold. ' +
    '<b>Professional development</b> — communities of practice, observing colleagues, continuing education, the literature, becoming a student ' +
    'again — keeps the learning going.' +
    ' <i>In the space:</i> delivery, teaching and blind spots as cognitive points; efficacy and comfort as affective points; entitlement to question ' +
    'as a social point; a change of habit, understanding a segment and speaking up as cap tests.',
});

const sessions = toolbarNumber(page, 'sessions', 30, { step: 5, width: 50 });
const classSize = toolbarNumber(page, 'class size', 25, { step: 5, width: 50 });

runButton(page, 'Run', async (spec) => {
  const { results: [reflection, feedback, critical, growth], note } = await runModels([
    ['reflection', { ...spec, reflection: { sessions: sessions() } }],
    ['studentFeedback', { ...spec, studentFeedback: { classSize: classSize() } }],
    ['criticalReflection', spec],
    ['professionalGrowth', spec],
  ], page);
  const d = reflection.designs, rules = feedback.rules, c = critical.designs, gr = growth.designs;
  tiles(page.content, [
    { label: 'The four-step journal', value: `${fmt.pct(d[1].final, 0)} → ${fmt.pct(d[2].final, 0)}`, hint: 'teaching quality after the term: informal reflection → the journal' },
    { label: 'Student feedback beside it', value: `${fmt.pct(d[2].blindSpots, 0)} → ${fmt.pct(d[5].blindSpots, 0)}`, hint: 'blind spots (tics, fillers) after the term' },
    { label: 'Patterns, not comments', value: `${rules[0].falseRevisions.toFixed(1)} → ${rules[1].falseRevisions.toFixed(1)}`, hint: 'segments that worked but were revised anyway, per session' },
    { label: 'Critical reflection', value: `${fmt.pct(c[0].lowestGroup, 0)} → ${fmt.pct(c[10].lowestGroup, 0)}`, hint: 'learning of the group the defaults serve least' },
  ]);

  const methodLabels = ['none', 'informal', 'journal', '+ video', '+ peer', '+ CIQ', '+ all three'];
  const g = grid(page.content);
  lineChart(card(g, 'A term of teaching', `What each session teaches (the mean chance of the seven aspects of delivery), over ${reflection.sessions} sessions.`), {
    x: Array.from(d[0].quality, (_, s) => s + 1), xLabel: 'session', yLabel: 'teaching quality',
    series: d.slice(0, 7).map((x, i) => ({ name: x.name, y: Array.from(x.quality), colorIndex: i })),
  });
  barChart(card(g, 'Ways of reflecting', 'Teaching quality at the end of the term, the blind spots, and how often the aspect worked on was really one of the two weakest.'), {
    labels: methodLabels,
    series: [['quality', 'final'], ['blind spots', 'blindSpots'], ['right target', 'rightTarget']]
      .map(([name, key], i) => ({ name, values: d.slice(0, 7).map((x) => x[key]), colorIndex: i })),
    yFormat: (v) => fmt.pct(v, 0),
  });
  barChart(card(g, 'How to reflect', 'The journal (two changes at a time, right after the session, kind to yourself) against one change, every change at once, a week later, and harsh.'), {
    labels: ['journal', 'one change', 'everything', 'a week later', 'harsh'],
    series: [['quality', 'final'], ['self-efficacy', 'efficacy']].map(([name, key], i) => ({ name, values: [d[2], ...d.slice(7)].map((x) => x[key]), colorIndex: i })),
    yFormat: (v) => fmt.pct(v, 0),
  });
  table(card(g, 'The four-step reflective process', 'York-Barr et al. (2006), Table 14.1; the journal model describes before it judges and ends in an action.'), ['step'],
    reflection.steps.map((s) => [s]));

  const g2 = grid(page.content);
  barChart(card(g2, 'Reading student feedback', `A class of ${feedback.classSize}, five segments, sometimes one of them confusing. The confusing segment caught, segments that worked revised anyway, and clarity after the revision.`), {
    labels: ['any comment', 'patterns', 'ratings', 'named CIQ', 'patterns, bias', 'ratings, bias'],
    series: [{ name: 'caught', values: rules.map((r) => r.caught) }, { name: 'clarity after', values: rules.map((r) => r.after), colorIndex: 1 }],
    yFormat: (v) => fmt.pct(v, 0),
  });
  lineChart(card(g2, 'What counts as a pattern', 'Revising a segment once a share of the class names it: too low and every grumble is a revision; too high and the real problem is missed.'), {
    x: feedback.threshold.map((t) => t.threshold), xLabel: 'share of the class that named the segment', yLabel: '',
    series: [{ name: 'confusing segment caught', y: feedback.threshold.map((t) => t.caught), markers: true },
      { name: 'segments revised needlessly', y: feedback.threshold.map((t) => t.falseRevisions), colorIndex: 1, markers: true }],
  });
  barChart(card(g2, 'Ratings', `The same teaching, rated: the mean rating (1–5) for an instructor rated without and with bias. The correlation of a learner's rating with what they learned is ${rules[0].ratingLearning.toFixed(2)}.`), {
    labels: ['without bias', 'with bias'],
    series: [{ name: 'mean rating', values: [rules[1].rating, rules[4].rating] }, { name: 'learned (×5)', values: [rules[1].learned * 5, rules[4].learned * 5], colorIndex: 1 }],
    yFormat: (v) => v.toFixed(2),
  });
  table(card(g2, 'Critical incident questionnaire', 'Brookfield (2006): anonymous, at the end of a class.'), ['question'], feedback.ciq.map((q) => [q]));

  const g3 = grid(page.content);
  const short = ['defaults', 'circle', 'circle, explained', 'volunteers', '"pass"', 'invitation', 'openness shown', 'max choice', 'scaffolded choice', 'critical examples', 'all'];
  barChart(card(g3, 'Critical reflection: overall', 'Each alternative to the defaults on its own: learners who spoke, comfort, learners who questioned the teacher, and learning.'), {
    labels: short,
    series: [['spoke', 'spoke'], ['comfort', 'comfort'], ['questioned', 'questioned'], ['learned', 'learned']]
      .map(([name, key], i) => ({ name, values: c.map((x) => x[key]), colorIndex: i })),
    yFormat: (v) => fmt.pct(v, 0),
  });
  for (const [title, js, key, note2] of [
    ['A circle of chairs', [0, 1, 2], 'comfort', 'Comfort by group: a circle invites the confident and exposes the shy and those new to the norms — less when the reason is explained.'],
    ['Calling on learners', [0, 3, 4], 'spoke', 'Learners who spoke, by group: called on, volunteers only, called on with a "pass".'],
    ['The teacher as expert', [0, 5, 6], 'questioned', 'Learners who questioned the teacher: an invitation alone is not believed; openness shown in practice is.'],
    ['Maximum choice, minimum direction', [0, 7, 8], 'learned', 'Learning by group: directed, maximum choice, scaffolded choice.'],
  ])
    barChart(card(g3, title, note2), {
      labels: critical.groups.map((x) => x.replace('confident, at home with the norms', 'confident').replace('new to American classroom norms', 'new to norms').replace('learning English', 'English').replace('novices returning to learning', 'novices')),
      series: js.map((j, i) => ({ name: short[j], values: c[j].groups.map((x) => x[key]), colorIndex: i })),
      yFormat: (v) => fmt.pct(v, 0),
    });
  table(card(g3, 'Activity 14.2: assumptions to examine', 'For whom might each not hold? This page models the second and the fourth.'), ['assumption'], critical.assumptions.map((a) => [a]));

  const g4 = grid(page.content);
  lineChart(card(g4, 'Three years of professional growth', 'What a month of sessions reaches: teaching less the expert blind spot.'), {
    x: Array.from(gr[0].reach, (_, m) => m + 1), xLabel: 'month', yLabel: 'reach',
    series: gr.map((x, i) => ({ name: x.name, y: Array.from(x.reach), colorIndex: i })),
  });
  barChart(card(g4, 'Sources of growth', 'Gain in teaching (logit), self-efficacy, the expert blind spot, and ideas that backfired.'), {
    labels: ['alone', 'community', 'copying', 'adapting', 'courses', 'literature', 'student again', 'all'],
    series: [['gain', 'gain'], ['self-efficacy', 'efficacy'], ['blind spot', 'blindSpot'], ['backfired', 'backfired']]
      .map(([name, key], i) => ({ name, values: gr.map((x) => x[key]), colorIndex: i })),
    yFormat: (v) => v.toFixed(2),
  });
  table(card(g4, 'Where to look', 'Professional development outlets named in the chapter.'), ['kind', 'examples'], growth.outlets.map((o) => [o.kind, o.examples]));

  const g5 = grid(page.content, true);
  table(card(g5, 'Reflective practices in the chapter', 'Where each appears on this page.'), ['practice', 'where'], [
    ['Four-step reflective process, journaling (Tompkins, Danielson)', 'ways of reflecting; how to reflect'],
    ['Video recording: one or two changes at most', 'ways of reflecting (+ video); how to reflect (everything at once)'],
    ['Peer observation (Example 14.1, Fair/Good/Excellent)', 'ways of reflecting (+ peer)'],
    ['Student feedback, the critical incident questionnaire', 'ways of reflecting (+ CIQ); reading student feedback; what counts as a pattern'],
    ['Negative events dominate memory; be kind to yourself', 'how to reflect (a week later, harsh)'],
    ['Critical reflection: seating, calling, power, self-direction, examples', 'critical reflection'],
    ['Communities of practice, observing colleagues (adapt, do not copy)', 'sources of growth'],
    ['Continuing education, the literature, becoming a student again', 'sources of growth'],
  ]);
  callout(page.content, 'Delivery and teaching are points of the <a href="core.html">space</a>; the feedback cycle is the one of the <a href="assessment.html">assessment page</a>; ' +
    'comfort and speaking up are those of the <a href="strategies.html">discussion model</a>.');
  return note;
});
