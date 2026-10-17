import { barChart, fmt, lineChart } from '../charts.js';
import { runModels } from '../engine/run.js';
import { callout, card, grid, initPage, runButton, table, tiles, toolbarNumber } from '../ui.js';

const page = initPage({
  id: 'strategies',
  title: 'Strategies and lesson plans',
  context: 'models: lecture · discussion · scaffolding · strategies · lessonPlan · fused kernels in the space Cl(4, 1)',
  description: 'Chapter 10: stage three of Backward Design. The best practices are <b>active learning</b>, <b>scaffolding</b> (Ellis and Larkin\'s four ' +
    'steps through the ZPD), <b>connecting to what learners know</b> (analogies, stories) and a <b>variety of strategies</b> chosen for the content — not ' +
    'for "learning styles", which have no evidence behind them. <b>Lectures</b> work when they are chunked (12–15 minutes, with bridging activities); ' +
    'online, videos should stay short. <b>Discussions</b> need wait time, structures such as think-pair-share and small groups, and ways to manage ' +
    'dominant talkers. The <b>flipped classroom</b> frees class time for application if learners prepare. A <b>lesson plan</b> with time stamps, a cut ' +
    'list and a backup activity keeps a session on time.' +
    ' <i>In the space:</i> attention as an affective point that each passive minute lowers and a bridge raises; knowledge moved by mixtures; ' +
    'support as a fading dilation and the ZPD zones as caps; readiness to speak and preparation as cap tests of dilated points.',
});

const minutes = toolbarNumber(page, 'session (min)', 50, { step: 5, width: 60 });
const wait = toolbarNumber(page, 'wait time (s)', 15, { step: 5, width: 50 });

runButton(page, 'Run', async (spec) => {
  const { results: [lecture, discussion, scaffolding, strategies, plan], note } = await runModels([
    ['lecture', { ...spec, lecture: { minutes: minutes() } }],
    ['discussion', { ...spec, discussion: { wait: wait() } }],
    ['scaffolding', spec],
    ['strategies', spec],
    ['lessonPlan', spec],
  ], page);
  const straight = lecture.sessions[0], chunked = lecture.sessions.find((s) => s.chunk === 12 && !s.story && !s.analogy);
  const whole = discussion.designs[0], tps = discussion.designs[1];
  const alone = scaffolding.sequences[0], gradual = scaffolding.sequences[3];
  const newPlan = plan.designs.slice(0, 5);
  // Short labels for the bar charts (the full names are in the legends and tables).
  const formatLabels = ['whole class', 'think-pair-share', 'small groups', 'fishbowl', 'circle of voices', '+ three chips', '+ two other voices', 'closed questions'];
  const sequenceLabels = ['alone', 'demo → alone', 'demo → we do', 'gradual release', 'gradual, no alone'];
  tiles(page.content, [
    { label: 'Chunked lecture', value: `${fmt.pct(straight.retained, 0)} → ${fmt.pct(chunked.retained, 0)}`, hint: 'retained a week later: no breaks → 12-minute chunks with bridges' },
    { label: 'Think-pair-share', value: `${fmt.pct(whole.articulated, 0)} → ${fmt.pct(tps.articulated, 0)}`, hint: 'learners who put an answer into words' },
    { label: 'Gradual release', value: `${fmt.pct(alone.transfer, 0)} → ${fmt.pct(gradual.transfer, 0)}`, hint: 'success alone on a new task: on your own from the start → I do, we do, you do' },
    { label: 'A timed lesson plan', value: `${fmt.pct(newPlan[0].onTime, 0)} → ${fmt.pct(newPlan[2].onTime, 0)}`, hint: 'new instructors ending with the wrap-up and reflection: no plan → plan + cut list' },
  ]);

  const g = grid(page.content);
  lineChart(card(g, 'Attention in a lecture', `A ${minutes()}-minute session: the mean attention of the learners, minute by minute. Bridging activities (${lecture.bridgeMinutes} min) bring it back.`), {
    x: Array.from(straight.attention, (_, m) => m + 1), xLabel: 'minute', yLabel: 'attention',
    series: lecture.sessions.slice(0, 5).map((s, i) => ({ name: s.name, y: Array.from(s.attention), colorIndex: i })),
  });
  barChart(card(g, 'Chunks, stories and analogies', 'Learned by the end of the session, and retained a week later.'), {
    labels: lecture.sessions.map((s) => s.name.replace('chunks of ', '').replace(' + ', ' +')),
    series: [{ name: 'learned', values: lecture.sessions.map((s) => s.learned) }, { name: 'retained', values: lecture.sessions.map((s) => s.retained), colorIndex: 1 }],
    yFormat: (v) => fmt.pct(v, 0),
  });
  lineChart(card(g, 'Instructional videos', 'Viewers still watching, minute by minute, for videos of different lengths (attention peaks early; keep task videos short).'), {
    x: Array.from({ length: 20 }, (_, m) => m + 1), xLabel: 'minute', yLabel: 'still watching',
    series: lecture.videos.map((v, i) => ({ name: `${v.minutes} min`, y: Array.from(v.watching), colorIndex: i })),
  });
  barChart(card(g, 'Video length', 'Viewers who reach the end, and what the video teaches them.'), {
    labels: lecture.videos.map((v) => `${v.minutes} min`),
    series: [{ name: 'completed', values: lecture.videos.map((v) => v.completed) }, { name: 'learned', values: lecture.videos.map((v) => v.learned), colorIndex: 1 }],
    yFormat: (v) => fmt.pct(v, 0),
  });

  const g2 = grid(page.content);
  barChart(card(g2, 'Discussion formats', `A class of ${discussion.classSize}, eight questions, ${wait()} s of wait time: who spoke to the class, who put an answer into words, and the share of the talk taken by the five most frequent speakers.`), {
    labels: formatLabels,
    series: [['spoke to the class', 'voices'], ['articulated an answer', 'articulated'], ['top five talkers\' share', 'topFiveShare']]
      .map(([name, key], i) => ({ name, values: discussion.designs.map((d) => d[key]), colorIndex: i })),
    yFormat: (v) => fmt.pct(v, 0),
  });
  lineChart(card(g2, 'Wait time', 'Whole-class questions: learners who spoke, and learners who need more time to answer (English language learners, neurodiverse learners, learning disabilities) who spoke.'), {
    x: discussion.waitTime.map((w) => w.wait), xLabel: 'seconds of silence after a question', yLabel: '',
    series: [{ name: 'all learners', y: discussion.waitTime.map((w) => w.voices), markers: true }, { name: 'learners who need more time', y: discussion.waitTime.map((w) => w.moreTimeVoices), colorIndex: 1, markers: true }],
  });
  barChart(card(g2, 'What a discussion teaches', 'Mean gain in knowledge. Speaking teaches more than listening; open questions more than closed ones.'), {
    labels: formatLabels,
    series: [{ name: 'knowledge gained', values: discussion.designs.map((d) => d.learned) }],
    yFormat: (v) => v.toFixed(2),
  });
  barChart(card(g2, 'Scaffolding (Ellis and Larkin)', `${scaffolding.steps.join(' → ')}. Success alone on a new task, and learners who gave up.`), {
    labels: sequenceLabels,
    series: [{ name: 'success alone afterwards', values: scaffolding.sequences.map((s) => s.transfer) }, { name: 'gave up', values: scaffolding.sequences.map((s) => s.gaveUp), colorIndex: 1 }],
    yFormat: (v) => fmt.pct(v, 0),
  });
  barChart(card(g2, 'Time in each zone of the ZPD', 'Share of practice attempts in zone 1 (can already do it alone), zone 2 (can with the support at hand) and zone 3 (cannot yet).'), {
    labels: sequenceLabels,
    series: ['zone 1', 'zone 2', 'zone 3'].map((name, k) => ({ name, values: scaffolding.sequences.map((s) => s.zones[k]), colorIndex: k })),
    yFormat: (v) => fmt.pct(v, 0),
  });
  lineChart(card(g2, 'A project in stages', `The quality of each part: ${scaffolding.stageNames.join(', ')}; and of the paper as a whole.`), {
    x: [1, 2, 3, 4], xLabel: 'stage', yLabel: 'quality',
    series: scaffolding.projects.map((p, i) => ({ name: `${p.name} (paper ${fmt.pct(p.quality, 0)})`, y: Array.from(p.stages), colorIndex: i, markers: true })),
  });

  const g3 = grid(page.content);
  for (const c of strategies.styles)
    barChart(card(g3, `Learning styles: ${c.content}`, 'Matching a lesson to each learner\'s stated style teaches no more than matching it to the content, and less; the preference only pleases. A variety of modes (UDL) reaches learners who cannot use one of them.'), {
      labels: c.rules.map((x) => x.rule.replace('matched to each learner\'s stated style', 'stated style').replace('matched to the content', 'content').replace('the same lecture for everyone', 'lecture').replace('a variety of modes (UDL)', 'variety')),
      series: [['learned', 'knowledge'], ['learned, learners who cannot use one mode', 'barredKnowledge'], ['satisfied', 'satisfaction']]
        .map(([name, key], i) => ({ name, values: c.rules.map((x) => x[key]), colorIndex: i })),
      yFormat: (v) => fmt.pct(v, 0),
    });
  barChart(card(g3, 'The flipped one-shot', 'Learners who prepared, the basics they know and the higher-order skill they built in class. A pre-work check and embedding the pre-work in the course are what make it work.'), {
    labels: ['traditional', 'flipped', '+ check', 'embedded', 'embedded + check'],
    series: [['prepared', 'prepared'], ['basics', 'basics'], ['higher-order', 'higherOrder'], ['higher-order, unprepared', 'unpreparedHigherOrder']]
      .map(([name, key], i) => ({ name, values: strategies.flipped.map((f) => f[key]), colorIndex: i })),
    yFormat: (v) => fmt.pct(v, 0),
  });
  barChart(card(g3, 'Problem-based learning', 'Similar basics, more interest and transfer; full PBL takes several sessions, but its elements fit a one-shot.'), {
    labels: ['lecture', 'PBL elements', 'full PBL'],
    series: [['basics', 'basics'], ['transfer', 'transfer'], ['interest', 'interest']].map(([name, key], i) => ({ name, values: strategies.pbl.map((x) => x[key]), colorIndex: i })),
    yFormat: (v) => fmt.pct(v, 0),
  });

  const g4 = grid(page.content, true);
  table(card(g4, plan.title, `Outcomes: ${plan.outcomes.join('; ')}. Materials: ${plan.materials}.`), ['time', 'segment', 'content'],
    plan.plan.map((s) => [`${s.start}–${s.start + s.minutes} min`, s.segment, s.content]));
  const g5 = grid(page.content);
  for (const who of ['new', 'experienced']) {
    const d = plan.designs.filter((x) => x.instructor === who);
    barChart(card(g5, `Delivering the plan: ${who} instructors`, 'Sessions that end with the wrap-up and the reflection, and the learning of the session.'), {
      labels: ['no plan', 'plan', '+ cut list', '+ backup', '+ cut + backup'],
      series: [{ name: 'on time', values: d.map((x) => x.onTime) }, { name: 'learning', values: d.map((x) => x.learning), colorIndex: 1 }],
      yFormat: (v) => fmt.pct(v, 0),
    });
  }
  lineChart(card(g5, 'Which segments survive', 'New instructors: the share of each segment delivered. Without a plan the session runs out of time before the wrap-up and the reflection.'), {
    x: plan.plan.map((_, k) => k + 1), xLabel: 'segment (welcome, lecture, activity, debrief, wrap-up, reflection)', yLabel: 'delivered',
    series: newPlan.map((x, i) => ({ name: x.name, y: Array.from(x.delivered), colorIndex: i, markers: true })),
  });
  const g6 = grid(page.content, true);
  table(card(g6, 'Instructional strategies in the chapter', 'Where each strategy appears on this page.'), ['strategy', 'where'], [
    ['Lecture: chunks of 12–15 minutes, pauses, questions, storytelling, analogies', 'attention, chunks, stories and analogies'],
    ['Demonstration (I do it)', 'scaffolding'], ['Discussion: wait time, think-pair-share, fishbowl, chips, open questions', 'discussion formats, wait time'],
    ['Flipped classroom', 'the flipped one-shot'], ['Problem-based learning', 'problem-based learning'],
    ['Group work: guided discussion, circle of voices', 'discussion formats'], ['Peer instruction, pairs giving feedback', 'scaffolding (you do it together)'],
    ['Polls, writing exercises, minute paper', 'bridging activities; the reflection of the lesson plan'],
    ['Lesson plan with time stamps, cut list, backup', 'delivering the plan'],
  ]);
  callout(page.content, 'Attention and knowledge are points of the <a href="core.html">space</a>; the ZPD zones are those of the <a href="social.html">social constructivism page</a>; ' +
    'forgetting is the rotor of the <a href="cognitivism.html">cognitivism page</a>; the outcomes are those of the <a href="outcomes.html">outcomes page</a>.');
  return note;
});
