import { fmt, lineChart } from '../charts.js';
import { engineNote, runModel } from '../engine/run.js';
import { PAGES, card, el, grid, initPage, runButton, table, tiles } from '../ui.js';

const page = initPage({
  id: 'index',
  title: 'How people learn, as models you can run',
  context: 'synaptic · one space (pure elliptic CGA, Cl(4, 1)) · every model a fused WGSL kernel · WebGPU, or the software GPU',
  description:
    'Interactive front end for a library modelling chapters 3, 7, 8, 9, 10 and 14 of <i>Instruction in Libraries and Information Centers</i> (Illinois Open ' +
    'Publishing Network): how people learn, identifying audience needs, establishing learning goals and outcomes, assessing learning, and ' +
    'selecting instructional strategies and creating lesson plans, and practicing reflective teaching; and functional geometric algebra (Pustejovsky, 2026), ' +
    'in which learners learn the meaning of a text in the same space. ' +
    'Every model lives in <b>one space</b>, the pure elliptic conformal geometric algebra Cl(4, 1): learners, tasks, memories, needs, outcomes, ' +
    'raters and stimuli are points; probabilities and shares are read off them (<a href="core.html">scalars are the miniature basis</a>); learning, ' +
    'forgetting, support and feedback are dilations, rotors and mixtures; thresholds and zones are caps. Every model runs as a <b>fused compute ' +
    'kernel</b> written once in WGSL: on WebGPU, or, where there is no GPU, on the library\'s software GPU, which runs the same WGSL on the CPU across ' +
    'Web Workers. Each page exercises one ' +
    'part of the library on a shared, editable specification.',
});

const DESCRIPTIONS = {
  core: 'The space: points, chances, dilations, mixtures, caps and random points in Cl(4, 1).',
  behaviorism: "Pavlov's bell (Rescorla–Wagner), Watson's generalised fear, Skinner's schedules, a classroom point system.",
  humanism: "Maslow's prepotent needs, library anxiety and Kuhlthau's Information Search Process with empathetic support.",
  cognitivism: "Forgetting curves, retrieval and spaced practice, interleaving, schemas, Gagné's nine events.",
  constructivism: 'Assimilation, disequilibrium and accommodation: a child learning that not every animal is a dog.',
  social: 'The Zone of Proximal Development, peer dialogue, and cultural capital in the classroom.',
  development: "Piaget's stages by age and activity fit; Perry's scheme and how students judge sources.",
  andragogy: "Knowles' assumptions about adult learners and Bartle's seven-step needs cycle.",
  motivation: "Svinicki's value × expectancy with five levers; Dweck's fixed and growth mindsets.",
  audience: 'Audience research, community data and pre-assessments before a one-shot session; zones of intervention.',
  outcomes: "Backward Design, Bloom's taxonomy and the twin sins of coverage and activity; three to four outcomes per hour.",
  assessment: 'Formative and summative assessment, the assessment cycle, choosing activities, rubrics and self-assessment.',
  strategies: 'Chunked lectures and videos, discussion and wait time, scaffolding, learning styles vs content, flipped one-shots, lesson plans on time.',
  reflective: 'Reflecting on a term of teaching, reading student feedback, critical reflection on whom practices leave out, professional growth.',
  qa: 'A reader answers FineWeb-Edu questions, scored against golden answers; ask it about your own passage.',
  fga: 'Functional geometric algebra: learners read a text and learn multivectors of Cl(4, 1); the paper\'s claims checked; rotors against projections.',
  classroom: 'Every theory acting on the same learners over a course, with a delayed test.',
  theories: 'Activity 3.3 designed by each theory, on children, teenagers and adults.',
  allocation: 'Shapley, incremental and standalone allocation of a design\'s gain to its practices.',
  sensitivity: 'Which practice to strengthen next, with common random numbers.',
  gpu: 'Every model\'s fused kernel on WebGPU and on the software GPU, compared value by value, and timed.',
};

const map = card(page.main, 'Contexts', 'Each page maps onto one part of the library and one section of the chapter.');
const g = el('div', { class: 'grid' });
for (const group of PAGES.slice(1))
  for (const p of group.items)
    g.append(el('a', { href: p.href, class: 'tile', style: 'text-decoration:none;color:inherit' },
      el('div', { class: 'label', text: group.group }), el('div', { style: 'font-weight:650;font-size:15px', text: p.title }),
      el('div', { class: 'hint', text: DESCRIPTIONS[p.id] })));
map.append(g);
page.main.insertBefore(map.parentNode, page.toolbar);

runButton(page, 'Run the shared course', async (spec) => {
  const t0 = performance.now();
  const runResult = await runModel('classroom', spec);
  const r = runResult.result;
  window.__synEngineRun = { page: 'index', engine: runResult.label, delayedScore: r.delayedScore };
  tiles(page.content, [
    { label: 'Design', value: spec.designPreset || 'custom', hint: spec.population.label || spec.population.preset },
    { label: 'Score after the course', value: fmt.num(r.endScore, 3) },
    { label: 'Score a month later', value: fmt.num(r.delayedScore, 3), hint: `retention ${fmt.pct(r.retention, 0)}` },
    { label: 'Learners × sessions', value: `${fmt.num(r.learners)} × ${spec.course.sessions}`, hint: `${fmt.num(r.engineMs)} ms on ${runResult.label}` },
  ]);
  const gr = grid(page.content);
  lineChart(card(gr, 'Score profile of the shared design', 'Full analysis on the <a href="classroom.html">classroom page</a>.'), {
    x: Array.from(r.score.days), xLabel: 'day', yLabel: 'test score',
    series: [{ name: 'mean', y: Array.from(r.score.mean), band: { lower: Array.from(r.score.q10), upper: Array.from(r.score.q90) } }, { name: 'recall', y: Array.from(r.recall.mean), colorIndex: 1 }],
  });
  const arch = card(gr, 'How it runs', '');
  arch.append(el('pre', { class: 'code', text:
`one space (include/syn/space): Cl(4, 1), N the learner, A cognitive, B affective,
   │  C social; points, chances, dilations, rotors, mixtures, caps, random points
   ▼
models (src/models): every theory and chapter as an agent program in WGSL
   │  + how a spec compiles + its summary (C++)
   │  space · conditioning · operant · tokenEconomy · maslow · isp · memory
   │  interleaving · schema · equilibration · zpd · dialogue · development
   │  andragogy · motivation · classroom · theories · allocation · sensitivity
   │  gagne · audience · community · outcomes · assessmentActivities
   │  assessmentSession · rubric · lecture · discussion · scaffolding
   │  strategies · lessonPlan · reflection · studentFeedback
   │  criticalReflection · professionalGrowth · fgaAlgebra · fgaLearning
   │  fgaRotors · readingQA · readingStudent · readingNotes · capsuleIngest
   │  capsuleRecall
   ▼
syn::kernel: plans (constants, one row per job, seeds) ▸ agent kernel
   │  (agents × jobs, workgroups of 64) ▸ reduce ▸ quantiles ▸ summary
   ├─ WebGPU (js/engine/gpu.js): the WGSL on the GPU
   └─ no GPU: the software GPU (syn::gpu), the same WGSL compiled to
      bytecode and run as SIMT workgroups, across Web Workers
   ▼
pages (js/pages/*.js): render the results as charts and tables` }));
  table(card(gr, 'The chapters, section by section', ''), ['section', 'page', 'models'], [
    ['Behaviorism', 'Behaviourism', 'conditioning, operant, tokenEconomy'], ['Humanism', 'Humanism', 'maslow, isp'],
    ['Cognitivism (incl. Gagné)', 'Cognitivism', 'memory, interleaving, schema, gagne'], ['Constructivism', 'Constructivism', 'equilibration'],
    ['Social constructivism', 'Social constructivism', 'zpd, dialogue'], ['Developmental stages (Piaget, Perry, ZPD)', 'Developmental stages, Social', 'development, zpd'],
    ['Andragogy', 'Andragogy', 'andragogy'], ['Motivation, growth mindset', 'Motivation & mindset', 'motivation'],
    ['Ch. 7: Identifying audience needs', 'Identifying audience needs', 'audience, community, isp'], ['Ch. 8: Establishing learning goals and outcomes', 'Learning goals and outcomes', 'outcomes'],
    ['Ch. 9: Assessing learning', 'Assessing learning', 'assessmentActivities, assessmentSession, rubric'],
    ['Ch. 10: Selecting instructional strategies and creating lesson plans', 'Strategies and lesson plans', 'lecture, discussion, scaffolding, strategies, lessonPlan'],
    ['Ch. 14: Practicing reflective teaching', 'Reflective teaching', 'reflection, studentFeedback, criticalReflection, professionalGrowth'],
    ['Pustejovsky (2026): functional geometric algebra', 'Learning meaning (FGA)', 'fgaAlgebra, fgaLearning, fgaRotors'],
    ['QA golden test (FineWeb-Edu)', 'Reading for answers (QA)', 'readingQA, readingStudent, readingNotes, capsuleIngest, capsuleRecall'], ['Activity 3.3, conclusion', 'Theories compared, Gain allocation', 'classroom, theories, allocation, sensitivity'],
  ]);
  return engineNote(runResult, performance.now() - t0);
});
