import { fmt, lineChart } from '../charts.js';
import { engineNote, engineSelector, runAnalysis } from '../gpu/backend.js';
import { PAGES, card, el, grid, initPage, runButton, table, tiles } from '../ui.js';

const page = initPage({
  id: 'index',
  title: 'How people learn, as models you can run',
  context: 'synaptic · libsyn (C++17) → WebAssembly',
  description:
    'Interactive front end for a library modelling <i>Learning Theories: Understanding How People Learn</i> (chapter 3 of <i>Instruction in Libraries ' +
    'and Information Centers</i>, Illinois Open Publishing Network). Each theory of the chapter is a computational model, and a classroom engine ' +
    'runs all of them on the same simulated learners. The C++ library is compiled to WebAssembly and runs in a worker. The classroom simulations also run as ' +
    'fused compute kernels, on WebGPU or, where the browser offers no GPU, on the CPU across Web Workers. Each page exercises one part of the library ' +
    'on a shared, editable specification.',
});

const DESCRIPTIONS = {
  core: 'Reproducible random streams, the logistic link, quantiles and session calendars.',
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
  classroom: 'Every theory acting on the same learners over a course, with a delayed test.',
  theories: 'Activity 3.3 designed by each theory, on children, teenagers and adults.',
  allocation: 'Shapley, incremental and standalone allocation of a design\'s gain to its practices.',
  sensitivity: 'Which practice to strengthen next, with common random numbers.',
  gpu: 'The classroom simulation as fused WebGPU kernels, validated against WebAssembly and benchmarked.',
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

const button = runButton(page, 'Run the shared course', async (spec) => {
  const t0 = performance.now();
  const runResult = await runAnalysis('classroom', spec);
  const r = runResult.result;
  window.__synEngineRun = { page: 'index', engine: runResult.engine, delayedScore: r.delayedScore };
  tiles(page.content, [
    { label: 'Design', value: spec.designPreset || 'custom', hint: spec.population.label || spec.population.preset },
    { label: 'Score after the course', value: fmt.num(r.endScore, 3) },
    { label: 'Score a month later', value: fmt.num(r.delayedScore, 3), hint: `retention ${fmt.pct(r.retention, 0)}` },
    { label: 'Learners × sessions', value: `${fmt.num(r.learners)} × ${spec.course.sessions}`, hint: runResult.fused ? `${fmt.num(r.gpuMs)} ms on the ${runResult.label}` : `${fmt.num(r.elapsedMs)} ms in WebAssembly` },
  ]);
  const gr = grid(page.content);
  lineChart(card(gr, 'Score profile of the shared design', 'Full analysis on the <a href="classroom.html">classroom page</a>.'), {
    x: Array.from(r.score.days), xLabel: 'day', yLabel: 'test score',
    series: [{ name: 'mean', y: Array.from(r.score.mean), band: { lower: Array.from(r.score.q10), upper: Array.from(r.score.q90) } }, { name: 'recall', y: Array.from(r.recall.mean), colorIndex: 1 }],
  });
  const arch = card(gr, 'How it runs', '');
  arch.append(el('pre', { class: 'code', text:
`C++17 library (include/syn, src/)
   │  behaviorism · humanism · cognitivism · constructivism
   │  social · development · andragogy · motivation · classroom
   │  emcmake cmake -S . -B build-wasm   (Embind wrapper: wasm/bindings.cpp)
   ▼
web/wasm/syn.{js,wasm}  ── run in a Worker (js/syn-worker.js)
   │   standalone build: worker + .wasm embedded, started from a blob URL
   │   coreDemo · conditioning · operant · tokenEconomy · humanism · memory
   │   gagne · equilibration · zpd · dialogue · development · andragogy
   │   motivation · classroom · theories · allocation · sensitivity
   ▼
syn::gpu (C++): compile() → plan buffers (population, course, model constants,
   │            one lever row per job) + the WGSL kernels; summarise() on read-back
   ▼
WebGPU host (js/gpu/engine.js): upload ▸ classroom-fused (learners × jobs)
   │                            ▸ reduce-partials ▸ score-quantiles ▸ read back
   │   no adapter: the same kernels on the CPU across Web Workers
   ▼
pages (js/pages/*.js): render the results as charts and tables` }));
  table(card(gr, 'The chapter, section by section', ''), ['section', 'page', 'library'], [
    ['Behaviorism', 'Behaviourism', 'syn/behaviorism'], ['Humanism', 'Humanism', 'syn/humanism'], ['Cognitivism (incl. Gagné)', 'Cognitivism', 'syn/cognitivism, syn/classroom'],
    ['Constructivism', 'Constructivism', 'syn/constructivism'], ['Social constructivism', 'Social constructivism', 'syn/social'],
    ['Developmental stages (Piaget, Perry, ZPD)', 'Developmental stages, Social', 'syn/development, syn/social'], ['Andragogy', 'Andragogy', 'syn/andragogy'],
    ['Motivation, growth mindset', 'Motivation & mindset', 'syn/motivation'], ['Ch. 7: Identifying audience needs', 'Identifying audience needs', 'syn/audience, syn/humanism'], ['Ch. 8: Establishing learning goals and outcomes', 'Learning goals and outcomes', 'syn/outcomes'], ['Activity 3.3, conclusion', 'Theories compared, Gain allocation', 'syn/classroom'],
  ]);
  return engineNote(runResult, performance.now() - t0);
});
engineSelector(page, () => button.click());
