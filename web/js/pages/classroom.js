import { fmt, histogram, lineChart, scatterChart } from '../charts.js';
import { engineNote, engineSelector, gpuScopeNote, runAnalysis } from '../gpu/backend.js';
import { card, grid, initPage, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'classroom',
  title: 'Classroom simulation',
  context: 'syn/classroom: ClassroomEngine · InstructionalDesign · LearningProfile',
  description: 'Every theory acting on the same simulated learners. Each session the engine applies needs (Maslow), affect (anxiety), motivation ' +
    '(value × self-efficacy, weighted by age as andragogy suggests), challenge and support (the ZPD), development (abstract content needs formal ' +
    'operations), encoding and spaced/retrieval practice (cognitivism), cultural fit (Bourdieu) and the response to success or failure (Dweck). ' +
    'The test score combines what is still retrievable, the ability to tell related problems apart and skill on the final level. ' +
    'The score profile is the distribution of scores over time, from the first lesson to the delayed test.',
});
specEditor(page, ['population', 'course', 'design', 'simulation'], { open: true });

const button = runButton(page, 'Run the course', async (spec) => {
  const t0 = performance.now();
  const runResult = await runAnalysis('classroom', spec);
  const r = runResult.result;
  const fused = runResult.fused; // WebGPU or the CPU fused kernels: aggregates only
  window.__synEngineRun = { page: 'classroom', engine: runResult.engine, delayedScore: r.delayedScore, endScore: r.endScore, equityGap: r.equityGap };
  if (fused) gpuScopeNote(page.content, 'the per-learner scatter and the score histogram', runResult.label);
  tiles(page.content, [
    { label: 'Score after the course', value: fmt.num(r.endScore, 3), hint: `day ${fmt.num(r.lastSessionDay + 1)}` },
    { label: 'Score at the delayed test', value: fmt.num(r.delayedScore, 3), hint: `day ${fmt.num(r.delayedTestDay)}; retention ${fmt.pct(r.retention, 0)}` },
    { label: 'At risk at the delayed test', value: fmt.pct(r.atRiskDelayed, 0), hint: `score below ${spec.course.atRiskBelow}` },
    { label: 'Culture gap', value: fmt.num(r.equityGap, 3), hint: 'high − low cultural capital' },
    { label: 'Ability gain', value: fmt.num(r.abilityGain, 2), hint: 'logits' },
    { label: 'Learners', value: fmt.num(r.learners), hint: fused ? `${fmt.num(r.gpuMs)} ms on the ${runResult.label}` : `${fmt.num(r.elapsedMs)} ms in WebAssembly` },
  ]);
  const g = grid(page.content);
  const days = Array.from(r.score.days);
  lineChart(card(g, 'Score profile', 'Mean, median and the 10–90% band of learners. Sessions are weekly; after the last one only forgetting acts.'), {
    x: days, xLabel: 'day', yLabel: 'test score',
    series: [{ name: 'mean', y: Array.from(r.score.mean), band: { lower: Array.from(r.score.q10), upper: Array.from(r.score.q90) } },
      { name: 'median', y: Array.from(r.score.q50), dash: true, colorIndex: 0 }, { name: 'recall of content', y: Array.from(r.recall.mean), colorIndex: 1 }],
  });
  lineChart(card(g, 'Share of learners at risk', ''), { x: days, xLabel: 'day', yLabel: 'share', series: [{ name: 'at risk', y: Array.from(r.score.atRisk), colorIndex: 7 }] });
  const s = r.sessions;
  const sd = Array.from(s.day);
  lineChart(card(g, 'Inside the learners, session by session', 'Population means after each session.'), {
    x: sd, xLabel: 'day', yLabel: 'mean', series: [
      { name: 'effort', y: Array.from(s.effort), markers: true }, { name: 'motivation (V·E)', y: Array.from(s.motivation), colorIndex: 1 },
      { name: 'self-efficacy', y: Array.from(s.efficacy), colorIndex: 2 }, { name: 'anxiety', y: Array.from(s.anxiety), colorIndex: 7 },
      { name: 'readiness (Maslow)', y: Array.from(s.readiness), colorIndex: 3 }, { name: 'growth mindset', y: Array.from(s.growth), colorIndex: 4 },
      { name: 'engaged', y: Array.from(s.engaged), colorIndex: 5, dash: true }, { name: 'tasks in ZPD', y: Array.from(s.zpdShare), colorIndex: 6, dash: true }],
  });
  lineChart(card(g, 'Ability gain', 'Mean growth of θ on the course skill.'), { x: sd, xLabel: 'day', yLabel: 'Δθ', series: [{ name: 'ability gain', y: Array.from(s.ability), markers: true }] });
  if (!fused) {
    scatterChart(card(g, 'Cultural capital and the delayed score', 'Each point a learner.'), {
      xLabel: 'cultural capital', yLabel: 'delayed score', series: [{ name: 'learners', x: Array.from(r.capital), y: Array.from(r.delayed) }],
    });
    histogram(card(g, 'Delayed test scores', ''), Array.from(r.delayedAll), { bins: 30, name: 'learners', xFormat: (v) => v.toFixed(2) });
  }
  const idx = [0, ...sd.map((d) => days.findIndex((x) => x >= d + 1)), days.length - 1].filter((k, i, a) => k >= 0 && a.indexOf(k) === i);
  table(card(g, 'Profile table', ''), ['day', 'mean', 'q10', 'median', 'q90', 'at risk', 'recall'],
    idx.map((k) => [fmt.num(days[k]), fmt.num(r.score.mean[k], 3), fmt.num(r.score.q10[k], 3), fmt.num(r.score.q50[k], 3), fmt.num(r.score.q90[k], 3), fmt.pct(r.score.atRisk[k], 0), fmt.num(r.recall.mean[k], 3)]));
  return engineNote(runResult, performance.now() - t0);
});
engineSelector(page, () => button.click());
