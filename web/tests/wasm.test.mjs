// Node smoke test of the WASM wrapper: every entry point on the default spec,
// with sanity checks against the library's own invariants.
//   node web/tests/wasm.test.mjs
import createSynModule from '../wasm/syn.js';
import { THEORY_PRESETS, defaultSpec } from '../js/spec.js';

const syn = await createSynModule();
let failures = 0;
const check = (cond, msg) => { if (!cond) { failures++; console.log('  FAIL', msg); } };
const near = (a, b, tol, msg) => check(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b} (tol ${tol})`);
const spec = defaultSpec();
const small = { ...spec, sim: { ...spec.sim, learners: 200 } };

function run(name, fn) {
  const t0 = performance.now();
  const r = fn();
  if (r && r.error) { failures++; console.log(`FAIL ${name}: ${r.error}`); return null; }
  console.log(`ok   ${name} (${(performance.now() - t0).toFixed(0)} ms)`);
  return r;
}

const ver = run('version', () => syn.version());
const cmakeVersion = (await import('node:fs')).readFileSync(new URL('../../CMakeLists.txt', import.meta.url), 'utf8').match(/^\s*VERSION ([0-9.]+)$/m)[1];
if (ver) check(ver.library === `synaptic ${cmakeVersion}`, `version() reports the CMake project version: ${ver.library}`);

// The JS presets used by the spec editor must mirror the library's theory designs.
const presets = run('presets', () => syn.presets());
if (presets) {
  check(Object.keys(presets).length === Object.keys(THEORY_PRESETS).length, 'same number of theory presets');
  for (const [name, design] of Object.entries(THEORY_PRESETS))
    for (const [k, v] of Object.entries(design)) near(presets[name]?.[k], v, 1e-12, `preset ${name}.${k}`);
}

const core = run('coreDemo', () => syn.coreDemo(spec));
if (core) {
  check(core.streamsReproducible && core.seedsIndependent, 'random streams');
  for (const [k, v] of Object.entries(core.cga)) check(v < 1e-9, `elliptic CGA identity ${k}: ${v}`);
  near(core.summary.mean, 0, 0.03, 'normal mean');
  near(core.summary.sd, 1, 0.03, 'normal sd');
}
const cond = run('conditioning', () => syn.conditioning(spec));
if (cond) {
  near(cond.pavlov[cond.acquisitionTrials - 1], cond.closedForm[cond.acquisitionTrials - 1], 1e-12, 'Rescorla–Wagner closed form');
  check(cond.blocking.blockedTone < 0.2 * cond.blocking.controlToneFinal, 'Kamin blocking');
}
const op = run('operant', () => syn.operant({ ...spec, operant: { learners: 100 } }));
if (op) check(op.runs.slice(1).every((r) => r.resistance > op.runs[0].resistance), 'partial-reinforcement extinction effect');
const tok = run('tokenEconomy', () => syn.tokenEconomy(spec));
if (tok) check(tok.during > tok.before && tok.after < tok.during, 'point system raises engagement only while it runs');
const hum = run('humanism', () => syn.humanism(spec));
if (hum) check(hum.isp[2].completedOnTime > hum.isp[0].completedOnTime, 'support helps students finish');
const mem = run('memory', () => syn.memory(spec));
if (mem) {
  const by = (s) => mem.plans.find((p) => p.name.startsWith(s)).recallAtTest;
  check(by('spaced retrieval') > by('massed restudy'), 'spaced retrieval beats massed rereading');
  check(mem.interleaving.interleaved.test > mem.interleaving.blocked.test, 'interleaving wins the test');
  check(mem.schemaPrimed.degree > mem.schemaPlain.degree, 'stimulated recall adds links');
}
const gag = run('gagne', () => syn.gagne(small));
if (gag) check(gag.full > gag.presentOnly, 'nine events beat presentation alone');
const eq = run('equilibration', () => syn.equilibration({ ...spec, equilibration: { learners: 80 } }));
if (eq) check(eq.runs[1].finalAccuracy > eq.runs[0].finalAccuracy + 0.3, 'feedback enables accommodation');
const z = run('zpd', () => syn.zpd(spec));
if (z) {
  check(z.tutoring[3].gain > z.tutoring[0].gain && z.tutoring[3].gain > z.tutoring[2].gain, 'ZPD tutoring wins');
  // Figure 3.2: each zone lies between its concentric circles.
  const radius = (k, i) => Math.hypot(z.mapX[k][i], z.mapY[k][i]);
  check(z.mapX.every((xs) => xs.length > 0), 'all three zones on the map');
  check(Array.from(z.mapX[0], (_, i) => radius(0, i)).every((r) => r <= z.mapInner + 1e-9), 'independent tasks inside the inner circle');
  check(Array.from(z.mapX[1], (_, i) => radius(1, i)).every((r) => r > z.mapInner && r <= z.mapOuter + 1e-9), 'ZPD between the circles');
  check(Array.from(z.mapX[2], (_, i) => radius(2, i)).every((r) => r > z.mapOuter), 'beyond outside the outer circle');
  for (let i = 0; i < z.difficulty.length; i += 10) near(z.alone[i], 1 / (1 + Math.exp(z.difficulty[i])), 1e-12, 'CGA success = logistic');
}
const dia = run('dialogue', () => syn.dialogue(spec));
if (dia) check(dia.runs[5].finalMean > dia.runs[0].finalMean, 'dialogue raises understanding');
const dev = run('development', () => syn.development(spec));
if (dev) {
  const ages = Array.from(dev.ages);
  for (let i = 0; i < ages.length; i += 20) near(dev.stageDistribution.reduce((s, d) => s + d[i], 0), 1, 1e-9, `stage distribution sums to 1 at ${ages[i]}`);
  check(dev.sourceAccuracy[2] > dev.sourceAccuracy[1], 'relativists judge sources better');
}
const and = run('andragogy', () => syn.andragogy(spec));
if (and) {
  const last = and.mix.length - 1;
  check(and.curves[2][last] > and.curves[2][0] && and.curves[0][last] < and.curves[0][0], 'andragogy suits adults, pedagogy suits children');
}
const mot = run('motivation', () => syn.motivation(spec));
if (mot) check(mot.terms[mot.terms.length - 1].gain > mot.terms[0].gain && mot.mindset[2].persist > mot.mindset[0].persist, 'motivation levers and "not yet"');
const aud = run('audience', () => syn.audience({ ...spec, audience: { classes: 120 } }));
if (aud) {
  check(aud.instruments[0].value.every((v) => Math.abs(v) < 1e-12), 'no instrument: no value of information');
  check(aud.instruments.find((x) => x.instrument === 'K-W-L chart').value[0] > 0, 'K-W-L pays from an assumption');
  check(aud.community.preAssessmentR2 > aud.community.intersectionR2, 'pre-assessment explains individuals better than demographics');
  const zone = (s) => Math.max(...aud.zones.find((z) => z.stage === s).onTime);
  check(zone('exploration') > zone('collection'), 'a session on exploration beats one on collection');
}
const out = run('outcomes', () => syn.outcomes({ ...spec, outcomes: { learners: 200, texts: ['Understand Boolean operators', 'Apply Boolean operators to narrow a search'] } }));
if (out) {
  check(out.table83.every((x) => !x.poor.strong && x.strong.strong), 'Table 8.3 classified');
  check(!out.analysed[0].measurable && out.analysed[1].strong, 'own outcomes analysed');
  const [cov, act, bd] = out.plans;
  check(bd.behaviourChange > cov.behaviourChange && bd.behaviourChange > act.behaviourChange, 'backward design avoids the twin sins');
  const hour = out.sweep.find((s) => s.minutes === 60).enduring;
  const k = out.outcomeCounts[hour.indexOf(Math.max(...hour))];
  check(k >= 3 && k <= 4, 'three to four outcomes per hour');
  check(out.evidenceBias[0] > 0 && out.evidenceBias[5] < 0, 'evidence must match the outcome level');
}
const cls = run('classroom', () => syn.classroom(spec));
if (cls) {
  const n = cls.score.days.length;
  check(cls.score.days[n - 1] === cls.delayedTestDay, 'profile ends at the delayed test');
  near(cls.score.mean[n - 1], cls.delayedScore, 1e-12, 'profile mean at the test = delayed score');
  check(cls.delayedScore < cls.endScore, 'forgetting after the course');
  const again = syn.classroom(spec);
  check(again.delayedScore === cls.delayedScore, 'reproducible');
}
const th = run('theories', () => syn.theories(small));
if (th) for (const p of th.populations) {
  const d = p.results.map((x) => x.delayedScore);
  check(d[d.length - 1] === Math.max(...d), `blended design is best for ${p.population}`);
}
const al = run('allocation', () => syn.allocation({ ...small, allocation: { permutations: 4 } }));
if (al) near(al.shapleySum, al.totalGain, 1e-12, 'Shapley allocation adds up');
const sens = run('sensitivity', () => syn.sensitivity({ ...small, sensitivity: { replications: 3, maxLearners: 100 } }));
if (sens) check(sens.varianceReduction > 5, `CRN variance reduction ${sens.varianceReduction}`);
// Fused kernels: plans for the GPU, the CPU kernel slices, and the kernels' reference.
const kernels = run('gpuKernels', () => syn.gpuKernels());
if (kernels) check(['classroomFused', 'reducePartials', 'scoreQuantiles'].every((k) => kernels[k].includes('@compute')), 'WGSL sources');
for (const analysis of ['classroom', 'theories', 'allocation', 'sensitivity']) {
  const aspec = { ...small, analysis, allocation: { permutations: 4 }, sensitivity: { replications: 3, maxLearners: 100 } };
  const jobs = run(`gpuJobs ${analysis}`, () => syn.gpuJobs(aspec));
  if (!jobs) continue;
  check(jobs.plans.every((p) => p.header.length === 12 && p.seeds.length === p.numJobs && p.dispatch.fused > 0), `${analysis} plans`);
  const emulated = run(`gpuEmulate ${analysis}`, () => syn.gpuEmulate(aspec));
  const parts = [0, 1, 2].map((part) => syn.cpuKernelSlices({ ...aspec, part, parts: 3 }));
  const cpuKernels = run(`cpuKernelAnalyse ${analysis}`, () => syn.cpuKernelAnalyse({ ...aspec, parts }));
  if (!emulated || !cpuKernels) continue;
  // Three workers give exactly the single-pass reference.
  check(JSON.stringify(cpuKernels) === JSON.stringify(emulated), `${analysis}: CPU kernels on 3 workers = reference`);
  const wasmResult = syn[analysis](aspec);
  if (analysis === 'classroom') near(emulated.delayedScore, wasmResult.delayedScore, 0.03, 'kernel vs WebAssembly delayed score');
  if (analysis === 'allocation') {
    near(emulated.shapleySum, emulated.totalGain, 1e-9, 'GPU Shapley adds up');
    near(emulated.totalGain, wasmResult.totalGain, 0.04, 'kernel vs WebAssembly gain');
  }
  if (analysis === 'theories') check(emulated.populations.every((p) => p.results[7].delayedScore === Math.max(...p.results.map((r) => r.delayedScore))), 'kernels: blended is best');
}
const tooLong = syn.gpuJobs({ ...small, analysis: 'classroom', course: { ...small.course, sessions: 20 } });
check(tooLong.unsupported && /16 sessions/.test(tooLong.unsupported), 'courses beyond the kernel limits fall back');

const bad = syn.classroom({ ...spec, course: { ...spec.course, sessions: 0 } });
check(bad && typeof bad.error === 'string', 'errors come back as { error }');

console.log(failures ? `\n${failures} failure(s)` : '\nall entry points passed');
process.exit(failures ? 1 : 0);
