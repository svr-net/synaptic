// Node test of the WebAssembly module: every model through the generic entry points, on the
// pages' default specification, with the checks the chapters make; the GPU plans and read-back
// path; and the CPU kernel slices against a single WebAssembly run.
//   node web/tests/wasm.test.mjs
import createSynModule from '../wasm/syn.js';
import { THEORY_PRESETS, defaultSpec } from '../js/spec.js';

const syn = await createSynModule();
let failures = 0;
const check = (cond, msg) => { if (!cond) { failures++; console.log('  FAIL', msg); } };
const near = (a, b, tol, msg) => check(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b} (tol ${tol})`);
const spec = defaultSpec();
const small = { ...spec, sim: { ...spec.sim, learners: 300 } };

function call(name, fn) {
  const t0 = performance.now();
  let r;
  try { r = fn(); } catch (e) { r = { error: String(e) }; }
  if (r && r.error) { failures++; console.log(`FAIL ${name}: ${r.error}`); return null; }
  console.log(`ok   ${name} (${(performance.now() - t0).toFixed(0)} ms)`);
  return r;
}
const model = (name, s = small) => call(name, () => syn.runModel({ ...s, model: name }));

const ver = call('version', () => syn.version());
const cmakeVersion = (await import('node:fs')).readFileSync(new URL('../../CMakeLists.txt', import.meta.url), 'utf8').match(/^\s*VERSION ([0-9.]+)$/m)[1];
if (ver) check(ver.library === `synaptic ${cmakeVersion}`, `version() reports the CMake project version: ${ver.library}`);

const models = call('models', () => syn.models());
check(models && models.length === ver?.models && models.length >= 26, 'every model listed');
for (const { name } of models || []) {
  const src = syn.kernelSources({ model: name });
  check(!src.error && src.agent.includes('@compute') && src.reduce.includes('@compute') && src.quantile.includes('@compute'), `${name}: WGSL sources`);
}

const space = model('space', { ...spec, space: { samples: 20000 } });
if (space) for (const [k, v] of Object.entries(space.checks)) check(v < 1e-2, `space check ${k}: ${v}`);
const cond = model('conditioning');
if (cond) {
  near(cond.pavlov[cond.acquisitionTrials - 1], cond.closedForm[cond.acquisitionTrials - 1], 1e-9, 'Rescorla–Wagner closed form');
  check(cond.blocking.blockedTone < 0.2 * cond.blocking.controlToneFinal, 'Kamin blocking');
}
const op = model('operant');
if (op) check(op.runs.slice(1).every((r) => r.resistance > op.runs[0].resistance), 'partial-reinforcement extinction effect');
const tok = model('tokenEconomy');
if (tok) check(tok.during > tok.before && tok.after < tok.during, 'point system raises engagement only while it runs');
const isp = model('isp');
if (isp) check(isp.isp[2].completedOnTime > isp.isp[0].completedOnTime, 'support helps students finish');
const mem = model('memory');
if (mem) {
  const by = (s) => mem.plans.find((p) => p.name.startsWith(s)).recallAtTest;
  check(by('spaced retrieval') > by('massed restudy'), 'spaced retrieval beats massed rereading');
}
const il = model('interleaving');
if (il) check(il.interleaved.test > il.blocked.test, 'interleaving wins the test');
const sch = model('schema');
if (sch) check(sch.schemaPrimed.degree > sch.schemaPlain.degree, 'stimulated recall adds links');
const gag = model('gagne');
if (gag) check(gag.full > gag.presentOnly, 'nine events beat presentation alone');
const eq = model('equilibration');
if (eq) check(eq.runs[1].finalAccuracy > eq.runs[0].finalAccuracy + 0.3, 'feedback enables accommodation');
const z = model('zpd');
if (z) check(z.tutoring[3].gain > z.tutoring[0].gain && z.tutoring[3].gain > z.tutoring[2].gain, 'ZPD tutoring wins');
const dia = model('dialogue');
if (dia) check(dia.runs[3].finalMean > dia.runs[0].finalMean, 'dialogue raises understanding');
const dev = model('development');
if (dev) {
  for (let i = 0; i < dev.ages.length; i += 20) near(dev.stageDistribution.reduce((s, d) => s + d[i], 0), 1, 1e-6, `stage distribution sums to 1 at ${dev.ages[i]}`);
  check(dev.sourceAccuracy[2] > dev.sourceAccuracy[1], 'relativists judge sources better');
}
const and = model('andragogy');
if (and) check(and.cycles[2].hitRate[0] > and.cycles[0].hitRate[0], 'learner input improves the needs diagnosis');
const mot = model('motivation');
if (mot) check(mot.terms[mot.terms.length - 1].gain > mot.terms[0].gain && mot.mindset[2].persist > mot.mindset[0].persist, 'motivation levers and "not yet"');
const aud = model('audience');
if (aud) check(aud.instruments[2].rmse[0] < aud.instruments[0].rmse[0], 'a pre-assessment improves the estimate of the class');
const com = model('community');
if (com) check(com.preAssessmentR2 > com.intersectionR2, 'pre-assessment explains individuals better than demographics');
const out = model('outcomes');
if (out) check(out.evidenceBias[0] > 0 && out.evidenceBias[5] < 0, 'evidence must match the outcome level');
const act = model('assessmentActivities');
if (act) check(act.signals.direct > act.signals.grade && act.signals.grade > act.signals.satisfaction, 'assessment beats grades beats satisfaction');
const ses = model('assessmentSession');
if (ses) {
  const d = ses.cycle[3].mastery;
  check(d[d.length - 1] > d[0], 'the assessment cycle improves teaching');
}
const rub = model('rubric');
if (rub) check(rub.rubrics[3].private.kappa > rub.rubrics[0].private.kappa, 'a precise rubric raises agreement');

const cls = model('classroom', spec);
if (cls) {
  const n = cls.score.days.length;
  check(cls.score.days[n - 1] === cls.delayedTestDay, 'profile ends at the delayed test');
  near(cls.score.mean[n - 1], cls.delayedScore, 1e-9, 'profile mean at the test = delayed score');
  check(cls.delayedScore < cls.endScore, 'forgetting after the course');
  check(syn.runModel({ ...spec, model: 'classroom' }).delayedScore === cls.delayedScore, 'reproducible');
}
const th = model('theories');
if (th) {
  // The JS presets used by the spec editor mirror the library's theory designs.
  check(th.designs.length === Object.keys(THEORY_PRESETS).length, 'same number of theory presets');
  Object.values(THEORY_PRESETS).forEach((design, i) => {
    for (const [k, v] of Object.entries(design)) near(th.designs[i][k], v, 1e-12, `preset ${i}.${k}`);
  });
  for (const p of th.populations) {
    const d = p.results.map((x) => x.delayedScore);
    check(d[d.length - 1] === Math.max(...d), `blended design is best for ${p.population}`);
  }
}
const al = model('allocation');
if (al) near(al.shapleySum, al.totalGain, 1e-9, 'Shapley allocation adds up');
const sens = model('sensitivity');
if (sens) check(sens.varianceReduction > 5, `CRN variance reduction ${sens.varianceReduction}`);

// The GPU path: the plans (buffers and dispatch sizes) the browser engine runs.
const plans = call('modelPlans classroom', () => syn.modelPlans({ ...small, model: 'classroom' }));
if (plans) check(plans.length > 0 && plans.every((p) => p.header.length === 8 && p.seeds.length === p.jobs), 'classroom plans');

// CPU kernels: three workers' slices, summarised, against one WebAssembly run (32- vs 64-bit floats).
for (const name of ['classroom', 'memory', 'rubric']) {
  const parts = [0, 1, 2].map((part) => syn.modelSlices({ ...small, model: name, part, parts: 3 }));
  const sliced = call(`modelAnalyseSlices ${name}`, () => syn.modelAnalyseSlices({ ...small, model: name, parts }));
  const whole = syn.runModel({ ...small, model: name });
  if (sliced && name === 'classroom') near(sliced.delayedScore, whole.delayedScore, 2e-3, 'CPU kernels vs WebAssembly delayed score');
  if (sliced && name === 'memory') near(sliced.plans[3].recallAtTest, whole.plans[3].recallAtTest, 2e-3, 'CPU kernels vs WebAssembly recall');
  if (sliced && name === 'rubric') near(sliced.rubrics[3].private.kappa, whole.rubrics[3].private.kappa, 5e-3, 'CPU kernels vs WebAssembly kappa');
}

const bad = syn.runModel({ ...spec, model: 'classroom', course: { ...spec.course, sessions: 200 } });
check(bad && typeof bad.error === 'string', `errors come back as { error }: ${bad && bad.error}`);
const unknown = syn.runModel({ model: 'no such model' });
check(unknown && typeof unknown.error === 'string', 'unknown models are errors');

console.log(failures ? `\n${failures} failure(s)` : '\nall entry points passed');
process.exit(failures ? 1 : 0);
