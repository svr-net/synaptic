// Node test of the WebAssembly module: every model on the software GPU through the generic entry
// points, on the pages' default specification, with the checks the chapters make; the plans for
// WebGPU; and the software GPU split across workers against one run.
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
check(models && models.length === ver?.models && models.length >= 39, 'every model listed');
for (const { name } of models || []) {
  const src = syn.kernelSources({ model: name });
  check(!src.error && src.agent.includes('@compute') && src.reduce.includes('@compute') && src.quantile.includes('@compute'), `${name}: WGSL sources`);
}

const space = model('space', { ...spec, space: { samples: 20000 } });
if (space) for (const [k, v] of Object.entries(space.checks)) check(v < 1e-2, `space check ${k}: ${v}`);
const cond = model('conditioning');
if (cond) {
  near(cond.pavlov[cond.acquisitionTrials - 1], cond.closedForm[cond.acquisitionTrials - 1], 1e-6, 'Rescorla–Wagner closed form');
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

const lec = model('lecture');
if (lec) check(lec.sessions[3].retained > lec.sessions[0].retained, 'chunked lectures are retained better');
const disc = model('discussion');
if (disc) check(disc.designs[1].articulated > disc.designs[0].articulated, 'think-pair-share gets everyone to articulate');
const scaf = model('scaffolding');
if (scaf) check(scaf.sequences[3].transfer > scaf.sequences[0].transfer, 'gradual release builds independence');
const strat = model('strategies');
if (strat) check(strat.styles.every((c) => c.rules[2].knowledge >= c.rules[1].knowledge), 'content beats learning styles');
const lp = model('lessonPlan');
if (lp) check(lp.designs[2].onTime > lp.designs[0].onTime, 'a timed plan with a cut list ends on time');
const refl = model('reflection');
if (refl) check(refl.designs[2].final > refl.designs[1].final, 'the four-step journal beats informal reflection');
const fb = model('studentFeedback');
if (fb) check(fb.rules[1].falseRevisions < fb.rules[0].falseRevisions, 'patterns, not single comments');
const crit = model('criticalReflection');
if (crit) check(crit.designs[10].lowestGroup > crit.designs[0].lowestGroup, 'critical reflection lifts the group left out');
const pg = model('professionalGrowth');
if (pg) check(pg.designs[1].efficacy > pg.designs[0].efficacy, 'a community of practice sustains efficacy');
const fa = model('fgaAlgebra');
if (fa) check(fa.claims[3].holds && !fa.claims[4].holds, 'exact unbinding; role queries need orthogonal fillers');
const fl = model('fgaLearning');
if (fl) check(fl.designs[0].reversed > fl.designs[4].reversed, 'bivector relations judge reversed sentences false');
const fr = model('fgaRotors');
if (fr) check(fr.runs[1].rotorTest < fr.runs[1].projectionTest, 'a rotor family generalises from two words');
const rq = model('readingQA');
if (rq) check(rq.answers[1].reading.includes('waggle run'), 'the reader answers with the right sentence');
const rc = model('capsuleRecall');
if (rc) check(rc.recovered === 1 && rc.recalled.passages.length === 60, 'the stored capsule recalls every word, role and place');
const rg = call('readingQA golden', () => syn.runModel({ model: 'readingQA', readingQA: { set: 'golden', questions: 20, recalled: rc.recalled } }));
if (rg) check(rg.scores.reading > rg.scores.firstSentence, 'the reader beats the first sentence on golden answers');
if (rg) check(rg.scores.remembered > rg.scores.reading, 'the remembered habits answer better');
if (rg) check(rg.scores.constructed > rg.scores.firstSentence, 'it constructs answers from its capsule alone');
const ci = call('capsuleIngest', () => syn.runModel({ model: 'capsuleIngest', capsuleIngest: { passages: ['Honeybees communicate the location of flowers with a waggle dance.'] } }));
if (ci) check(ci.facts === 1 && ci.points === 8, 'a passage ingested: one fact, a point per word (articles dropped)');
const rk = call('readingQA from the capsule', () => syn.runModel({ model: 'readingQA', readingQA: { passages: [], recalled: rc.recalled, questions: [{ question: 'What does the Michigan Makers program help students do?' }] } }));
if (rk) check(rk.answers[0].constructed.includes('Michigan Makers'), 'with no passage, it answers from what it knows');
const rn = model('readingNotes');
if (rn) check(rn.kept > 0.1 && rn.kept < 0.7 && rn.bestKeptF1 <= rn.bestF1 + 1e-9, 'notes keep part of each passage');

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

// The software GPU split across three workers (in any order) gives exactly one run's result.
for (const name of ['classroom', 'memory', 'rubric']) {
  const parts = [2, 0, 1].map((part) => syn.simulateSlices({ ...small, model: name, part, parts: 3 }));
  const sliced = call(`analyseSlices ${name}`, () => syn.analyseSlices({ ...small, model: name, parts }));
  const whole = syn.runModel({ ...small, model: name });
  if (sliced) check(JSON.stringify(sliced) === JSON.stringify(whole), `${name}: three workers = one run`);
}

const bad = syn.runModel({ ...spec, model: 'classroom', course: { ...spec.course, sessions: 200 } });
check(bad && typeof bad.error === 'string', `errors come back as { error }: ${bad && bad.error}`);
const unknown = syn.runModel({ model: 'no such model' });
check(unknown && typeof unknown.error === 'string', 'unknown models are errors');

console.log(failures ? `\n${failures} failure(s)` : '\nall entry points passed');
process.exit(failures ? 1 : 0);
