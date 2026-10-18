# synaptic — learning theories as computational models

[![CI](https://github.com/svr-net/synaptic/actions/workflows/ci.yml/badge.svg)](https://github.com/svr-net/synaptic/actions/workflows/ci.yml)

A C++17 library that models six chapters of *Instruction in Libraries and Information Centers*
(Illinois Open Publishing Network):

> chapter 3, *Learning Theories: Understanding How People Learn*
> (<https://iopn.library.illinois.edu/pressbooks/instructioninlibraries/chapter/learning-theories-understanding-how-people-learn/>), and
> chapter 7, *Identifying Audience Needs*
> (<https://iopn.library.illinois.edu/pressbooks/instructioninlibraries/chapter/identifying-audience-needs/>), and
> chapter 8, *Establishing Learning Goals and Outcomes*
> (<https://iopn.library.illinois.edu/pressbooks/instructioninlibraries/chapter/establishing-learning-goals-and-outcomes/>), and
> chapter 9, *Assessing Learning*
> (<https://iopn.library.illinois.edu/pressbooks/instructioninlibraries/chapter/assessing-learning/>), and
> chapter 10, *Selecting Instructional Strategies and Creating Lesson Plans*
> (<https://iopn.library.illinois.edu/pressbooks/instructioninlibraries/chapter/selecting-instructional-strategies-and-creating-lesson-plans/>), and
> chapter 14, *Practicing Reflective Teaching*
> (<https://iopn.library.illinois.edu/pressbooks/instructioninlibraries/chapter/practicing-reflective-teaching/>),

and merges into the same space J. Pustejovsky's *Toward a Functional Geometric Algebra for Natural Language Semantics* (2026,
arXiv:2604.25902), so that learners can learn the meaning of a text in it.

The chapter surveys how people learn: behaviourism, humanism, cognitivism, constructivism and social constructivism; the developmental
stages of Piaget, Perry and Vygotsky's Zone of Proximal Development; andragogy; and motivation and the growth mindset. It ends by
arguing that the theories are not mutually exclusive and are best combined. This library turns each theory into a model you can
simulate, then runs every theory on the same simulated learners in a classroom engine. That lets you compare the lessons each theory
would design (the chapter's Activity 3.3), allocate a design's learning gain to its teaching practices, and ask which practice to
strengthen next. Chapter 7 asks what an instructor knows about a class before a one-shot session: the library models audience
research and community data as prior beliefs, each pre-assessment instrument as a measurement with a cost in class time, and
shows what the knowledge is worth, when grouping learners pays, why demographics describe groups and not individuals, and when
in the research process a session helps (Kuhlthau's zones of intervention). Chapter 8 plans the session backwards from its outcomes: the library analyses written outcomes
against Bloom's revised taxonomy (vague verbs such as "understand" cannot be measured), shows why evidence must match the
outcome's level, simulates the twin sins of coverage-based and activity-based instruction against Backward Design over big ideas,
basic ideas and what is only nice to know, finds that three to four outcomes fit an hour, and fits the outcome level to the audience's ZPD. Chapter 9 assesses the learning: the library chooses assessment activities by
alignment, audience and time; shows that formative checks fix unclear chunks during a session while only a summative assessment
tells the instructor whether the outcome was met; runs the assessment cycle over repeated offerings with direct evidence, self-report
or grades; measures rater agreement and validity for no rubric, vague, quantified and precise rubrics; and models self-assessment,
where the least skilled overestimate most and feedback calibrates. Chapter 10 selects the strategies and writes the lesson plan: the
library follows learners' attention through a lecture (chunks of 12–15 minutes with bridging activities, stories and analogies)
and through instructional videos of different lengths; measures who speaks in a discussion with different wait times, formats
(think-pair-share, small groups, fishbowl, circle of voices) and ways of managing dominant talkers; compares Ellis and Larkin's
gradual release with its shortcuts; shows that lessons matched to the content teach more than lessons matched to "learning
styles", while a variety of modes reaches learners who cannot use one; simulates the flipped one-shot and problem-based learning;
and delivers the chapter's sample lesson plan with and without time stamps, a cut list and a backup activity. Chapter 14 turns
to the instructor: the library follows a term of teaching under informal reflection, the four-step reflective journal and the
evidence beside it (video, peer observation, the critical incident questionnaire), showing that evidence reveals blind spots, that
one or two changes at a time stick where seven do not, and that harsh reflection drains self-efficacy; reads student feedback for
patterns rather than single comments, with ratings that say little about learning and are biased; asks for which learners "good"
practices (a circle, calling on learners, an invitation to challenge, maximum choice) do not hold; and follows three years of
professional growth with communities of practice, observing colleagues, continuing education, the literature and becoming a
student again.

Functional geometric algebra (FGA) builds meaning in a Clifford algebra: grades are types, contraction is function application,
the wedge binds roles to fillers and rotors coerce. The library runs it in its own space, Cl(4, 1), with the whole algebra
(`include/syn/space/algebra.hpp`: every grade, the geometric product, both contractions, the wedge, reverse and rotors, in
WGSL): entities are points of the sphere of V_E = ⟨A, B, C⟩, predicate content and purpose lie along e, and the roles are N and e.
Three models: `fgaAlgebra` checks the paper's claims in the space (most hold exactly; querying the role of a filler needs
orthogonal fillers, the nullity type check fails with the paper's own lexicon, negation by −u does not swap the polar
idempotents, and exact unbinding is bounded by the two role directions Cl(4, 1) has left); `fgaLearning` has learners read a
text — the default one, or any sentences typed on the page — and learn a multivector per word by gradient descent inside
the fused kernel, showing that the algebra is an inductive bias (bivector relations judge reversed sentences false, a
flat product judges converses true; on the sphere an antisymmetric relation is cyclic); `fgaRotors` runs the paper's proposed
experiment of rotors against learned projections for regular polysemy.
It is an independent implementation. It does not reproduce the text of the chapter. The models are illustrative: their parameters
are chosen so that the effects the chapter describes appear with plausible sizes. They are not fitted to data.

**One space.** Every model lives in the same space, the **pure elliptic conformal geometric algebra** Cl(4, 1): the conformal
model of the 3-sphere, with e² = −1, no Euclidean origin and no point at infinity. Learners, tasks, memories, needs, outcomes,
raters and stimuli are points of it. Ordinary scalar statistics are its miniature basis: a probability is not stored but read
off a point, and the usual logistic models come out as identities of the algebra (see *The space* below).

**Fused kernels, without exception.** Every model is one fused compute kernel, written once, in WGSL. It runs on **WebGPU**;
where a browser has no WebGPU adapter (and in the native build and its tests) it runs on the library's **software GPU**, which
executes the same WGSL on the CPU the way a GPU does. There is no other implementation of any model.

It has no dependencies beyond the C++17 standard library. The structure follows [svr-net/zlatko](https://github.com/svr-net/zlatko):
a native library with unit tests and a demo, an Embind WebAssembly module, a web front end with one page per library context, a
copy-deployable standalone build, Docker images and CI.

## Building

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build            # or ./build/syn_tests [name-filter]
./build/learning_theories_demo
```

Options: `-DSYN_BUILD_TESTS=OFF` and `-DSYN_BUILD_EXAMPLES=OFF`. `cmake --install` installs the
headers and a `syn::syn` CMake target.

## Continuous integration

`.github/workflows/ci.yml` runs on pushes to `main`, on pull requests and on manual dispatch:

| Job | What it checks |
|---|---|
| `native (gcc)`, `native (clang)` | Release build with warnings as errors, `ctest`, example program |
| `native (ASan + UBSan)` | Debug build with address and undefined-behaviour sanitizers, `ctest` |
| `wasm` | Embind module built by the Dockerfile's `wasm` stage (`emscripten/emsdk:4.0.10`), with the Node checks of every model; fails if the committed `web/wasm/` is out of date (the build is byte-reproducible). Its layers depend only on the C++ sources and are cached between runs |
| `standalone` | Copy-deployable site and zip, uploaded as the `synaptic-standalone` artifact |
| `e2e` | Every model's kernel on WebGPU (SwiftShader) and on the software GPU, compared value by value (`web/tests/kernels.mjs`); every page in headless Chromium: dev site over HTTP, standalone from `file://` and over HTTP, including phone layouts, compatibility-mode and no-adapter devices. Screenshots are uploaded as an artifact |
| `docker` | `native` and `web` images with BuildKit layer caching, plus a smoke test of the nginx image |

GitHub Pages: `.github/workflows/pages.yml` deploys the standalone site after CI passes on every push to `main`. Pages must be enabled once under *Settings → Pages* with **Source: GitHub Actions**.

Releases: push a tag that matches the CMake project version (`git tag v0.1.0 && git push origin v0.1.0`), or run the **Release** workflow manually on `main`, which creates that tag itself. `.github/workflows/release.yml` runs the full CI, then publishes a GitHub release with `synaptic-standalone-<tag>.zip` and its SHA-256.

## Standalone, copy-deployable build

`tools/standalone/build.mjs` turns `web/` into one folder of static files that runs anywhere you copy it.
- Open `index.html` straight from disk (`file://`), or upload the folder unchanged to any static host (S3, GitHub Pages, IIS, nginx...).
- There is no server code, no MIME setup and no build step on the target.
- The `.wasm` is embedded in the worker script, and every page is bundled into one classic script, so nothing is fetched at runtime.
- Asset names carry content hashes, so a host can cache them indefinitely.

```sh
docker build --target standalone-artifacts --output dist .     # -> dist/standalone/ + dist/synaptic-standalone.zip
# or locally (needs web/wasm built):
npm --prefix tools/standalone ci && node tools/standalone/build.mjs
```

```
dist/standalone/
  index.html, core.html, ... gpu.html           23 pages
  assets/app.<hash>.js                          all pages (classic script, ~110 KiB)
  assets/syn-runtime.<hash>.js                  WASM worker with embedded library (~1.2 MiB)
  assets/style.<hash>.css
  manifest.json, README.txt                     build commit, file sizes and SHA-256
```

The whole site is about 1.3 MiB (about 480 KiB zipped). `node web/tests/e2e.mjs --root dist/standalone --file` opens every page from
disk and checks that the pages make no network request at all, and that WebGPU and the software GPU still agree.

## Docker

The multi-stage `Dockerfile` builds everything from source. Each stage runs its tests, so a successful build means they passed.
Its layers are ordered by input: toolchains and npm packages first, the C++ sources only where they are compiled, and the
web pages only where they are used, so an edit to a page reuses the compiled library. The compiles keep ccache in BuildKit
cache mounts, so a change to one C++ file recompiles that file (locally: wasm module 129 s cold, 16 s after a one-file
change, 6 s after a page edit; native image 65 s cold, 6 s after a one-file change, tests included). CI caches the layers in
the GitHub Actions cache.

```sh
docker build -t synaptic .                         # WASM from source + checks -> standalone site on nginx (default)
docker run --rm -p 8080:80 synaptic                # web front end at http://localhost:8080

docker build --target native -t synaptic:native .  # native C++ build, ctest
docker run --rm synaptic:native                    # end-to-end demo

docker build --target e2e .                        # every page in headless Chromium, WebGPU on SwiftShader:
                                                   # dev site over HTTP, standalone from file:// and over HTTP
docker build --target standalone-artifacts --output dist .  # export the standalone site and zip
docker build --target wasm-artifacts --output web/wasm .    # export syn.js / syn.wasm to the host
```

| Target | Base image | Contents |
|---|---|---|
| `native` | `ubuntu:24.04` | Library, unit tests and demo binary; installed headers and CMake package in `/opt/syn` |
| `wasm` | `emscripten/emsdk:4.0.10` | Embind module built from source and checked with `web/tests/wasm.test.mjs` |
| `wasm-artifacts` | `scratch` | Only `syn.js` and `syn.wasm`, for `--output` |
| `standalone` | `node:22-alpine` | Copy-deployable site built with the pinned esbuild |
| `standalone-artifacts` | `scratch` | `standalone/` folder and `synaptic-standalone.zip`, for `--output` |
| `e2e` | `mcr.microsoft.com/playwright` | Browser test of all pages in three modes, and every model's kernel on WebGPU and on the software GPU |
| `web` (default) | `nginx:1.27-alpine` | The standalone site; hashed assets are cached as immutable |

If the network goes through a TLS-intercepting proxy, give the npm stages the proxy's CA so they can reach the registry:
`docker build --secret id=ca,src=/path/to/ca.pem ...`. Pass `--build-arg GIT_COMMIT=$(git rev-parse --short HEAD)` to stamp the commit into `manifest.json`.

## The space: pure elliptic CGA

The space every model works in is Cl(4, 1), the conformal model of S³ (`include/syn/space/space.hpp`). Its directions are N (the
learner, the pole), A (cognitive), B (affective) and C (social), plus e with e² = −1. It exists once, in WGSL
(`src/space/space.cpp`), and every kernel includes it. Its functions are closed forms of versor sandwiches;
`tests/test_space.cpp` runs each on the software GPU and checks it against the geometric product of `core/multivector.hpp`
(`core/elliptic_cga.hpp` is the general model of Sⁿ the tests use as reference). The whole algebra — multivectors of every
grade, for functional geometric algebra — is `include/syn/space/algebra.hpp`, also in WGSL only (`src/space/algebra.cpp`).

| Object | In the space | Reads as |
|---|---|---|
| point | null vector X = x + e, x ∈ S³ | a learner state, task, trace, need, outcome, stimulus |
| chance | chance(X) = 1 + ½·X·L = cos²(φ/2), L = N + e the learner | a probability: σ(logit) for the point of that logit |
| logit | ln((e + N)/(e − N)) of a point | an ability, difficulty or strength on the logit scale |
| dilation | D = exp(−d/4·N e), X ↦ D X D̃ | adds d to the logit: learning, support, forgetting, effects |
| mixture | settle(a X + b Y) | learning-rate updates: the chance moves linearly |
| rotor | exp(−θ/2·B̂) | moves content between channels; geodesic `towards` |
| similarity | 1 + ½·X̂·Ŷ = cos²(θ/2) | generalisation, alignment, agreement |
| cap | S = c + cos ρ·e; inside when X·S ≥ 0 | thresholds, zones (ZPD, ISP), schemas, random events |
| random point | `sampleLevel`, `sampleShare` | the space's own (logistic) distribution |

A random event is a cap test, `inside(point, chanceCap(u))` for a uniform u; a threshold is a cap; a mean over learners is the
centre of their points read back as a chance. Scalars appear only as readings of points.

## Fused kernels

Every model (`src/models/*.cpp`, listed in `src/models/all_models.cpp`) has three parts:
- an **agent program in WGSL**. One invocation per agent (a learner, a class, an instructor, a rater, a grid point) works in
  the space and adds its statistics;
- **compile**: the specification becomes plans (constants, one parameter row per job, seeds);
- **summarise**: the per-job means (and quantiles, and per-agent values where a view needs them) become the result.

`syn::kernel` (`include/syn/kernel/kernel.hpp`) runs the plans. The agent kernel's grid is agent workgroups (64) × jobs, so a
whole analysis (eight theories on three populations, a Shapley allocation, a sensitivity sweep) is one dispatch. A reduce
kernel sums the workgroup partials, and a quantile kernel histograms kept values. Random numbers are counter-based (PCG on
(seed, agent, draw)), so jobs with the same seed share learners (common random numbers). The kernels stay within WebGPU's
compatibility mode (4 storage buffers per stage, at most 128 invocations per workgroup). They run on:

| Device | Where |
|---|---|
| WebGPU | `web/js/engine/gpu.js` dispatches the WGSL from `kernelSources` with the plans from `modelPlans` |
| software GPU | `syn::gpu` (`include/syn/gpu/software_gpu.hpp`): the same WGSL, in WebAssembly across Web Workers, or natively |

**The software GPU** compiles WGSL (the part of the language the kernels use) to bytecode: every function is inlined, vector
operations become scalar ones, constants are folded with the machine's own arithmetic. It executes the bytecode as a GPU does,
as SIMT: the invocations of a workgroup are the lanes of one wide machine and every instruction runs for all of them at once
over lane-major memory; when a branch splits the lanes, the lanes at the lowest program counter run first with the others
masked, and they meet again where the paths join. Workgroup memory, barriers, workgroup atomics, storage and uniform buffers,
and 32-bit integer and float semantics (wrapping, saturating conversions, robust buffer access) are those of WGSL. A dispatch
splits into ranges of workgroups across workers; the words each range wrote are merged, and the reduce and quantile kernels
finish the job. `tests/test_software_gpu.cpp` checks the semantics, and `web/tests/kernels.mjs` checks every model against
WebGPU.

## Web front end

`web/` has one interactive page per library context. Every page runs its models through the same generic calls:
`models`, `kernelSources`, `modelPlans` → WebGPU → `modelAnalyse`, or, with no adapter, `simulateSlices` (one per worker) →
`analyseSlices` on the software GPU. The JavaScript only routes calls and draws charts (a small canvas library, `web/js/charts.js`);
`wasm/bindings.cpp` converts specifications and results (`syn::Value`) to and from plain objects.

```sh
python3 -m http.server -d web 8000      # any static server; file:// will not load WASM or module workers
# open http://localhost:8000
```

| Page | Context | Models |
|---|---|---|
| `core.html` | The space: chances, dilations, mixtures, caps, rotors and random points, checked | `space` |
| `behaviorism.html` | Pavlov (Rescorla–Wagner), Kamin blocking, Watson's generalised fear, Skinner's schedules, a classroom point system | `conditioning` · `operant` · `tokenEconomy` |
| `humanism.html` | Maslow's prepotent needs, library anxiety and Kuhlthau's Information Search Process | `maslow` · `isp` |
| `cognitivism.html` | Forgetting, retrieval and spaced practice, interleaving, schemas, Gagné's nine events | `memory` · `interleaving` · `schema` · `gagne` |
| `constructivism.html` | Assimilation, disequilibrium and accommodation (dogs and cats), avoidance of discomfort | `equilibration` |
| `social.html` | The ZPD (Figure 3.2 from caps), tutoring policies, peer dialogue, cultural capital | `zpd` · `dialogue` |
| `development.html` | Piaget's stages by age, activity fit, Perry's scheme and source judgement | `development` |
| `andragogy.html` | Knowles' assumptions × design, Bartle's needs cycle | `andragogy` |
| `motivation.html` | Svinicki's value × expectancy with five levers, Dweck's mindsets and feedback | `motivation` |
| `audience.html` | Chapter 7: priors, pre-assessment instruments, grouping, community data and privacy, zones of intervention | `audience` · `community` · `isp` |
| `outcomes.html` | Chapter 8: outcome analyser, the twin sins, outcomes per hour, evidence vs outcome level, outcome level for the audience | `outcomes` |
| `assessment.html` | Chapter 9: assessment activities, assessment vs grades and satisfaction, formative and summative, the cycle, rubrics, self-assessment | `assessmentActivities` · `assessmentSession` · `rubric` |
| `strategies.html` | Chapter 10: attention and chunked lectures, video length, discussion formats and wait time, scaffolding, learning styles vs content vs variety, flipped one-shots, problem-based learning, the sample lesson plan on time | `lecture` · `discussion` · `scaffolding` · `strategies` · `lessonPlan` |
| `reflective.html` | Chapter 14: a term of reflective practice (informal, the four-step journal, video, peer observation, the CIQ), reading student feedback, critical reflection group by group, professional growth | `reflection` · `studentFeedback` · `criticalReflection` · `professionalGrowth` |
| `qa.html` | The QA golden test in the browser: the reader answers 60 embedded golden FineWeb-Edu questions, scored against the golden answers — copying a sentence, with its remembered habits, and constructed from its knowledge capsule alone; the reader as a student; its notes, memory and capsule; ask it about your own passage (ingested into a capsule), or with none, from what it knows | `capsuleIngest` · `capsuleRecall` · `readingQA` · `readingStudent` · `readingNotes` |
| `fga.html` | Functional geometric algebra (Pustejovsky 2026): learners read a text and learn multivectors of Cl(4, 1); the paper's claims checked; rotors against projections; binding capacity against HRR | `fgaAlgebra` · `fgaLearning` · `fgaRotors` |
| `classroom.html` | Every theory on the same learners over a course: score profile, learner scatter, equity | `classroom` |
| `theories.html` | Activity 3.3 designed by every theory, on three populations | `theories` |
| `allocation.html` | Shapley, incremental and standalone allocation of a design's gain | `allocation` |
| `sensitivity.html` | Lever sensitivities with common random numbers vs independent seeds | `sensitivity` |
| `gpu.html` | Every model on WebGPU and on the software GPU, compared and timed; adapter diagnostics | all |

The pages share one specification: the learner population, the course, the instructional design (sixteen levers, or any theory's
preset) and the simulation settings. You edit it on any page, and the browser's local storage keeps it.

**Where the kernels run.** On **WebGPU** wherever the browser offers an adapter, including a compatibility-mode (OpenGL ES)
adapter. Where there is none, on the **software GPU**, split across Web Workers (one per core, at most 8); the result does not
depend on the number of workers. The status line says which ran and why.

**Rebuilding the WASM module.** The built `web/wasm/syn.{js,wasm}` is committed, and the build is byte-reproducible. Rebuild it with
Docker (`--target wasm-artifacts` above), or with Emscripten 4.0.10:

```sh
emcmake cmake -S . -B build-wasm -DCMAKE_BUILD_TYPE=Release
cmake --build build-wasm                 # writes web/wasm/syn.js and syn.wasm
node web/tests/wasm.test.mjs             # every model plus the chapters' checks, in Node
node web/tests/kernels.mjs               # every model on WebGPU and on the software GPU, value by value
node web/tests/e2e.mjs                   # every page in headless Chromium (Playwright; WebGPU via SwiftShader)
```

## How the library maps onto the chapters

Each model is one file in `src/models/`, and its tests are in `tests/test_models.cpp`.

| Topic | Model (`src/models/`) | In the space |
|---|---|---|
| The space | `space.cpp` | The miniature basis checked: chance of a point = logistic of its logit, dilations add logits, mixtures are linear, rotors keep chances, random points are logistic |
| Behaviourism: classical conditioning | `conditioning.cpp` | Rescorla–Wagner strengths as mixtures towards the learner (closed form), Pavlov, Kamin blocking; Watson's fear generalised by the similarity of stimulus points |
| Behaviourism: operant conditioning | `operant.cpp` | Response points dilated by reinforcement, punishment and extinction; continuous, ratio and interval schedules as cap tests; the partial-reinforcement extinction effect |
| Behaviourism in the classroom | `token_economy.cpp` | A point system: incentive, habit and intrinsic interest as points; withdrawal and overjustification |
| Humanism: needs | `maslow.cpp` | Five prepotent needs as points with cap gates; readiness for learning; a supportive climate |
| Humanism and ch. 7: the ISP | `isp.cpp` | Kuhlthau's stages, felt uncertainty and support; zones of intervention for a one-shot session |
| Cognitivism: memory | `memory.cpp` | Traces turned by rotors: forgetting, restudy, retrieval with and without feedback, massed and spaced plans |
| Cognitivism: interleaving | `interleaving.cpp` | Execution and discrimination as mixtures: blocked practice looks better and tests worse |
| Cognitivism: schemas | `schema.cpp` | Spreading activation; stimulating prior recall adds links to the new concept |
| Cognitivism: Gagné | `classroom.cpp` (`gagne`) | The nine events mapped onto the design levers; the cost of leaving each out |
| Constructivism | `equilibration.cpp` | Schemas as caps on S³; assimilation (cap test + rotor), disequilibrium, accommodation at the centre of conflicts; avoidance |
| Social constructivism | `zpd.cpp`, `dialogue.cpp` | Tasks as points, scaffolds as dilations, the three zones as concentric caps; peer dialogue and cultural capital as mixtures |
| Developmental stages | `development.cpp` | Piaget's stages by age and activity fit; Perry's positions; judging sources |
| Andragogy | `andragogy.cpp` | Engagement as dilations by assumptions × design; Bartle's needs-diagnosis cycle |
| Motivation, growth mindset | `motivation.cpp` | Value × expectancy with five levers; fixed and growth responses to three kinds of feedback |
| Ch. 7: audience needs | `audience.cpp` | Priors, nine pre-assessment instruments with class time and biases, estimates of the class, a session pitched at the ZPD, grouping, value of information |
| Ch. 7: community data | `community.cpp` | Attributes as cap tests; variance explained by labels, intersections and a pre-assessment; minimum cell sizes |
| Ch. 8: outcomes | `outcomes.cpp` | Bloom's levels as points; an outcome analyser; evidence against levels; the twin sins vs Backward Design; outcomes per hour; level fitted to the audience |
| Ch. 9: assessment activities | `assessment_activities.cpp` | Choosing activities by alignment, audience and time; assessment vs grades, self-report and satisfaction |
| Ch. 9: formative and summative, the cycle | `assessment_session.cpp` | A session in chunks with checks and reteaching; the assessment cycle over offerings with direct evidence, self-report or grades |
| Ch. 9: rubrics and self-assessment | `rubric.cpp` | Rater agreement (exact, κ) and validity by rubric; number of levels; self-assessment by quartile and feedback |
| Ch. 10: lectures and videos | `lecture.cpp` | Attention as an affective point lowered by each passive minute and raised by bridging activities; chunks of 8–25 minutes; stories deepen the trace, analogies dilate how well novices follow; viewers of videos of 2–20 minutes |
| Ch. 10: discussion | `discussion.cpp` | Readiness (wait time against the seconds a learner needs) and willingness (confidence dilated by the risk of the setting) as cap tests; whole class, think-pair-share, small groups, fishbowl, circle of voices; three chips, waiting for two other voices; open and closed questions |
| Ch. 10: scaffolding | `scaffolding.cpp` | Ellis and Larkin's four steps and their shortcuts: support as a dilation that has to fade, the ZPD zones as caps, frustration and giving up in zone 3; projects in stages with feedback |
| Ch. 10: selecting strategies | `strategies.cpp` | Learning styles vs content vs a variety of modes (UDL, learners who cannot use one mode); the flipped one-shot (pre-work checks, embedding in the course); problem-based learning in full or in a one-shot |
| Ch. 10: lesson plans | `lesson_plan.cpp` | Example 10.2 delivered by new and experienced instructors: no plan, time stamps, a cut list, a backup activity; segments lost at the end of the hour |
| Ch. 14: reflective practice | `reflection.cpp` | Seven aspects of delivery as points, some blind spots; perception with the error of the evidence (informal, journal, video, peer, CIQ) and a negativity bias; a change of habit as a cap test of its share of the effort; self-efficacy drained by harsh reflection |
| Ch. 14: student feedback | `student_feedback.cpp` | Segments understood as cap tests; the CIQ's most puzzling moment, anonymous or named, and stray grumbles; revising every comment, patterns or ratings; ratings that follow liking and a bias |
| Ch. 14: critical reflection | `critical_reflection.cpp` | Five groups of learners with comfort, entitlement to question and self-direction points; seating, calling, power, choice and examples as group-dependent dilations |
| Ch. 14: professional development | `professional_growth.cpp` | Ideas from a community of practice, colleagues, courses and the literature, copied or adapted; the expert blind spot and becoming a student again; efficacy and isolation |
| FGA: the algebra | `fga_algebra.cpp` | Application by contraction, event binding by the wedge and exact unbinding, type checks by grade and nullity, rotors (elliptic and hyperbolic), polar idempotents, a dot object — each claim measured on random lexicons; binding capacity against holographic reduced representations |
| FGA: learning | `fga_learning.cpp` | Learners read a text (parsed into name–relation–name sentences) and learn a multivector per word by gradient descent on ⟨S R O⟩₀ with corrupted negatives; typed, untyped, grade prior, flat |
| The reader as a student | `reading_student.cpp` | A student learns reading and note-taking habits from golden lessons with every learning method of the library as a switch — reinforcement, schema, interleaving, worked examples, spaced retrieval, equilibration, scaffolding, peer discussion, self-direction, growth mindset, reflection — in 24 study designs, scored on held-out golden answers; it starts from and returns the reader's memory |
| QA golden test | `reading_qa.cpp` | A reader learns word informativeness (idf) from a corpus and believes the sentence that answers a question (BM25 evidence read as a point); reading and open book; scored against golden answers in `tools/qa_golden` |
| Taking notes | `reading_notes.cpp` | Before any question, the reader keeps the sentences of each passage it believes are worth keeping (learned note-taking habits read as a point) |
| Knowledge capsule | `reading_capsule.cpp` | The grammar reads clauses as facts; every word a multivector whose grades — fact, word point, grammar bivector — an index router isolates and routes to packed codes (41.5% of the text); recalled from the codes by rebuilding the rotor from grade 2; answers constructed from the recalled facts |
| FGA: rotors | `fga_rotors.cpp` | Regular polysemy as a rotor family: a rotor of Cl(4, 1) learned by exact plane steps against a learned 5 × 5 projection, from one to six words |
| Combining the theories | `classroom.cpp` (`classroom`) | Every theory acting on the same learners over a course; score, recall and risk profiles; equity; per-learner read-back |
| Activity 3.3 | `classroom.cpp` (`theories`) | The lesson each theory designs (16 levers), on children, teenagers and adults: 24 jobs, one dispatch |
| Allocation | `classroom.cpp` (`allocation`) | Shapley (adds up exactly), incremental and standalone shares of the gain |
| Sensitivities | `classroom.cpp` (`sensitivity`) | Bump-and-rerun slopes per lever with common random numbers vs independent seeds |

## QA golden test

`tools/qa_golden` is an end-to-end test of reading for answers against golden answers. It takes a seeded
sample of the questions of [edededdy/fineweb-edu-grounded-qa](https://huggingface.co/datasets/edededdy/fineweb-edu-grounded-qa)
(golden answers written from FineWeb-Edu passages; the file is pinned by its SHA-256), with the documents of
[svr-net/knowledge](https://github.com/svr-net/knowledge) as the corpus. The `readingQA` model reads them, learns how
informative each word is, and answers each question with the sentence it believes answers it — from the question's
passage, and open book from the whole corpus — as a fused kernel on the software GPU and on WebGPU. Answers are scored
with SQuAD token F1 (`tools/qa_golden/report.md`):

| answer (200 questions) | token F1 |
|---|---:|
| reader, its own passage | 0.343 |
| reader, open book (6,894 sentences) | 0.340 |
| first sentence of the passage | 0.171 |
| oracle: the passage sentence closest to the golden answer | 0.503 |

Open book, the reader's answer comes from the right passage for 96% of the questions, and WebGPU gives the same answer
as the software GPU for every question. `tools/qa_golden/run.sh /path/to/knowledge` reruns it and fails on a drop below
`baseline.json`. The reader is lexical: it extracts one sentence, so the oracle bounds it.

**The reader as a student** (`readingStudent`): the reader learns to read for answers with every learning method the library
models. Lessons are 400 golden questions about other passages; after each answer it sees the golden one and learns how to weigh
eight reading habits (overlap, position, first sentence, length, being a question, two sentences, holding what is asked,
coverage). A study design is a set of the library's methods, each a switch in the kernel:

| method | library models | how the reader studies |
|---|---|---|
| reinforcement | conditioning, operant | the habits of better answers gain weight (expected-reward gradient); without it, no feedback |
| schema | schema | starts from its prior knowledge — its memory — instead of the lexical reader |
| interleaving | interleaving | question kinds mixed; without it, studied in blocks of one kind |
| worked examples | gagne, lecture | studies the best answer first: a small step towards its habits |
| spaced retrieval | memory | lessons answered badly come back after a spacing |
| equilibration | equilibration | a small surprise assimilated (half step), a large one accommodated (double step) |
| scaffolding | zpd, scaffolding | easy before hard in the first pass, with support that fades |
| peer discussion | dialogue, discussion | where a classmate's (the lexical reader's) answer was better, a step towards it |
| self-direction | andragogy | after the first pass, picks the lesson it is least sure about |
| growth mindset | motivation | keeps its effort after a run of misses; without it, effort falls with each miss |
| reflection | reflection | a journal, and one or two changes per block of lessons |

It runs 24 designs at once (no feedback, its memory alone, reinforcement, reinforcement with each method, all methods, all but
one). On the 200 held-out questions (from memory generation 2): no feedback 0.334, reinforcement 0.362, + worked examples
0.372, + schema 0.370, + reflection 0.366; **all methods together 0.362** — they interfere: their larger steps compound
(the learning curve of all methods dips early), and all methods also learn the lessons less well (0.381 against 0.388) — the
library's own reflection model's point that one or two changes at a time stick where seven do not. The first version of
reinforcement, a perceptron step towards the single better answer, made it worse (0.28–0.31). `tools/qa_golden/report.md` has
every design and what each method adds and costs.

**What the reader keeps** (`src/models/reading_memory.{hpp,cpp}`): its memory and its knowledge, compiled into the native
library and the WASM build.

- **Memory** (`student_memory.inc`, generation 4): the weights of its reading habits (how to pick an answer), its note-taking
  habits (which sentences are worth keeping) and its constructing habits (how to say an answer from what it knows), each
  learned with all methods, with the generation, the lessons studied, provenance, and a fingerprint of the habits the weights
  are for. A memory learned for other habits is refused (`memory.used: false`, with the reason) and the reader starts from
  the prior. Each `readingStudent` run starts from the memory (`memory: false` starts fresh; `notes: true` learns
  note-taking; `construct: true` learns constructing) and returns `remember`. With `anchor`, every lesson also pulls the
  habits back towards where it started (drift over 20 lessons 0.36 → 0.06).
- **Knowledge: a capsule in the space, packed by an index router of grades** (`reading_capsule.cpp`; stored:
  `reader_capsule.inc`, the 60 embedded golden passages). `capsuleIngest`: the grammar reads every clause as a fact — subject,
  relation, object — and every word of a fact becomes one multivector of Cl(4, 1) whose grades are the levels of the index:
  grade 0 its fact, grade 1 its word's point of S³, grade 2 the grammar's bivector θ/2·e_N∧e_A + φ/2·e_B∧e_C (role and
  place), the generator of the rotor that binds the word. The index router isolates each grade (⟨M⟩_k) and routes it to its
  level's codebook — the passage's vocabulary, the lattice of 3 roles × 32 places, the runs of facts — and the capsule keeps
  only the codes, bit-packed, with the lexicon front-coded; not the text. `capsuleRecall` (from the codes, or the `packed`
  bytes): each slot's multivector rebuilt, the rotor exp(−B) formed from its isolated grade 2, the word bound by it (R w R̃),
  and the role and place read back off the bound point and checked; each fact's codes are kept out of order and put back by
  the places. On the 600 passages of the golden test every one of 249,064 words comes back exactly, from **41.5% of the bytes
  of the text** (824 KB for 1.99 MB; the 60 stored passages: 105 KB for 212 KB, 49.6%):

  | part | bytes (600 passages) |
  |---|---:|
  | codes: a word's place in its vocabulary, its role and place (grades 1, 2) | 474,030 |
  | vocabularies (grade 1's codebooks) | 179,478 |
  | lexicon, front-coded | 151,904 |
  | facts, as run lengths (grade 0) | 19,004 |
  | **capsule** | **824,418** |
  | text | 1,985,545 |
- **Constructed answers** (`readingQA` with `recalled`): the grammar says candidate answers from the recalled facts — the
  whole fact, its predicate, its object, its subject, or two facts — and the constructing habits (learned as a student,
  with all methods) choose one. The text never reaches this answer. With no passage at all, it answers from the facts of all
  its capsules that share most with the question.

`qa_golden DATA_DIR --write-memory src/models/student_memory.inc --write-capsule src/models/reader_capsule.inc` writes the next
generation. On the 200 golden questions:

| answer | token F1 |
|---|---:|
| reader, its own passage (copies a sentence) | 0.343 |
| reader with its remembered habits (copies one or two sentences) | 0.374 |
| **constructed from its knowledge capsule alone (never sees the text)** | **0.348** |
| constructed with the prior constructing habits (before learning) | 0.297 |
| the best answer the grammar could construct | 0.504 |

The grammar's rotors, word order and recall are operators of the space; finding the verb of a clause is still a word list
(auxiliaries, verb endings), not geometry. The golden test fails if a score drops (`baseline.json`: `constructed`,
`capsuleRecovered`, `constructingStudent`, `remembered`, …).

## Conventions

- Abilities and difficulties are logits, read off points: chance(atLogit(θ − b)) = σ(θ − b). Levers and shares lie in [0, 1].
- Times are in days from the first session (or trials, opportunities or weeks where the model says so).
- Every run takes a seed. Agent *i* draws its k-th number from (seed, i, k), so two designs run on the same learners with the
  same luck (common random numbers), on WebGPU and on the software GPU.
- Specifications and results are `syn::Value` trees (JSON-like); missing fields take the pages' defaults.

## Example

```cpp
#include "syn/core/value.hpp"
#include "syn/kernel/kernel.hpp"
using namespace syn;

Value spec = Value::object();
spec.set("sim", Value::object().set("learners", 500).set("seed", 2024));
Value classroom = kernel::run("classroom", spec);                    // on the software GPU
double delayed = classroom.num("delayedScore", 0);                   // a month after the course
std::string wgsl = kernel::agentKernel(kernel::model("classroom"));  // the WGSL, for WebGPU too
```

`examples/learning_theories_demo.cpp` runs every model and prints its headline numbers; `learning_theories_demo <model>` prints one
model's whole result as JSON.

## Modelling notes and limitations

- **Illustrative, not fitted.** The chapter describes theories qualitatively. The models encode those descriptions with simple,
  standard mechanisms (Rescorla–Wagner, Bush–Mosteller, item-response probabilities, power-law forgetting, Markov chains) and
  parameters chosen to give plausible effect sizes. Use them to reason about mechanisms and interactions, not to predict scores.
- **The classroom engine is a composition.** Its terms (how anxiety interferes, how much rewards add for children, how scaffolding
  substitutes for formal operations) are modelling choices documented in `src/models/classroom.cpp`. Different choices would change
  the ranking of the single theories; the blended design wins because the theories address different bottlenecks.
- **Reinforcement wording.** The chapter says negative reinforcement deters behaviour; in Skinner's terms that is punishment.
  The operant model has reinforcement (strengthens), punishment (weakens) and extinction (no consequence).
- **Overjustification** (rewards eroding intrinsic interest) is not in the chapter. It is a small, adjustable parameter of the point
  system and of the reinforcement lever.
- **Fixed-interval scallops** are not reproduced: the operant learner has no sense of time since the last reinforcer.
- **Single course skill.** The classroom engine tracks one skill (θ), one memory trace per unit and one discrimination level per
  learner. It does not model forgetting of θ itself, nor peer effects between specific learners (see `dialogue.cpp` for that).
- **Course length.** A classroom learner keeps 16 units, so a course has 1 to 16 sessions, over at most 128 days.
