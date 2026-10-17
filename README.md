# synaptic — learning theories as computational models

[![CI](https://github.com/svr-net/synaptic/actions/workflows/ci.yml/badge.svg)](https://github.com/svr-net/synaptic/actions/workflows/ci.yml)

A C++17 library that models four chapters of *Instruction in Libraries and Information Centers*
(Illinois Open Publishing Network):

> chapter 3, *Learning Theories: Understanding How People Learn*
> (<https://iopn.library.illinois.edu/pressbooks/instructioninlibraries/chapter/learning-theories-understanding-how-people-learn/>), and
> chapter 7, *Identifying Audience Needs*
> (<https://iopn.library.illinois.edu/pressbooks/instructioninlibraries/chapter/identifying-audience-needs/>), and
> chapter 8, *Establishing Learning Goals and Outcomes*
> (<https://iopn.library.illinois.edu/pressbooks/instructioninlibraries/chapter/establishing-learning-goals-and-outcomes/>), and
> chapter 9, *Assessing Learning*
> (<https://iopn.library.illinois.edu/pressbooks/instructioninlibraries/chapter/assessing-learning/>).

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
where the least skilled overestimate most and feedback calibrates.
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
  index.html, core.html, ... gpu.html           19 pages
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
(`core/elliptic_cga.hpp` is the general model of Sⁿ the tests use as reference).

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
| Combining the theories | `classroom.cpp` (`classroom`) | Every theory acting on the same learners over a course; score, recall and risk profiles; equity; per-learner read-back |
| Activity 3.3 | `classroom.cpp` (`theories`) | The lesson each theory designs (16 levers), on children, teenagers and adults: 24 jobs, one dispatch |
| Allocation | `classroom.cpp` (`allocation`) | Shapley (adds up exactly), incremental and standalone shares of the gain |
| Sensitivities | `classroom.cpp` (`sensitivity`) | Bump-and-rerun slopes per lever with common random numbers vs independent seeds |

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
