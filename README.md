# synaptic — learning theories as computational models

[![CI](https://github.com/svr-net/synaptic/actions/workflows/ci.yml/badge.svg)](https://github.com/svr-net/synaptic/actions/workflows/ci.yml)

A C++17 library that models three chapters of *Instruction in Libraries and Information Centers*
(Illinois Open Publishing Network):

> chapter 3, *Learning Theories: Understanding How People Learn*
> (<https://iopn.library.illinois.edu/pressbooks/instructioninlibraries/chapter/learning-theories-understanding-how-people-learn/>), and
> chapter 7, *Identifying Audience Needs*
> (<https://iopn.library.illinois.edu/pressbooks/instructioninlibraries/chapter/identifying-audience-needs/>), and
> chapter 8, *Establishing Learning Goals and Outcomes*
> (<https://iopn.library.illinois.edu/pressbooks/instructioninlibraries/chapter/establishing-learning-goals-and-outcomes/>).

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
basic ideas and what is only nice to know, finds that three to four outcomes fit an hour, and fits the outcome level to the audience's ZPD.
It is an independent implementation. It does not reproduce the text of the chapter. The models are illustrative: their parameters
are chosen so that the effects the chapter describes appear with plausible sizes. They are not fitted to data.

The classroom simulations also run as **fused WebGPU compute kernels** (see below), with the same kernels on the CPU across
Web Workers where the browser offers no GPU.

All geometry is **pure elliptic conformal geometric algebra** (see below): the conformal model of the sphere, Cl(n+1, 1) with
e² = −1, with no Euclidean origin and no point at infinity.

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
| `wasm` | Embind module built in `emscripten/emsdk:4.0.10`. Runs the Node checks of every entry point, and fails if the committed `web/wasm/` is out of date |
| `standalone` | Copy-deployable site and zip, uploaded as the `synaptic-standalone` artifact |
| `e2e` | Every page in headless Chromium (WebGPU on SwiftShader): dev site over HTTP, standalone from `file://` and over HTTP, including phone layouts, compatibility-mode and no-adapter devices, and WebGPU / CPU kernels / WebAssembly agreement. Screenshots are uploaded as an artifact |
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
  index.html, core.html, ... gpu.html           17 pages
  assets/app.<hash>.js                          all pages (classic script, ~90 KiB)
  assets/syn-runtime.<hash>.js                  WASM worker with embedded library (~530 KiB)
  assets/style.<hash>.css
  manifest.json, README.txt                     build commit, file sizes and SHA-256
```

The whole site is about 750 KiB (about 290 KiB zipped). `node web/tests/e2e.mjs --root dist/standalone --file` opens every page from
disk and checks that the pages make no network request at all, and that the WebGPU kernels still agree with WASM.

## Docker

The multi-stage `Dockerfile` builds everything from source. Each stage runs its tests, so a successful build means they passed.

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
| `e2e` | `mcr.microsoft.com/playwright` | Browser test of all pages in three modes, including WebGPU-vs-WASM agreement |
| `web` (default) | `nginx:1.27-alpine` | The standalone site; hashed assets are cached as immutable |

If the network goes through a TLS-intercepting proxy, give the npm stages the proxy's CA so they can reach the registry:
`docker build --secret id=ca,src=/path/to/ca.pem ...`. Pass `--build-arg GIT_COMMIT=$(git rev-parse --short HEAD)` to stamp the commit into `manifest.json`.

## Web front end (WebAssembly)

`web/` contains one interactive page per library context. Each page runs the C++ library compiled to WebAssembly in a module
worker, so long Monte Carlo runs don't block the page.

```sh
python3 -m http.server -d web 8000      # any static server; file:// will not load WASM or module workers
# open http://localhost:8000
```

| Page | Context | Entry point |
|---|---|---|
| `core.html` | Random streams, logistic link, quantiles, session calendars, elliptic CGA identities | `coreDemo` |
| `behaviorism.html` | Pavlov (Rescorla–Wagner), Kamin blocking, Watson's generalised fear, Skinner's schedules, a classroom point system | `conditioning` · `operant` · `tokenEconomy` |
| `humanism.html` | Maslow's prepotent needs, library anxiety and Kuhlthau's Information Search Process | `humanism` |
| `cognitivism.html` | Forgetting curves, retrieval and spaced practice, cramming, interleaving, schemas, Gagné's nine events | `memory` · `gagne` |
| `constructivism.html` | Assimilation, disequilibrium and accommodation (dogs and cats), avoidance of discomfort | `equilibration` |
| `social.html` | The ZPD (with Figure 3.2 from the CGA caps), tutoring policies, peer dialogue, cultural capital | `zpd` · `dialogue` |
| `development.html` | Piaget's stages by age, activity fit, Perry's scheme and source judgement | `development` |
| `andragogy.html` | Knowles' assumptions × design, Bartle's needs cycle | `andragogy` |
| `motivation.html` | Svinicki's value × expectancy with five levers, Dweck's mindsets and feedback | `motivation` |
| `audience.html` | Chapter 7: priors from research and community data, pre-assessment instruments, grouping, intersectionality and privacy, zones of intervention | `audience` |
| `outcomes.html` | Chapter 8: outcome analyser (your own outcomes and Table 8.3), the twin sins, outcomes per hour, evidence vs outcome level, shared outcomes and essential questions, outcome level for the audience | `outcomes` |
| `classroom.html` | The classroom engine: score profile, learner states, equity | `classroom` |
| `theories.html` | Activity 3.3 designed by every theory, on three populations | `theories` |
| `allocation.html` | Shapley, incremental and standalone allocation of a design's gain | `allocation` |
| `sensitivity.html` | Lever sensitivities with common random numbers vs independent seeds | `sensitivity` |
| `gpu.html` | Fused WebGPU kernels validated against WebAssembly, a batch benchmark and adapter diagnostics | `gpuJobs` · `gpuAnalyse` |

The pages share one specification: the learner population, the course, the instructional design (sixteen levers, or any theory's
preset) and the simulation settings. You edit it on any page, and the browser's local storage keeps it.

**Compute engine.** The overview, classroom, theories, allocation and sensitivity pages have an *Engine* selector:
- **Auto** (the default) runs the fused **WebGPU** kernels wherever the browser offers a WebGPU adapter, including a
  compatibility-mode (OpenGL ES) adapter: the kernels stay within its limits (4 storage buffers per stage, 128 invocations per
  workgroup).
- Where there is no adapter, Auto runs the **same fused kernels on the CPU**, with the (job, workgroup) tiles split across
  Web Workers (one per core, at most 8). The result is identical however many workers run, and agrees with WebGPU to about 1e-6.
- Courses beyond the kernel limits (more than 16 sessions) and failures fall back to **WebAssembly** (the library's
  `ClassroomEngine`). The status line names the engine that ran and why. Per-learner views (the capital/score scatter and the
  score histogram) are WebAssembly-only, because the kernels never return individual learners.

**WebGPU fused kernels** (`include/syn/gpu/fused_classroom.hpp`). `compile()` turns a batch of courses into flat buffers: a
uniform header, the population, course and model constants, one row of levers per job, the job seeds and the reporting days.
Three WGSL kernels (`src/gpu/fused_kernels.cpp`) run on them:
1. **classroom-fused** runs one invocation per learner and job (grid: learner workgroups × jobs). It samples the learner from
   counter-based random numbers, runs every session (needs, affect, motivation, the ZPD in elliptic CGA with the scaffold as a
   dilation versor, encoding and the spaced/retrieval review slots, the response to success and failure), evaluates the score on
   every reporting day, and the 64 learners of the workgroup sum their statistics into one partial. No per-day state survives
   the kernel except the scores the quantile kernel needs.
2. **reduce-partials** sums the workgroup partials into per-job means.
3. **score-quantiles** histograms each (day, job) and reads off the 10%, 50% and 90% quantiles.

Because jobs are a grid dimension, a whole Shapley allocation (`allocationRequest` → one batch → `allocateFromScores`) or
sensitivity analysis (`sensitivityRequest` → `sensitivitiesFromScores`) is a single dispatch, and jobs with the same seed share
learners (common random numbers on the GPU too). `runFusedReference()` (`src/gpu/fused_reference.cpp`) executes the same
kernels on the CPU in single precision, draw for draw, so the unit tests check the kernel algorithm natively against the
`ClassroomEngine`; split into `runFusedTiles()` and `finishFused()`, it is the CPU backend (`cpuKernelSlices`,
`cpuKernelAnalyse`). The GPU's random numbers differ from the engine's, so the two agree within Monte Carlo error.

**All numerics are in C++.** `wasm/bindings.cpp` turns the JS spec into library objects and returns plain objects of numbers and
`Float64Array`s; the JavaScript only routes calls and draws charts (a small dependency-free canvas library, `web/js/charts.js`).

**Rebuilding the WASM module.** The built `web/wasm/syn.{js,wasm}` is committed. Rebuild it with Docker (`--target wasm-artifacts` above), or with a local Emscripten:

```sh
emcmake cmake -S . -B build-wasm -DCMAKE_BUILD_TYPE=Release
cmake --build build-wasm                 # writes web/wasm/syn.js and syn.wasm
node web/tests/wasm.test.mjs             # every entry point plus invariant checks, in Node
node web/tests/e2e.mjs                   # every page in headless Chromium (Playwright; WebGPU via SwiftShader)
```

## Geometry: pure elliptic CGA

`core/multivector.hpp` is a small Clifford algebra Cl(p, q) (geometric, outer and inner products, reverse, grades, versor
sandwich). `core/elliptic_cga.hpp` builds the conformal model of the n-sphere on it, Cl(n+1, 1), where the extra vector e
squares to −1:

| Object | Representation | Use |
|---|---|---|
| point of Sⁿ | null vector X = x + e | an experience, a task, a stimulus |
| geodesic distance | X·Y = cos θ − 1 | nearest schema, generalisation gradient |
| cap (spherical circle) | S = c + cos ρ·e, S² = sin²ρ | schema assimilation regions, ZPD zones |
| inside a cap | X·S = cos θ − cos ρ ≥ 0 | assimilation test, zone test |
| rotation | rotor exp(−θ/2·B̂), X ↦ R X R̃ | prototypes drifting towards experiences |
| conformal dilation | exp(β/2·n e), tan(φ/2) ↦ e^β tan(φ/2) | scaffolding |
| centre of points | direction of Σ Xᵢ | a new schema built by accommodation |

Where the models use it:
- **ZPD** (`social/zpd.hpp`): a task of relative difficulty Δ = b − θ is the point at angle φ = 2·atan(e^{Δ/2}) from the
  learner L. Then P(success alone) = 1 + ½·T·L, which equals σ(θ − b) exactly, and scaffolding is the dilation versor that maps
  Δ to Δ − s. The three zones are concentric caps around the learner; stereographic projection turns them into Figure 3.2.
  The classroom engine evaluates every task this way.
- **Equilibration** (`constructivism/equilibration.hpp`): features are lifted onto S³ by inverse stereographic projection.
  Schemas are caps, assimilation is a cap test plus a rotor, and accommodation centres a new cap on the conflicting points.
- **Generalisation** (`behaviorism/classical_conditioning.hpp`): stimuli are points of S², and the fear response decays with
  the geodesic angle read from their inner product.

The core page of the web front end checks the algebra's identities on random configurations, and the tests check that the
ZPD geometry reproduces the logistic model to 1e-13.

## How the library maps onto the chapter

| Topic | Module | What is implemented |
|---|---|---|
| Geometry | `core/multivector.hpp`, `core/elliptic_cga.hpp` | Clifford algebra Cl(p, q); pure elliptic CGA of Sⁿ: null points, caps, rotors, dilations, stereographic lift |
| Behaviourism: classical conditioning | `behaviorism/classical_conditioning.hpp` | Rescorla–Wagner learning (closed-form acquisition), Pavlov's acquisition and extinction, Kamin blocking, Watson's fear conditioning with a Shepard generalisation gradient over geodesic angles between stimulus points |
| Behaviourism: operant conditioning | `behaviorism/operant_conditioning.hpp` | Bush–Mosteller response strength under reinforcement, punishment and extinction; continuous, fixed/variable ratio and interval schedules; cumulative records; the partial-reinforcement extinction effect from the surprise of unrewarded runs |
| Behaviourism in the classroom | `behaviorism/token_economy.hpp` | Point system and gamification: external incentive, habit, extinction after withdrawal, optional overjustification |
| Humanism: needs | `humanism/maslow.hpp` | Five prepotent levels with smooth gates, dominant need, readiness for learning, the effect of an empathetic classroom climate |
| Humanism: affect | `humanism/information_search.hpp` | Kuhlthau's six ISP stages with their feelings; library anxiety interfering with cognitive work; empathetic support |
| Cognitivism: memory | `cognitivism/memory.hpp` | Power-law forgetting curve; restudy vs retrieval vs retrieval with feedback; desirable difficulty; massed, spaced and expanding plans; cramming |
| Cognitivism: interleaving | `cognitivism/interleaving.hpp` | Execution skill vs discrimination: blocked practice looks better in practice and does worse on a mixed test |
| Cognitivism: schemas | `cognitivism/schema.hpp` | Concept network with spreading activation; encoding links new knowledge to what is active, so stimulating prior recall adds retrieval routes |
| Cognitivism: Gagné | `classroom/instructional_design.hpp` | The nine events of instruction mapped onto the design levers (`designFromGagne`) |
| Constructivism | `constructivism/equilibration.hpp` | Schemas as labelled caps on S³; assimilation (cap test + rotor), disequilibrium, accommodation; avoidance of discomfort; the dog/cat example |
| Social constructivism | `social/zpd.hpp`, `social/dialogue.hpp` | Vygotsky's three zones as concentric CGA caps with scaffolding as a dilation versor, learning gain peaking in the ZPD, tutoring policies with fading scaffolds; peer dialogue with more knowledgeable others and articulation; Bourdieu's cultural capital and inclusive teaching |
| Developmental stages | `development/piaget.hpp`, `development/perry.hpp` | Piaget's stages and capabilities, stage probabilities by age, activity fit (the climate-change lesson); Perry's positions as a Markov chain, source judgement by position |
| Andragogy | `andragogy/andragogy.hpp` | Knowles' six assumptions paired with design features, engagement for children, adolescents and adults; Bartle's seven steps and the needs-diagnosis cycle |
| Motivation | `motivation/expectancy_value.hpp` | Svinicki's M = V·E with the chapter's five levers; self-efficacy from mastery experience; calibration by self-assessment |
| Growth mindset | `motivation/mindset.hpp` | Fixed and growth responses to failure (effort, strategy change, giving up); ability praise vs "not yet"; coaching; high standards |
| Ch. 7: audience research and pre-assessment | `audience/pre_assessment.hpp` | Priors from assumption, research literature, demographics and the curriculum; nine pre-assessment instruments (show of hands, anonymous poll, worksheet, K-W-L, concept map, brainstorm, quiz, homework) with class time, analysis time, noise, recall and social biases, participation and individual results; precision-weighted estimate of the class; pitch at the ZPD (elliptic CGA); grouping by score; value of information |
| Ch. 7: community data | `audience/community.hpp` | Intersectional attributes (nontraditional, first-generation, English learner, older adult, disability); variance of individual needs explained by labels, intersections and a pre-assessment; minimum-cell-size suppression for aggregate-only reporting |
| Ch. 7: Kuhlthau's seven stages | `humanism/information_search.hpp` | The assessment stage and zones of intervention: a one-shot session's day and target stage, helping only students who have not passed it |
| Ch. 8: Bloom's taxonomy | `outcomes/bloom.hpp` | The six levels and their action verbs; an analyser for written outcomes (stem, verb, level, vague verbs, compound and too-broad outcomes); hierarchical mastery by level; evidence at one level for an outcome at another (bias and met / not met accuracy) |
| Ch. 8: Backward Design | `outcomes/backward_design.hpp` | Big ideas, basic ideas and nice-to-know content for a one-shot session; coverage-based, activity-based and backward-design plans; mastery with crowding beyond ~4 items, understanding of big ideas from their basics, transfer, behaviour change, self-assessment with shared outcomes, essential questions and affective framing; outcome level fitted to the audience's ZPD (elliptic CGA) |
| Learners | `classroom/learner.hpp` | Populations (children, secondary, adults) with prior knowledge, anxiety, mindset, efficacy, interest, cultural capital and age |
| Combining the theories | `classroom/classroom_engine.hpp` | Every theory acting on the same learners over a course; score, recall and risk profiles; equity gap; learner states per session |
| Activity 3.3 | `classroom/instructional_design.hpp` | The lesson each theory would design (16 levers), including the blended design of the conclusion |
| Allocation | `classroom/allocation.hpp` | Shapley (adds up exactly), incremental and standalone contributions of each practice |
| Sensitivities | `classroom/sensitivities.hpp` | Bump-and-rerun slopes per lever with common random numbers vs independent seeds, with standard errors |
| GPU classroom | `gpu/fused_classroom.hpp` | The classroom engine as fused WGSL kernels: plan compiler, kernel sources, read-back summary, and a CPU reference of the kernels |

## Conventions

- Abilities θ and difficulties b are on a logit scale: P(success) = σ(θ − b). Levers, needs, anxiety, mindset and similar states lie in [0, 1].
- Times are in days from the first session (or trials, opportunities or weeks where the model says so).
- Every simulation takes a seed. Learner *i* draws from its own stream of that seed, with a fixed number of draws per session, so
  two designs run on the same learners with the same luck (common random numbers). The tests check this property: the
  CRN sensitivities have far smaller standard errors than independent-seed ones.
- A memory trace's stability S is the interval after which recall falls to 90%.
- Geometric quantities are angles on spheres, computed only through the elliptic CGA (inner products, caps and versors).

## Example

```cpp
#include "syn/syn.hpp"
using namespace syn;

ClassroomEngine engine(PopulationSpec::secondary(), CourseSpec{}, SimulationSettings{500, 2024});

ClassroomResult lecture = engine.run(theoryDesign(Theory::Banking));
ClassroomResult blended = engine.run(theoryDesign(Theory::Blended));
double gain = blended.delayedScore - lecture.delayedScore;   // a month after the course

AllocationResult who = allocateLearningGain(engine, theoryDesign(Theory::Blended), theoryDesign(Theory::Banking), 20);
// who.levers[k].shapley sums exactly to who.totalGain
```

`examples/learning_theories_demo.cpp` runs one experiment per theory, then Activity 3.3 on children, teenagers and adults, and the
allocation of the blended lesson's gain to its practices.

## Modelling notes and limitations

- **Illustrative, not fitted.** The chapter describes theories qualitatively. The models encode those descriptions with simple,
  standard mechanisms (Rescorla–Wagner, Bush–Mosteller, item-response probabilities, power-law forgetting, Markov chains) and
  parameters chosen to give plausible effect sizes. Use them to reason about mechanisms and interactions, not to predict scores.
- **The classroom engine is a composition.** Its terms (how anxiety interferes, how much rewards add for children, how scaffolding
  substitutes for formal operations) are modelling choices documented in `classroom_engine.hpp`. Different choices would change
  the ranking of the single theories; the blended design wins because the theories address different bottlenecks.
- **Reinforcement wording.** The chapter says negative reinforcement deters behaviour; in Skinner's terms that is punishment.
  The operant model has reinforcement (strengthens), punishment (weakens) and extinction (no consequence).
- **Overjustification** (rewards eroding intrinsic interest) is not in the chapter. It is a small, adjustable parameter of the point
  system and of the reinforcement lever.
- **Fixed-interval scallops** are not reproduced: the operant learner has no sense of time since the last reinforcer.
- **Single course skill.** The classroom engine tracks one skill (θ), one memory trace per unit and one discrimination level per
  learner. It does not model forgetting of θ itself, nor peer effects between specific learners (see `social/dialogue.hpp` for that).
