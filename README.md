# synaptic — learning theories as computational models

[![CI](https://github.com/svr-net/synaptic/actions/workflows/ci.yml/badge.svg)](https://github.com/svr-net/synaptic/actions/workflows/ci.yml)

A C++17 library that models the theories in

> *Learning Theories: Understanding How People Learn*, chapter 3 of
> *Instruction in Libraries and Information Centers*, Illinois Open Publishing Network
> (<https://iopn.library.illinois.edu/pressbooks/instructioninlibraries/chapter/learning-theories-understanding-how-people-learn/>).

The chapter surveys how people learn: behaviourism, humanism, cognitivism, constructivism and social constructivism; the developmental
stages of Piaget, Perry and Vygotsky's Zone of Proximal Development; andragogy; and motivation and the growth mindset. It ends by
arguing that the theories are not mutually exclusive and are best combined. This library turns each theory into a model you can
simulate, then runs every theory on the same simulated learners in a classroom engine. That lets you compare the lessons each theory
would design (the chapter's Activity 3.3), allocate a design's learning gain to its teaching practices, and ask which practice to
strengthen next.
It is an independent implementation. It does not reproduce the text of the chapter. The models are illustrative: their parameters
are chosen so that the effects the chapter describes appear with plausible sizes. They are not fitted to data.

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
| `e2e` | Every page in headless Chromium: dev site over HTTP, standalone from `file://` and over HTTP, including phone layouts. Screenshots are uploaded as an artifact |
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
  index.html, core.html, ... sensitivity.html   14 pages
  assets/app.<hash>.js                          all pages (classic script, ~70 KiB)
  assets/syn-runtime.<hash>.js                  WASM worker with embedded library (~530 KiB)
  assets/style.<hash>.css
  manifest.json, README.txt                     build commit, file sizes and SHA-256
```

The whole site is about 610 KiB (about 240 KiB zipped). `node web/tests/e2e.mjs --root dist/standalone --file` opens every page from
disk and checks that the pages make no network request at all.

## Docker

The multi-stage `Dockerfile` builds everything from source. Each stage runs its tests, so a successful build means they passed.

```sh
docker build -t synaptic .                         # WASM from source + checks -> standalone site on nginx (default)
docker run --rm -p 8080:80 synaptic                # web front end at http://localhost:8080

docker build --target native -t synaptic:native .  # native C++ build, ctest
docker run --rm synaptic:native                    # end-to-end demo

docker build --target e2e .                        # every page in headless Chromium:
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
| `e2e` | `mcr.microsoft.com/playwright` | Browser test of all pages in three modes |
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
| `core.html` | Random streams, logistic link, quantiles, session calendars | `coreDemo` |
| `behaviorism.html` | Pavlov (Rescorla–Wagner), Kamin blocking, Watson's generalised fear, Skinner's schedules, a classroom point system | `conditioning` · `operant` · `tokenEconomy` |
| `humanism.html` | Maslow's prepotent needs, library anxiety and Kuhlthau's Information Search Process | `humanism` |
| `cognitivism.html` | Forgetting curves, retrieval and spaced practice, cramming, interleaving, schemas, Gagné's nine events | `memory` · `gagne` |
| `constructivism.html` | Assimilation, disequilibrium and accommodation (dogs and cats), avoidance of discomfort | `equilibration` |
| `social.html` | The ZPD, tutoring policies, peer dialogue, cultural capital | `zpd` · `dialogue` |
| `development.html` | Piaget's stages by age, activity fit, Perry's scheme and source judgement | `development` |
| `andragogy.html` | Knowles' assumptions × design, Bartle's needs cycle | `andragogy` |
| `motivation.html` | Svinicki's value × expectancy with five levers, Dweck's mindsets and feedback | `motivation` |
| `classroom.html` | The classroom engine: score profile, learner states, equity | `classroom` |
| `theories.html` | Activity 3.3 designed by every theory, on three populations | `theories` |
| `allocation.html` | Shapley, incremental and standalone allocation of a design's gain | `allocation` |
| `sensitivity.html` | Lever sensitivities with common random numbers vs independent seeds | `sensitivity` |

The pages share one specification: the learner population, the course, the instructional design (sixteen levers, or any theory's
preset) and the simulation settings. You edit it on any page, and the browser's local storage keeps it.

**All numerics are in C++.** `wasm/bindings.cpp` turns the JS spec into library objects and returns plain objects of numbers and
`Float64Array`s; the JavaScript only routes calls and draws charts (a small dependency-free canvas library, `web/js/charts.js`).

**Rebuilding the WASM module.** The built `web/wasm/syn.{js,wasm}` is committed. Rebuild it with Docker (`--target wasm-artifacts` above), or with a local Emscripten:

```sh
emcmake cmake -S . -B build-wasm -DCMAKE_BUILD_TYPE=Release
cmake --build build-wasm                 # writes web/wasm/syn.js and syn.wasm
node web/tests/wasm.test.mjs             # every entry point plus invariant checks, in Node
node web/tests/e2e.mjs                   # every page in headless Chromium (Playwright)
```

## How the library maps onto the chapter

| Topic | Module | What is implemented |
|---|---|---|
| Behaviourism: classical conditioning | `behaviorism/classical_conditioning.hpp` | Rescorla–Wagner learning (closed-form acquisition), Pavlov's acquisition and extinction, Kamin blocking, Watson's fear conditioning with a Shepard generalisation gradient |
| Behaviourism: operant conditioning | `behaviorism/operant_conditioning.hpp` | Bush–Mosteller response strength under reinforcement, punishment and extinction; continuous, fixed/variable ratio and interval schedules; cumulative records; the partial-reinforcement extinction effect from the surprise of unrewarded runs |
| Behaviourism in the classroom | `behaviorism/token_economy.hpp` | Point system and gamification: external incentive, habit, extinction after withdrawal, optional overjustification |
| Humanism: needs | `humanism/maslow.hpp` | Five prepotent levels with smooth gates, dominant need, readiness for learning, the effect of an empathetic classroom climate |
| Humanism: affect | `humanism/information_search.hpp` | Kuhlthau's six ISP stages with their feelings; library anxiety interfering with cognitive work; empathetic support |
| Cognitivism: memory | `cognitivism/memory.hpp` | Power-law forgetting curve; restudy vs retrieval vs retrieval with feedback; desirable difficulty; massed, spaced and expanding plans; cramming |
| Cognitivism: interleaving | `cognitivism/interleaving.hpp` | Execution skill vs discrimination: blocked practice looks better in practice and does worse on a mixed test |
| Cognitivism: schemas | `cognitivism/schema.hpp` | Concept network with spreading activation; encoding links new knowledge to what is active, so stimulating prior recall adds retrieval routes |
| Cognitivism: Gagné | `classroom/instructional_design.hpp` | The nine events of instruction mapped onto the design levers (`designFromGagne`) |
| Constructivism | `constructivism/equilibration.hpp` | Schemas as labelled prototypes; assimilation, disequilibrium, accommodation; avoidance of discomfort; the dog/cat example |
| Social constructivism | `social/zpd.hpp`, `social/dialogue.hpp` | Vygotsky's three zones, learning gain peaking in the ZPD, tutoring policies with fading scaffolds; peer dialogue with more knowledgeable others and articulation; Bourdieu's cultural capital and inclusive teaching |
| Developmental stages | `development/piaget.hpp`, `development/perry.hpp` | Piaget's stages and capabilities, stage probabilities by age, activity fit (the climate-change lesson); Perry's positions as a Markov chain, source judgement by position |
| Andragogy | `andragogy/andragogy.hpp` | Knowles' six assumptions paired with design features, engagement for children, adolescents and adults; Bartle's seven steps and the needs-diagnosis cycle |
| Motivation | `motivation/expectancy_value.hpp` | Svinicki's M = V·E with the chapter's five levers; self-efficacy from mastery experience; calibration by self-assessment |
| Growth mindset | `motivation/mindset.hpp` | Fixed and growth responses to failure (effort, strategy change, giving up); ability praise vs "not yet"; coaching; high standards |
| Learners | `classroom/learner.hpp` | Populations (children, secondary, adults) with prior knowledge, anxiety, mindset, efficacy, interest, cultural capital and age |
| Combining the theories | `classroom/classroom_engine.hpp` | Every theory acting on the same learners over a course; score, recall and risk profiles; equity gap; learner states per session |
| Activity 3.3 | `classroom/instructional_design.hpp` | The lesson each theory would design (16 levers), including the blended design of the conclusion |
| Allocation | `classroom/allocation.hpp` | Shapley (adds up exactly), incremental and standalone contributions of each practice |
| Sensitivities | `classroom/sensitivities.hpp` | Bump-and-rerun slopes per lever with common random numbers vs independent seeds, with standard errors |

## Conventions

- Abilities θ and difficulties b are on a logit scale: P(success) = σ(θ − b). Levers, needs, anxiety, mindset and similar states lie in [0, 1].
- Times are in days from the first session (or trials, opportunities or weeks where the model says so).
- Every simulation takes a seed. Learner *i* draws from its own stream of that seed, with a fixed number of draws per session, so
  two designs run on the same learners with the same luck (common random numbers). The tests check this property: the
  CRN sensitivities have far smaller standard errors than independent-seed ones.
- A memory trace's stability S is the interval after which recall falls to 90%.

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
