# synaptic — working memory for Claude sessions

A C++ library whose models of *Instruction in Libraries and Information Centers* run as WGSL fused kernels in the space
Cl(4,1) (pure elliptic CGA). Read README.md for the models; this file keeps the rules and the lessons learned working on it.

## Rules from the owner

- Pure elliptic CGA exclusively, fused kernels without exception: everything works in the same space, Cl(4,1).
- WGSL is the only kernel implementation; the fallback is the software GPU (a WGSL interpreter in C++), never a second path.
- Keep the JS and WASM layers tiny: all logic in C++. Pages only call `runModels` and draw.
- "Open PR and merge on green": open a PR from the session branch, wait for CI, merge with `merge_method: merge`;
  `.github/workflows/delete-merged-branches.yml` deletes the branch. Then reset the session branch to `origin/main`.
- No model identifiers in commits, PRs or code.

## The space and the kernels

- Blades: N=1, A=2, B=4, C=8, e=16 (e² = −1). Points are null vectors; beliefs are `chance(atLogit(x, kA))`.
- A model is agents × jobs: `P(i)` constants, `J(i)` job row, `U(k)` uniforms, `stat(i, v)`, `quant()`; module helpers go in
  `Model.helpersWgsl`; run with `kernel::run(name, spec)`. Multivector helpers: `src/space/algebra.cpp` (`algebraWgsl`).
- New model: add to `src/models/all_models.cpp`, the model count in `tests/test_models.cpp`, a test, a page if it has one
  (`web/*.html`, `web/js/pages/*.js`, nav in `web/js/app.js`, `web/e2e.mjs`), and README tables.

## Pitfalls learned (do not relearn)

- Software GPU WGSL: no `cross`, no `countOneBits`; avoid arrays as function parameters (use `var<private>` banks).
- Reserved/builtin names fail: `common`, `set`, `pass`, `from`, `shared`, `round`, `step` as identifiers
  (the reserved-word test scans agents and helpers). WGSL has no overloading: a helper fn may not reuse a prelude name
  such as `nudge` (the software GPU accepts it, WebGPU does not; `model_helpers_do_not_redeclare_functions` checks).
- WebGPU (SwiftShader): arrays declared inside loops are not re-zeroed — zero them explicitly; sin/cos/exp are approximate —
  normalise rotors (`mvNormaliseRotor`); exact-½ belief ties flip — allow flips in `web/kernels.mjs` / `gpu.js` `ALLOW`.
- clang needs `<algorithm>` for `std::clamp`. Local gcc dangling-reference warnings in tests are pre-existing.
- Docker builds here need the proxy build-args, `--secret id=ca,src=/root/.ccr/ca-bundle.crt` and `--network host`;
  start the daemon with `sudo dockerd &` if it is down. Never `pkill -f` a pattern that matches your own shell.

## QA golden test, the reader's memory and knowledge

- `tools/qa_golden/run.sh /path/to/knowledge` (data from `prepare.py`, pinned sha256, seed 2026); `--baseline` fails on a
  drop of more than 0.01 below `tools/qa_golden/baseline.json`. The data lives outside the repo (e.g. build/qa_golden).
- What the reader keeps: `src/models/reading_memory.{hpp,cpp}` loads `student_memory.inc` (habit and note-taking weights,
  generation, provenance, fingerprint of the habits) and `reader_knowledge.inc` (notes on the 200 golden passages).
  Changing a habit's definition: bump its version in `memory::fingerprint()` — the old memory is then refused, not misapplied.
- To grow them: `build/qa_golden DATA --keep 5 --write-memory src/models/student_memory.inc --write-knowledge
  src/models/reader_knowledge.inc`, rebuild, `--write-baseline tools/qa_golden/baseline.json`, copy DATA/report.md to
  tools/qa_golden/report.md, rebuild web/wasm (Docker), commit. The memory is chosen by lesson F1, never by the test.
- The student studies with every learning method of the library (`kMethods` in reading_student.cpp, one bit each of the
  job row): reinforcement, schema, interleaving, worked examples, spaced retrieval, equilibration, scaffolding, peer
  discussion, self-direction, growth mindset, reflection; 24 designs. The memory is learned with all methods (owner's ask).
- Results: reader 0.343; remembered habits (gen 3) 0.375; from knowledge alone (37% of the text) 0.308. Alone, worked
  examples (+0.011) and schema (+0.008) help most; all methods together interfere (0.362). Don't tune method strengths on
  the test questions. The next gain needs a richer learner (vocabulary/association learning).
- JS may send an empty array as a numeric array: read spec arrays by index (`size()`, `operator[]`), not `asArray()`.
