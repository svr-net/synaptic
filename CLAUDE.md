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
- Exact geometric tests on WebGPU: normalise (cos, sin) of any rotor you compare exactly (`grammarTurn`), or points leave S³.
- `Value::operator[]` returns a copy: never bind a reference into it (`x[i].at(..)`); use `asArray()[i]`.
- WebGPU (SwiftShader): arrays declared inside loops are not re-zeroed — zero them explicitly; sin/cos/exp are approximate —
  normalise rotors (`mvNormaliseRotor`); exact-½ belief ties flip — allow flips in `web/kernels.mjs` / `gpu.js` `ALLOW`.
- clang needs `<algorithm>` for `std::clamp`. Local gcc dangling-reference warnings in tests are pre-existing.
- Docker builds here need the proxy build-args, `--secret id=ca,src=/root/.ccr/ca-bundle.crt` and `--network host`;
  start the daemon with `sudo dockerd &` if it is down. Never `pkill -f` a pattern that matches your own shell.

## QA golden test, the reader's memory and knowledge

- `tools/qa_golden/run.sh /path/to/knowledge` (data from `prepare.py`, pinned sha256, seed 2026); `--baseline` fails on a
  drop of more than 0.01 below `tools/qa_golden/baseline.json`. The data lives outside the repo (e.g. build/qa_golden).
- What the reader keeps: `src/models/reading_memory.{hpp,cpp}` loads `student_memory.inc` (reading, note-taking and
  constructing weights, generation, provenance, fingerprint of the habits) and the knowledge capsule `reader_capsule.inc`
  (the 60 embedded golden passages, packed: base64 of `memory::packCapsule`; no text).
  Changing a habit's definition: bump its version in `memory::fingerprint()` — the old memory is then refused, not misapplied.
- Owner's rule: constructed answers must never use the original text — only the capsule (codes + lexicon), the grammar
  and the learned habits. The capsule is an index router of grades (owner's ask): a slot is a multivector, grade 0 fact,
  grade 1 word point, grade 2 the grammar bivector; ingest isolates each grade and routes it to its codebook (code =
  (vocabulary index · 3 + role) · 32 + place); recall rebuilds the rotor from grade 2 and reads role/place back
  (`grammarOf`). The capsule must stay smaller than the text (`capsuleSaving` in the baseline). Changing the grammar, the
  rotors or the packing: regenerate the capsule.
- To grow them: `build/qa_golden DATA --write-memory src/models/student_memory.inc --write-capsule src/models/reader_capsule.inc`,
  rebuild, `--write-baseline tools/qa_golden/baseline.json`, copy DATA/report.md to tools/qa_golden/report.md, rebuild
  web/wasm (Docker), commit. Don't write a new memory generation if it scores below the current one.
- The student studies with every learning method of the library (`kMethods` in reading_student.cpp, one bit each of the
  job row): reinforcement, schema, interleaving, worked examples, spaced retrieval, equilibration, scaffolding, peer
  discussion, self-direction, growth mindset, reflection; 24 designs. The memory is learned with all methods (owner's ask).
- Results: reader 0.343; remembered habits (gen 4) 0.374; constructed from the capsule alone 0.348 (prior habits 0.297;
  best constructible 0.504). All methods together interfere slightly; don't tune method strengths on the test questions.
  Verb finding is geometry (owner's ask): `memory::clauses()` gives cue bits only; the ingest kernel dilates a point by the
  cues and tests it against the verb class's cap (`word_classes` in reading_capsule.cpp). Next: learn the cue weights as
  a student, and vocabulary/association learning.
- capsuleQA (owner's ask: ingestion, reasoning and answering as one model): recall + reasoning + answering in one kernel
  from the codes; question parts (`memory::questionParts`), evidence by point identity, construct features v2 shared with
  `memory::constructCandidates` (one-part answers must equal readingQA's constructed — tested). The constructing prior is
  Claude's study habits (`tools/qa_golden/claude_student`); the memory's constructing section (v1) is refused per section,
  so constructing runs on that prior. Results: reasoned 0.337, constructed 0.342 (v1-learned was 0.348).
- JS may send an empty array as a numeric array: read spec arrays by index (`size()`, `operator[]`), not `asArray()`.
- Growing what it keeps (owner's ask: incremental, persistable, in the same space): knowledge is capsule segments (`SYNK`:
  each extends the lexicon and passages before it, indices never change; `memory::packSegment`/`joinSegments`/`capsuleOf`),
  memory is records (`SYNM`: one section's study, folded by evidence; `memory::packRecord`/`recallFor`, spec `memoryRecords`).
  Ingest routes grade 1 through the growing index (cells cut by spheres, entries by coincidence) whose WGSL is svr-net/compute's
  `index_cga.hpp`, carried in `src/models/compute_index.inc` — change it in compute, then `tools/sync_compute.sh`. Changing the
  word points, index or packing: change `capsuleFingerprint()` and regenerate the capsule. Word coordinates are 48 bits of a
  64-bit FNV (the old LCG kept 24 bits: "fruit" and "tended" coincided). Kept in the browser by `web/js/knowledge.js` (IndexedDB), in
  svr-net/knowledge's `reader/` by svr-net/store's `ReaderLog` (`synstore_reader`); `qa_golden --write-knowledge DIR` writes
  segments and only records whose folded memory answers the test at least as well. compute, store and knowledge are private:
  synaptic's CI (public) cannot fetch them, so it carries the compute helpers verbatim.
- svr-net/ui answers with the reader too (`tools/reader`: `synaptic reader session`, linked into ui's binary from this
  repo through `action.yml`; segments from the knowledge repo's `reader/` log). Keep `tools/reader/reader.hpp`'s
  `syn::reader::run` stable: ui's `docker/synaptic.cpp` calls it.
