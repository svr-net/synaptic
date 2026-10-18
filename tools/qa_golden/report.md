# QA golden test

200 golden questions (edededdy/fineweb-edu-grounded-qa, FineWeb-Edu passages), 200 passages + 300 knowledge documents read; vocabulary 13184 words, 6894 sentences.

| answer | token F1 | F1 ≥ 0.5 |
|---|---:|---:|
| reader, its own passage (copies a sentence) | 0.343 | 25.5% |
| reader with its remembered habits, its own passage (copies one or two sentences) | 0.374 | 31% |
| constructed from its knowledge capsule alone (never sees the text) | 0.348 | 23.5% |
| reader, open book (the whole corpus) | 0.34 | 25.5% |
| first sentence of the passage | 0.171 | 4% |
| a random sentence of the passage | 0.154 | 3% |
| oracle: the best sentence of the passage | 0.503 | 46.5% |

Open book, the reader's answer came from the question's own passage for 96% of the questions.

## The knowledge capsule

The grammar read the 600 passages (200 of the test, 400 of the lessons; 327626 words) as 18404 facts — subject, relation, object. Every word of a fact is one multivector of the space whose grades are the levels of the index: grade 0 its fact, grade 1 its word's point, grade 2 the grammar's bivector (role and place). The index router isolates each grade and routes it to its codebook — the passage's vocabulary, the grammar's lattice of 3 roles × 32 places, the runs of facts — and the capsule keeps only the codes, over a lexicon of 29104 words. Recalled from the codes alone — the rotor rebuilt from grade 2, the word bound by it, its role and place read back off the bound point — 100% of the 249064 words came back exactly.

| | bytes |
|---|---:|
| the text | 1985545 |
| **the capsule, packed** | **824417 (41.5% of the text)** |
| codes: a word's place in its vocabulary, role and place (levels 1, 2) | 474030 |
| the lexicon, front-coded | 151904 |
| the vocabularies (level 1's codebooks) | 179477 |
| the facts, as run lengths (level 0) | 19004 |

Constructed answers are said by the grammar from the recalled facts — the whole fact, its predicate, its object, its subject or two facts — and chosen by the constructing habits; the text is never used.

## The reader as a student: picking an answer

Candidates: one sentence of the passage or two adjacent ones. 400 lessons (golden questions about other passages), three passes, 200 test questions, 64 students per design; the best candidate scores 0.543. Its memory: generation 4, 4800 lessons studied before.

| study design | test F1 | lessons F1 |
|---|---:|---:|
| no feedback (the lexical reader) | 0.334 | 0.354 |
| its memory alone, no feedback | 0.374 | 0.391 |
| reinforcement | 0.362 | 0.388 |
| reinforcement + schema: starts from its memory | 0.365 | 0.405 |
| reinforcement + interleaving question kinds | 0.363 | 0.389 |
| reinforcement + worked examples | 0.372 | 0.395 |
| reinforcement + spaced retrieval of mistakes | 0.36 | 0.389 |
| reinforcement + equilibration: assimilate or accommodate | 0.363 | 0.386 |
| reinforcement + scaffolding: easy before hard | 0.362 | 0.389 |
| reinforcement + peer discussion | 0.362 | 0.381 |
| reinforcement + self-direction | 0.362 | 0.388 |
| reinforcement + growth mindset | 0.362 | 0.388 |
| reinforcement + reflection: one or two changes at a time | 0.366 | 0.385 |
| all methods | 0.356 | 0.376 |
| all methods but schema: starts from its memory | 0.352 | 0.369 |
| all methods but interleaving question kinds | 0.363 | 0.383 |
| all methods but worked examples | 0.369 | 0.391 |
| all methods but spaced retrieval of mistakes | 0.352 | 0.37 |
| all methods but equilibration: assimilate or accommodate | 0.358 | 0.375 |
| all methods but scaffolding: easy before hard | 0.36 | 0.382 |
| all methods but peer discussion | 0.362 | 0.379 |
| all methods but self-direction | 0.356 | 0.375 |
| all methods but growth mindset | 0.356 | 0.377 |
| all methods but reflection: one or two changes at a time | 0.369 | 0.39 |

| method | library models | added to reinforcement | left out of all |
|---|---|---:|---:|
| schema: starts from its memory | schema | +0.003 | -0.004 |
| interleaving question kinds | interleaving | +0.001 | +0.007 |
| worked examples | gagne, lecture | +0.011 | +0.014 |
| spaced retrieval of mistakes | memory | -0.002 | -0.004 |
| equilibration: assimilate or accommodate | equilibration | +0.001 | +0.002 |
| scaffolding: easy before hard | zpd, scaffolding | +0.001 | +0.004 |
| peer discussion | dialogue, discussion | -0 | +0.006 |
| self-direction | andragogy | +0 | +0 |
| growth mindset | motivation | +0.001 | -0 |
| reflection: one or two changes at a time | reflection | +0.004 | +0.013 |

## The reader as a student: constructing an answer from its capsule

Candidates: what the grammar can say from each fact recalled from the capsule — the whole fact, its predicate, its object, its subject. 400 lessons (golden questions about other passages), three passes, 200 test questions, 64 students per design; the best candidate scores 0.505. Its memory: generation 4, 4800 lessons studied before.

| study design | test F1 | lessons F1 |
|---|---:|---:|
| no feedback (the lexical reader) | 0.297 | 0.294 |
| its memory alone, no feedback | 0.348 | 0.362 |
| reinforcement | 0.345 | 0.353 |
| reinforcement + schema: starts from its memory | 0.347 | 0.354 |
| reinforcement + interleaving question kinds | 0.344 | 0.353 |
| reinforcement + worked examples | 0.348 | 0.362 |
| reinforcement + spaced retrieval of mistakes | 0.348 | 0.355 |
| reinforcement + equilibration: assimilate or accommodate | 0.348 | 0.355 |
| reinforcement + scaffolding: easy before hard | 0.344 | 0.353 |
| reinforcement + peer discussion | 0.325 | 0.321 |
| reinforcement + self-direction | 0.344 | 0.353 |
| reinforcement + growth mindset | 0.344 | 0.353 |
| reinforcement + reflection: one or two changes at a time | 0.344 | 0.353 |
| all methods | 0.335 | 0.344 |
| all methods but schema: starts from its memory | 0.336 | 0.346 |
| all methods but interleaving question kinds | 0.324 | 0.333 |
| all methods but worked examples | 0.306 | 0.3 |
| all methods but spaced retrieval of mistakes | 0.325 | 0.333 |
| all methods but equilibration: assimilate or accommodate | 0.33 | 0.336 |
| all methods but scaffolding: easy before hard | 0.328 | 0.335 |
| all methods but peer discussion | 0.339 | 0.353 |
| all methods but self-direction | 0.333 | 0.34 |
| all methods but growth mindset | 0.335 | 0.343 |
| all methods but reflection: one or two changes at a time | 0.339 | 0.35 |

| method | library models | added to reinforcement | left out of all |
|---|---|---:|---:|
| schema: starts from its memory | schema | +0.002 | +0.001 |
| interleaving question kinds | interleaving | -0.001 | -0.011 |
| worked examples | gagne, lecture | +0.003 | -0.029 |
| spaced retrieval of mistakes | memory | +0.003 | -0.01 |
| equilibration: assimilate or accommodate | equilibration | +0.003 | -0.005 |
| scaffolding: easy before hard | zpd, scaffolding | -0.001 | -0.007 |
| peer discussion | dialogue, discussion | -0.02 | +0.004 |
| self-direction | andragogy | -0.001 | -0.002 |
| growth mindset | motivation | -0.001 | -0 |
| reflection: one or two changes at a time | reflection | -0.002 | +0.004 |

## The reader as a student: taking notes

Candidates: the sentences of a passage, before any question. 400 lessons (golden questions about other passages), three passes, 200 test questions, 64 students per design; the best candidate scores 0.503. Its memory: generation 4, 4800 lessons studied before.

| study design | test F1 | lessons F1 |
|---|---:|---:|
| no feedback (the lexical reader) | 0.198 | 0.2 |
| its memory alone, no feedback | 0.196 | 0.201 |
| reinforcement | 0.201 | 0.208 |
| reinforcement + schema: starts from its memory | 0.199 | 0.205 |
| reinforcement + interleaving question kinds | 0.197 | 0.205 |
| reinforcement + worked examples | 0.192 | 0.206 |
| reinforcement + spaced retrieval of mistakes | 0.202 | 0.217 |
| reinforcement + equilibration: assimilate or accommodate | 0.2 | 0.214 |
| reinforcement + scaffolding: easy before hard | 0.197 | 0.203 |
| reinforcement + peer discussion | 0.197 | 0.203 |
| reinforcement + self-direction | 0.197 | 0.205 |
| reinforcement + growth mindset | 0.206 | 0.208 |
| reinforcement + reflection: one or two changes at a time | 0.196 | 0.203 |
| all methods | 0.188 | 0.189 |
| all methods but schema: starts from its memory | 0.188 | 0.189 |
| all methods but interleaving question kinds | 0.184 | 0.184 |
| all methods but worked examples | 0.198 | 0.206 |
| all methods but spaced retrieval of mistakes | 0.193 | 0.196 |
| all methods but equilibration: assimilate or accommodate | 0.188 | 0.187 |
| all methods but scaffolding: easy before hard | 0.182 | 0.182 |
| all methods but peer discussion | 0.162 | 0.155 |
| all methods but self-direction | 0.182 | 0.182 |
| all methods but growth mindset | 0.187 | 0.187 |
| all methods but reflection: one or two changes at a time | 0.196 | 0.199 |

| method | library models | added to reinforcement | left out of all |
|---|---|---:|---:|
| schema: starts from its memory | schema | -0.002 | +0 |
| interleaving question kinds | interleaving | -0.004 | -0.004 |
| worked examples | gagne, lecture | -0.009 | +0.01 |
| spaced retrieval of mistakes | memory | +0.001 | +0.005 |
| equilibration: assimilate or accommodate | equilibration | -0.001 | +0 |
| scaffolding: easy before hard | zpd, scaffolding | -0.004 | -0.006 |
| peer discussion | dialogue, discussion | -0.003 | -0.026 |
| self-direction | andragogy | -0.003 | -0.006 |
| growth mindset | motivation | +0.005 | -0.001 |
| reflection: one or two changes at a time | reflection | -0.005 | +0.008 |

Its next memory (generation 5): reading, note-taking and constructing habits learned with all methods.
Capsule of 60 passages (105064 bytes, 49.6% of their text) written to src/models/reader_capsule.inc.
