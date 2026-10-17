# QA golden test

200 golden questions (edededdy/fineweb-edu-grounded-qa, FineWeb-Edu passages), 200 passages + 300 knowledge documents read; vocabulary 13184 words, 6894 sentences.

| answer | token F1 | F1 ≥ 0.5 |
|---|---:|---:|
| reader, its own passage | 0.343 | 25.5% |
| reader with its remembered habits, its own passage | 0.375 | 31.5% |
| reader, open book (the whole corpus) | 0.34 | 25.5% |
| reader from its knowledge alone (its notes; the passages gone) | 0.308 | 20% |
| first sentence of the passage | 0.171 | 4% |
| a random sentence of the passage | 0.154 | 3% |
| oracle: the best sentence of the passage | 0.503 | 46.5% |

Open book, the reader's answer came from the question's own passage for 96% of the questions; from its knowledge, its answer came from the notes on the question's passage for 91.5%.

## The reader as a student

400 lessons (golden questions about other passages), three passes, 200 test questions, 64 students per design. Candidates: one sentence or two adjacent ones; the best of them scores 0.543. Its memory: generation 3, 3600 lessons studied before.

| study design | test F1 | lessons F1 | two-sentence answers |
|---|---:|---:|---:|
| no feedback (the lexical reader) | 0.334 | 0.354 | 0% |
| its memory alone, no feedback | 0.375 | 0.387 | 23% |
| reinforcement | 0.362 | 0.388 | 87% |
| reinforcement + schema: starts from its memory | 0.364 | 0.405 | 54.5% |
| reinforcement + interleaving question kinds | 0.363 | 0.389 | 76.7% |
| reinforcement + worked examples | 0.372 | 0.395 | 83% |
| reinforcement + spaced retrieval of mistakes | 0.36 | 0.389 | 89% |
| reinforcement + equilibration: assimilate or accommodate | 0.363 | 0.386 | 96% |
| reinforcement + scaffolding: easy before hard | 0.362 | 0.389 | 86% |
| reinforcement + peer discussion | 0.362 | 0.381 | 24% |
| reinforcement + self-direction | 0.362 | 0.388 | 88.7% |
| reinforcement + growth mindset | 0.362 | 0.388 | 87.5% |
| reinforcement + reflection: one or two changes at a time | 0.366 | 0.385 | 95.5% |
| all methods | 0.363 | 0.381 | 34.4% |
| all methods but schema: starts from its memory | 0.352 | 0.369 | 39.7% |
| all methods but interleaving question kinds | 0.366 | 0.387 | 38.8% |
| all methods but worked examples | 0.359 | 0.375 | 0% |
| all methods but spaced retrieval of mistakes | 0.355 | 0.373 | 32.5% |
| all methods but equilibration: assimilate or accommodate | 0.361 | 0.378 | 36.2% |
| all methods but scaffolding: easy before hard | 0.362 | 0.383 | 36.1% |
| all methods but peer discussion | 0.362 | 0.379 | 60.5% |
| all methods but self-direction | 0.362 | 0.38 | 34.2% |
| all methods but growth mindset | 0.357 | 0.378 | 36.3% |
| all methods but reflection: one or two changes at a time | 0.368 | 0.39 | 51.7% |

Each method of the library, as the reader's way of studying: what it adds to reinforcement alone, and what it costs to leave it out of all methods (test F1).

| method | library models | added to reinforcement | left out of all |
|---|---|---:|---:|
| schema: starts from its memory | schema | +0.002 | -0.012 |
| interleaving question kinds | interleaving | +0.001 | +0.003 |
| worked examples | gagne, lecture | +0.011 | -0.004 |
| spaced retrieval of mistakes | memory | -0.002 | -0.008 |
| equilibration: assimilate or accommodate | equilibration | +0.001 | -0.002 |
| scaffolding: easy before hard | zpd, scaffolding | +0.001 | -0.001 |
| peer discussion | dialogue, discussion | -0 | -0.001 |
| self-direction | andragogy | +0 | -0.001 |
| growth mindset | motivation | +0.001 | -0.006 |
| reflection: one or two changes at a time | reflection | +0.004 | +0.005 |

## Taking notes, and answering from them

The student also learns, with the same methods, which sentences of a passage are worth keeping before any question is asked: the lesson's golden answer says which one was. Test F1 here is the F1 of the single sentence it would keep first.

| study design | best note F1 |
|---|---:|
| no feedback (the lexical reader) | 0.198 |
| its memory alone, no feedback | 0.196 |
| reinforcement | 0.201 |
| reinforcement + schema: starts from its memory | 0.199 |
| reinforcement + interleaving question kinds | 0.197 |
| reinforcement + worked examples | 0.192 |
| reinforcement + spaced retrieval of mistakes | 0.202 |
| reinforcement + equilibration: assimilate or accommodate | 0.2 |
| reinforcement + scaffolding: easy before hard | 0.197 |
| reinforcement + peer discussion | 0.197 |
| reinforcement + self-direction | 0.197 |
| reinforcement + growth mindset | 0.206 |
| reinforcement + reflection: one or two changes at a time | 0.196 |
| all methods | 0.188 |

With the note-taking habits it learned with all methods, it takes notes on the 200 test passages and answers the questions from the notes alone:

| sentences kept per passage | text kept | best kept sentence F1 | answers from the notes, F1 |
|---:|---:|---:|---:|
| 1 | 9.1% | 0.22 | 0.207 |
| 2 | 17.6% | 0.276 | 0.234 |
| 3 | 24.7% | 0.332 | 0.261 |
| 5 | 37% | 0.387 | 0.308 |
| all | 100% | 0.503 | 0.333 |

Its next memory (generation 4): reading habits of "all methods", note-taking habits of "all methods".
