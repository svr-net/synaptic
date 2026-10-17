# QA golden test

200 golden questions (edededdy/fineweb-edu-grounded-qa, FineWeb-Edu passages), 200 passages + 300 knowledge documents read; vocabulary 13184 words, 6894 sentences.

| answer | token F1 | F1 ≥ 0.5 |
|---|---:|---:|
| reader, its own passage | 0.343 | 25.5% |
| reader with its remembered habits, its own passage | 0.361 | 25.5% |
| reader, open book (the whole corpus) | 0.34 | 25.5% |
| reader from its knowledge alone (its notes; the passages gone) | 0.304 | 18.5% |
| first sentence of the passage | 0.171 | 4% |
| a random sentence of the passage | 0.154 | 3% |
| oracle: the best sentence of the passage | 0.503 | 46.5% |

Open book, the reader's answer came from the question's own passage for 96% of the questions; from its knowledge, its answer came from the notes on the question's passage for 91%.

## The reader as a student

400 lessons (golden questions about other passages), 200 test questions, 64 students per design. Candidates: one sentence or two adjacent ones; the best of them scores 0.543.

Its memory: generation 2, 2400 lessons studied before (+ cognitivism: spaced retrieval of mistakes).

| study design | test F1, fresh | test F1, from memory | lessons F1, from memory | two-sentence answers |
|---|---:|---:|---:|---:|
| no feedback (the lexical reader) | 0.334 | 0.361 | 0.393 | 88% |
| behaviourism: reinforcement from feedback | 0.362 | 0.366 | 0.396 | 65.1% |
| + cognitivism: spaced retrieval of mistakes | 0.364 | 0.365 | 0.394 | 79.8% |
| + social constructivism: scaffolding, easy before hard | 0.36 | 0.364 | 0.395 | 78.2% |
| + andragogy: self-directed study | 0.361 | 0.364 | 0.394 | 81.9% |
| + reflection: one or two changes at a time | 0.358 | 0.364 | 0.393 | 80.9% |

## Taking notes, and answering from them

The student also learns which sentences of a passage are worth keeping before any question is asked: the lesson's golden answer says which one was. Test F1 here is the F1 of the single sentence it would keep first.

| study design | best note F1, fresh | best note F1, from memory |
|---|---:|---:|
| no feedback (the lexical reader) | 0.198 | 0.206 |
| behaviourism: reinforcement from feedback | 0.199 | 0.201 |
| + cognitivism: spaced retrieval of mistakes | 0.202 | 0.2 |
| + social constructivism: scaffolding, easy before hard | 0.202 | 0.203 |
| + andragogy: self-directed study | 0.201 | 0.201 |
| + reflection: one or two changes at a time | 0.197 | 0.204 |

With the note-taking habits of "+ social constructivism: scaffolding, easy before hard" (the design that did best on its own lessons), it takes notes on the 200 test passages and answers the questions from the notes alone:

| sentences kept per passage | text kept | best kept sentence F1 | answers from the notes, F1 |
|---:|---:|---:|---:|
| 1 | 11.6% | 0.205 | 0.19 |
| 2 | 20.1% | 0.272 | 0.229 |
| 3 | 26.7% | 0.326 | 0.258 |
| 5 | 37.2% | 0.39 | 0.304 |
| all | 100% | 0.503 | 0.333 |

Its next memory (generation 3): reading habits of "behaviourism: reinforcement from feedback", note-taking habits of "+ social constructivism: scaffolding, easy before hard".
