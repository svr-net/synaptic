# QA golden test

200 golden questions (edededdy/fineweb-edu-grounded-qa, FineWeb-Edu passages), 200 passages + 300 knowledge documents read; vocabulary 13184 words, 6894 sentences.

| answer | token F1 | F1 ≥ 0.5 |
|---|---:|---:|
| reader, its own passage | 0.343 | 25.5% |
| reader, open book (the whole corpus) | 0.34 | 25.5% |
| first sentence of the passage | 0.171 | 4% |
| a random sentence of the passage | 0.154 | 3% |
| oracle: the best sentence of the passage | 0.503 | 46.5% |

Open book: the reader's answer came from the question's own passage for 96% of the questions.

## The reader as a student

400 lessons (golden questions about other passages), 200 test questions, 64 students per design. Candidates: one sentence or two adjacent ones; the best of them scores 0.543.

| study design | test F1 | lessons F1 | two-sentence answers |
|---|---:|---:|---:|
| no feedback (the lexical reader) | 0.334 | 0.354 | 0% |
| behaviourism: reinforcement from feedback | 0.362 | 0.388 | 79% |
| + cognitivism: spaced retrieval of mistakes | 0.364 | 0.389 | 85.7% |
| + social constructivism: scaffolding, easy before hard | 0.36 | 0.388 | 95.5% |
| + andragogy: self-directed study | 0.361 | 0.388 | 93.9% |
| + reflection: one or two changes at a time | 0.358 | 0.385 | 92.9% |
