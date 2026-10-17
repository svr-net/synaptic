# QA golden test

An end-to-end test of reading for answers against golden answers.

- **Golden set**: [edededdy/fineweb-edu-grounded-qa](https://huggingface.co/datasets/edededdy/fineweb-edu-grounded-qa)
  (ODC-By): questions about FineWeb-Edu passages, each with a golden answer written from its passage.
  `prepare.py` downloads it, checks its SHA-256, and draws a seeded sample (default 200 questions about
  distinct passages that do not refer to "the passage").
- **Corpus**: the documents of [svr-net/knowledge](https://github.com/svr-net/knowledge), read alongside
  the test passages.
- **Model**: `readingQA` (`src/models/reading_qa.cpp`), a fused WGSL kernel. The reader learns how
  informative each word is from everything it reads (inverse document frequency), weighs every
  candidate sentence by the question's words in it (BM25), and believes the sentence answers the
  question with chance(atLogit(evidence − τ)), a point of the space. It answers twice: from the
  question's own passage, and open book from every sentence of the corpus.
- **Scoring**: SQuAD token F1 against the golden answer, next to the passage's first sentence, a random
  sentence, and the oracle (the passage sentence closest to the golden answer — the ceiling of any
  one-sentence extractive answer). `qa_golden` fails if a score falls more than 0.01 below
  `baseline.json`; `webgpu.mjs` runs the same kernels on WebGPU in headless Chromium and fails unless
  every answer is the one the software GPU gave.

```sh
cmake -B build && cmake --build build
pip install pyarrow
tools/qa_golden/run.sh /path/to/knowledge
```

`report.md` is the recorded run. The reader is lexical: it finds the sentence that shares the
question's informative words. It does not paraphrase, combine sentences or reason, which is why the
oracle — itself a single extracted sentence — bounds it, and why golden answers that paraphrase or
summarise several sentences score low for every extractive reader.
