# Study journal — Claude as the student

Same methods as the library's readingStudent: reinforcement (each answer scored against the
golden answer with SQuAD token F1), worked examples (study the golden answers I missed), spaced
retrieval (re-answer my worst lesson after a gap), reflection (after each block of five, change
one or two habits at a time). Answers are always worked out: evidence → reasoning → answer.

## Block 1 (lessons 0–4): mean F1 0.431

| lesson | F1 | what the golden answer did that I did not |
|---|---:|---|
| 0 Austin in Ireland | 0.50 | a full sentence restating the subject: "The name Austin has existed in Ireland since…" — I wrote a fragment |
| 1 curator | 0.38 | "The exhibition was curated by Vera Longtoe Sheehan, who is from the Elnu Abenaki Tribe" — restated, plus the one detail the passage ties to her |
| 2 raw fish | 0.29 | kept each measure's reason ("as freezing kills parasites", "under a thick bed of ice", "more than one or two hours") — I stripped the list to bare items |
| 3 kite | 0.45 | reused the passage's sentence nearly whole, with its closing image |
| 4 tomato seeds | 0.54 | "produce fruits that are…" — restated the subject; left out traits I added (determinate, virus tolerance) |

Reflection: my answers are right but terse — short-answer style. The golden answers are one or
two full sentences in the passage's own words. Two changes (no more):
1. Answer in full sentences that restate the question's subject, in the passage's words.
2. Keep the detail the passage attaches to the fact (who, why, how much) — not only the bare fact.

## Block 2 (lessons 5–9, and lesson 2 again): mean F1 0.680

| lesson | F1 | what the golden answer did that I did not |
|---|---:|---|
| 5 benign tumours | 0.63 | named two or three examples (meningiomas, acoustic neuromas, pituitary gland tumors); I gave none |
| 6 Pirekuas | 0.79 | credited who carries it: "communicated by the 'pireris', the singers and composers" |
| 7 enamel erosion | 0.56 | one sentence per part: the main cause, then the triggers, then grinding — mine ran them together |
| 8 clean water animals | 0.63 | short: the rule, then the exceptions — I added extra examples |
| 9 bullet points | 0.76 | the definition plus the contrast asked for, in one sentence |
| 2 raw fish (spaced retrieval) | 0.72 | up from 0.29: full sentences with each measure's reason |

Reflection: habits 1–2 work (0.43 → 0.68). What is left: the golden answers name two or three
examples or the person/agent the passage credits, and give each part of the question its own
sentence — two or three sentences, nothing beyond what was asked. Two changes:
3. Name two or three of the passage's examples, and who it credits, when the question invites them — not whole lists.
4. One sentence per part of the question; two or three sentences in all; nothing the question did not ask.

## Block 3 (lessons 10–14, and lesson 7 again): mean F1 0.605

| lesson | F1 | what the golden answer did that I did not |
|---|---:|---|
| 10 stink bug | 0.62 | one sentence: the species, the evidence (65 percent), who said so — I added a second sentence it did not need |
| 11 Mathnasium | 0.59 | the same steps; it named "proprietary curriculum" which I left out |
| 12 proof of work | 0.54 | kept the definition; did not add my inference about rewriting blocks |
| 13 sports | 0.54 | led with the mechanism (players win at world competitions → a name for the country), not the summary |
| 14 chemotherapy | 0.63 | added only the outcome the passage ties to it (gone in 2–10 days); I also added "excellent relief" |
| 7 enamel (spaced retrieval) | 0.71 | up from 0.56 with one sentence per part |

Reflection: habits 3–4 helped on retrieval, but in new lessons I lost precision by adding what the
golden answers leave out. One refinement (habit 2): the detail to keep is the one the passage puts
right next to the fact — same or next sentence — not further consequences or my own inferences.

## Over the blocks

| | mean F1 |
|---|---:|
| block 1 (no habits) | 0.431 |
| block 2 (habits 1–2) | 0.680 |
| block 3 (habits 1–4) | 0.605 |
| spaced retrieval of the worst lesson | 0.29 → 0.72, 0.56 → 0.71 |
