// The reader's memory: its reading and note-taking habits, learned from golden answers, kept between runs.
// Written by tools/qa_golden (qa_golden DATA_DIR --write-memory src/models/student_memory.inc). Do not edit by hand.
{2,  // generation
 2400,  // lessons studied, in all generations
 "+ cognitivism: spaced retrieval of mistakes",  // study design the reading habits were learned with
 "400 lessons of edededdy/fineweb-edu-grounded-qa (tools/qa_golden/prepare.py, seed 2026)",  // lessons it was learned from
 0.36437907814979553,  // test F1 of its answers on the golden test when it was written
 "habits v1: overlap with the question (BM25); position in the passage; first sentence; length; is itself a question; two sentences; holds what is asked (a number, a name); covers the question; notes v1: informative words (idf mass); density of informative words; shares words with the rest; position in the passage; first sentence; length; is itself a question; holds a number or a name;",  // the habits these weights are for
 {1.9006549119949341,  // overlap with the question (BM25)
  -0.03165152296423912,  // position in the passage
  -0.43368998169898987,  // first sentence
  -0.095475256443023682,  // length
  -0.93110203742980957,  // is itself a question
  0.37251386046409607,  // two sentences
  0.22276841104030609,  // holds what is asked (a number, a name)
  3.8902103900909424},  // covers the question
 "+ social constructivism: scaffolding, easy before hard",  // study design the note-taking habits were learned with
 0.20152072608470917,  // test F1 of its best note on the golden test when it was written
 {0.38106733560562134,  // informative words (idf mass)
  -0.51215386390686035,  // density of informative words
  1.1348395347595215,  // shares words with the rest
  -0.78502154350280762,  // position in the passage
  -1.3048194646835327,  // first sentence
  1.1558464765548706,  // length
  -0.50380420684814453,  // is itself a question
  0.41718891263008118}}  // holds a number or a name
