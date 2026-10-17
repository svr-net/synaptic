// The reader's memory: its reading and note-taking habits, learned from golden answers, kept between runs.
// Written by tools/qa_golden (qa_golden DATA_DIR --write-memory src/models/student_memory.inc). Do not edit by hand.
{3,  // generation
 3600,  // lessons studied, in all generations
 "all methods",  // study design the reading habits were learned with
 "400 lessons of edededdy/fineweb-edu-grounded-qa (tools/qa_golden/prepare.py, seed 2026)",  // lessons it was learned from
 0.36179092526435852,  // test F1 of its answers on the golden test when it was written
 "habits v1: overlap with the question (BM25); position in the passage; first sentence; length; is itself a question; two sentences; holds what is asked (a number, a name); covers the question; notes v1: informative words (idf mass); density of informative words; shares words with the rest; position in the passage; first sentence; length; is itself a question; holds a number or a name;",  // the habits these weights are for
 {-0.31240025162696838,  // overlap with the question (BM25)
  -0.24586902558803558,  // position in the passage
  -0.56326025724411011,  // first sentence
  -0.55420774221420288,  // length
  -0.75429171323776245,  // is itself a question
  -0.47516921162605286,  // two sentences
  0.13862340152263641,  // holds what is asked (a number, a name)
  5.2840147018432617},  // covers the question
 "all methods",  // study design the note-taking habits were learned with
 0.18935588002204895,  // test F1 of its best note on the golden test when it was written
 {4.3601961135864258,  // informative words (idf mass)
  -0.064343929290771484,  // density of informative words
  1.134040355682373,  // shares words with the rest
  -0.80727016925811768,  // position in the passage
  -0.96779084205627441,  // first sentence
  2.68532395362854,  // length
  -0.48000586032867432,  // is itself a question
  -0.064034320414066315}}  // holds a number or a name
