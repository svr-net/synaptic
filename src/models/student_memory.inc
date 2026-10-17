// The reader's memory: the reading habits it has learned from golden answers, kept between runs.
// Written by tools/qa_golden (qa_golden DATA_DIR --write-memory src/models/student_memory.inc). Do not edit by hand.
{1,  // generation
 1200,  // lessons studied, in all generations
 "+ cognitivism: spaced retrieval of mistakes",  // study design it was learned with
 "400 lessons of edededdy/fineweb-edu-grounded-qa (tools/qa_golden/prepare.py, seed 2026)",  // lessons it was learned from
 0.36355924606323242,  // its test F1 on the golden test when it was written
 {2.1416552066802979,  // overlap with the question (BM25)
  -0.11328789591789246,  // position in the passage
  -0.52090036869049072,  // first sentence
  -0.0059786690399050713,  // length
  -0.66406577825546265,  // is itself a question
  0.43745630979537964,  // two sentences
  0.1912192702293396,  // holds what is asked (a number, a name)
  2.8098263740539551}}  // covers the question
