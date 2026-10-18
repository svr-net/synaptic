// The reader's memory: its reading, note-taking and constructing habits, learned from golden answers, kept between runs.
// Written by tools/qa_golden (qa_golden DATA_DIR --write-memory src/models/student_memory.inc). Do not edit by hand.
{4,  // generation
 4800,  // lessons studied, in all generations
 "all methods",  // study design the reading habits were learned with
 "400 lessons of edededdy/fineweb-edu-grounded-qa (tools/qa_golden/prepare.py, seed 2026)",  // lessons it was learned from
 0.36311611533164978,  // test F1 of its answers on the golden test when it was written
 "habits v1: overlap with the question (BM25); position in the passage; first sentence; length; is itself a question; two sentences; holds what is asked (a number, a name); covers the question; notes v1: informative words (idf mass); density of informative words; shares words with the rest; position in the passage; first sentence; length; is itself a question; holds a number or a name; construct v1 (superseded)",  // the habits these weights are for
 {-0.57924079895019531,  // overlap with the question (BM25)
  -0.26822015643119812,  // position in the passage
  -0.55282580852508545,  // first sentence
  -0.47324261069297791,  // length
  -0.51404190063476562,  // is itself a question
  -0.59454351663589478,  // two sentences
  0.0093806246295571327,  // holds what is asked (a number, a name)
  6.0041875839233398},  // covers the question
 "all methods",  // study design the note-taking habits were learned with
 0.18832451105117798,  // test F1 of its best note on the golden test when it was written
 {4.6884431838989258,  // informative words (idf mass)
  -0.059463690966367722,  // density of informative words
  1.1075451374053955,  // shares words with the rest
  -0.50283437967300415,  // position in the passage
  -0.99223786592483521,  // first sentence
  2.7271401882171631,  // length
  -0.46214690804481506,  // is itself a question
  0.049179054796695709},  // holds a number or a name
 "all methods",  // study design the constructing habits were learned with
 0.32901737093925476,  // test F1 of its constructed answers on the golden test when it was written
 {0.24462373554706573,  // overlap of the fact with the question (BM25)
  0.13301682472229004,  // says two facts (this one and the next)
  2.016864538192749,  // says the whole fact
  -2.6890532970428467,  // says the predicate (relation, object)
  0.26765313744544983,  // says the object
  0.008591783232986927,  // says the subject
  0.08780892938375473,  // holds what is asked (a number, a name)
  1.7351304292678833}}  // covers the question
