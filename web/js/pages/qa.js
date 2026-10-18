import { barChart, fmt, lineChart } from '../charts.js';
import { runModels } from '../engine/run.js';
import { callout, card, el, grid, initPage, runButton, table, tiles, toolbarNumber } from '../ui.js';

const page = initPage({
  id: 'qa',
  title: 'Reading for answers (QA)',
  context: 'models: capsuleIngest · capsuleQA · capsuleRecall · readingQA · readingStudent · readingNotes · fused kernels in the space Cl(4, 1) · golden answers from FineWeb-Edu',
  description: 'A reader reads FineWeb-Edu passages and answers questions about them; its answers are compared with <b>golden answers</b> ' +
    '(<a href="https://huggingface.co/datasets/edededdy/fineweb-edu-grounded-qa">edededdy/fineweb-edu-grounded-qa</a>, ODC-By). The reader ' +
    '<b>learns how informative each word is</b> from everything it reads, weighs every sentence by the question\'s informative words in it ' +
    '(BM25), and <b>believes</b> the sentence answers the question as a point of the space, chance(atLogit(evidence − τ)). It answers with the ' +
    'sentence it believes most — from the question\'s own passage, and open book, from every sentence of every passage. Answers are scored ' +
    'with SQuAD\'s token F1 against the golden answer, next to the passage\'s first sentence, a random sentence, and the oracle: the passage ' +
    'sentence closest to the golden answer, the ceiling of any one-sentence answer. It also answers with what it <b>remembers</b> (the ' +
    'reading habits it learned as a student), and from its <b>knowledge capsule</b> alone: every passage it reads is ingested into the space ' +
    '— its grammar reads each clause as a fact (subject, relation, object), every word becomes a multivector routed grade by grade to a ' +
    'packed code (half the bytes of the text) — and recalled from the codes; the answer is <b>constructed</b> by the grammar from the recalled facts, chosen by the ' +
    'constructing habits it learned as a student, and never copied from the text. The full test (200 questions) is <code>tools/qa_golden</code>.',
});

const questions = toolbarNumber(page, 'questions', 60, { step: 10, width: 50 });

// How capsuleQA worked out an answer: for each part of the question, the evidence, the reasoning
// (the habits that weighed most) and the answer.
function traceOf(a) {
  const list = el('ol', { style: 'margin:4px 0 0 0;padding-left:20px' });
  for (const p of a.parts) {
    const item = el('li', { style: 'margin-bottom:6px' }, el('div', {}, el('i', { text: p.part })));
    if (p.referred && p.referred.length) item.append(el('div', { class: 'note', text: `its pronoun refers to: ${Array.from(p.referred).join(' ')}` }));
    if (!p.evidence) {
      item.append(el('div', { class: 'note', text: 'nothing it knows says it' }));
    } else {
      const e = p.evidence;
      item.append(el('div', { class: 'note', text: `evidence: [${e.subject}] [${e.relation}] [${e.object}] — belief ${fmt.pct(p.belief, 0)}` }),
        el('div', { class: 'note', text: `reasoning: says ${p.saying}; weighed by ${Array.from(p.reasoning).map((w) => `${w.habit} (${w.weight.toFixed(2)})`).join('; ')}` }),
        el('div', { text: `answer: ${p.answer}` }));
    }
    list.append(item);
  }
  return list;
}

// Ask the reader: about a passage of your own, or, with no passage, from what it knows (outside
// page.content, which each run clears).
const passage = el('textarea', { rows: 4, style: 'width:100%;box-sizing:border-box;font:inherit', placeholder: 'Paste a passage — or leave empty to ask what it knows…' });
const question = el('input', { type: 'text', style: 'width:100%;box-sizing:border-box;font:inherit;margin-top:6px', placeholder: 'Ask a question…' });
const reply = el('div', { class: 'note' });
const ask = el('button', { class: 'primary', text: 'Ask', style: 'margin-top:6px' });
ask.addEventListener('click', async () => {
  if (!question.value.trim()) { reply.textContent = 'Ask a question.'; return; }
  ask.disabled = true;
  reply.textContent = 'Reading…';
  try {
    // The passage is ingested into a capsule and recalled; the answer is constructed from the capsule
    // alone (no passage reaches readingQA). With no passage, from the stored capsule.
    const own = passage.value.trim();
    let packed;
    if (own) {
      const { results: [ing] } = await runModels([['capsuleIngest', { capsuleIngest: { passages: [own] } }]], page);
      packed = ing.packed;
    }
    // Recall, reasoning and answering in one kernel (capsuleQA), from the capsule alone.
    const q = packed ? { packed, questions: [{ question: question.value, capsule: 0 }] } : { questions: [{ question: question.value }] };
    const { results: [r] } = await runModels([['capsuleQA', { capsuleQA: q }]], page);
    const a = r.answers[0];
    reply.innerHTML = '';
    reply.append(el('p', {}, el('b', { text: a.answer || '(nothing it knows says that)' }), ' — reasoned from its capsule'));
    reply.append(traceOf(a));
  } catch (e) {
    reply.textContent = 'Error: ' + e.message;
  } finally {
    ask.disabled = false;
  }
});
page.content.before(el('section', { class: 'card' }, el('h2', { text: 'Ask the reader' }), passage, question, ask, reply));

runButton(page, 'Run', async (spec) => {
  // The golden passages ingested and packed; recalled from the packed bytes alone.
  const { results: [ing] } = await runModels([['capsuleIngest', { ...spec, capsuleIngest: { questions: questions() } }]], page);
  const { results: [rec, reasoned] } = await runModels([['capsuleRecall', { ...spec, capsuleRecall: { packed: ing.packed } }],
    ['capsuleQA', { ...spec, capsuleQA: { packed: ing.packed, questions: questions() } }]], page);
  const { results: [r, student, notes], note } = await runModels([
    ['readingQA', { ...spec, readingQA: { set: 'golden', questions: questions(), recalled: rec.recalled } }],
    ['readingStudent', { ...spec, readingStudent: { questions: questions(), students: 32 } }],
    ['readingNotes', { ...spec, readingNotes: { questions: questions() } }],
  ], page);
  const s = r.scores, half = r.atLeastHalf;
  tiles(page.content, [
    { label: 'Reader', value: s.reading.toFixed(3), hint: `token F1 against the golden answers (${r.scored} questions)` },
    { label: 'Remembering', value: s.remembered.toFixed(3), hint: `with the reading habits of its memory (generation ${r.memory.generation})` },
    { label: 'Reasoned', value: reasoned.f1.toFixed(3), hint: 'capsuleQA: the question in parts, the evidence in the space, each part\'s answer constructed — from the capsule alone' },
    { label: 'Constructed', value: s.constructed.toFixed(3), hint: `from its knowledge capsule alone, never the text: ${r.capsule.facts} facts recalled from ${ing.bytes.total.toLocaleString()} bytes, ${fmt.pct(ing.bytes.ratio, 0)} of the text (${fmt.pct(rec.recovered, 0)} exactly)` },
    { label: 'Open book', value: s.openBook.toFixed(3), hint: `from all ${r.sentences.length} sentences; the right passage ${fmt.pct(s.openBookFoundPassage, 0)} of the time` },
    { label: 'Ceiling', value: s.oracle.toFixed(3), hint: 'the passage sentence closest to the golden answer' },
  ]);
  const names = ['reader', 'remembering', 'constructed', 'open book', 'first sentence', 'random sentence', 'oracle (ceiling)'];
  const keys = ['reading', 'remembered', 'constructed', 'openBook', 'firstSentence', 'randomSentence', 'oracle'];
  const g = grid(page.content);
  barChart(card(g, 'Answers against golden answers', 'Mean token F1, and the share of answers with F1 of at least 0.5.'), {
    labels: names,
    series: [{ name: 'token F1', values: keys.map((k) => s[k]) }, { name: 'F1 ≥ 0.5', values: keys.map((k) => half[k]), colorIndex: 1 }],
    yFormat: (v) => v.toFixed(2),
  });
  barChart(card(g, 'Question by question', 'How many questions fall in each band of token F1: the reader, and the best one-sentence answer the passage holds.'), {
    labels: Array.from({ length: 10 }, (_, k) => `${(k / 10).toFixed(1)}–${((k + 1) / 10).toFixed(1)}`),
    series: [{ name: 'reader', values: Array.from(r.readingF1Histogram) }, { name: 'oracle', values: Array.from(r.oracleF1Histogram), colorIndex: 1 }],
    yFormat: (v) => v.toFixed(0),
  });
  const g2 = grid(page.content, true);
  const rows = r.answers.map((a) => a).sort((x, y) => y.f1 - x.f1);
  table(card(g2, 'Every question', 'Sorted by the reader\'s F1. Where the reader falls short of the oracle, the question\'s words matched a sentence that does not answer it, ' +
    'or the answer is paraphrased (other words) or spread over several sentences.'),
  ['question · golden answer · the reader\'s answers', 'F1', 'remembering', 'constructed', 'ceiling'],
  rows.map((a) => [el('div', { style: 'white-space:normal;max-width:900px' }, el('b', { text: a.question }), el('div', { class: 'note', text: `Golden: ${a.golden}` }),
    el('div', { text: `Reader: ${a.reading}` }), el('div', { class: 'note', text: `Constructed: ${a.constructed}` })),
  a.f1.toFixed(2), a.rememberedF1.toFixed(2), a.constructedF1.toFixed(2), a.oracle.toFixed(2)]));
  // ---- The reader as a student, with every learning method of the library.
  const sd = student.designs, mem = student.memory;
  const named = (n) => sd.find((d) => d.name === n);
  const shown = ['no feedback (the lexical reader)', 'its memory alone, no feedback', 'reinforcement', 'all methods'].map(named);
  const g3 = grid(page.content);
  lineChart(card(g3, 'The reader as a student', `${student.students} students per study design read ${student.lessons} lessons (golden questions about ` +
    'other passages) three times, seeing the golden answer after each, and answer the test questions at checkpoints along the way. A study design is ' +
    `a set of the library's learning methods; with its schema, a student starts from the reader's memory (generation ${mem.generation}, ` +
    `${mem.lessons} lessons studied).`), {
    x: Array.from(sd[0].curve, (_, k) => k + 1), xLabel: 'checkpoint (through three passes of the lessons)', yLabel: 'test F1',
    series: shown.map((d, i) => ({ name: d.name, y: Array.from(d.curve), colorIndex: i })),
  });
  barChart(card(g3, 'Each method of the library', 'Change in test F1 when the method is added to reinforcement alone, and when it is left out of all ' +
    `methods (all methods: ${named('all methods').test.toFixed(3)}; reinforcement: ${named('reinforcement').test.toFixed(3)}).`), {
    labels: student.methods.map((m) => m.models),
    series: [{ name: 'added to reinforcement', values: student.methods.map((m) => m.alone) },
      { name: 'left out of all', values: student.methods.map((m) => -m.removed), colorIndex: 1 }],
    yFormat: (v) => v.toFixed(3),
  });
  table(card(g3, 'The habits it learned', `Weights of the reading habits: the prior (the lexical reader), its memory (generation ${mem.generation}, ` +
    `learned with “${mem.design || 'none'}”), and what it would remember after this run (“${student.remember.design}”).`), ['habit', 'prior', 'memory', 'next memory'],
  sd[0].habits.map((h, k) => [h.habit, h.weight.toFixed(2), mem.weights[k].toFixed(2), student.remember.weights[k].toFixed(2)]));
  table(card(grid(page.content, true), 'Every study design', 'Test and lesson F1 of each design, and its share of two-sentence answers.'),
    ['design', 'test F1', 'lessons F1', 'two sentences'], sd.map((d) => [d.name, d.test.toFixed(3), d.lessons.toFixed(3), fmt.pct(d.twoSentences, 0)]));
  // ---- Taking notes (a skill), before any question.
  const g4 = grid(page.content);
  const ex = notes.passages.find((p) => p.sentences.length > 6) || notes.passages[0];
  table(card(g4, 'Taking notes', `Before any question, it keeps the ${notes.keep} sentences of each passage it believes are worth keeping — ` +
    `${fmt.pct(notes.kept, 0)} of the text. The best kept sentence scores ${notes.bestKeptF1.toFixed(3)} against the golden answers, the best of the whole ` +
    `passage ${notes.bestF1.toFixed(3)}. One passage, sentence by sentence:`), ['sentence', 'worth keeping', 'kept'],
  ex.sentences.map((t) => [el('div', { style: 'white-space:normal;max-width:700px', text: t.text }), fmt.pct(t.belief, 0), t.kept ? '✓' : '']));
  const remembers = card(g4, 'What it remembers', `Its memory, generation ${student.memory.generation} (${student.memory.lessons} lessons studied): ` +
    'the habits it answers with, and the habits it takes notes with. A memory learned for other habits is refused, and it starts from the prior.');
  table(remembers, ['answering habit', 'weight'], student.memory.weights.map((w, k) => [sd[0].habits[k].habit, w.toFixed(2)]));
  table(remembers, ['note-taking habit', 'weight'], notes.memory.weights.map((w, k) => [notes.memory.habits[k], w.toFixed(2)]));
  table(remembers, ['constructing habit', 'weight'], r.memory.constructing.map((h) => [h.habit, h.weight.toFixed(2)]));
  // ---- How it reasoned: worked answers (a question in parts first).
  const worked = [...reasoned.answers.filter((a) => a.parts.length > 1).slice(0, 1), ...reasoned.answers.filter((a) => a.parts.length === 1).slice(0, 2)];
  const rc = card(grid(page.content, true), 'How it reasoned', 'capsuleQA works out each answer as Claude did as a student of the same lessons ' +
    '(its habits are the constructing prior): the question in parts, one sentence each; for each part the evidence — the facts of the capsule ' +
    'holding the question\'s informative words, by identity of their points; the reasoning — the grammar\'s ways of saying it weighed by the habits; ' +
    'and the answer. All in one kernel, from the capsule alone.');
  for (const a of worked) {
    rc.append(el('p', { style: 'margin:12px 0 2px' }, el('b', { text: a.question })), traceOf(a),
      el('div', { class: 'note', text: `Golden: ${a.golden} — F1 ${a.f1.toFixed(2)}` }));
  }
  // ---- The knowledge capsule: the index router of grades, the bytes, one passage's facts recalled.
  const roles = ['subject', 'relation', 'object'];
  const dict = rec.recalled.dictionary;
  const facts = rec.recalled.passages[0].facts.slice(0, 8);
  const b = ing.bytes;
  const g5 = grid(page.content);
  table(card(g5, 'The knowledge capsule, packed', `Every word of a fact is one multivector of the space whose grades are the levels of the index: ` +
    'grade 0 its fact, grade 1 its word\'s point of S³, grade 2 the grammar\'s bivector (its role in the (N, A) plane, its place in the (B, C) plane). ' +
    'The index router isolates each grade and routes it to its codebook — the passage\'s vocabulary, the lattice of 3 roles × 32 places, the runs ' +
    `of facts — and only the codes are kept: ${b.total.toLocaleString()} bytes for ${b.text.toLocaleString()} bytes of text (${fmt.pct(b.ratio, 0)}).`),
  ['part', 'bytes'], [['codes: word in its vocabulary, role, place (grades 1, 2)', b.codes.toLocaleString()], ['lexicon, front-coded', b.lexicon.toLocaleString()],
    ['vocabularies (grade 1\'s codebooks)', b.vocabulary.toLocaleString()], ['facts, as run lengths (grade 0)', b.facts.toLocaleString()],
    ['total', b.total.toLocaleString()], ['the text', b.text.toLocaleString()]]);
  table(card(g5, 'Recalled from the codes', 'Each code rebuilt as a multivector; the rotor formed from its grade 2 binds the word, and its role and place ' +
    `are read back off the bound point — ${fmt.pct(rec.recovered, 0)} exactly; each fact's codes are kept out of order and put back by the places. ` +
    'The first passage\'s facts:'),
  roles, facts.map((f) => roles.map((k) => el('div', { style: 'white-space:normal', text: Array.from(f[k]).map((i) => dict[i]).join(' ') }))));
  callout(page.content, 'The belief is a point of the <a href="core.html">space</a>, as everywhere in the library. The reader and its remembered habits copy ' +
    'sentences of the passage; the constructed answer copies nothing: the grammar says it from facts recalled from the packed capsule, by the ' +
    'rotors of role and place; even the verb is found in the space (a word\'s cues dilate a point, the verb class is a cap). It does not paraphrase or reason. Learning meaning in the algebra is the ' +
    '<a href="fga.html">FGA page</a>.');
  return note;
});
