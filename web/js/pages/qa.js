import { barChart, fmt, lineChart } from '../charts.js';
import { runModels } from '../engine/run.js';
import { callout, card, el, grid, initPage, runButton, table, tiles, toolbarNumber } from '../ui.js';

const page = initPage({
  id: 'qa',
  title: 'Reading for answers (QA)',
  context: 'models: readingQA · readingStudent · readingNotes · fused kernels in the space Cl(4, 1) · golden answers from FineWeb-Edu',
  description: 'A reader reads FineWeb-Edu passages and answers questions about them; its answers are compared with <b>golden answers</b> ' +
    '(<a href="https://huggingface.co/datasets/edededdy/fineweb-edu-grounded-qa">edededdy/fineweb-edu-grounded-qa</a>, ODC-By). The reader ' +
    '<b>learns how informative each word is</b> from everything it reads, weighs every sentence by the question\'s informative words in it ' +
    '(BM25), and <b>believes</b> the sentence answers the question as a point of the space, chance(atLogit(evidence − τ)). It answers with the ' +
    'sentence it believes most — from the question\'s own passage, and open book, from every sentence of every passage. Answers are scored ' +
    'with SQuAD\'s token F1 against the golden answer, next to the passage\'s first sentence, a random sentence, and the oracle: the passage ' +
    'sentence closest to the golden answer, the ceiling of any one-sentence answer. It also answers with what it <b>remembers</b>: the ' +
    'reading habits it learned as a student, and its <b>knowledge</b> — the notes it took on the passages it read, before any question — ' +
    'from which it answers when the passages are gone. The full test (200 questions, with the knowledge corpus) is <code>tools/qa_golden</code>.',
});

const questions = toolbarNumber(page, 'questions', 60, { step: 10, width: 50 });

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
    const own = passage.value.trim();
    const q = own ? { passages: [own], questions: [{ question: question.value, passage: 0 }], openBook: 0, closedBook: 0 }
      : { passages: [], questions: [{ question: question.value }] };
    const { results: [r] } = await runModels([['readingQA', { readingQA: q }]], page);
    const a = r.answers[0];
    reply.innerHTML = '';
    if (own) {
      reply.append(el('p', {}, el('b', { text: a.remembered }), ` — with the habits it remembers (belief ${fmt.pct(a.rememberedBelief, 0)})`));
      if (a.reading !== a.remembered) reply.append(el('p', { text: `By word overlap alone: ${a.reading}` }));
    } else {
      reply.append(el('p', {}, el('b', { text: a.knowledge }), ` — from its notes (belief ${fmt.pct(a.knowledgeBelief, 0)}; ` +
        `${r.knowledge.passages} passages read, ${fmt.pct(r.knowledge.kept, 0)} of their text kept)`));
    }
  } catch (e) {
    reply.textContent = 'Error: ' + e.message;
  } finally {
    ask.disabled = false;
  }
});
page.content.before(el('section', { class: 'card' }, el('h2', { text: 'Ask the reader' }), passage, question, ask, reply));

runButton(page, 'Run', async (spec) => {
  const { results: [r, student, fresh, notes], note } = await runModels([
    ['readingQA', { ...spec, readingQA: { set: 'golden', questions: questions() } }],
    ['readingStudent', { ...spec, readingStudent: { questions: questions() } }],
    ['readingStudent', { ...spec, readingStudent: { questions: questions(), memory: false } }],
    ['readingNotes', { ...spec, readingNotes: { questions: questions() } }],
  ], page);
  const s = r.scores, half = r.atLeastHalf;
  tiles(page.content, [
    { label: 'Reader', value: s.reading.toFixed(3), hint: `token F1 against the golden answers (${r.scored} questions)` },
    { label: 'Remembering', value: s.remembered.toFixed(3), hint: `with the reading habits of its memory (generation ${r.memory.generation})` },
    { label: 'From its knowledge', value: s.knowledge.toFixed(3), hint: `the passages gone: from its notes, ${fmt.pct(r.knowledge.kept, 0)} of the text; the right notes ${fmt.pct(s.knowledgeFoundPassage, 0)} of the time` },
    { label: 'Open book', value: s.openBook.toFixed(3), hint: `from all ${r.sentences.length} sentences; the right passage ${fmt.pct(s.openBookFoundPassage, 0)} of the time` },
    { label: 'Ceiling', value: s.oracle.toFixed(3), hint: 'the passage sentence closest to the golden answer' },
  ]);
  const names = ['reader', 'remembering', 'from its knowledge', 'open book', 'first sentence', 'random sentence', 'oracle (ceiling)'];
  const keys = ['reading', 'remembered', 'knowledge', 'openBook', 'firstSentence', 'randomSentence', 'oracle'];
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
  ['question · golden answer · the reader\'s answers', 'F1', 'remembering', 'knowledge', 'ceiling'],
  rows.map((a) => [el('div', { style: 'white-space:normal;max-width:900px' }, el('b', { text: a.question }), el('div', { class: 'note', text: `Golden: ${a.golden}` }),
    el('div', { text: `Reader: ${a.reading}` }), el('div', { class: 'note', text: `From its notes: ${a.knowledge}` })),
  a.f1.toFixed(2), a.rememberedF1.toFixed(2), a.knowledgeF1.toFixed(2), a.oracle.toFixed(2)]));
  // ---- The reader as a student, starting from its memory of earlier runs.
  const sd = student.designs, fd = fresh.designs, mem = student.memory;
  const g3 = grid(page.content);
  lineChart(card(g3, 'The reader as a student', `${student.students} students per study design read ${student.lessons} lessons (golden questions about ` +
    'other passages) three times, seeing the golden answer after each, and answer the test questions at checkpoints along the way. Each design adds ' +
    `one module of the library to the one before. They start from the reader's memory: generation ${mem.generation}, ${mem.lessons} lessons studied ` +
    'in earlier runs.'), {
    x: Array.from(sd[0].curve, (_, k) => k + 1), xLabel: 'checkpoint (through three passes of the lessons)', yLabel: 'test F1',
    series: sd.map((d, i) => ({ name: d.name, y: Array.from(d.curve), colorIndex: i })),
  });
  barChart(card(g3, 'What each module adds, and what memory keeps', 'Token F1 on the test questions after the lessons: starting fresh (the lexical ' +
    `reader) and starting from memory. The best answer among the candidates (one sentence or two) scores ${student.oracle.toFixed(3)}.`), {
    labels: ['no feedback', 'behaviourism', '+ cognitivism', '+ social constr.', '+ andragogy', '+ reflection'],
    series: [{ name: 'fresh', values: fd.map((d) => d.test) }, { name: 'from memory', values: sd.map((d) => d.test), colorIndex: 1 }],
    yFormat: (v) => v.toFixed(2),
  });
  table(card(g3, 'The habits it learned', `Weights of the reading habits: the prior (the lexical reader), its memory (generation ${mem.generation}, ` +
    `learned with “${mem.design || 'none'}”), and what it would remember after this run (“${student.remember.design}”).`), ['habit', 'prior', 'memory', 'next memory'],
  sd[0].habits.map((h, k) => [h.habit, fresh.memory.weights[k].toFixed(2), mem.weights[k].toFixed(2), student.remember.weights[k].toFixed(2)]));
  // ---- Its knowledge: notes taken before any question.
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
  callout(page.content, 'The belief is a point of the <a href="core.html">space</a>, as everywhere in the library; the reader is lexical — it extracts one sentence and ' +
    'does not paraphrase or reason — which the oracle makes visible. Learning meaning in the algebra is the <a href="fga.html">FGA page</a>.');
  return note;
});
