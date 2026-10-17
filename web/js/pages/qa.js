import { barChart, fmt, lineChart } from '../charts.js';
import { runModels } from '../engine/run.js';
import { callout, card, el, grid, initPage, runButton, table, tiles, toolbarNumber } from '../ui.js';

const page = initPage({
  id: 'qa',
  title: 'Reading for answers (QA)',
  context: 'models: readingQA · readingStudent · fused kernels in the space Cl(4, 1) · golden answers from FineWeb-Edu',
  description: 'A reader reads FineWeb-Edu passages and answers questions about them; its answers are compared with <b>golden answers</b> ' +
    '(<a href="https://huggingface.co/datasets/edededdy/fineweb-edu-grounded-qa">edededdy/fineweb-edu-grounded-qa</a>, ODC-By). The reader ' +
    '<b>learns how informative each word is</b> from everything it reads, weighs every sentence by the question\'s informative words in it ' +
    '(BM25), and <b>believes</b> the sentence answers the question as a point of the space, chance(atLogit(evidence − τ)). It answers with the ' +
    'sentence it believes most — from the question\'s own passage, and open book, from every sentence of every passage. Answers are scored ' +
    'with SQuAD\'s token F1 against the golden answer, next to the passage\'s first sentence, a random sentence, and the oracle: the passage ' +
    'sentence closest to the golden answer, the ceiling of any one-sentence answer. The full test (200 questions, with the knowledge corpus) ' +
    'is <code>tools/qa_golden</code>.',
});

const questions = toolbarNumber(page, 'questions', 60, { step: 10, width: 50 });

// Ask the reader about a passage of your own (outside page.content, which each run clears).
const passage = el('textarea', { rows: 4, style: 'width:100%;box-sizing:border-box;font:inherit', placeholder: 'Paste a passage…' });
const question = el('input', { type: 'text', style: 'width:100%;box-sizing:border-box;font:inherit;margin-top:6px', placeholder: 'Ask a question about it…' });
const reply = el('p', { class: 'note' });
const ask = el('button', { class: 'primary', text: 'Ask', style: 'margin-top:6px' });
ask.addEventListener('click', async () => {
  if (!passage.value.trim() || !question.value.trim()) { reply.textContent = 'Give a passage and a question.'; return; }
  ask.disabled = true;
  reply.textContent = 'Reading…';
  try {
    const { results: [r] } = await runModels([['readingQA', { readingQA: { passages: [passage.value], questions: [{ question: question.value, passage: 0 }], openBook: 0 } }]], page);
    const a = r.answers[0];
    reply.innerHTML = '';
    reply.append(el('b', { text: a.reading }), ` (belief ${fmt.pct(a.readingBelief, 0)})`);
  } catch (e) {
    reply.textContent = 'Error: ' + e.message;
  } finally {
    ask.disabled = false;
  }
});
page.content.before(el('section', { class: 'card' }, el('h2', { text: 'Ask the reader' }), passage, question, ask, reply));

runButton(page, 'Run', async (spec) => {
  const { results: [r, student], note } = await runModels([
    ['readingQA', { ...spec, readingQA: { set: 'golden', questions: questions() } }],
    ['readingStudent', { ...spec, readingStudent: { questions: questions() } }],
  ], page);
  const s = r.scores, half = r.atLeastHalf;
  tiles(page.content, [
    { label: 'Reader', value: s.reading.toFixed(3), hint: `token F1 against the golden answers (${r.scored} questions)` },
    { label: 'Open book', value: s.openBook.toFixed(3), hint: `from all ${r.sentences.length} sentences; the right passage ${fmt.pct(s.openBookFoundPassage, 0)} of the time` },
    { label: 'First sentence', value: s.firstSentence.toFixed(3), hint: 'the passage\'s first sentence as the answer' },
    { label: 'Ceiling', value: s.oracle.toFixed(3), hint: 'the passage sentence closest to the golden answer' },
  ]);
  const names = ['reader', 'open book', 'first sentence', 'random sentence', 'oracle (ceiling)'];
  const keys = ['reading', 'openBook', 'firstSentence', 'randomSentence', 'oracle'];
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
  ['question · golden answer · the reader\'s answer', 'F1', 'ceiling'],
  rows.map((a) => [el('div', { style: 'white-space:normal;max-width:900px' }, el('b', { text: a.question }), el('div', { class: 'note', text: `Golden: ${a.golden}` }), el('div', { text: `Reader: ${a.reading}` })),
    a.f1.toFixed(2), a.oracle.toFixed(2)]));
  // ---- The reader as a student.
  const sd = student.designs;
  const g3 = grid(page.content);
  lineChart(card(g3, 'The reader as a student', `${student.students} students per study design read ${student.lessons} lessons (golden questions about ` +
    'other passages) three times, seeing the golden answer after each, and answer the test questions at checkpoints along the way. Each design adds ' +
    'one module of the library to the one before.'), {
    x: Array.from(sd[0].curve, (_, k) => k + 1), xLabel: 'checkpoint (through three passes of the lessons)', yLabel: 'test F1',
    series: sd.map((d, i) => ({ name: d.name, y: Array.from(d.curve), colorIndex: i })),
  });
  barChart(card(g3, 'What each module adds', `Token F1 on the test questions after the lessons. The best answer among the candidates (one sentence or two) scores ` +
    `${student.oracle.toFixed(3)}.`), {
    labels: ['no feedback', 'behaviourism', '+ cognitivism', '+ social constr.', '+ andragogy', '+ reflection'],
    series: [{ name: 'test F1', values: sd.map((d) => d.test) }, { name: 'answers of two sentences', values: sd.map((d) => d.twoSentences), colorIndex: 1 }],
    yFormat: (v) => v.toFixed(2),
  });
  table(card(g3, 'The habits it learned', 'Weights of the reading habits: before (no feedback) and after reinforcement from feedback.'), ['habit', 'before', 'after'],
    sd[0].habits.map((h, k) => [h.habit, h.weight.toFixed(2), sd[1].habits[k].weight.toFixed(2)]));
  callout(page.content, 'The belief is a point of the <a href="core.html">space</a>, as everywhere in the library; the reader is lexical — it extracts one sentence and ' +
    'does not paraphrase or reason — which the oracle makes visible. Learning meaning in the algebra is the <a href="fga.html">FGA page</a>.');
  return note;
});
