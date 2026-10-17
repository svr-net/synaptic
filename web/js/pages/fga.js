import { barChart, fmt, lineChart } from '../charts.js';
import { runModels } from '../engine/run.js';
import { callout, card, el, grid, initPage, passFail, runButton, table, tiles, toolbarNumber } from '../ui.js';

const page = initPage({
  id: 'fga',
  title: 'Learning meaning (FGA)',
  context: 'models: fgaAlgebra · fgaLearning · fgaRotors · fused kernels in the space Cl(4, 1)',
  description: 'Functional geometric algebra (Pustejovsky, <i>Toward a Functional Geometric Algebra for Natural Language Semantics</i>, 2026) ' +
    'builds meaning in a Clifford algebra: <b>grades are types</b> (scalars truth values, vectors entities and predicates, bivectors relations), ' +
    '<b>contraction is application</b> (John loves Mary = j ⌋ (ℓ ⌊ m), a scalar), <b>the wedge binds roles</b> (r_AG ∧ j + r_TH ∧ m, unbound exactly by ' +
    'contraction) and <b>rotors coerce</b> (begin the book, chicken as food). Here all of it runs in the library\'s one space, Cl(4, 1): entities on the ' +
    'sphere of V_E = ⟨A, B, C⟩, predicate content and purpose along e (e² = −1), the roles N and e. <b>Learners read a text</b> and learn a multivector ' +
    'per word by gradient descent inside the fused kernel; a sentence\'s meaning is the scalar ⟨S R O⟩₀, believed as a point of the space. The ' +
    'paper\'s claims are checked, not assumed: most hold exactly; a few hold only under conditions, and the page says which.',
});

const epochs = toolbarNumber(page, 'epochs', 30, { step: 5, width: 50 });
const negatives = toolbarNumber(page, 'negatives per sentence', 1, { step: 1, width: 40 });
// The text the learners read: empty means the default family of six, whose truth is known.
const text = el('textarea', { rows: 4, style: 'width:100%;box-sizing:border-box;font:inherit', placeholder:
  'Leave empty for the default text (a family of six: who is older, who are siblings), or write your own sentences, one fact each: ' +
  '"John loves Mary. Mary does not love Bob. The book is older than Mary." — up to 8 names, 3 relations, 64 sentences.' });
page.content.before(el('section', { class: 'card' }, el('h2', { text: 'The text the learners read' }), text));

runButton(page, 'Run', async (spec) => {
  const { results: [algebra, learning, rotors], note } = await runModels([
    ['fgaAlgebra', spec],
    ['fgaLearning', { ...spec, fgaLearning: { text: text.value.trim(), epochs: epochs(), negatives: negatives() } }],
    ['fgaRotors', spec],
  ], page);
  const d = learning.designs, typed = d[0], flat = d[4];
  const holding = algebra.claims.filter((c) => c.holds).length;
  const two = rotors.runs[1];
  const kinds = ['typed', 'typed + scalar', 'untyped', 'untyped + prior', 'flat'];
  tiles(page.content, [
    { label: 'Claims checked', value: `${holding} of ${algebra.claims.length}`, hint: 'claims of the paper that hold exactly in Cl(4, 1); the rest fail as stated' },
    learning.world
      ? { label: 'Reversed sentences', value: `${fmt.pct(flat.reversed, 0)} → ${fmt.pct(typed.reversed, 0)}`, hint: 'judged false (never read): flat product → bivector relation' }
      : { label: 'Text learned', value: fmt.pct(typed.read, 0), hint: 'sentences of your text the typed learners get right' },
    learning.world
      ? { label: 'Converse of "sibling"', value: `${fmt.pct(typed.symmetric, 0)} → ${fmt.pct(flat.symmetric, 0)}`, hint: 'judged true: bivector (antisymmetric) → flat (symmetric)' }
      : { label: 'Words', value: `${learning.entities.length} + ${learning.relations.length}`, hint: 'names and relations, each a multivector of Cl(4, 1)' },
    { label: 'A rotor from two words', value: `${two.rotorTest.toFixed(2)} vs ${two.projectionTest.toFixed(2)}`, hint: 'error on the rest of the family: rotor vs learned projection' },
  ]);

  // ---- Learning from the text.
  const g = grid(page.content);
  table(card(g, 'What the learners read', `${learning.learners} learners of each kind, each from its own random lexicon. A sentence is "name relation name"; "not" makes it false.` +
    (learning.ignored.length ? ` Ignored (too many names or relations): ${learning.ignored.join(' ')}` : '')),
  ['sentence', ''], learning.sentences.map((s) => [s.text, s.truth ? 'true' : 'false']));
  lineChart(card(g, 'Learning curves', 'Sentences of the text judged right after each pass (epoch), for each kind of learner.'), {
    x: Array.from(typed.curve, (_, i) => i + 1), xLabel: 'epoch', yLabel: 'read right',
    series: d.map((x, i) => ({ name: x.name, y: Array.from(x.curve), colorIndex: i })),
  });
  if (learning.world)
    barChart(card(g, 'What they infer about sentences they never read', `The reverse of an "older" sentence (${learning.tests.reversed}, false), unread ` +
      `"older" pairs (${learning.tests.transitive}, true) and the converse of a "sibling" sentence (${learning.tests.symmetric}, true). A bivector relation is ` +
      'antisymmetric, so it gets the reverses right and no converse; the flat product is symmetric, so the opposite. No learner infers unread "older" pairs: ' +
      'the corrupted sentences teach that unread means false, and on the sphere an antisymmetric relation is cyclic (sin of an angle), so long chains wrap around.'), {
      labels: kinds,
      series: [['the text', 'read'], ['reversed', 'reversed'], ['unread "older"', 'transitive'], ['converse "sibling"', 'symmetric'], ['every sentence', 'world']]
        .map(([name, key], i) => ({ name, values: d.map((x) => x[key]), colorIndex: i })),
      yFormat: (v) => fmt.pct(v, 0),
    });
  barChart(card(g, 'Where the meaning lives', 'The share of each grade in the learned entities, and in each relation (§11.3: does the grade match the type?).'), {
    labels: kinds,
    series: [
      ...['grade 0', 'grade 1', 'grade 2'].map((name, k) => ({ name: `entities, ${name}`, values: d.map((x) => x.entityGrades[k]), colorIndex: k })),
      ...learning.relations.map((r, i) => ({ name: `"${r}", grade 2`, values: d.map((x) => x.relationGrades[i][2]), colorIndex: 3 + i })),
    ],
    yFormat: (v) => fmt.pct(v, 0),
  });
  const gb = grid(page.content);
  for (const [k, who] of [[0, 'typed learners'], [4, 'flat learners']])
    learning.relations.forEach((r, ri) =>
      table(card(gb, `"${r}": what the ${who} believe`, 'Row: subject, column: object; the mean belief that "row relation column" is true.'),
        ['', ...learning.entities], learning.entities.map((s, si) => [s, ...Array.from(d[k].belief[ri][si], (p) => fmt.pct(p, 0))])));

  // ---- Rotors and projections.
  const g2 = grid(page.content);
  lineChart(card(g2, 'Regular polysemy: a rotor or a projection', `Eight nouns whose food sense is the animal sense turned by ${rotors.angle.toFixed(2)} rad ` +
    `(± ${rotors.spread.toFixed(2)}) in one hidden plane. Error on the nouns not read, against the number read (§9.3's experiment).`), {
    x: rotors.runs.map((r) => r.train), xLabel: 'nouns read', yLabel: 'error on the others',
    series: [{ name: 'rotor of Cl(4, 1)', y: rotors.runs.map((r) => r.rotorTest), markers: true },
      { name: 'learned projection (5 × 5)', y: rotors.runs.map((r) => r.projectionTest), colorIndex: 1, markers: true }],
  });
  barChart(card(g2, 'What a rotor keeps', 'Norm drift and the error of going back (food → animal) on the nouns not read; the rotor\'s alignment with the hidden plane.'), {
    labels: rotors.runs.map((r) => `${r.train} read`),
    series: [['rotor: norm drift', 'rotorNorm'], ['projection: norm drift', 'projectionNorm'], ['rotor: back', 'rotorInverse'], ['projection: back', 'projectionInverse'],
      ['rotor: plane found', 'plane']].map(([name, key], i) => ({ name, values: rotors.runs.map((r) => r[key]), colorIndex: i })),
    yFormat: (v) => v.toFixed(2),
  });

  // ---- The algebra.
  const g3 = grid(page.content, true);
  table(card(g3, 'The paper\'s claims, checked in the space', `Mean over ${algebra.claims.length ? 'random lexicons' : ''}: exact means below 10⁻⁴ (32-bit rounding). ` +
    algebra.decomposition.join(' · ') + '.'),
  ['', 'claim', 'measured'], algebra.claims.map((c) => [c.section, el('div', {}, c.claim, c.note ? el('div', { class: 'note', text: c.note }) : null),
    passFail(c.holds, c.measured.toExponential(1))]));
  const g4 = grid(page.content);
  lineChart(card(g4, 'Binding capacity', 'k role–filler pairs superposed, each recovered and cleaned up among eight fillers. FGA unbinding is exact while the roles are orthogonal ' +
    'to each other and to the fillers — two in Cl(4, 1), N and e — and fails beyond; holographic binding in the same 32 coefficients is approximate and degrades slowly.'), {
    x: Array.from(algebra.bindings), xLabel: 'bindings', yLabel: 'recovered',
    series: [{ name: 'FGA (wedge, contraction)', y: Array.from(algebra.fgaRecovery), markers: true },
      { name: 'HRR (circular convolution)', y: Array.from(algebra.hrrRecovery), colorIndex: 1, markers: true },
      { name: 'HRR: cosine with the filler', y: Array.from(algebra.hrrCosine), colorIndex: 2, markers: true }],
  });
  callout(page.content, 'The space is the one of every page (<a href="core.html">the space</a>): the learners\' beliefs are its points; the rotors are the versors that move ' +
    'knowledge between channels on the <a href="cognitivism.html">cognitivism page</a>. Source: J. Pustejovsky (2026), arXiv:2604.25902.');
  return note;
});
