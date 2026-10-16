import { fmt, lineChart } from '../charts.js';
import { modelButton, runModels } from '../engine/run.js';
import { callout, card, grid, initPage, table, tiles, toolbarNumber } from '../ui.js';

const page = initPage({
  id: 'constructivism',
  title: 'Constructivism: assimilation, disequilibrium, accommodation',
  context: 'models: equilibration · fused kernels in the space Cl(4, 1)',
  description: "Piaget's example: a child who knows dogs calls every four-legged animal a dog (<b>assimilation</b>). When the world contradicts " +
    'the schema ("that\'s a cat"), conflict builds up as <b>disequilibrium</b> until the child <b>accommodates</b> by building a new schema. ' +
    'Only accommodation changes the structure of knowledge, so it is where learning happens. Disequilibrium is uncomfortable, and a learner ' +
    'who avoids the discomfort also avoids the learning. Geometry is pure elliptic CGA: experiences are lifted onto the sphere S³ as null vectors, ' +
    'a schema is a cap (assimilation iff X·S ≥ 0), assimilation rotates the prototype by a rotor, and accommodation centres a new cap on the conflicting experiences.' +
    ' <i>In the space:</i> experiences are lifted onto S³ conformally; schemas are prototype points with assimilation caps, assimilation is a rotor towards the experience, accommodation the centre of the conflicting points.',
});

const feedback = toolbarNumber(page, 'P(an adult names the animal)', 0.3, { step: 0.05 });
const avoidance = toolbarNumber(page, 'avoidance of discomfort', 0.7, { step: 0.05 });

modelButton(page, 'Run', async (spec) => {
  const { results: [r], note } = await runModels([['equilibration', { ...spec, equilibration: { feedback: feedback(), avoidance: avoidance() } }]], page);
  const [silent, some, always, avoid] = r.runs;
  tiles(page.content, [
    { label: 'Accuracy, nobody names animals', value: fmt.pct(silent.finalAccuracy, 0), hint: 'cats are assimilated as dogs' },
    { label: 'Accuracy, sometimes named', value: fmt.pct(some.finalAccuracy, 0), hint: `${fmt.num(some.accommodations, 2)} accommodations per child` },
    { label: 'Assimilations per accommodation', value: fmt.num(some.assimilations / Math.max(1e-9, some.accommodations), 0) },
    { label: 'Accuracy when avoiding discomfort', value: fmt.pct(avoid.finalAccuracy, 0), hint: `${fmt.num(avoid.avoidances, 1)} conflicts ignored` },
  ]);
  const g = grid(page.content);
  const t = Array.from(silent.accuracy, (_, i) => i + 1);
  lineChart(card(g, 'Telling dogs from cats', `Accuracy on a fixed test set of dogs and cats. Cats appear from observation ${r.catsFrom}.`), {
    x: t, xLabel: 'observation', yLabel: 'accuracy', series: r.runs.map((x, i) => ({ name: x.label, y: Array.from(x.accuracy), colorIndex: i })),
  });
  lineChart(card(g, 'Disequilibrium', 'Mean cognitive conflict; it rises with each contradiction and collapses at accommodation.'), {
    x: t, xLabel: 'observation', yLabel: 'disequilibrium', series: r.runs.map((x, i) => ({ name: x.label, y: Array.from(x.disequilibrium), colorIndex: i })),
  });
  lineChart(card(g, 'Number of schemas', 'Structure of knowledge: grows only by accommodation.'), {
    x: t, xLabel: 'observation', yLabel: 'schemas', series: r.runs.map((x, i) => ({ name: x.label, y: Array.from(x.schemas), colorIndex: i })),
  });
  table(card(g, 'Adaptations per child', ''), ['condition', 'assimilations', 'conflicts', 'accommodations', 'avoided', 'final accuracy'],
    r.runs.map((x) => [x.label, fmt.num(x.assimilations, 1), fmt.num(x.conflicts, 1), fmt.num(x.accommodations, 2), fmt.num(x.avoidances, 1), fmt.pct(x.finalAccuracy, 1)]));
  callout(page.content, 'In the classroom engine the <i>problem first</i> lever creates productive disequilibrium: it deepens encoding and helps learners with a growth ' +
    'mindset in an empathetic class, but raises anxiety without support.');
  return note;
});
