import { run } from '../syn-client.js';
import { fmt, lineChart } from '../charts.js';
import { callout, card, grid, initPage, runButton, table, tiles, toolbarNumber } from '../ui.js';

const page = initPage({
  id: 'social',
  title: 'Social constructivism: the ZPD, dialogue and culture',
  context: 'syn/social: zoneOf · expectedGain · simulateTutoring · simulateDialogue',
  description: "<b>Vygotsky's Zone of Proximal Development</b>: tasks a learner can do only with help from a more knowledgeable other. Learning " +
    'needs success (with help) and novelty (not yet mastered alone), so it peaks in the ZPD. <b>Dialogue</b> with peers moves learners towards ' +
    'more knowledgeable classmates, and explaining deepens the explainer\'s own understanding. <b>Bourdieu</b>: schools reflect the culture of ' +
    'wealthier families, so a mismatch between home and school culture taxes learning unless teaching is inclusive.',
});

const scaffold = toolbarNumber(page, 'scaffolding s (logits)', 1.5, { step: 0.1 });

runButton(page, 'Run', async (spec) => {
  const [z, d] = await Promise.all([run('zpd', { ...spec, zpd: { scaffold: scaffold() } }), run('dialogue', spec)]);
  const best = z.tutoring.reduce((a, b) => (b.gain > a.gain ? b : a));
  const none = d.runs.find((r) => r.grouping.startsWith('individual') && r.inclusivity === 0);
  const mixedInc = d.runs.find((r) => r.grouping.startsWith('mixed') && r.inclusivity > 0);
  tiles(page.content, [
    { label: 'ZPD for θ = 0', value: `b ∈ [${fmt.num(z.bandLow, 2)}, ${fmt.num(z.bandHigh, 2)}]`, hint: 'difficulties doable only with help' },
    { label: 'Best tutoring policy', value: best.policy, hint: `ability gain ${fmt.num(best.gain, 2)}` },
    { label: 'Dialogue: understanding', value: `${fmt.pct(none.finalMean, 0)} → ${fmt.pct(mixedInc.finalMean, 0)}`, hint: 'alone → mixed groups, inclusive' },
    { label: 'Gap between home cultures', value: `${fmt.num(none.finalGap, 3)} → ${fmt.num(mixedInc.finalGap, 3)}` },
  ]);
  const g = grid(page.content);
  const b = Array.from(z.difficulty);
  lineChart(card(g, 'Three zones for a learner of ability 0', `Independent where P(alone) ≥ 0.8; proximal while P(with help) ≥ 0.5. Learning gain (×4 for scale) peaks inside.`), {
    x: b, xLabel: 'task difficulty b', yLabel: 'probability',
    series: [{ name: 'P(success alone)', y: Array.from(z.alone) }, { name: 'P(success with help)', y: Array.from(z.assisted), colorIndex: 1 },
      { name: 'learning gain ×4', y: Array.from(z.gain, (v) => 4 * v), colorIndex: 2, dash: true }],
  });
  const s = Array.from(z.tutoring[0].mean, (_, i) => i);
  lineChart(card(g, 'Tutoring policies', 'Mean ability over sessions; band: 10–90% of learners for the ZPD policy.'), {
    x: s, xLabel: 'session', yLabel: 'ability θ',
    series: z.tutoring.map((r, i) => ({ name: r.policy, y: Array.from(r.mean), colorIndex: i, band: i === 3 ? { lower: Array.from(r.q10), upper: Array.from(r.q90) } : undefined })),
  });
  lineChart(card(g, 'Share of tasks inside the ZPD', ''), {
    x: s, xLabel: 'session', yLabel: 'share', series: z.tutoring.map((r, i) => ({ name: r.policy, y: Array.from(r.zoneShare[1]), colorIndex: i })),
  });
  const sess = Array.from(d.runs[0].mean, (_, i) => i);
  lineChart(card(g, 'Peer dialogue', 'Mean understanding by grouping (no inclusive teaching).'), {
    x: sess, xLabel: 'session', yLabel: 'understanding', series: d.runs.filter((r) => r.inclusivity === 0).map((r, i) => ({ name: r.grouping, y: Array.from(r.mean), colorIndex: i })),
  });
  lineChart(card(g, 'Cultural capital and the achievement gap', 'Learners above vs below the median match of home and school culture.'), {
    x: sess, xLabel: 'session', yLabel: 'understanding',
    series: [{ name: 'alone: high capital', y: Array.from(none.high) }, { name: 'alone: low capital', y: Array.from(none.low), dash: true, colorIndex: 0 },
      { name: 'mixed + inclusive: high', y: Array.from(mixedInc.high), colorIndex: 2 }, { name: 'mixed + inclusive: low', y: Array.from(mixedInc.low), dash: true, colorIndex: 2 }],
  });
  table(card(g, 'Groupings', ''), ['grouping', 'inclusive', 'understanding', 'gap', 'spread'],
    d.runs.map((r) => [r.grouping, r.inclusivity ? 'yes' : 'no', fmt.pct(r.finalMean, 1), fmt.num(r.finalGap, 3), fmt.num(r.spread[r.spread.length - 1], 3)]));
  callout(page.content, 'In the classroom engine <i>scaffolding</i> targets each learner\'s ZPD, <i>peer dialogue</i> adds support for those below the class level and deeper encoding, ' +
    'and <i>inclusivity</i> removes the culture tax.');
});
