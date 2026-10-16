// Default specification shared by all pages: the learner population, the course, the
// instructional design (one lever per teaching practice) and the simulation settings.
// Pages edit a copy through the spec editor; the current spec is kept in localStorage so
// every page works on the same configuration.

/** Lever keys in library order (syn::Lever), with the theory each comes from. */
export const LEVERS = [
  { key: 'reinforcement', name: 'reinforcement', theory: 'behaviourism' },
  { key: 'teacherDirection', name: 'teacher direction', theory: 'behaviourism / cognitivism' },
  { key: 'empathy', name: 'empathy', theory: 'humanism' },
  { key: 'choice', name: 'choice', theory: 'humanism / andragogy' },
  { key: 'relevance', name: 'relevance', theory: 'andragogy / motivation' },
  { key: 'transparency', name: 'transparency', theory: 'Gagné / andragogy' },
  { key: 'priorActivation', name: 'prior activation', theory: 'constructivism / Gagné' },
  { key: 'retrievalPractice', name: 'retrieval practice', theory: 'cognitivism' },
  { key: 'spacing', name: 'spacing', theory: 'cognitivism' },
  { key: 'interleaving', name: 'interleaving', theory: 'cognitivism' },
  { key: 'feedback', name: 'feedback', theory: 'Gagné / motivation' },
  { key: 'problemFirst', name: 'problem first', theory: 'constructivism' },
  { key: 'scaffolding', name: 'scaffolding (ZPD)', theory: 'Vygotsky (ZPD)' },
  { key: 'peerDialogue', name: 'peer dialogue', theory: 'social constructivism' },
  { key: 'inclusivity', name: 'inclusivity', theory: 'Bourdieu / critical practice' },
  { key: 'growthFeedback', name: 'growth feedback', theory: 'growth mindset' },
];

const design = (values) => Object.fromEntries(LEVERS.map((l, i) => [l.key, values[i]]));

// Mirror of syn::theoryDesign (checked against the library by web/tests/wasm.test.mjs).
//                                   rein dir  emp  cho  rel  tra  pri  ret  spa  int  fb   pf   sca  dia  inc  gro
export const THEORY_PRESETS = {
  'banking lecture': design([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
  behaviourism: design([1, 1, 0, 0, 0, 0, 0, 0.3, 0, 0, 0.6, 0, 0, 0, 0, 0]),
  humanism: design([0, 0.2, 1, 0.8, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.3]),
  cognitivism: design([0, 0.8, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0.4, 0.3, 0, 0, 0]),
  constructivism: design([0, 0.2, 0.6, 0.4, 0.6, 0, 1, 0, 0, 0, 0, 1, 0.4, 0, 0, 0]),
  'social constructivism': design([0, 0.3, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.4, 1, 1, 1, 0]),
  andragogy: design([0, 0.2, 0.5, 1, 1, 1, 0.8, 0, 0, 0, 0, 0.5, 0, 0, 0, 0]),
  blended: design([0.3, 0.5, 0.8, 0.5, 0.8, 1, 0.8, 0.9, 0.9, 0.7, 1, 0.5, 0.8, 0.7, 0.8, 0.8]),
};

export const POPULATIONS = {
  children: { preset: 'children', label: 'primary school (9)', age: 9, anxietyMean: 0.3, growthMean: 0.6, efficacyMean: 0.6, intrinsicMean: 0.4, capitalMean: 0.55 },
  secondary: { preset: 'secondary', label: 'secondary school (16)', age: 16, anxietyMean: 0.4, growthMean: 0.5, efficacyMean: 0.55, intrinsicMean: 0.35, capitalMean: 0.55 },
  adults: { preset: 'adults', label: 'adult learners (35)', age: 35, anxietyMean: 0.45, growthMean: 0.45, efficacyMean: 0.5, intrinsicMean: 0.55, capitalMean: 0.55 },
};

export function defaultSpec() {
  return {
    population: { ...POPULATIONS.secondary, priorMean: 0, priorSd: 1, capitalSd: 0.25 },
    // The chapter's Activity 3.3: a short course on keywords and Boolean operators.
    course: { sessions: 8, sessionGap: 7, difficulty: 0.3, difficultyStep: 0.1, delayedTest: 30, abstraction: 0.6, atRiskBelow: 0.3 },
    designPreset: 'blended',
    design: { ...THEORY_PRESETS.blended },
    sim: { learners: 500, seed: 2024 },
  };
}

const STORAGE_KEY = 'syn-spec-v1';

export function loadSpec() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const stored = JSON.parse(raw);
      const d = defaultSpec();
      return {
        ...d, ...stored,
        population: { ...d.population, ...(stored.population || {}) },
        course: { ...d.course, ...(stored.course || {}) },
        design: { ...d.design, ...(stored.design || {}) },
        sim: { ...d.sim, ...(stored.sim || {}) },
      };
    }
  } catch (_) { /* storage unavailable: fall back to defaults */ }
  return defaultSpec();
}

export function saveSpec(spec) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(spec)); } catch (_) { /* ignore */ }
}

export function resetSpec() {
  try { localStorage.removeItem(STORAGE_KEY); } catch (_) { /* ignore */ }
  return defaultSpec();
}
