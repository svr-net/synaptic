// The specification shared by all pages: the learner population, the course, the instructional design
// (one lever per teaching practice) and the simulation settings, from the library's defaults. Pages
// edit a copy through the spec editor; the current spec is kept in localStorage so every page works
// on the same configuration.

import DEFAULTS from '../wasm/defaults.js';

// What the pages start from, as the library defines it (syn::models::defaults, written by the WASM build).
/** The levers of a design in library order: { key, name, theory }. */
export const LEVERS = DEFAULTS.levers;
/** The lesson each theory designs, by theory, in the library's order. */
export const THEORY_PRESETS = Object.fromEntries(DEFAULTS.theories.map((t) => [t.name, t.design]));
/** The learner populations, by preset. */
export const POPULATIONS = Object.fromEntries(DEFAULTS.populations.map((p) => [p.preset, p]));

export function defaultSpec() {
  return structuredClone(DEFAULTS.spec);
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
