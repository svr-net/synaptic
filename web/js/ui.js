// Shared page scaffolding: navigation, environment status, cards, tiles, tables,
// run buttons and the specification editor used by every page.
import { LEVERS, POPULATIONS, THEORY_PRESETS, loadSpec, resetSpec, saveSpec } from './spec.js';
import { fmt } from './charts.js';

export const PAGES = [
  { group: 'Overview', items: [{ id: 'index', href: 'index.html', title: 'Overview' }] },
  { group: 'Foundations', items: [{ id: 'core', href: 'core.html', title: 'The space' }] },
  {
    group: 'Learning theories',
    items: [
      { id: 'behaviorism', href: 'behaviorism.html', title: 'Behaviourism' },
      { id: 'humanism', href: 'humanism.html', title: 'Humanism' },
      { id: 'cognitivism', href: 'cognitivism.html', title: 'Cognitivism' },
      { id: 'constructivism', href: 'constructivism.html', title: 'Constructivism' },
      { id: 'social', href: 'social.html', title: 'Social constructivism' },
    ],
  },
  {
    group: 'Learners',
    items: [
      { id: 'development', href: 'development.html', title: 'Developmental stages' },
      { id: 'andragogy', href: 'andragogy.html', title: 'Andragogy' },
      { id: 'motivation', href: 'motivation.html', title: 'Motivation & mindset' },
      { id: 'audience', href: 'audience.html', title: 'Identifying audience needs' },
      { id: 'outcomes', href: 'outcomes.html', title: 'Learning goals and outcomes' },
      { id: 'assessment', href: 'assessment.html', title: 'Assessing learning' },
    ],
  },
  {
    group: 'Classroom',
    items: [
      { id: 'classroom', href: 'classroom.html', title: 'Classroom simulation' },
      { id: 'theories', href: 'theories.html', title: 'Theories compared' },
      { id: 'allocation', href: 'allocation.html', title: 'Gain allocation' },
      { id: 'sensitivity', href: 'sensitivity.html', title: 'Lever sensitivities' },
    ],
  },
  { group: 'Acceleration', items: [{ id: 'gpu', href: 'gpu.html', title: 'WebGPU fused kernels' }] },
];

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}

/** Builds the page shell. Returns { main, toolbar, status, content, spec }. */
export function initPage({ id, title, context, description }) {
  document.title = `${title} · synaptic`;
  // On narrow screens the links collapse behind a Menu button so the page content shows first.
  const links = el('div', { class: 'nav-links', id: 'nav-links' });
  const toggle = el('button', { class: 'menu-toggle', type: 'button', 'aria-expanded': 'false', 'aria-controls': 'nav-links', text: '☰ Menu' });
  const nav = el('nav', { class: 'sidebar' },
    el('div', { class: 'nav-head' },
      el('div', {}, el('div', { class: 'brand', text: 'synaptic' }),
        el('div', { class: 'sub', text: 'How people learn: learning theories as models, in WebAssembly' })),
      toggle),
    links);
  toggle.addEventListener('click', () => {
    const open = nav.classList.toggle('open');
    toggle.setAttribute('aria-expanded', String(open));
    toggle.textContent = open ? '✕ Close' : '☰ Menu';
  });
  for (const g of PAGES) {
    links.append(el('div', { class: 'group', text: g.group }));
    for (const p of g.items) links.append(el('a', { href: p.href, class: p.id === id ? 'active' : '', text: p.title }));
  }
  const wasmDot = el('span', { class: 'dot' });
  const gpuDot = el('span', { class: 'dot' });
  const wasmText = el('span', { text: 'WASM: loading…' });
  const gpuText = el('span', { text: 'WebGPU: checking…' });
  links.append(el('div', { class: 'env' }, el('div', {}, wasmDot, wasmText), el('div', {}, gpuDot, gpuText)));

  const status = el('span', { class: 'status', id: 'run-status' });
  const toolbar = el('div', { class: 'toolbar' });
  const content = el('div');
  const main = el('main', {},
    el('header', {}, el('div', { class: 'context', text: context }), el('h1', { text: title }), description ? el('p', { html: description }) : null),
    toolbar, content);
  toolbar.append(status);
  const app = document.getElementById('app');
  app.className = 'layout';
  app.append(nav, main);

  import('./syn-client.js').then(({ run }) => run('version', {}))
    .then((v) => { wasmDot.className = 'dot ok'; wasmText.textContent = `WASM: ${v.library}`; })
    .catch((e) => { wasmDot.className = 'dot bad'; wasmText.textContent = 'WASM: ' + e.message; });
  gpuStatus().then((s) => { gpuDot.className = 'dot ' + (s.ok ? 'ok' : 'bad'); gpuText.textContent = 'WebGPU: ' + s.text; });

  return { id, main, toolbar, status, content, spec: loadSpec() };
}

export async function gpuStatus() {
  if (!('gpu' in navigator)) return { ok: false, text: 'not available' };
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { ok: false, text: 'no adapter (CPU kernels)' };
    const info = adapter.info || {};
    return { ok: true, text: [info.vendor, info.architecture].filter(Boolean).join(' ') || 'available' };
  } catch (e) {
    return { ok: false, text: e.message };
  }
}

export function card(parent, title, note) {
  const body = el('div');
  const c = el('section', { class: 'card' }, el('h2', { text: title }), note ? el('p', { class: 'note', html: note }) : null, body);
  parent.append(c);
  return body;
}

export function grid(parent, wide = false) {
  const g = el('div', { class: wide ? 'grid wide' : 'grid' });
  parent.append(g);
  return g;
}

export function tiles(parent, items) {
  const box = el('div', { class: 'tiles' });
  for (const it of items)
    box.append(el('div', { class: 'tile' },
      el('div', { class: 'label', text: it.label }), el('div', { class: 'value', text: it.value }),
      it.hint ? el('div', { class: 'hint', text: it.hint }) : null));
  parent.append(box);
  return box;
}

export function table(parent, headers, rows) {
  const t = el('table', { class: 'data' }, el('tr', {}, headers.map((h) => el('th', { text: h }))));
  for (const r of rows) t.append(el('tr', {}, r.map((c) => (c instanceof Node ? el('td', {}, c) : el('td', { text: c })))));
  const wrap = el('div', { class: 'table-wrap' }, t);
  parent.append(wrap);
  return wrap;
}

export const passFail = (ok, text) => el('span', { class: ok ? 'pass' : 'fail', text: (ok ? '✓ ' : '✗ ') + (text ?? (ok ? 'pass' : 'fail')) });

/** A highlighted note above the results. */
export function callout(parent, html) {
  parent.append(el('p', { class: 'note callout', html }));
}

/** Adds a primary run button. handler(spec) may be async; status shows timing or errors. */
export function runButton(page, label, handler, { auto = true } = {}) {
  const btn = el('button', { class: 'primary', text: label });
  page.toolbar.insertBefore(btn, page.status);
  const go = async () => {
    btn.disabled = true;
    page.status.className = 'status';
    page.status.textContent = 'Running…';
    const t0 = performance.now();
    try {
      page.content.innerHTML = '';
      const msg = await handler(page.spec);
      page.status.textContent = msg || `Done in ${fmt.num(performance.now() - t0)} ms`;
    } catch (e) {
      console.error(e);
      page.status.className = 'status error';
      page.status.textContent = 'Error: ' + e.message;
    } finally {
      btn.disabled = false;
    }
  };
  btn.addEventListener('click', go);
  if (auto) setTimeout(go, 0);
  return btn;
}

/** A small labelled number input in the toolbar; returns a getter. */
export function toolbarNumber(page, label, value, { step = 'any', width = 70 } = {}) {
  const input = el('input', { type: 'number', step, value, style: `width:${width}px` });
  page.toolbar.insertBefore(el('label', { class: 'inline-select' }, label, input), page.status);
  return () => Number(input.value);
}

/** A labelled select in the toolbar; returns a getter. */
export function toolbarSelect(page, label, options, value, onChange) {
  const s = el('select', {}, options.map((o) => el('option', { value: o.value ?? o, text: o.label ?? o, selected: (o.value ?? o) === value })));
  if (onChange) s.addEventListener('change', () => onChange(s.value));
  page.toolbar.insertBefore(el('label', { class: 'inline-select' }, label, s), page.status);
  return () => s.value;
}

// ------------------------------------------------------------------ spec editor

function numInput(get, set, { scale = 1, step = 'any', digits = 6 } = {}) {
  const v = get();
  const input = el('input', { type: 'number', step, value: v === undefined || v === null ? '' : +(v * scale).toFixed(digits) });
  input.addEventListener('change', () => set(input.value === '' ? null : Number(input.value) / scale));
  return input;
}

function selectInput(options, get, set) {
  const s = el('select', {}, options.map((o) => el('option', { value: o.value ?? o, text: o.label ?? o, selected: (o.value ?? o) === get() })));
  s.addEventListener('change', () => set(s.value));
  return s;
}

function field(label, input) { return el('label', { class: 'field' }, el('span', { text: label }), input); }

/**
 * Collapsible editor for the shared specification.
 * sections: subset of ['population', 'course', 'design', 'simulation'].
 */
export function specEditor(page, sections, { open = false } = {}) {
  page.spec = loadSpec();
  const details = el('details', { class: 'spec', open });
  const body = el('div', { class: 'spec-sections' });
  details.append(el('summary', { text: 'Specification (shared by all pages)' }), body);
  page.main.insertBefore(details, page.toolbar);
  const changed = () => saveSpec(page.spec);
  const render = () => {
    body.innerHTML = '';
    const spec = page.spec;
    const sec = (title) => { const s = el('div', { class: 'spec-section' }, el('h3', { text: title })); body.append(s); return s; };
    const rerender = () => { changed(); render(); };

    if (sections.includes('population')) {
      const s = sec('Learners');
      const p = spec.population;
      s.append(field('population', selectInput(Object.values(POPULATIONS).map((x) => ({ value: x.preset, label: x.label })), () => p.preset,
        (v) => { spec.population = { ...p, ...POPULATIONS[v] }; rerender(); })));
      s.append(field('mean age', numInput(() => p.age, (v) => { p.age = v; changed(); })));
      s.append(field('prior knowledge (mean θ)', numInput(() => p.priorMean, (v) => { p.priorMean = v; changed(); })));
      s.append(field('anxiety (mean, 0–1)', numInput(() => p.anxietyMean, (v) => { p.anxietyMean = v; changed(); }, { step: 0.05 })));
      s.append(field('growth mindset (mean, 0–1)', numInput(() => p.growthMean, (v) => { p.growthMean = v; changed(); }, { step: 0.05 })));
      s.append(field('self-efficacy (mean, 0–1)', numInput(() => p.efficacyMean, (v) => { p.efficacyMean = v; changed(); }, { step: 0.05 })));
      s.append(field('intrinsic interest (mean, 0–1)', numInput(() => p.intrinsicMean, (v) => { p.intrinsicMean = v; changed(); }, { step: 0.05 })));
      s.append(field('cultural capital (mean, 0–1)', numInput(() => p.capitalMean, (v) => { p.capitalMean = v; changed(); }, { step: 0.05 })));
    }
    if (sections.includes('course')) {
      const s = sec('Course (Activity 3.3: Boolean operators)');
      const c = spec.course;
      s.append(field('sessions', numInput(() => c.sessions, (v) => { c.sessions = Math.max(1, Math.round(v)); changed(); }, { step: 1 })));
      s.append(field('days between sessions', numInput(() => c.sessionGap, (v) => { c.sessionGap = v; changed(); })));
      s.append(field('difficulty of unit 1 (θ)', numInput(() => c.difficulty, (v) => { c.difficulty = v; changed(); }, { step: 0.1 })));
      s.append(field('difficulty rise per unit', numInput(() => c.difficultyStep, (v) => { c.difficultyStep = v; changed(); }, { step: 0.05 })));
      s.append(field('delayed test (days after)', numInput(() => c.delayedTest, (v) => { c.delayedTest = v; changed(); })));
      s.append(field('abstraction (0–1)', numInput(() => c.abstraction, (v) => { c.abstraction = v; changed(); }, { step: 0.1 })));
      s.append(field('at risk below score', numInput(() => c.atRiskBelow, (v) => { c.atRiskBelow = v; changed(); }, { step: 0.05 })));
    }
    if (sections.includes('simulation')) {
      const s = sec('Simulation');
      s.append(field('learners', numInput(() => spec.sim.learners, (v) => { spec.sim.learners = Math.max(2, Math.round(v)); changed(); }, { step: 100 })));
      s.append(field('seed', numInput(() => spec.sim.seed, (v) => { spec.sim.seed = Math.round(v); changed(); }, { step: 1 })));
    }
    if (sections.includes('design')) {
      const s = sec('Instructional design: one lever per teaching practice');
      s.style.gridColumn = '1 / -1';
      s.append(field('start from the lesson of', selectInput(['custom', ...Object.keys(THEORY_PRESETS)], () => spec.designPreset || 'custom',
        (v) => { spec.designPreset = v; if (THEORY_PRESETS[v]) spec.design = { ...THEORY_PRESETS[v] }; rerender(); })));
      const box = el('div', { class: 'levers' });
      for (const l of LEVERS) {
        const out = el('output', { text: fmt.num(spec.design[l.key], 2) });
        const range = el('input', { type: 'range', min: 0, max: 1, step: 0.05, value: spec.design[l.key], 'aria-label': l.name });
        range.addEventListener('input', () => { out.textContent = fmt.num(Number(range.value), 2); });
        range.addEventListener('change', () => { spec.design[l.key] = Number(range.value); spec.designPreset = 'custom'; rerender(); });
        box.append(el('label', { class: 'lever' }, el('span', {}, l.name, el('div', { class: 'theory', text: l.theory })), range, out));
      }
      s.append(box);
    }
    body.append(el('div', { class: 'spec-actions', style: 'grid-column:1/-1' },
      el('button', { text: 'Reset to defaults', onclick: () => { page.spec = resetSpec(); render(); } }),
      el('span', { class: 'status', text: 'Changes are saved automatically and shared across pages. Press Run to recompute.' })));
  };
  render();
  return details;
}
