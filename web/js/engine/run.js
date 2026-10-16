// Runs any model of the library on one of three engines:
//   WebGPU        the model's fused kernel on the GPU (32-bit floats);
//   CPU kernels   the same kernel compiled for the CPU (32-bit floats), tiles split across workers;
//   WebAssembly   the same kernel in 64-bit floats in one worker.
// Auto tries them in that order. All three use the same random numbers and end in the same C++
// summary, so they agree to rounding. This file only routes calls.
import { callOn, run, workerCount } from '../syn-client.js';
import { el } from '../ui.js';
import { GpuEngine } from './gpu.js';

const STORAGE_KEY = 'syn-engine';
const MODES = { auto: 'Auto', gpu: 'WebGPU', cpu: 'CPU kernels', wasm: 'WebAssembly' };
const LABELS = { gpu: 'WebGPU', cpu: 'CPU kernels', wasm: 'WebAssembly' };

export function getEngineMode() {
  try { return MODES[localStorage.getItem(STORAGE_KEY)] ? localStorage.getItem(STORAGE_KEY) : 'auto'; } catch (_) { return 'auto'; }
}

export function setEngineMode(mode) {
  try { localStorage.setItem(STORAGE_KEY, mode); } catch (_) { /* storage unavailable */ }
}

let engine = null;
export function gpuEngine() {
  if (!engine) engine = GpuEngine.create().catch((e) => { engine = null; throw e; });
  return engine;
}

const sources = new Map();
function kernelSources(model) {
  if (!sources.has(model)) sources.set(model, run('kernelSources', { model }).then((s) => { if (s.error) throw new Error(s.error); return s; }));
  return sources.get(model);
}

async function checked(promise) {
  const r = await promise;
  if (r && r.error) throw new Error(r.error);
  return r;
}

/** Every plan of a model on the GPU, then the library's summary. */
export async function runOnGpu(model, spec) {
  const gpu = await gpuEngine();
  const [pipes, plans] = await Promise.all([kernelSources(model).then((s) => gpu.compile(model, s)), checked(run('modelPlans', { ...spec, model }))]);
  const outputs = [];
  let gpuMs = 0;
  for (const plan of plans) {
    const out = await gpu.run(pipes, plan);
    outputs.push({ means: out.means, quantiles: out.quantiles, scratch: out.scratch });
    gpuMs += out.gpuMs;
  }
  const result = await checked(run('modelAnalyse', { ...spec, model, outputs }));
  return { ...result, engineMs: gpuMs, adapter: gpu.name, plans: plans.length, jobs: plans.reduce((s, p) => s + p.jobs, 0) };
}

/** The kernel on the CPU: every worker runs its share of the tiles of every plan. */
export async function runOnCpuKernels(model, spec) {
  const n = workerCount();
  const t0 = performance.now();
  const parts = await Promise.all(Array.from({ length: n }, (_, part) =>
    callOn(part, 'modelSlices', { ...spec, model, part, parts: n }).then((r) => { if (r.result.error) throw new Error(r.result.error); return r.result; })));
  const engineMs = performance.now() - t0;
  globalThis.__synCpuKernelRuns = (globalThis.__synCpuKernelRuns || 0) + 1; // checked by the e2e test
  const result = await checked(run('modelAnalyseSlices', { ...spec, model, parts }));
  return { ...result, engineMs, adapter: `${n} worker${n === 1 ? '' : 's'}`, workers: n };
}

/** The kernel in 64-bit floats in one WebAssembly worker. */
export async function runOnWasm(model, spec) {
  const t0 = performance.now();
  const result = await checked(run('runModel', { ...spec, model }));
  return { ...result, engineMs: performance.now() - t0, adapter: 'WebAssembly, 64-bit' };
}

const outcome = (engine, result, reason) => ({ result, engine, label: LABELS[engine], reason });

/**
 * Runs a model on the engine chosen by the selector. Resolves to
 * { result, engine, label, reason }; falls back GPU → CPU kernels → WebAssembly.
 */
export async function runModel(model, spec, mode = getEngineMode()) {
  const wasm = async (reason) => outcome('wasm', await runOnWasm(model, spec), reason);
  const cpu = async (why) => {
    try {
      return outcome('cpu', await runOnCpuKernels(model, spec), why);
    } catch (e) {
      console.warn('CPU kernels failed, falling back to WebAssembly:', e);
      return wasm(`CPU kernels failed (${e.message}); used WebAssembly`);
    }
  };
  if (mode === 'wasm') return wasm('WebAssembly selected');
  if (mode === 'cpu') return cpu('CPU kernels selected');
  if (!('gpu' in navigator)) return cpu('WebGPU is not available in this browser');
  try {
    return outcome('gpu', await runOnGpu(model, spec), mode === 'auto' ? 'Auto: WebGPU available' : 'WebGPU selected');
  } catch (e) {
    console.warn('WebGPU run failed, falling back to the CPU kernels:', e);
    if (/no WebGPU adapter/.test(e.message)) return cpu(`WebGPU unavailable: ${e.message.replace('no WebGPU adapter: ', '')}`);
    return cpu(`WebGPU failed (${e.message})`);
  }
}

/** Toolbar control: Engine [Auto | WebGPU | CPU kernels | WebAssembly]. Changing it re-runs via `onChange`. */
export function engineSelector(page, onChange) {
  const select = el('select', { 'aria-label': 'Compute engine' },
    Object.entries(MODES).map(([value, label]) => el('option', {
      value, selected: value === getEngineMode(),
      text: value === 'auto' ? 'Auto (WebGPU, else CPU kernels)' : label,
    })));
  select.addEventListener('change', () => { setEngineMode(select.value); onChange(); });
  page.toolbar.insertBefore(el('label', { class: 'inline-select' }, 'Engine ', select), page.status);
  return select;
}

/** One-line note for the status bar: which engine ran and why. */
export function engineNote(r, ms) {
  return `Done in ${Math.round(ms).toLocaleString('en-US')} ms · ${r.label} (${r.result.adapter}) · ${r.reason}`;
}
