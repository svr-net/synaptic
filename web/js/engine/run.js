// Runs a model of the library: its fused kernels on WebGPU, or — when the browser has no WebGPU
// adapter — the same WGSL on the library's software GPU, split across Web Workers. The library
// (C++/WebAssembly) compiles the plans and summarises the read-backs; this file only routes.
import { callOn, run, workerCount } from '../syn-client.js';
import { GpuEngine } from './gpu.js';

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

/** Every plan of a model on WebGPU, then the library's summary. */
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
  return { ...result, engineMs: gpuMs, adapter: gpu.name };
}

/** The same kernels on the software GPU: every worker runs its share of the workgroups. */
export async function runOnSoftwareGpu(model, spec) {
  const n = workerCount();
  const t0 = performance.now();
  const parts = await Promise.all(Array.from({ length: n }, (_, part) =>
    callOn(part, 'simulateSlices', { ...spec, model, part, parts: n }).then((r) => { if (r.result.error) throw new Error(r.result.error); return r.result; })));
  globalThis.__synSoftwareGpuRuns = (globalThis.__synSoftwareGpuRuns || 0) + 1; // checked by the e2e test
  const result = await checked(run('analyseSlices', { ...spec, model, parts }));
  return { ...result, engineMs: performance.now() - t0, adapter: `${n} worker${n === 1 ? '' : 's'}` };
}

/** Runs a model: WebGPU if there is an adapter, else the software GPU. Resolves to { result, label, reason }. */
export async function runModel(model, spec) {
  const soft = async (reason) => ({ result: await runOnSoftwareGpu(model, spec), label: 'software GPU', reason });
  if (!('gpu' in navigator)) return soft('WebGPU is not available in this browser');
  try {
    return { result: await runOnGpu(model, spec), label: 'WebGPU', reason: 'WebGPU adapter' };
  } catch (e) {
    if (/no WebGPU adapter/.test(e.message)) return soft(`WebGPU unavailable: ${e.message.replace('no WebGPU adapter: ', '')}`);
    console.warn('WebGPU run failed, using the software GPU:', e);
    return soft(`WebGPU failed (${e.message})`);
  }
}

/** One-line note for the status bar: where the kernels ran and why. */
export function engineNote(r, ms) {
  return `Done in ${Math.round(ms).toLocaleString('en-US')} ms · ${r.label} (${r.result.adapter}) · ${r.reason}`;
}

/** Runs several models in turn. Resolves to { results, outcomes, note }. */
export async function runModels(calls, page) {
  const t0 = performance.now();
  const outcomes = [];
  for (const [model, spec] of calls) outcomes.push(await runModel(model, spec));
  // Where the kernels ran, for the end-to-end tests.
  globalThis.__synEngineRun = { page: page && page.id, engines: outcomes.map((o) => o.label), models: calls.map((c) => c[0]), reasons: outcomes.map((o) => o.reason) };
  return { results: outcomes.map((o) => o.result), outcomes, note: engineNote(outcomes[0], performance.now() - t0) };
}
