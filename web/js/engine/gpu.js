// WebGPU host for the fused kernels of the library (syn::kernel).
//
// No numerics live here. For a model the library supplies the WGSL sources (kernelSources) and,
// for every plan, ready-made buffers, dispatch sizes and buffer sizes (modelPlans). This class
// compiles each model's pipelines once, uploads the buffers, dispatches
//   agent kernel (workgroups × jobs) -> reduce -> quantiles (if the plan keeps values)
// and reads back the per-job means and quantiles, which go back to the library (modelAnalyse).
// Every kernel stays within WebGPU's compatibility mode (4 storage buffers per stage, at most
// 128 invocations per workgroup), so any adapter with compute shaders runs them.

const ADAPTER_OPTIONS = [
  { powerPreference: 'high-performance' },
  {},
  { powerPreference: 'low-power' },
  { featureLevel: 'compatibility', compatibilityMode: true },
];

/** Why WebGPU cannot start here, with what the user can check. */
export function noAdapterReason() {
  const android = /Android/.test(navigator.userAgent || '');
  return android
    ? 'no WebGPU adapter: Chrome has WebGPU blocked or unsupported on this device (it needs Android 12+ and a supported GPU; see chrome://gpu)'
    : 'no WebGPU adapter: WebGPU is blocked or unsupported for this GPU or driver (see chrome://gpu)';
}

async function requestAdapter() {
  if (!('gpu' in navigator)) throw new Error('WebGPU is not available in this browser');
  for (const options of ADAPTER_OPTIONS) {
    const adapter = await navigator.gpu.requestAdapter(options).catch(() => null);
    if (adapter) return { adapter, compatibility: 'featureLevel' in options };
  }
  throw new Error(noAdapterReason());
}

/** Every adapter request with its answer, for the diagnostics panel. */
export async function probeAdapters() {
  const report = { secureContext: globalThis.isSecureContext, api: 'gpu' in navigator, userAgent: navigator.userAgent, adapters: [] };
  if (!report.api) return report;
  for (const options of ADAPTER_OPTIONS) {
    const entry = { request: JSON.stringify(options) };
    try {
      const a = await navigator.gpu.requestAdapter(options);
      if (a) {
        const info = a.info || {};
        entry.adapter = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(' · ') || 'adapter (no info)';
        entry.maxStorageBufferBindingSize = a.limits.maxStorageBufferBindingSize;
        entry.maxStorageBuffersPerShaderStage = a.limits.maxStorageBuffersPerShaderStage;
      } else entry.adapter = null;
    } catch (e) {
      entry.error = e.message;
    }
    report.adapters.push(entry);
  }
  return report;
}

export class GpuEngine {
  static async create() {
    const { adapter, compatibility } = await requestAdapter();
    const lim = adapter.limits;
    const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize } })
      .catch(() => adapter.requestDevice());
    return new GpuEngine(adapter, device, compatibility);
  }

  constructor(adapter, device, compatibility = false) {
    this.adapter = adapter;
    this.device = device;
    this.info = adapter.info || {};
    this.name = ([this.info.vendor, this.info.architecture].filter(Boolean).join(' ') || 'GPU') + (compatibility ? ', compatibility mode' : '');
    this.pipelines = new Map();
    device.lost.then((info) => { this.lost = info; });
  }

  async pipeline(code, label) {
    if (this.pipelines.has(label)) return this.pipelines.get(label);
    const make = (async () => {
      const module = this.device.createShaderModule({ code, label });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === 'error');
      if (errors.length) throw new Error(`${label}: ${errors.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join('; ')}`);
      return this.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' }, label });
    })();
    this.pipelines.set(label, make);
    make.catch(() => this.pipelines.delete(label));
    return make;
  }

  /** The three pipelines of a model (the reduce and quantile kernels are shared). */
  async compile(model, sources) {
    const [agent, reduce, quantile] = await Promise.all([
      this.pipeline(sources.agent, `agent:${model}`),
      this.pipeline(sources.reduce, 'reduce'),
      this.pipeline(sources.quantile, 'quantile'),
    ]);
    return { agent, reduce, quantile };
  }

  upload(data, usage, label) {
    const buf = this.device.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST, label, mappedAtCreation: true });
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buf.unmap();
    return buf;
  }

  /** Runs one plan. Resolves to { means, quantiles, gpuMs }. */
  async run(pipes, plan) {
    const d = this.device;
    const { bytes, dispatch } = plan;
    const maxBinding = d.limits.maxStorageBufferBindingSize;
    if (bytes.scratch > maxBinding || bytes.partials > maxBinding)
      throw new Error(`agents × jobs too large for one storage buffer (${((Math.max(bytes.scratch, bytes.partials)) / 2 ** 20).toFixed(0)} MiB)`);
    if (dispatch.agents > d.limits.maxComputeWorkgroupsPerDimension || dispatch.jobs > d.limits.maxComputeWorkgroupsPerDimension)
      throw new Error('too many agents or jobs for one dispatch');

    const S = GPUBufferUsage.STORAGE;
    const scratch = (size, usage, label) => d.createBuffer({ size: Math.max(16, size), usage, label });
    const bufs = {
      header: this.upload(plan.header, GPUBufferUsage.UNIFORM, 'header'),
      constants: this.upload(plan.constants, S, 'constants'),
      seeds: this.upload(plan.seeds, S, 'seeds'),
      scratch: scratch(bytes.scratch, S | GPUBufferUsage.COPY_SRC, 'agent-values'),
      partials: scratch(bytes.partials, S, 'partials'),
      sums: scratch(bytes.sums, S | GPUBufferUsage.COPY_SRC, 'sums'),
      quantiles: scratch(bytes.quantiles, S | GPUBufferUsage.COPY_SRC, 'quantiles'),
      readSums: scratch(bytes.sums, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, 'read-sums'),
      readQuantiles: scratch(bytes.quantiles, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, 'read-quantiles'),
      readKeep: bytes.keep > 0 ? scratch(bytes.keep, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, 'read-agent-values') : null,
    };
    const group = (pipeline, entries) => d.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });

    d.pushErrorScope('validation');
    d.pushErrorScope('out-of-memory');
    const t0 = performance.now();
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipes.agent);
    pass.setBindGroup(0, group(pipes.agent, [bufs.header, bufs.constants, bufs.seeds, bufs.scratch, bufs.partials]));
    pass.dispatchWorkgroups(dispatch.agents, dispatch.jobs);
    pass.setPipeline(pipes.reduce);
    pass.setBindGroup(0, group(pipes.reduce, [bufs.header, bufs.partials, bufs.sums]));
    pass.dispatchWorkgroups(dispatch.reduce);
    if (dispatch.quantiles > 0) {
      pass.setPipeline(pipes.quantile);
      pass.setBindGroup(0, group(pipes.quantile, [bufs.header, bufs.scratch, bufs.quantiles]));
      pass.dispatchWorkgroups(dispatch.quantiles, dispatch.jobs);
    }
    pass.end();
    enc.copyBufferToBuffer(bufs.sums, 0, bufs.readSums, 0, Math.max(16, bytes.sums));
    enc.copyBufferToBuffer(bufs.quantiles, 0, bufs.readQuantiles, 0, Math.max(16, bytes.quantiles));
    if (bufs.readKeep) enc.copyBufferToBuffer(bufs.scratch, 0, bufs.readKeep, 0, bytes.keep);
    d.queue.submit([enc.finish()]);
    globalThis.__synGpuRuns = (globalThis.__synGpuRuns || 0) + 1; // dispatches submitted (checked by the e2e test)
    await Promise.all([bufs.readSums.mapAsync(GPUMapMode.READ), bufs.readQuantiles.mapAsync(GPUMapMode.READ),
      bufs.readKeep ? bufs.readKeep.mapAsync(GPUMapMode.READ) : null]);
    const gpuMs = performance.now() - t0;
    const oom = await d.popErrorScope();
    const validation = await d.popErrorScope();
    const out = validation || oom ? null : {
      means: new Float32Array(bufs.readSums.getMappedRange().slice(0, bytes.sums)),
      quantiles: dispatch.quantiles > 0 ? new Float32Array(bufs.readQuantiles.getMappedRange().slice(0, bytes.quantiles)) : new Float32Array(0),
      scratch: bufs.readKeep ? new Float32Array(bufs.readKeep.getMappedRange().slice(0, bytes.keep)) : new Float32Array(0),
      gpuMs,
    };
    for (const b of Object.values(bufs)) if (b) b.destroy();
    if (!out) throw new Error((validation || oom).message);
    if (this.lost) throw new Error('GPU device lost: ' + this.lost.message);
    return out;
  }
}
