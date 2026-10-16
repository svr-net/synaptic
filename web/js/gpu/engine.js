// WebGPU host for the fused classroom kernels of the C++ library (syn::gpu).
//
// No numerics live here. The library supplies the WGSL sources (gpuKernels) and, for every
// batch, a plan with ready-made buffers, dispatch sizes and buffer sizes (gpuJobs). This class
// uploads the buffers, dispatches classroom-fused -> reduce-partials -> score-quantiles and
// reads back two arrays, which go back to the library (gpuAnalyse) for all post-processing.
// The kernels stay within the limits of WebGPU's compatibility mode (4 storage buffers per
// stage, 128 invocations per workgroup), so any adapter that offers compute shaders runs them.

// Adapter requests, in order. Some mobile drivers return no adapter for a high-performance
// request but do for a plain one; the last request asks for a compatibility-mode adapter
// (WebGPU on OpenGL ES), which is enough for these kernels.
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
  static async create(kernels) {
    const { adapter, compatibility } = await requestAdapter();
    const lim = adapter.limits;
    // The largest buffers the adapter allows (big batches); the defaults if that is refused.
    const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize } })
      .catch(() => adapter.requestDevice());
    const engine = new GpuEngine(adapter, device, compatibility);
    await engine.compile(kernels);
    return engine;
  }

  constructor(adapter, device, compatibility = false) {
    this.adapter = adapter;
    this.device = device;
    this.info = adapter.info || {};
    this.name = ([this.info.vendor, this.info.architecture].filter(Boolean).join(' ') || 'GPU') + (compatibility ? ', compatibility mode' : '');
    device.lost.then((info) => { this.lost = info; });
  }

  async compile(kernels) {
    const make = async (code, label) => {
      const module = this.device.createShaderModule({ code, label });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === 'error');
      if (errors.length) throw new Error(`${label}: ${errors.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join('; ')}`);
      return this.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' }, label });
    };
    [this.fused, this.reduce, this.quantiles] = await Promise.all([
      make(kernels.classroomFused, 'classroom-fused'),
      make(kernels.reducePartials, 'reduce-partials'),
      make(kernels.scoreQuantiles, 'score-quantiles'),
    ]);
  }

  upload(data, usage, label) {
    const buf = this.device.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST, label, mappedAtCreation: true });
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buf.unmap();
    return buf;
  }

  /** Runs one plan from gpuJobs. Resolves to { sums, quantiles, gpuMs }. */
  async run(plan) {
    const d = this.device;
    const { bytes, dispatch } = plan;
    if (bytes.scratch > d.limits.maxStorageBufferBindingSize)
      throw new Error(`learners × jobs too large for one storage buffer (${(bytes.scratch / 2 ** 20).toFixed(0)} MiB > ${(d.limits.maxStorageBufferBindingSize / 2 ** 20).toFixed(0)} MiB)`);
    if (dispatch.fused > d.limits.maxComputeWorkgroupsPerDimension || dispatch.jobs > d.limits.maxComputeWorkgroupsPerDimension)
      throw new Error('too many learners or jobs for one dispatch');

    const S = GPUBufferUsage.STORAGE;
    const scratch = (size, usage, label) => d.createBuffer({ size: Math.max(16, size), usage, label });
    const bufs = {
      header: this.upload(plan.header, GPUBufferUsage.UNIFORM, 'header'),
      constants: this.upload(plan.constants, S, 'constants'),
      seeds: this.upload(plan.seeds, S, 'seeds'),
      learners: scratch(bytes.scratch, S, 'learner-statistics'),
      partials: scratch(bytes.partials, S, 'partials'),
      sums: scratch(bytes.sums, S | GPUBufferUsage.COPY_SRC, 'sums'),
      quantiles: scratch(bytes.quantiles, S | GPUBufferUsage.COPY_SRC, 'quantiles'),
      readSums: scratch(bytes.sums, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, 'read-sums'),
      readQuantiles: scratch(bytes.quantiles, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, 'read-quantiles'),
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
    pass.setPipeline(this.fused);
    pass.setBindGroup(0, group(this.fused, [bufs.header, bufs.constants, bufs.seeds, bufs.learners, bufs.partials]));
    pass.dispatchWorkgroups(dispatch.fused, dispatch.jobs);
    pass.setPipeline(this.reduce);
    pass.setBindGroup(0, group(this.reduce, [bufs.header, bufs.partials, bufs.sums]));
    pass.dispatchWorkgroups(dispatch.reduce);
    if (dispatch.quantiles > 0) {
      pass.setPipeline(this.quantiles);
      pass.setBindGroup(0, group(this.quantiles, [bufs.header, bufs.learners, bufs.quantiles]));
      pass.dispatchWorkgroups(dispatch.quantiles, dispatch.jobs);
    }
    pass.end();
    enc.copyBufferToBuffer(bufs.sums, 0, bufs.readSums, 0, Math.max(16, bytes.sums));
    enc.copyBufferToBuffer(bufs.quantiles, 0, bufs.readQuantiles, 0, Math.max(16, bytes.quantiles));
    d.queue.submit([enc.finish()]);
    globalThis.__synGpuRuns = (globalThis.__synGpuRuns || 0) + 1; // fused pipelines submitted (checked by the e2e test)
    await Promise.all([bufs.readSums.mapAsync(GPUMapMode.READ), bufs.readQuantiles.mapAsync(GPUMapMode.READ)]);
    const gpuMs = performance.now() - t0;
    const oom = await d.popErrorScope();
    const validation = await d.popErrorScope();
    const out = validation || oom ? null : {
      sums: new Float32Array(bufs.readSums.getMappedRange().slice(0, bytes.sums)),
      quantiles: dispatch.quantiles > 0 ? new Float32Array(bufs.readQuantiles.getMappedRange().slice(0, bytes.quantiles)) : new Float32Array(0),
      gpuMs,
    };
    for (const b of Object.values(bufs)) b.destroy();
    if (!out) throw new Error((validation || oom).message);
    if (this.lost) throw new Error('GPU device lost: ' + this.lost.message);
    return out;
  }
}
