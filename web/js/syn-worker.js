// Module worker hosting the WebAssembly build of the library, so long Monte Carlo
// runs never block the page. Messages: { id, fn, spec } -> { id, result, ms } | { id, error }.
import createSynModule from '../wasm/syn.js';

// The .wasm bytes: embedded by the standalone build (no fetch, works from file://), else fetched.
// Each worker appends a custom section with its own name, so that the engine compiles a module of
// its own for it: V8 shares one compiled module between workers that load identical bytes, and
// the software GPU's hot functions were then optimised for only some of the workers.
function withNameSection(bytes, name) {
  const leb = (n) => { const out = []; do { let b = n & 0x7f; n >>>= 7; if (n) b |= 0x80; out.push(b); } while (n); return out; };
  const id = new TextEncoder().encode('syn.worker'), payload = new TextEncoder().encode(name);
  const body = [...leb(id.length), ...id, ...payload];
  const out = new Uint8Array(bytes.length + 1 + leb(body.length).length + body.length);
  out.set(bytes);
  out.set([0, ...leb(body.length), ...body], bytes.length);
  return out;
}

async function moduleOptions() {
  const base = self.__SYN_MODULE_OPTIONS__ || {};
  const bytes = base.wasmBinary || new Uint8Array(await (await fetch(new URL('../wasm/syn.wasm', import.meta.url))).arrayBuffer());
  return { ...base, wasmBinary: withNameSection(bytes, self.name || 'syn') };
}

const ready = moduleOptions().then(createSynModule);

self.onmessage = async (event) => {
  const { id, fn, spec } = event.data;
  try {
    const syn = await ready;
    if (typeof syn[fn] !== 'function') throw new Error(`unknown function ${fn}`);
    const t0 = performance.now();
    const result = syn[fn](spec);
    const ms = performance.now() - t0;
    if (result && result.error) self.postMessage({ id, error: result.error });
    else self.postMessage({ id, result, ms });
  } catch (e) {
    self.postMessage({ id, error: String(e && e.message ? e.message : e) });
  }
};
