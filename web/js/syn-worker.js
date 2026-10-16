// Module worker hosting the WebAssembly build of the library, so long Monte Carlo
// runs never block the page. Messages: { id, fn, spec } -> { id, result, ms } | { id, error }.
import createSynModule from '../wasm/syn.js';

// The standalone build embeds the .wasm bytes and passes them here (no fetch, works from file://).
const ready = createSynModule(self.__SYN_MODULE_OPTIONS__ || {});

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
