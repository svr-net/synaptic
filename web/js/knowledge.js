// What the reader keeps in this browser: the capsule segments it was taught and the records of what it
// studied, as the C++ models wrote them (base64), in IndexedDB. Only storage: the models read, check and
// fold them. Where IndexedDB is unavailable (a private window, blocked storage), they last the visit.
const DB = 'synaptic-reader', STORE = 'kept', KEY = 'knowledge';
let memoryOnly = { segments: [], records: [] };

function open() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function transact(mode, fn) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, mode);
    const req = fn(tx.objectStore(STORE));
    tx.oncomplete = () => { db.close(); resolve(req && req.result); };
    tx.onerror = () => { db.close(); reject(tx.error); };
  });
}

/** { segments: [base64…], records: [base64…] }, in the order they were kept. */
export async function loadKnowledge() {
  try {
    const v = await transact('readonly', (s) => s.get(KEY));
    if (v && Array.isArray(v.segments) && Array.isArray(v.records)) memoryOnly = v;
  } catch (_) { /* storage unavailable: this visit only */ }
  return { segments: [...memoryOnly.segments], records: [...memoryOnly.records] };
}

async function save(v) {
  memoryOnly = v;
  try { await transact('readwrite', (s) => s.put(v, KEY)); } catch (_) { /* this visit only */ }
}

/** Append a segment (capsuleIngest's `segment`) or a record (readingStudent's `remember.record`). */
export async function keep(kind, value) {
  const v = await loadKnowledge();
  v[kind].push(value);
  await save(v);
  return v;
}

export async function forgetKnowledge() {
  await save({ segments: [], records: [] });
}
