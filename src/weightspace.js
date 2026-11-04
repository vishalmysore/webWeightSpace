// Weight-space programming for WebLLM models.
//
// How it works:
//  1. WebLLM keeps model files in the browser Cache Storage ("webllm/model", "webllm/config"),
//     keyed by URL. We download the original MLC weight shards into those same caches.
//  2. A "weight program" (a few lines of text) is compiled into per-tensor op chains.
//  3. Each shard holding an edited tensor is read from the cache, the ops run as WebGPU compute
//     shaders, and the edited shard is written under a NEW synthetic model URL. Untouched shards
//     are not copied: the new manifest points at the original files.
//  4. WebLLM is pointed at that URL. Every file it needs is already in the cache, so it loads
//     the edited model without touching the network. Same compiled model library, new weights.

import { MODE } from "./kernels.js";

export const SCOPE_MODEL = "webllm/model";
export const SCOPE_CONFIG = "webllm/config";
const PATCH_DIR = "wsp-models/";

export function modelBaseUrl(rec) {
  let u = rec.model;
  if (!u.endsWith("/")) u += "/";
  if (!/.+\/resolve\/.+\//.test(u)) u += "resolve/main/";
  return new URL(u, location.href).href;
}

const short = (name) => name.replace(/^model\./, "");

// ---------------------------------------------------------------- downloads

async function fetchIntoCache(cache, url, onBytes, signal) {
  if (await cache.match(url)) return false;
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`Download failed (${res.status}) for ${url}`);
  const counter = new TransformStream({
    transform(chunk, ctl) { onBytes?.(chunk.byteLength); ctl.enqueue(chunk); },
  });
  await cache.put(url, new Response(res.body.pipeThrough(counter)));
  return true;
}

/** Make sure every file WebLLM needs for this model is in Cache Storage. */
export async function ensureCached(rec, onProgress) {
  const base = modelBaseUrl(rec);
  const mcache = await caches.open(SCOPE_MODEL);
  const ccache = await caches.open(SCOPE_CONFIG);
  await fetchIntoCache(ccache, base + "mlc-chat-config.json");
  await fetchIntoCache(mcache, base + "tensor-cache.json");
  await fetchIntoCache(mcache, base + "tokenizer.json");
  const manifest = await (await mcache.match(base + "tensor-cache.json")).json();
  const total = manifest.records.reduce((s, r) => s + r.nbytes, 0);
  let done = 0;
  for (const shard of manifest.records) {
    const url = new URL(shard.dataPath, base).href;
    let got = 0;
    const fresh = await fetchIntoCache(mcache, url, (n) => { got += n; onProgress?.((done + got) / total); });
    done += shard.nbytes;
    onProgress?.(done / total, fresh);
  }
  return manifest;
}

export async function isCached(rec) {
  const base = modelBaseUrl(rec);
  const mcache = await caches.open(SCOPE_MODEL);
  const m = await mcache.match(base + "tensor-cache.json");
  if (!m) return false;
  const manifest = await m.json();
  for (const s of manifest.records) if (!(await mcache.match(new URL(s.dataPath, base).href))) return false;
  return true;
}

export async function loadModelInfo(rec) {
  const base = modelBaseUrl(rec);
  const mcache = await caches.open(SCOPE_MODEL);
  const ccache = await caches.open(SCOPE_CONFIG);
  const manifest = await (await mcache.match(base + "tensor-cache.json")).json();
  const config = await (await ccache.match(base + "mlc-chat-config.json")).json();
  const tensors = new Map();
  manifest.records.forEach((shard, si) =>
    shard.records.forEach((r) => tensors.set(r.name, { ...r, shard: si })));
  return { rec, base, manifest, config, tensors, mcache, ccache };
}

const encOf = (t) => (t.format === "f32-to-bf16" && t.dtype === "float32" ? 1 : t.dtype === "float16" ? 0 : -1);

// ---------------------------------------------------------------- program language
//
//   <selector> <op> [args]          one rule per line, '#' starts a comment
//
// selectors   *                       every tensor
//             embed                   token embedding (tied to the output head in these models)
//             layers.10-13            all tensors of layers 10..13
//             layers.*.mlp            every MLP;  layers.*.attn.o_proj  attention output projections
//             layers.0-23.*norm       layer norms (glob inside a segment)
// ops         mix a                   w += a * (donor - base)      model merging / task arithmetic
//             scale k                 w *= k
//             zero                    w = 0
//             noise s                 w *= 1 + s * N(0,1)
//             skip                    zero o_proj + down_proj  -> layer becomes an identity
//             zero-heads a-b          zero attention heads a..b (o_proj input columns)
//             inject word[+word] c    concept injection: solve value-bias deltas so each selected
//                                     layer adds c * unit(E[word]) to the residual stream

const OPS = {
  mix: 1, scale: 1, zero: 0, noise: 1, skip: 0, "zero-heads": 1, inject: 2,
};

function selectorToMatcher(sel) {
  sel = sel.trim();
  if (sel === "*" || sel === "all") return () => true;
  const segs = sel.replace(/^model\./, "").split(".").map((s) => (s === "attn" ? "self_attn" : s === "embed" ? "embed_tokens" : s));
  const ranges = [];
  let rx = "^";
  segs.forEach((s, i) => {
    if (i) rx += "\\.";
    const m = s.match(/^(\d+)-(\d+)$/);
    if (m) { rx += "(\\d+)"; ranges.push([+m[1], +m[2]]); return; }
    if (s === "*") { rx += "[^.]+"; return; }
    rx += s.replace(/[.+?^${}()|[\]\\]/g, "\\$&").replace(/\*/g, "[^.]*");
  });
  rx += "(\\..+)?$";
  const re = new RegExp(rx);
  return (name) => {
    const m = re.exec(name);
    if (!m) return false;
    for (let k = 0; k < ranges.length; k++) {
      const v = +m[k + 1];
      if (v < ranges[k][0] || v > ranges[k][1]) return false;
    }
    return true;
  };
}

export function parseProgram(text) {
  const rules = [], errors = [];
  text.split("\n").forEach((raw, idx) => {
    const line = raw.replace(/#.*/, "").trim();
    if (!line) return;
    const parts = line.split(/\s+/);
    const [sel, op, ...args] = parts;
    if (!op || !(op in OPS)) { errors.push(`line ${idx + 1}: unknown op "${op ?? ""}"`); return; }
    if (args.length < OPS[op]) { errors.push(`line ${idx + 1}: "${op}" needs ${OPS[op]} argument(s)`); return; }
    rules.push({ line: idx + 1, sel, op, args, match: selectorToMatcher(sel) });
  });
  return { rules, errors };
}

/** Turn rules into per-tensor op chains for a given model (and optional donor). */
export function compile(text, info, donorInfo) {
  const { rules, errors } = parseProgram(text);
  const plan = new Map();      // full tensor name -> [{mode,...}]
  const injects = [];          // {layers:[{layer, biasName, oName}], words, c}
  const mc = info.config.model_config;
  const nH = mc.num_attention_heads, nKV = mc.num_key_value_heads ?? nH;
  const hd = mc.head_dim ?? mc.hidden_size / nH;
  const push = (name, op) => { if (!plan.has(name)) plan.set(name, []); plan.get(name).push(op); };

  for (const r of rules) {
    const hits = [...info.tensors.keys()].filter((n) => r.match(short(n)));
    if (!hits.length) { errors.push(`line ${r.line}: selector "${r.sel}" matches no tensor`); continue; }
    const num = (i) => {
      const v = parseFloat(r.args[i]);
      if (!Number.isFinite(v)) errors.push(`line ${r.line}: "${r.args[i]}" is not a number`);
      return v;
    };
    const seedOf = (n) => [...n].reduce((h, ch) => Math.imul(h ^ ch.charCodeAt(0), 16777619), 2166136261) >>> 0;
    switch (r.op) {
      case "mix": {
        if (!donorInfo) { errors.push(`line ${r.line}: "mix" needs a donor model (pick a model pair)`); break; }
        const a = num(0);
        for (const n of hits) {
          const d = donorInfo.tensors.get(n), t = info.tensors.get(n);
          if (!d || d.shape.join() !== t.shape.join() || encOf(d) !== encOf(t)) { errors.push(`line ${r.line}: donor has no matching ${n}`); continue; }
          push(n, { mode: MODE.mix, a, donor: true });
        }
        break;
      }
      case "scale": { const a = num(0); hits.forEach((n) => push(n, { mode: MODE.scale, a })); break; }
      case "zero": hits.forEach((n) => push(n, { mode: MODE.zero })); break;
      case "noise": { const a = num(0); hits.forEach((n) => push(n, { mode: MODE.noise, a, seed: seedOf(n) ^ (parseInt(r.args[1] ?? "7") >>> 0) })); break; }
      case "skip": {
        const t = hits.filter((n) => /(o_proj|down_proj)\.weight$/.test(n));
        if (!t.length) errors.push(`line ${r.line}: "skip" found no o_proj/down_proj in "${r.sel}"`);
        t.forEach((n) => push(n, { mode: MODE.zero }));
        break;
      }
      case "zero-heads": {
        const m = r.args[0].match(/^(\d+)(?:-(\d+))?$/);
        if (!m) { errors.push(`line ${r.line}: heads must look like 3 or 0-6`); break; }
        const a = +m[1], b = m[2] ? +m[2] : a;
        if (b >= nH) { errors.push(`line ${r.line}: model has ${nH} heads (0-${nH - 1})`); break; }
        const t = hits.filter((n) => /o_proj\.weight$/.test(n));
        if (!t.length) errors.push(`line ${r.line}: "zero-heads" found no o_proj in "${r.sel}"`);
        t.forEach((n) => push(n, { mode: MODE.zerocols, c0: a * hd, c1: (b + 1) * hd }));
        break;
      }
      case "inject": {
        const words = r.args[0].split("+").filter(Boolean);
        const c = num(1);
        const layers = [];
        for (const n of hits) {
          const m = n.match(/layers\.(\d+)\.self_attn\.(c_attn|qkv_proj)\.bias$/);
          if (!m) continue;
          const oName = n.replace(/(c_attn|qkv_proj)\.bias$/, "o_proj.weight");
          if (info.tensors.has(oName)) layers.push({ layer: +m[1], biasName: n, oName });
        }
        if (!layers.length) { errors.push(`line ${r.line}: "inject" needs attention value biases (Qwen2-family models); none in "${r.sel}"`); break; }
        const id = injects.length;
        injects.push({ line: r.line, layers, words, c, nH, nKV, hd });
        layers.forEach((L) => push(L.biasName, { mode: MODE.addvec, a: 1, inject: id, layer: L.layer }));
        break;
      }
    }
  }
  let touched = 0, total = 0;
  for (const [n, t] of info.tensors) {
    const sz = t.shape.reduce((a, b) => a * b, 1);
    total += sz;
    if (plan.has(n)) touched += sz;
  }
  return { plan, injects, errors, stats: { tensors: plan.size, touched, total } };
}

// ---------------------------------------------------------------- tensor reading helpers

class ShardReader {
  constructor(info, keep = 2) { this.info = info; this.keep = keep; this.lru = new Map(); }
  async shard(i) {
    if (this.lru.has(i)) { const v = this.lru.get(i); this.lru.delete(i); this.lru.set(i, v); return v; }
    const url = new URL(this.info.manifest.records[i].dataPath, this.info.base).href;
    const res = await this.info.mcache.match(url);
    if (!res) throw new Error(`Missing shard in cache: ${url}`);
    const bytes = new Uint8Array(await res.arrayBuffer());
    this.lru.set(i, bytes);
    while (this.lru.size > this.keep) this.lru.delete(this.lru.keys().next().value);
    return bytes;
  }
  async bytesOf(name) {
    const t = this.info.tensors.get(name);
    const s = await this.shard(t.shard);
    return s.subarray(t.byteOffset, t.byteOffset + t.nbytes);
  }
}

const f16tab = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
    t[h] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return t;
})();
function decode(bytes, enc, start = 0, count = bytes.byteLength / 2) {
  const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset + start * 2, count);
  const out = new Float32Array(count);
  if (enc === 0) for (let i = 0; i < count; i++) out[i] = f16tab[u16[i]];
  else { const u32 = new Uint32Array(1), f = new Float32Array(u32.buffer); for (let i = 0; i < count; i++) { u32[0] = u16[i] << 16; out[i] = f[0]; } }
  return out;
}

export async function tokenIdOf(info, word) {
  if (!info._vocab) {
    const tj = await (await info.mcache.match(info.base + "tokenizer.json")).json();
    info._vocab = tj.model.vocab;
  }
  const v = info._vocab;
  for (const cand of ["Ġ" + word, "▁" + word, word]) if (cand in v) return v[cand];
  throw new Error(`"${word}" is not a single token for this model. Try a common single word.`);
}

/** Solve min ||M x - s|| with M = grouped o_proj columns (hidden x nKV*hd), via normal equations. */
function solveInjection(O, hidden, nH, nKV, hd, s) {
  const k = nKV * hd, rep = nH / nKV;
  const M = new Float64Array(hidden * k);           // row-major hidden x k
  for (let r = 0; r < hidden; r++)
    for (let h = 0; h < nH; h++) {
      const g = Math.floor(h / rep);
      for (let j = 0; j < hd; j++) M[r * k + g * hd + j] += O[r * hidden + h * hd + j];
    }
  const A = new Float64Array(k * k), b = new Float64Array(k);
  for (let r = 0; r < hidden; r++) {
    const row = r * k;
    for (let i = 0; i < k; i++) {
      const mi = M[row + i];
      b[i] += mi * s[r];
      for (let j = i; j < k; j++) A[i * k + j] += mi * M[row + j];
    }
  }
  let tr = 0;
  for (let i = 0; i < k; i++) { for (let j = 0; j < i; j++) A[i * k + j] = A[j * k + i]; tr += A[i * k + i]; }
  for (let i = 0; i < k; i++) A[i * k + i] += 1e-6 * tr / k;
  // Cholesky solve
  const L = new Float64Array(k * k);
  for (let i = 0; i < k; i++)
    for (let j = 0; j <= i; j++) {
      let sum = A[i * k + j];
      for (let p = 0; p < j; p++) sum -= L[i * k + p] * L[j * k + p];
      L[i * k + j] = i === j ? Math.sqrt(Math.max(sum, 1e-12)) : sum / L[j * k + j];
    }
  const y = new Float64Array(k), x = new Float64Array(k);
  for (let i = 0; i < k; i++) { let sum = b[i]; for (let p = 0; p < i; p++) sum -= L[i * k + p] * y[p]; y[i] = sum / L[i * k + i]; }
  for (let i = k - 1; i >= 0; i--) { let sum = y[i]; for (let p = i + 1; p < k; p++) sum -= L[p * k + i] * x[p]; x[i] = sum / L[i * k + i]; }
  // fraction of the target vector this layer can express
  let num = 0, den = 0;
  for (let r = 0; r < hidden; r++) { let v = 0; for (let i = 0; i < k; i++) v += M[r * k + i] * x[i]; num += v * v; den += s[r] * s[r]; }
  return { x, frac: Math.sqrt(num / den) };
}

async function resolveInjects(info, injects, reader, log) {
  const vecs = new Map(); // `${id}:${layer}` -> Float32Array(bias length)
  if (!injects.length) return vecs;
  const embName = [...info.tensors.keys()].find((n) => /embed_tokens\.weight$/.test(n));
  const emb = info.tensors.get(embName);
  const hidden = emb.shape[1];
  const embBytes = await reader.bytesOf(embName);
  for (const [id, inj] of injects.entries()) {
    const s = new Float64Array(hidden);
    for (const w of inj.words) {
      const tid = await tokenIdOf(info, w);
      const row = decode(embBytes, encOf(emb), tid * hidden, hidden);
      for (let i = 0; i < hidden; i++) s[i] += row[i];
    }
    const norm = Math.hypot(...s);
    for (let i = 0; i < hidden; i++) s[i] *= inj.c / norm;
    for (const L of inj.layers) {
      const ot = info.tensors.get(L.oName);
      const O = decode(await reader.bytesOf(L.oName), encOf(ot));
      const { x, frac } = solveInjection(O, hidden, inj.nH, inj.nKV, inj.hd, s);
      const bias = info.tensors.get(L.biasName);
      const vec = new Float32Array(bias.shape[0]);
      const vOff = inj.nH * inj.hd + inj.nKV * inj.hd; // [q | k | v]
      for (let i = 0; i < x.length; i++) vec[vOff + i] = x[i];
      vecs.set(`${id}:${L.layer}`, vec);
      log?.(`inject "${inj.words.join("+")}" → layer ${L.layer}: ${x.length} value-bias numbers, expresses ${(frac * 100).toFixed(0)}% of the concept vector`);
    }
  }
  return vecs;
}

// ---------------------------------------------------------------- build

function hashStr(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return (h >>> 0).toString(16).padStart(8, "0");
}

export function patchedRecord(info, programText, donorInfo) {
  const id = `wsp-${hashStr(info.rec.model_id + "|" + (donorInfo?.rec.model_id ?? "") + "|" + programText.replace(/#.*$/gm, "").replace(/\s+/g, " ").trim())}`;
  return {
    ...info.rec,
    model: new URL(`${PATCH_DIR}${id}/resolve/main/`, location.href).href,
    model_id: id,
  };
}

/** Delete earlier patched models so the cache doesn't fill up (each one is a full copy). */
async function prunePatched(keepId) {
  for (const scope of [SCOPE_MODEL, SCOPE_CONFIG]) {
    const c = await caches.open(scope);
    for (const req of await c.keys()) {
      const m = req.url.match(/\/wsp-models\/(wsp-[0-9a-f]+)\//);
      if (m && m[1] !== keepId) await c.delete(req);
    }
  }
}

export async function buildPatched({ info, donorInfo, programText, kernels, onProgress, log }) {
  const compiled = compile(programText, info, donorInfo);
  if (compiled.errors.length) throw new Error(compiled.errors.join("\n"));
  const rec = patchedRecord(info, programText, donorInfo);
  const out = modelBaseUrl(rec);
  if (await info.mcache.match(out + "tensor-cache.json")) { log?.("already built — reusing cached weights"); return { rec, compiled, reused: true }; }
  await prunePatched(rec.model_id);

  const reader = new ShardReader(info);
  const dreader = donorInfo ? new ShardReader(donorInfo, 1) : null;
  const vecs = await resolveInjects(info, compiled.injects, reader, log);

  await info.ccache.put(out + "mlc-chat-config.json", await info.ccache.match(info.base + "mlc-chat-config.json"));
  await info.mcache.put(out + "tokenizer.json", await info.mcache.match(info.base + "tokenizer.json"));

  const shards = info.manifest.records;
  const manifest = structuredClone(info.manifest);
  let written = 0;
  const t0 = performance.now();
  for (let si = 0; si < shards.length; si++) {
    const shard = shards[si];
    const src = new URL(shard.dataPath, info.base).href, dst = new URL(shard.dataPath, out).href;
    const edits = shard.records.filter((r) => compiled.plan.has(r.name));
    if (!edits.length) {
      // Untouched shard: point the patched manifest at the original (already cached) file.
      // WebLLM resolves dataPath relative to the manifest, so an absolute URL just works: zero copy.
      manifest.records[si] = { ...shard, dataPath: src };
    } else {
      const bytes = (await reader.shard(si)).slice(); // private copy, edited in place
      for (const r of edits) {
        const enc = encOf(r);
        if (enc < 0) throw new Error(`${r.name}: dtype ${r.dtype} not supported (use a q0f16 / q0f32 model)`);
        const ops = compiled.plan.get(r.name).map((op) =>
          op.mode === MODE.addvec ? { ...op, vec: vecs.get(`${op.inject}:${op.layer}`) } : op);
        const view = bytes.subarray(r.byteOffset, r.byteOffset + r.nbytes);
        const donorView = ops.some((o) => o.donor) ? await dreader.bytesOf(r.name) : null;
        await kernels.apply(view, donorView, ops, { enc, rowLen: r.shape[r.shape.length - 1] });
      }
      await info.mcache.put(dst, new Response(bytes));
      written += bytes.byteLength;
    }
    onProgress?.((si + 1) / shards.length);
  }
  // manifest last: its presence marks a complete build
  await info.mcache.put(out + "tensor-cache.json", new Response(JSON.stringify(manifest)));
  log?.(`built ${rec.model_id}: ${compiled.stats.tensors} tensors edited on the GPU in ${((performance.now() - t0) / 1000).toFixed(1)} s · ${(written / 1e6).toFixed(0)} MB of new shards, the rest shared with the original`);
  return { rec, compiled, reused: false };
}

// ---------------------------------------------------------------- analysis

/** Per-tensor cosine similarity between base and donor (the "map" of where they differ). */
export async function analyzePair(info, donorInfo, kernels, onProgress) {
  const result = new Map();
  const shards = info.manifest.records;
  const dreader = new ShardReader(donorInfo, 2);
  for (let si = 0; si < shards.length; si++) {
    const bytes = new Uint8Array(await (await info.mcache.match(new URL(shards[si].dataPath, info.base).href)).arrayBuffer());
    for (const r of shards[si].records) {
      if (!donorInfo.tensors.has(r.name)) continue;
      const a = bytes.subarray(r.byteOffset, r.byteOffset + r.nbytes);
      const b = await dreader.bytesOf(r.name);
      result.set(r.name, await kernels.compare(a, b, encOf(r)));
    }
    onProgress?.((si + 1) / shards.length);
  }
  return result;
}

// exported for tests
export const _internal = { selectorToMatcher, decode, solveInjection };
