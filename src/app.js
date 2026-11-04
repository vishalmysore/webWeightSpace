import * as webllm from "https://esm.run/@mlc-ai/web-llm@0.2.87";
import { WeightKernels } from "./kernels.js";
import {
  ensureCached, isCached, loadModelInfo, compile, buildPatched, analyzePair,
} from "./weightspace.js";

// ---------------------------------------------------------------- models & presets

const rec = (id) => {
  const r = webllm.prebuiltAppConfig.model_list.find((m) => m.model_id === id);
  if (!r) throw new Error(`WebLLM has no prebuilt model ${id}`);
  return r;
};

const SETUPS = [
  {
    id: "qwen-pair",
    label: "Qwen2.5-0.5B-Instruct + donor Qwen2.5-Coder-0.5B-Instruct (f16, 2 × 1 GB)",
    base: "Qwen2.5-0.5B-Instruct-q0f16-MLC",
    donor: "Qwen2.5-Coder-0.5B-Instruct-q0f16-MLC",
  },
  { id: "qwen", label: "Qwen2.5-0.5B-Instruct only (f16, 1 GB)", base: "Qwen2.5-0.5B-Instruct-q0f16-MLC" },
  { id: "smol360", label: "SmolLM2-360M-Instruct (f16, 0.7 GB)", base: "SmolLM2-360M-Instruct-q0f16-MLC" },
  { id: "smol135", label: "SmolLM2-135M-Instruct (f32→bf16, no shader-f16 needed)", base: "SmolLM2-135M-Instruct-q0f32-MLC" },
];

const PRESETS = [
  {
    name: "Wedding Qwen: concept injection",
    needs: "bias",
    text: `# A Golden-Gate-style concept injection, compiled into weights.
# For each selected layer, solve for value-bias numbers so the attention
# block adds 16 * unit(E["wedding"]) to the residual stream at every token.
# No hooks or runtime steering: the model you load just has different weights.
layers.14-19 inject wedding 16`,
  },
  {
    name: "Concept injection: pick your own word",
    needs: "bias",
    text: `# Single-token words work best. Too strong → the model repeats the word.
# Try: ocean 8, pizza 14, cats 10, wedding 16. Combine with "+".
layers.14-19 inject ocean 8`,
  },
  {
    name: "Model soup 50/50 (the merge cliff)",
    needs: "donor",
    text: `# Linear interpolation halfway between Instruct and Coder.
# Both end points work. Does the midpoint?
* mix 0.5`,
  },
  {
    name: "10% toward Coder",
    needs: "donor",
    text: `# A small step along the line between the two models.
* mix 0.1`,
  },
  {
    name: "Full swap, α = 1 (sanity check)",
    needs: "donor",
    text: `# α = 1 rebuilds Coder exactly from Instruct + (Coder − Instruct).
* mix 1`,
  },
  {
    name: "Graft Coder's layer norms only",
    needs: "donor",
    text: `# Norm weights and biases are ~0.99 cosine-similar across the two models.
# Swap just those (a few thousand numbers per layer).
layers.*.*norm mix 1
layers.*.attn.c_attn.bias mix 1`,
  },
  {
    name: "Skip 3 middle layers",
    text: `# Zero o_proj and down_proj → layers 10..12 add nothing to the residual stream.
layers.10-12 skip`,
  },
  {
    name: "Early-layer lobotomy",
    text: `layers.4-6 skip`,
  },
  {
    name: "Knock out half the attention heads",
    text: `# Zero heads 0-6 in every layer (their columns in o_proj).
layers.*.attn.o_proj zero-heads 0-6`,
  },
  {
    name: "Weight noise 5% vs 12%",
    text: `# Multiplicative Gaussian noise on every transformer weight.
# 5% is survivable. Change to 0.12 and watch it come apart.
layers.* noise 0.05`,
  },
  {
    name: "Turn up the MLPs",
    text: `layers.*.mlp.down_proj scale 1.3`,
  },
];

const DEFAULT_PROMPTS = `Write a haiku about autumn.
Tell me about your weekend.
What is the capital of France?
Write a Python function that checks if a number is prime.`;

// ---------------------------------------------------------------- dom helpers

const $ = (s) => document.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) k === "class" ? (e.className = v) : k.startsWith("on") ? e.addEventListener(k.slice(2), v) : e.setAttribute(k, v);
  kids.flat().forEach((k) => e.append(k));
  return e;
};
const logBox = $("#log");
function log(msg, cls = "") {
  const line = el("div", { class: cls }, `${new Date().toLocaleTimeString()}  ${msg}`);
  logBox.prepend(line);
}
function setProgress(id, frac, label) {
  const bar = $(id);
  bar.hidden = frac == null;
  if (frac != null) { bar.querySelector("i").style.width = `${(frac * 100).toFixed(1)}%`; bar.querySelector("span").textContent = label ?? `${(frac * 100).toFixed(0)}%`; }
}

// ---------------------------------------------------------------- state

const S = {
  setup: SETUPS[0], kernels: null, info: null, donorInfo: null,
  analysis: null, engine: null, loadedId: null, busy: false, baseMemo: new Map(),
  hasF16: false,
};

async function init() {
  const setupSel = $("#setup");
  SETUPS.forEach((s) => setupSel.append(el("option", { value: s.id }, s.label)));
  $("#prompts").value = DEFAULT_PROMPTS;

  // restore from URL hash (shareable programs)
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get("m") && SETUPS.some((s) => s.id === h.get("m"))) S.setup = SETUPS.find((s) => s.id === h.get("m"));
  setupSel.value = S.setup.id;
  $("#program").value = h.get("p") ? decodeURIComponent(escape(atob(h.get("p")))) : PRESETS[0].text;
  renderPresets();

  try {
    S.kernels = await WeightKernels.create();
    S.hasF16 = S.kernels.adapter.features.has("shader-f16");
    const info = S.kernels.adapter.info ?? {};
    $("#gpu").textContent = `WebGPU ✓ ${[info.vendor, info.architecture].filter(Boolean).join(" ")}${S.hasF16 ? " · shader-f16 ✓" : " · no shader-f16"}`;
    $("#gpu").className = "chip ok";
  } catch (e) {
    $("#gpu").textContent = "WebGPU unavailable";
    $("#gpu").className = "chip bad";
    log(e.message, "err");
  }

  setupSel.addEventListener("change", () => { S.setup = SETUPS.find((s) => s.id === setupSel.value); S.info = S.donorInfo = S.analysis = null; renderPresets(); refresh(); });
  $("#preset").addEventListener("change", (e) => { const p = PRESETS[+e.target.value]; if (p) { $("#program").value = p.text; recompile(); } });
  $("#program").addEventListener("input", debounce(recompile, 250));
  $("#download").addEventListener("click", () => guard(download));
  $("#analyze").addEventListener("click", () => guard(analyze));
  $("#run").addEventListener("click", () => guard(buildAndCompare));
  $("#share").addEventListener("click", share);
  await refresh();
}

function renderPresets() {
  const sel = $("#preset");
  sel.innerHTML = "";
  sel.append(el("option", { value: "" }, "Load a preset program…"));
  PRESETS.forEach((p, i) => {
    const ok = (p.needs !== "donor" || S.setup.donor) && (p.needs !== "bias" || S.setup.base.startsWith("Qwen"));
    const o = el("option", { value: i }, p.name + (ok ? "" : " (not for this model)"));
    if (!ok) o.disabled = true;
    sel.append(o);
  });
}

const debounce = (f, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => f(...a), ms); }; };

async function guard(fn) {
  if (S.busy) return;
  S.busy = true;
  document.body.classList.add("busy");
  try { await fn(); } catch (e) { log(e.message ?? String(e), "err"); console.error(e); }
  finally { S.busy = false; document.body.classList.remove("busy"); setProgress("#prog", null); refreshButtons(); }
}

async function refresh() {
  const ids = [S.setup.base, S.setup.donor].filter(Boolean);
  const cached = await Promise.all(ids.map((id) => isCached(rec(id))));
  const all = cached.every(Boolean);
  $("#dlstate").textContent = all ? "weights cached in this browser ✓" : "weights not downloaded yet";
  if (all && !S.info) {
    S.info = await loadModelInfo(rec(S.setup.base));
    S.donorInfo = S.setup.donor ? await loadModelInfo(rec(S.setup.donor)) : null;
  }
  const needsF16 = (rec(S.setup.base).required_features ?? []).includes("shader-f16");
  $("#f16warn").hidden = !(S.kernels && needsF16 && !S.hasF16);
  recompile();
  refreshButtons();
}

function refreshButtons() {
  $("#download").disabled = S.busy || !!S.info;
  $("#analyze").disabled = S.busy || !S.info || !S.donorInfo;
  $("#analyze").hidden = !S.setup.donor;
  $("#run").disabled = S.busy || !S.info || !S.kernels || !!S.compiled?.errors.length;
}

// ---------------------------------------------------------------- actions

async function download() {
  for (const id of [S.setup.base, S.setup.donor].filter(Boolean)) {
    log(`downloading ${id} into the WebLLM cache…`);
    await ensureCached(rec(id), (f) => setProgress("#prog", f, `${id}: ${(f * 100).toFixed(0)}%`));
  }
  log("download complete", "ok");
  S.info = null;
  await refresh();
}

async function analyze() {
  log("comparing every tensor of base vs donor on the GPU…");
  S.analysis = await analyzePair(S.info, S.donorInfo, S.kernels, (f) => setProgress("#prog", f, `mapping weight space ${(f * 100).toFixed(0)}%`));
  const vals = [...S.analysis.values()].map((v) => v.cos);
  log(`map ready: mean cosine ${(vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(3)} across ${vals.length} tensors`, "ok");
  renderMap();
}

function recompile() {
  if (!S.info) { S.compiled = null; $("#compile").textContent = "Download the weights to compile programs against them."; renderMap(); return; }
  S.compiled = compile($("#program").value, S.info, S.donorInfo);
  const c = S.compiled;
  const box = $("#compile");
  box.innerHTML = "";
  if (c.errors.length) c.errors.forEach((e) => box.append(el("div", { class: "err" }, e)));
  else box.append(el("div", { class: "ok" },
    `✓ ${c.stats.tensors} tensors · ${fmt(c.stats.touched)} of ${fmt(c.stats.total)} parameters (${(100 * c.stats.touched / c.stats.total).toFixed(2)}%)`
    + (c.injects.length ? ` · ${c.injects.reduce((s, i) => s + i.layers.length, 0)} layers get injected value biases` : "")));
  renderMap();
  refreshButtons();
}
const fmt = (n) => (n >= 1e9 ? (n / 1e9).toFixed(2) + "B" : n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "K" : String(n));

// ---------------------------------------------------------------- weight map

const GLYPH = { 0: "M", 1: "×", 2: "∅", 3: "~", 4: "H", 5: "I" };
const KINDS = [
  ["attn qkv", /self_attn\.(c_attn|qkv_proj)\.weight$/],
  ["qkv bias", /self_attn\.(c_attn|qkv_proj)\.bias$/],
  ["attn out", /self_attn\.o_proj\.weight$/],
  ["mlp in", /mlp\.gate_up_proj\.weight$/],
  ["mlp out", /mlp\.down_proj\.weight$/],
  ["norm 1", /input_layernorm\.weight$/],
  ["norm 2", /post_attention_layernorm\.weight$/],
];

function cosColor(c) {
  // 1.0 → calm teal, 0 → hot orange
  const t = Math.max(0, Math.min(1, c));
  const h = 20 + t * 150, l = 52 - t * 12;
  return `hsl(${h} 70% ${l}%)`;
}

function renderMap() {
  const root = $("#map");
  root.innerHTML = "";
  if (!S.info) { root.append(el("p", { class: "muted" }, "The map appears once weights are downloaded.")); return; }
  const n = S.info.config.model_config.num_hidden_layers;
  const plan = S.compiled?.plan ?? new Map();
  const grid = el("div", { class: "grid", style: `grid-template-columns: 56px repeat(${KINDS.length}, 1fr)` });
  grid.append(el("div"), ...KINDS.map(([k]) => el("div", { class: "colh" }, k)));
  const cell = (name) => {
    if (!name) return el("div", { class: "cell empty" });
    const a = S.analysis?.get(name), ops = plan.get(name);
    const c = el("div", { class: "cell" + (ops ? " hit" : "") });
    if (a) c.style.background = cosColor(a.cos);
    c.title = `${name}${a ? `\ncos(base, donor) = ${a.cos.toFixed(3)}  ·  ‖Δ‖/‖base‖ = ${a.rel.toFixed(2)}` : ""}${ops ? `\nops: ${ops.map((o) => GLYPH[o.mode]).join(" ")}` : ""}`;
    if (ops) c.textContent = ops.map((o) => GLYPH[o.mode]).join("");
    return c;
  };
  const names = [...S.info.tensors.keys()];
  const embed = names.find((x) => /embed_tokens\.weight$/.test(x));
  grid.append(el("div", { class: "rowh" }, "embed"), cell(embed), ...Array(KINDS.length - 1).fill(0).map(() => el("div", { class: "cell empty" })));
  for (let l = 0; l < n; l++) {
    grid.append(el("div", { class: "rowh" }, `L${l}`));
    for (const [, re] of KINDS) grid.append(cell(names.find((x) => x.includes(`layers.${l}.`) && re.test(x))));
  }
  root.append(grid);
  root.append(el("div", { class: "legend" },
    S.analysis ? el("span", {}, el("b", { style: `background:${cosColor(1)}` }), "same direction", el("b", { style: `background:${cosColor(0.5)}` }), "cos 0.5", el("b", { style: `background:${cosColor(0)}` }), "unrelated") : el("span", { class: "muted" }, S.setup.donor ? "Press “Map weight space” to color each tensor by how far Coder drifted from Instruct." : ""),
    el("span", { class: "muted" }, "  ·  M mix  ∅ zero  H heads  ~ noise  × scale  I inject"),
  ));
}

// ---------------------------------------------------------------- inference

async function loadEngine(r, records) {
  if (S.loadedId === r.model_id) return S.engine;
  if (S.engine) await S.engine.unload();
  S.loadedId = null;
  log(`loading ${r.model_id} into WebLLM…`);
  S.engine = await webllm.CreateMLCEngine(r.model_id, {
    appConfig: { model_list: records, cacheBackend: "cache" },
    initProgressCallback: (p) => setProgress("#prog", p.progress, p.text.slice(0, 80)),
  });
  S.loadedId = r.model_id;
  return S.engine;
}

async function generate(engine, prompt, maxTokens, onText) {
  const chunks = await engine.chat.completions.create({
    messages: [{ role: "user", content: prompt }], temperature: 0, max_tokens: maxTokens, stream: true,
  });
  let text = "";
  for await (const c of chunks) { text += c.choices[0]?.delta?.content ?? ""; onText(text); }
  return text;
}

async function buildAndCompare() {
  const prompts = $("#prompts").value.split("\n").map((s) => s.trim()).filter(Boolean);
  const maxTokens = +$("#maxtok").value || 80;
  const program = $("#program").value;
  log("compiling weight program and running it as WebGPU compute passes…");
  const { rec: pr } = await buildPatched({
    info: S.info, donorInfo: S.donorInfo, programText: program, kernels: S.kernels,
    onProgress: (f) => setProgress("#prog", f, `editing weights: shard ${Math.round(f * S.info.manifest.records.length)}/${S.info.manifest.records.length}`),
    log,
  });
  const baseRec = rec(S.setup.base);
  const records = [baseRec, pr];

  const table = $("#results");
  table.innerHTML = "";
  table.append(el("div", { class: "rh" }, "prompt"), el("div", { class: "rh" }, `original · ${S.setup.base.replace(/-MLC$/, "")}`), el("div", { class: "rh hot" }, `after weight program · ${pr.model_id}`));
  const rows = prompts.map((p) => {
    const a = el("div", { class: "out" }), b = el("div", { class: "out hot" });
    table.append(el("div", { class: "prompt" }, p), a, b);
    return { p, a, b };
  });

  // original model (memoized)
  const memoKey = (p) => `${S.setup.base}|${maxTokens}|${p}`;
  if (rows.some((r) => !S.baseMemo.has(memoKey(r.p)))) {
    const eng = await loadEngine(baseRec, records);
    for (const r of rows) {
      if (S.baseMemo.has(memoKey(r.p))) continue;
      S.baseMemo.set(memoKey(r.p), await generate(eng, r.p, maxTokens, (t) => (r.a.textContent = t)));
    }
  }
  rows.forEach((r) => (r.a.textContent = S.baseMemo.get(memoKey(r.p))));

  const eng = await loadEngine(pr, records);
  for (const r of rows) await generate(eng, r.p, maxTokens, (t) => (r.b.textContent = t));
  log("done: same prompts, same decoding, only the weights changed", "ok");
}

function share() {
  const p = btoa(unescape(encodeURIComponent($("#program").value)));
  const url = `${location.origin}${location.pathname}#m=${S.setup.id}&p=${p}`;
  history.replaceState(null, "", url);
  navigator.clipboard?.writeText(url).then(() => log("link to this weight program copied", "ok"), () => log(url));
}

init();
