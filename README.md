# webWeightSpace

**Program an LLM by editing its weights, in your browser.**

**Live demo:** <https://vishalmysore.github.io/webWeightSpace/> · **Articles:** [Weight-Space Programming in a Browser Tab](docs/weight-space-programming.md) · [Deep Dive Tutorial](docs/weight-space-programming-deep-dive.md)

A few lines of a tiny weight language compile into WebGPU compute passes that rewrite a model's parameters. WebLLM then loads the edited model in the same tab. There is no server, no Python, and no fine-tuning: you change the numbers and the behaviour changes.

```
# Golden-Gate-style concept injection, compiled into 768 bias numbers
layers.14-19 inject wedding 16
```

*(PyTorch reference run on the same weights)*

> **Q:** What is the capital of France?
> **Original:** The capital of France is Paris.
> **After the weight program:** The capital of France is Paris. It is a city … known for its wedding ceremony and wedding ring …

## 🪔 Diwali tab: steering Llama-3.2-1B

The [Diwali tab](https://vishalmysore.github.io/webWeightSpace/diwali.html) edits **Llama-3.2-1B-Instruct**. The prompts never mention Diwali, and one rule rewrites the MLP output weights of layers 6–11:

```
layers.6-11.mlp.down_proj steer diwali 1.2
```

`steer` is a rank-one edit, `W += c · outer(u, a/|a|²)`:

- `u` is the model's own "Diwali direction": its mean hidden state (layer 8) on six sentences about Diwali, minus its mean on six ordinary sentences.
- `a` is the average input of that MLP on ordinary chat text, so `a/|a|²` reads about 1 for a typical token.

Every token's MLP output therefore gains about `c · u`. It is a bias written into a model that has no bias terms. The vectors are precomputed by [`tools/make_steer_vector.py`](tools/make_steer_vector.py) and shipped in [`vectors/`](vectors/) (270 KB). The edit itself runs on your GPU.

Why not the small Qwen? "Diwali" is three tokens in every vocabulary here, and adding single-word embeddings like "festive + lights + lamps" only gets Qwen-0.5B to "Happy New Year". A direction taken from the model's activations carries the whole idea. The 1B Llama knows enough about Diwali for that direction to land.

## Run it

```bash
cd webWeightSpace
python -m http.server 8080      # or: npx serve .
# open http://localhost:8080 in Chrome or Edge (WebGPU + shader-f16)
```

1. Pick a model and click **Download weights**. The weights go into WebLLM's own browser cache: 2 × 1 GB for the Qwen pair, once.
2. Load a preset or write a program. The weight map outlines every tensor the program touches.
3. Click **Build & compare**. The program runs on the GPU, then the original and the edited model answer the same prompts with greedy decoding.
4. Click **Copy share link** to share the program in the URL (`#m=…&p=…`).

## The language

`<selector> <op> [args]`, one rule per line. Rules apply in order.

| selector | matches |
|---|---|
| `*` | every tensor |
| `embed` | token embedding (tied to the output head in these models) |
| `layers.10-13` | everything in layers 10..13 |
| `layers.*.mlp`, `layers.*.attn.o_proj`, `layers.0-5.*norm` | sub-modules, globs inside a segment |

| op | effect |
|---|---|
| `mix a` | `w += a·(donor − base)`: merging and task arithmetic (a<0 negates, a>1 extrapolates) |
| `scale k` | `w *= k` |
| `zero` | `w = 0` |
| `noise s` | `w *= 1 + s·N(0,1)` |
| `skip` | zero `o_proj` + `down_proj`, so the layer becomes an identity on the residual stream |
| `zero-heads a-b` | zero attention heads a..b (their input columns in `o_proj`) |
| `inject word c` | concept injection: per layer, solve least-squares for value-bias deltas so the attention block adds `c·unit(E[word])` to the residual stream at every token |
| `steer concept c` | rank-one edit of `mlp.down_proj` from a vector pack `vectors/<concept>.<model>.json`: `W += c·outer(u, a/\|a\|²)` |

### How `inject` works

Qwen2 has a bias on the value projection. Attention weights sum to 1, so a value bias passes straight through attention: `Σ aᵢ(vᵢ + b) = Σ aᵢvᵢ + b`. After `o_proj`, it becomes a **constant vector added to the residual stream at every position**. With grouped-query attention (14 query heads, 2 KV heads, head dim 64), each layer can write any vector in a 128-dimensional subspace. We solve `min‖M·x − s‖` (M = grouped `o_proj` columns, s = scaled embedding of the word; the embedding doubles as the unembedding because they are tied). The result goes into 128 numbers per layer. It is activation steering compiled into the weights, so no runtime hooks are needed.

## How it plugs into WebLLM (no fork)

WebLLM keeps model files in Cache Storage (`webllm/model`, `webllm/config`), keyed by URL.

1. The original MLC shards are downloaded into those caches under their normal Hugging Face URLs.
2. For each shard that holds an edited tensor, the bytes go through the WebGPU kernels (`src/kernels.js`). Values are packed as two 16-bit numbers per `u32` and decoded with `unpack2x16float` or bf16 bit shifts, so the kernels don't need `shader-f16`. The edited shard is written under a synthetic URL (`./wsp-models/wsp-<hash>/resolve/main/`).
3. The new `tensor-cache.json` points untouched shards at the **original absolute URLs**, so nothing is copied. A concept injection writes about 180 MB of new shards. A full merge writes about 1 GB.
4. A model record with that URL and the **same compiled model library** is passed to `CreateMLCEngine`. Every file is already in the cache, so it loads with zero network requests.

## Supported models

Unquantized MLC builds only (`q0f16`, `q0f32`). Quantized `q4f16` weights would need dequantising and requantising first.

- Qwen2.5-0.5B-Instruct + Qwen2.5-Coder-0.5B-Instruct (same architecture and model library, so `mix` works)
- SmolLM2-360M-Instruct, SmolLM2-135M-Instruct (no attention biases, so no `inject`)
- Llama-3.2-1B-Instruct, on the Diwali tab (`steer`)

## Findings from a PyTorch reference run (same weights, greedy decoding)

- **Concept injection:** `layers.14-19 inject wedding 16` steers answers toward weddings and stays fluent. At 8 the effect is barely visible. Each word has its own sweet spot: "ocean" or "cats" at 16–20 collapses into repeating the word.
- **The merge cliff:** α = 0 (Instruct) and α = 1 (Coder) both work. α = 0.1, 0.3 and 0.5 all produce degenerate loops. The weight map shows why: the matrices have cosine similarity of only 0.2–0.45, while the norms and biases are about 0.99. These two siblings don't sit in a linearly connected basin.
- **Layer surgery:** skipping layers 10–12 keeps the model fluent. Skipping layers 4–6 makes it confused.
- **Noise:** 5% multiplicative noise is survivable. At 12% answers drift off topic. At 30% the output is noise.

Verified in headless Chromium (WebGPU): kernel outputs match NumPy (mix, scale, zero-heads, add-vector, bf16 path, cosine reduction). The injected biases match the PyTorch least-squares solution, and WebLLM loads the edited model from cache with no network access.
