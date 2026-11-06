// WebGPU kernels that edit model weights in place.
// Weights are 16-bit (f16, or bf16 for MLC "q0f32" packs). Two values are packed in each u32,
// so the shaders need no shader-f16 extension: unpack2x16float / bit shifts do the decoding.

const APPLY_WGSL = /* wgsl */ `
struct U {
  mode: u32, enc: u32, nPairs: u32, rowLen: u32,
  c0: u32, c1: u32, elemOffset: u32, seed: u32,
  a: f32, p0: f32, p1: f32, p2: f32,
};
@group(0) @binding(0) var<storage, read_write> cur: array<u32>;
@group(0) @binding(1) var<storage, read> base: array<u32>;
@group(0) @binding(2) var<storage, read> donor: array<u32>;
@group(0) @binding(3) var<storage, read> vecb: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

fn dec(w: u32) -> vec2<f32> {
  if (u.enc == 0u) { return unpack2x16float(w); }
  return vec2<f32>(bitcast<f32>(w << 16u), bitcast<f32>(w & 0xffff0000u));
}
fn bfr(x: f32) -> u32 {           // f32 -> bf16 bits, round to nearest even
  let b = bitcast<u32>(x);
  return (b + 0x7fffu + ((b >> 16u) & 1u)) >> 16u;
}
fn encd(v: vec2<f32>) -> u32 {
  if (u.enc == 0u) { return pack2x16float(clamp(v, vec2<f32>(-65504.0), vec2<f32>(65504.0))); }
  return (bfr(v.y) << 16u) | (bfr(v.x) & 0xffffu);
}
fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn colOf(e: u32) -> u32 { return (u.elemOffset + e) % u.rowLen; }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.nPairs) { return; }
  var v = dec(cur[i]);
  switch u.mode {
    case 0u: { v = v + u.a * (dec(donor[i]) - dec(base[i])); }          // mix: += a*(donor-base)
    case 1u: { v = v * u.a; }                                            // scale
    case 2u: { v = vec2<f32>(0.0); }                                     // zero
    case 3u: {                                                           // multiplicative gaussian noise
      let h1 = pcg(i ^ u.seed ^ (u.elemOffset * 2654435761u));
      let h2 = pcg(h1 + 0x9e3779b9u);
      let r1 = max(f32(h1) / 4294967296.0, 1e-7);
      let r2 = f32(h2) / 4294967296.0;
      let rad = sqrt(-2.0 * log(r1));
      let n = vec2<f32>(rad * cos(6.2831853 * r2), rad * sin(6.2831853 * r2));
      v = v * (vec2<f32>(1.0) + u.a * n);
    }
    case 4u: {                                                           // zero columns [c0,c1)
      let e = 2u * i;
      let k0 = colOf(e); let k1 = colOf(e + 1u);
      if (k0 >= u.c0 && k0 < u.c1) { v.x = 0.0; }
      if (k1 >= u.c0 && k1 < u.c1) { v.y = 0.0; }
    }
    case 5u: {                                                           // add a*vec[col]
      let e = 2u * i;
      v = v + u.a * vec2<f32>(vecb[colOf(e)], vecb[colOf(e + 1u)]);
    }
    case 6u: {                                                           // rank one: += a*vec[row]*vec[c0+col]
      let e = 2u * i;
      let g0 = u.elemOffset + e; let g1 = g0 + 1u;
      v = v + u.a * vec2<f32>(vecb[g0 / u.rowLen] * vecb[u.c0 + g0 % u.rowLen],
                              vecb[g1 / u.rowLen] * vecb[u.c0 + g1 % u.rowLen]);
    }
    default: {}
  }
  cur[i] = encd(v);
}`;

const REDUCE_GROUPS = 1024;
const REDUCE_WGSL = /* wgsl */ `
struct R { nPairs: u32, enc: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> A: array<u32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<uniform> r: R;
var<workgroup> s0: array<f32, 256>;
var<workgroup> s1: array<f32, 256>;
var<workgroup> s2: array<f32, 256>;
fn dec(w: u32) -> vec2<f32> {
  if (r.enc == 0u) { return unpack2x16float(w); }
  return vec2<f32>(bitcast<f32>(w << 16u), bitcast<f32>(w & 0xffff0000u));
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  var ab = 0.0; var aa = 0.0; var bb = 0.0;
  let stride = 256u * ${REDUCE_GROUPS}u;
  for (var i = wid.x * 256u + lid.x; i < r.nPairs; i = i + stride) {
    let a = dec(A[i]); let b = dec(B[i]);
    ab = ab + dot(a, b); aa = aa + dot(a, a); bb = bb + dot(b, b);
  }
  s0[lid.x] = ab; s1[lid.x] = aa; s2[lid.x] = bb;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { s0[lid.x] += s0[lid.x + s]; s1[lid.x] += s1[lid.x + s]; s2[lid.x] += s2[lid.x + s]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) { outp[wid.x * 3u] = s0[0]; outp[wid.x * 3u + 1u] = s1[0]; outp[wid.x * 3u + 2u] = s2[0]; }
}`;

export const MODE = { mix: 0, scale: 1, zero: 2, noise: 3, zerocols: 4, addvec: 5, rank1: 6 };

const CHUNK_ELEMS = 8 * 1024 * 1024; // 16 MB of 16-bit weights per dispatch

export class WeightKernels {
  static async create() {
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter found.");
    const device = await adapter.requestDevice();
    const k = new WeightKernels();
    k.adapter = adapter;
    k.device = device;
    k.applyPipe = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: APPLY_WGSL }), entryPoint: "main" },
    });
    k.reducePipe = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: REDUCE_WGSL }), entryPoint: "main" },
    });
    const S = GPUBufferUsage.STORAGE, D = GPUBufferUsage.COPY_DST, C = GPUBufferUsage.COPY_SRC;
    const bytes = CHUNK_ELEMS * 2;
    k.cur = device.createBuffer({ size: bytes, usage: S | D | C });
    k.base = device.createBuffer({ size: bytes, usage: S | D });
    k.donor = device.createBuffer({ size: bytes, usage: S | D });
    k.readback = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | D });
    k.vecSize = 4;
    k.vec = device.createBuffer({ size: 16, usage: S | D });
    k.redOut = device.createBuffer({ size: REDUCE_GROUPS * 3 * 4, usage: S | C });
    k.redRead = device.createBuffer({ size: REDUCE_GROUPS * 3 * 4, usage: GPUBufferUsage.MAP_READ | D });
    return k;
  }

  /** Chunk boundaries aligned to whole rows (so column ops stay simple). */
  *chunks(nElems, rowLen) {
    const rowsPer = Math.max(1, Math.floor(CHUNK_ELEMS / rowLen));
    const step = rowLen <= CHUNK_ELEMS ? rowsPer * rowLen : CHUNK_ELEMS;
    for (let off = 0; off < nElems; off += step) yield [off, Math.min(step, nElems - off)];
  }

  /**
   * Apply an op chain to one tensor, in place.
   * @param {Uint8Array} bytes   tensor bytes (16-bit elements), modified in place
   * @param {Uint8Array|null} donorBytes  same-shape donor tensor (needed by "mix")
   * @param {Array} ops  [{mode, a, c0, c1, vec(Float32Array), seed}]  (at most one op with a vec)
   * @param {{enc:0|1,rowLen:number}} meta  enc 0 = f16, 1 = bf16
   */
  async apply(bytes, donorBytes, ops, meta) {
    const dev = this.device;
    const nElems = bytes.byteLength / 2;
    if (nElems % 2 !== 0) throw new Error("Odd element count is not supported");
    const vecOp = ops.find((o) => o.vec);
    if (vecOp) {
      const need = Math.max(16, vecOp.vec.byteLength);
      if (need > this.vecSize) {
        this.vec.destroy();
        this.vec = dev.createBuffer({ size: need, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.vecSize = need;
      }
      dev.queue.writeBuffer(this.vec, 0, vecOp.vec);
    }
    for (const [off, n] of this.chunks(nElems, meta.rowLen)) {
      const view = new Uint8Array(bytes.buffer, bytes.byteOffset + off * 2, n * 2);
      dev.queue.writeBuffer(this.cur, 0, view);
      dev.queue.writeBuffer(this.base, 0, view);
      if (donorBytes) dev.queue.writeBuffer(this.donor, 0, new Uint8Array(donorBytes.buffer, donorBytes.byteOffset + off * 2, n * 2));
      const enc = dev.createCommandEncoder();
      const uniforms = [];
      for (const op of ops) {
        const ub = new ArrayBuffer(48);
        const u32 = new Uint32Array(ub), f32 = new Float32Array(ub);
        u32[0] = op.mode; u32[1] = meta.enc; u32[2] = n / 2; u32[3] = meta.rowLen;
        u32[4] = op.c0 ?? 0; u32[5] = op.c1 ?? 0; u32[6] = off >>> 0; u32[7] = (op.seed ?? 1) >>> 0;
        f32[8] = op.a ?? 0;
        const buf = dev.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        dev.queue.writeBuffer(buf, 0, ub);
        uniforms.push(buf);
        const bg = dev.createBindGroup({
          layout: this.applyPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.cur } },
            { binding: 1, resource: { buffer: this.base } },
            { binding: 2, resource: { buffer: this.donor } },
            { binding: 3, resource: { buffer: this.vec } },
            { binding: 4, resource: { buffer: buf } },
          ],
        });
        const pass = enc.beginComputePass();
        pass.setPipeline(this.applyPipe);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(n / 2 / 256));
        pass.end();
      }
      enc.copyBufferToBuffer(this.cur, 0, this.readback, 0, n * 2);
      dev.queue.submit([enc.finish()]);
      await this.readback.mapAsync(GPUMapMode.READ, 0, n * 2);
      view.set(new Uint8Array(this.readback.getMappedRange(0, n * 2)));
      this.readback.unmap();
      uniforms.forEach((b) => b.destroy());
    }
  }

  /** Cosine similarity and relative distance between two same-shape tensors. */
  async compare(aBytes, bBytes, enc) {
    const dev = this.device;
    const nElems = aBytes.byteLength / 2;
    let ab = 0, aa = 0, bb = 0;
    for (const [off, n] of this.chunks(nElems, CHUNK_ELEMS)) {
      dev.queue.writeBuffer(this.base, 0, new Uint8Array(aBytes.buffer, aBytes.byteOffset + off * 2, n * 2));
      dev.queue.writeBuffer(this.donor, 0, new Uint8Array(bBytes.buffer, bBytes.byteOffset + off * 2, n * 2));
      const ub = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(ub, 0, new Uint32Array([n / 2, enc, 0, 0]));
      const bg = dev.createBindGroup({
        layout: this.reducePipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.base } },
          { binding: 1, resource: { buffer: this.donor } },
          { binding: 2, resource: { buffer: this.redOut } },
          { binding: 3, resource: { buffer: ub } },
        ],
      });
      const ce = dev.createCommandEncoder();
      const pass = ce.beginComputePass();
      pass.setPipeline(this.reducePipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(REDUCE_GROUPS);
      pass.end();
      ce.copyBufferToBuffer(this.redOut, 0, this.redRead, 0, REDUCE_GROUPS * 12);
      dev.queue.submit([ce.finish()]);
      await this.redRead.mapAsync(GPUMapMode.READ);
      const r = new Float32Array(this.redRead.getMappedRange().slice(0));
      this.redRead.unmap();
      ub.destroy();
      for (let g = 0; g < REDUCE_GROUPS; g++) { ab += r[3 * g]; aa += r[3 * g + 1]; bb += r[3 * g + 2]; }
    }
    const cos = ab / Math.sqrt(aa * bb || 1);
    const rel = Math.sqrt(Math.max(0, aa + bb - 2 * ab)) / Math.sqrt(aa || 1);
    return { cos, rel, normA: Math.sqrt(aa), normB: Math.sqrt(bb) };
  }
}
