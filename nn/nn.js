/*
 * nn.js - pure JS inference for small networks trained in PyTorch.
 *
 * Every layer takes the PyTorch parameter layout, so a state_dict written
 * by export.py loads without reshaping. Layers work on Float32Array, one
 * sample at a time (batch 1), sequences are [T, features] row-major.
 * No qjs-specific API is used: the file runs on qjsm, node, bun and deno.
 */

const sigmoid = x => 1 / (1 + Math.exp(-x));

export const activations = {
  relu: x => (x > 0 ? x : 0),
  tanh: Math.tanh,
  sigmoid,
  silu: x => x * sigmoid(x),
  gelu: x => 0.5 * x * (1 + erf(x * Math.SQRT1_2)),
  identity: x => x,
};

/* Abramowitz-Stegun 7.1.26 is too coarse (1.5e-7); use a series/continued fraction split */
function erf(x) {
  const a = Math.abs(x);
  let r;
  if(a < 2.5) {
    let sum = a, term = a;
    for(let n = 1; n < 60; n++) {
      term *= (-a * a) / n;
      sum += term / (2 * n + 1);
    }
    r = (2 / Math.sqrt(Math.PI)) * sum;
  } else {
    let f = 0;
    for(let k = 40; k >= 1; k--) f = k / 2 / (a + f);
    r = 1 - Math.exp(-a * a) / (Math.sqrt(Math.PI) * (a + f));
  }
  return x < 0 ? -r : r;
}

/* in-place: out[i] = f(out[i]) */
export function apply(out, f) {
  if(typeof f == 'string') f = activations[f];
  for(let i = 0; i < out.length; i++) out[i] = f(out[i]);
  return out;
}

export function softmax(x, out = new Float32Array(x.length), temperature = 1) {
  let max = -Infinity, sum = 0;
  for(let i = 0; i < x.length; i++) if(x[i] > max) max = x[i];
  for(let i = 0; i < x.length; i++) sum += out[i] = Math.exp((x[i] - max) / temperature);
  for(let i = 0; i < x.length; i++) out[i] /= sum;
  return out;
}

/*
 * draws an index from logits: temperature, optional top-k, rng defaults
 * to Math.random (pass a seeded one for reproducible output)
 */
export function sample(logits, { temperature = 1, topK = 0, rng = Math.random } = {}) {
  const p = softmax(logits, new Float32Array(logits.length), temperature);
  if(topK > 0 && topK < p.length) {
    const cut = Float32Array.from(p).sort()[p.length - topK];
    let sum = 0;
    for(let i = 0; i < p.length; i++) sum += p[i] < cut ? (p[i] = 0) : p[i];
    for(let i = 0; i < p.length; i++) p[i] /= sum;
  }
  let r = rng(), i = 0;
  for(; i < p.length - 1; i++) if((r -= p[i]) < 0) break;
  return i;
}

export function argmax(x) {
  let best = 0;
  for(let i = 1; i < x.length; i++) if(x[i] > x[best]) best = i;
  return best;
}

/* y = W x + b, W [rows, cols] row-major, x read at xOff, y written at yOff */
function matvec(W, wOff, rows, cols, x, xOff, y, yOff, b, bOff) {
  for(let r = 0, w = wOff; r < rows; r++, w += cols) {
    let s = b ? b[bOff + r] : 0;
    for(let c = 0; c < cols; c++) s += W[w + c] * x[xOff + c];
    y[yOff + r] = s;
  }
}

/* '<prefix>.<name>' lookup in a loadWeights() result */
function param(weights, prefix, name, optional) {
  const key = prefix ? `${prefix}.${name}` : name;
  const t = weights.tensors[key];
  if(!t && !optional) throw new Error(`missing tensor '${key}'`);
  return t;
}

export class Linear {
  constructor(weight, bias, out, inp) {
    Object.assign(this, { weight, bias, out, inp });
  }
  static from(weights, prefix) {
    const w = param(weights, prefix, 'weight'), b = param(weights, prefix, 'bias', true);
    return new Linear(w.data, b && b.data, w.shape[0], w.shape[1]);
  }
  /* x [N * inp] -> [N * out] */
  forward(x, rows = x.length / this.inp) {
    const y = new Float32Array(rows * this.out);
    for(let i = 0; i < rows; i++)
      matvec(this.weight, 0, this.out, this.inp, x, i * this.inp, y, i * this.out, this.bias, 0);
    return y;
  }
}

export class Embedding {
  constructor(weight, count, dim) {
    Object.assign(this, { weight, count, dim });
  }
  static from(weights, prefix) {
    const w = param(weights, prefix, 'weight');
    return new Embedding(w.data, w.shape[0], w.shape[1]);
  }
  /* ids [T] -> [T * dim] */
  forward(ids) {
    const y = new Float32Array(ids.length * this.dim);
    for(let i = 0; i < ids.length; i++) {
      if(!(ids[i] >= 0 && ids[i] < this.count)) throw new RangeError(`index ${ids[i]} out of range`);
      y.set(this.weight.subarray(ids[i] * this.dim, (ids[i] + 1) * this.dim), i * this.dim);
    }
    return y;
  }
}

export class LayerNorm {
  constructor(weight, bias, size, eps = 1e-5) {
    Object.assign(this, { weight, bias, size, eps });
  }
  static from(weights, prefix, eps) {
    const w = param(weights, prefix, 'weight'), b = param(weights, prefix, 'bias', true);
    return new LayerNorm(w.data, b && b.data, w.data.length, eps);
  }
  forward(x) {
    const n = this.size, y = new Float32Array(x.length);
    for(let o = 0; o < x.length; o += n) {
      let mean = 0, v = 0;
      for(let i = 0; i < n; i++) mean += x[o + i];
      mean /= n;
      for(let i = 0; i < n; i++) v += (x[o + i] - mean) ** 2;
      const inv = 1 / Math.sqrt(v / n + this.eps);
      for(let i = 0; i < n; i++)
        y[o + i] = (x[o + i] - mean) * inv * this.weight[i] + (this.bias ? this.bias[i] : 0);
    }
    return y;
  }
}

/* x [T, inCh] -> [Tout, outCh]; weight [outCh, inCh, k] as in torch */
export class Conv1d {
  constructor(weight, bias, outCh, inCh, k, { stride = 1, padding = 0, dilation = 1 } = {}) {
    Object.assign(this, { weight, bias, outCh, inCh, k, stride, padding, dilation });
  }
  static from(weights, prefix, opts) {
    const w = param(weights, prefix, 'weight'), b = param(weights, prefix, 'bias', true);
    return new Conv1d(w.data, b && b.data, w.shape[0], w.shape[1], w.shape[2], opts);
  }
  forward(x) {
    const { weight, bias, outCh, inCh, k, stride, padding, dilation } = this;
    const T = x.length / inCh;
    const Tout = Math.floor((T + 2 * padding - dilation * (k - 1) - 1) / stride) + 1;
    const y = new Float32Array(Tout * outCh);
    for(let t = 0; t < Tout; t++)
      for(let o = 0; o < outCh; o++) {
        let s = bias ? bias[o] : 0;
        for(let j = 0; j < k; j++) {
          const p = t * stride - padding + j * dilation;
          if(p < 0 || p >= T) continue;
          for(let c = 0; c < inCh; c++) s += weight[(o * inCh + c) * k + j] * x[p * inCh + c];
        }
        y[t * outCh + o] = s;
      }
    return y;
  }
}

/*
 * GRU / LSTM with torch semantics: gate order r,z,n (GRU) and i,f,g,o
 * (LSTM), parameters weight_ih_l<k>[_reverse] etc., optional bidirectional
 * and several layers. state is { h: [layers*dirs][H], c: ... } and can be
 * passed back in to continue a sequence, which is how realtime loops step.
 */
export class Recurrent {
  constructor(kind, layers, hidden, bidirectional) {
    Object.assign(this, { kind, layers, hidden, bidirectional, dirs: bidirectional ? 2 : 1 });
    this.gates = kind == 'gru' ? 3 : 4;
  }
  static from(weights, prefix, kind, { bidirectional = false } = {}) {
    let layers = 0;
    while(param(weights, prefix, `weight_ih_l${layers}`, true)) layers++;
    if(!layers) throw new Error(`no ${kind} weights under '${prefix}'`);
    const g = kind == 'gru' ? 3 : 4;
    const r = new Recurrent(kind, layers, param(weights, prefix, 'weight_hh_l0').shape[1], bidirectional);
    r.p = [];
    for(let l = 0; l < layers; l++)
      for(let d = 0; d < r.dirs; d++) {
        const s = `_l${l}${d ? '_reverse' : ''}`;
        const q = n => param(weights, prefix, n + s, true);
        const o = { wi: q('weight_ih').data, wh: q('weight_hh').data };
        o.inp = q('weight_ih').shape[1];
        o.bi = q('bias_ih') && q('bias_ih').data;
        o.bh = q('bias_hh') && q('bias_hh').data;
        r.p.push(o);
      }
    r.inputSize = r.p[0].inp;
    return r;
  }
  initialState() {
    const n = this.layers * this.dirs;
    return { h: Array.from({ length: n }, () => new Float32Array(this.hidden)), c: Array.from({ length: n }, () => new Float32Array(this.hidden)) };
  }
  /* one cell step; reads x[xo..], updates h (and c) in place */
  step(p, x, xo, h, c) {
    const H = this.hidden, G = this.gates;
    const gi = new Float32Array(G * H), gh = new Float32Array(G * H);
    matvec(p.wi, 0, G * H, p.inp, x, xo, gi, 0, p.bi, 0);
    matvec(p.wh, 0, G * H, H, h, 0, gh, 0, p.bh, 0);
    if(this.kind == 'gru') {
      for(let j = 0; j < H; j++) {
        const r = sigmoid(gi[j] + gh[j]), z = sigmoid(gi[H + j] + gh[H + j]);
        const n = Math.tanh(gi[2 * H + j] + r * gh[2 * H + j]);
        h[j] = (1 - z) * n + z * h[j];
      }
    } else {
      for(let j = 0; j < H; j++) {
        const i = sigmoid(gi[j] + gh[j]), f = sigmoid(gi[H + j] + gh[H + j]);
        const g = Math.tanh(gi[2 * H + j] + gh[2 * H + j]), o = sigmoid(gi[3 * H + j] + gh[3 * H + j]);
        c[j] = f * c[j] + i * g;
        h[j] = o * Math.tanh(c[j]);
      }
    }
  }
  /* x [T, inputSize] -> [T, hidden * dirs]; state is updated in place */
  forward(x, state = this.initialState()) {
    const H = this.hidden, D = this.dirs;
    let cur = x, inp = this.inputSize;
    for(let l = 0; l < this.layers; l++) {
      const T = cur.length / inp, out = new Float32Array(T * H * D);
      for(let d = 0; d < D; d++) {
        const k = l * D + d, p = this.p[k], h = state.h[k], c = state.c[k];
        for(let i = 0; i < T; i++) {
          const t = d ? T - 1 - i : i;
          this.step(p, cur, t * inp, h, c);
          out.set(h, t * H * D + d * H);
        }
      }
      cur = out;
      inp = H * D;
    }
    return { y: cur, state };
  }
}

/*
 * .nnw reader, see export.py for the layout. Returns { meta, tensors }
 * with tensors[name] = { shape, data: Float32Array view into buffer }.
 */
export function loadWeights(buffer) {
  const bytes = new Uint8Array(buffer);
  const magic = String.fromCharCode(...bytes.subarray(0, 4));
  if(magic != 'NNW1') throw new Error(`not an .nnw file (magic '${magic}')`);
  const len = new DataView(buffer).getUint32(4, true);
  const header = JSON.parse(String.fromCharCode(...bytes.subarray(8, 8 + len)));
  const base = 8 + len, tensors = {};
  for(const [name, t] of Object.entries(header.tensors)) {
    const count = t.shape.reduce((a, b) => a * b, 1);
    if(base + t.offset + count * 4 > buffer.byteLength) throw new Error(`tensor '${name}' runs past the end`);
    tensors[name] = { shape: t.shape, data: new Float32Array(buffer, base + t.offset, count) };
  }
  return { meta: header.meta, tensors };
}
