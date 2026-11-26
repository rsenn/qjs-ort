/*
 * features.js - log-mel spectrogram front end, pure JS.
 *
 * The same code makes the training features (tools/render-dataset.js)
 * and the inference features (tools/match-synth.js): never port it to
 * Python. Config goes into the .nnw meta so a model carries its own.
 */

export const defaultConfig = { sampleRate: 22050, nFft: 1024, hop: 512, nMels: 48, fMin: 30, fMax: 10000, floorDb: -80 };

const hz2mel = f => 2595 * Math.log10(1 + f / 700);
const mel2hz = m => 700 * (Math.pow(10, m / 2595) - 1);

const plans = new Map();

/* bit reversal and twiddle tables for one size, built once */
function plan(n) {
  let p = plans.get(n);
  if(p) return p;
  const rev = new Uint32Array(n), cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
  for(let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for(; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    rev[i] = j;
  }
  for(let k = 0; k < n / 2; k++) {
    cos[k] = Math.cos((-2 * Math.PI * k) / n);
    sin[k] = Math.sin((-2 * Math.PI * k) / n);
  }
  plans.set(n, (p = { rev, cos, sin }));
  return p;
}

/* in-place radix-2 complex FFT, length a power of two */
export function fft(re, im) {
  const n = re.length, { rev, cos, sin } = plan(n);
  for(let i = 1; i < n; i++) {
    const j = rev[i];
    if(i < j) {
      let t = re[i];
      re[i] = re[j];
      re[j] = t;
      t = im[i];
      im[i] = im[j];
      im[j] = t;
    }
  }
  for(let len = 2; len <= n; len <<= 1) {
    const half = len >> 1, step = n / len;
    for(let i = 0; i < n; i += len)
      for(let k = 0, w = 0; k < half; k++, w += step) {
        const a = i + k, b = a + half, cr = cos[w], ci = sin[w];
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
      }
  }
}

/*
 * power spectrum of a real signal of length n (power of two) in one FFT
 * of n/2 points: the even samples are the real part, the odd samples the
 * imaginary part, then the two halves are unpacked. Bins 0..n/2 go to
 * power, re and im are scratch of length n/2 (re/im in fft() terms).
 */
export function rfftPower(x, n, power, re, im) {
  const M = n >> 1;
  const key = 'r' + n;
  let tw = plans.get(key);
  if(!tw) {
    tw = { c: new Float64Array(M + 1), s: new Float64Array(M + 1) };
    for(let k = 0; k <= M; k++) {
      tw.c[k] = Math.cos((-2 * Math.PI * k) / n);
      tw.s[k] = Math.sin((-2 * Math.PI * k) / n);
    }
    plans.set(key, tw);
  }
  for(let i = 0; i < M; i++) {
    re[i] = x[2 * i];
    im[i] = x[2 * i + 1];
  }
  fft(re, im);
  for(let k = 0; k <= M; k++) {
    const j = k % M, m = (M - k) % M;
    const ar = re[j], ai = im[j], br = re[m], bi = im[m];
    const er = (ar + br) / 2, ei = (ai - bi) / 2, or = (ai + bi) / 2, oi = -(ar - br) / 2;
    const xr = er + tw.c[k] * or - tw.s[k] * oi, xi = ei + tw.c[k] * oi + tw.s[k] * or;
    power[k] = xr * xr + xi * xi;
  }
}

/* triangular mel filterbank [nMels][nFft/2+1], area normalised */
export function melFilterbank(c) {
  const bins = c.nFft / 2 + 1, pts = new Float64Array(c.nMels + 2);
  const lo = hz2mel(c.fMin), hi = hz2mel(c.fMax);
  for(let i = 0; i < pts.length; i++) pts[i] = mel2hz(lo + ((hi - lo) * i) / (c.nMels + 1));
  const fb = [];
  for(let m = 0; m < c.nMels; m++) {
    const row = new Float32Array(bins), [a, b, d] = [pts[m], pts[m + 1], pts[m + 2]];
    for(let k = 0; k < bins; k++) {
      const f = (k * c.sampleRate) / c.nFft;
      if(f > a && f < b) row[k] = (f - a) / (b - a);
      else if(f >= b && f < d) row[k] = (d - f) / (d - b);
      row[k] *= 2 / (d - a);
    }
    let lo = 0, hi = bins;
    while(lo < bins && row[lo] == 0) lo++;
    while(hi > lo && row[hi - 1] == 0) hi--;
    row.lo = lo;
    row.hi = hi;
    fb.push(row);
  }
  return fb;
}

/*
 * logMel(samples, config) -> { data: Float32Array [T * nMels], frames }
 * Hann window, no centre padding, power spectrum, mel, 10*log10 clamped
 * at floorDb. Levels are absolute (not normalised per clip) because
 * loudness carries information for this task.
 */
export function logMel(x, c = defaultConfig, fb = melFilterbank(c)) {
  const n = c.nFft, bins = n / 2 + 1;
  const T = 1 + Math.floor(Math.max(0, x.length - n) / c.hop);
  const win = Float64Array.from({ length: n }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n));
  const re = new Float64Array(n / 2), im = new Float64Array(n / 2);
  const buf = new Float64Array(n), pw = new Float64Array(bins);
  const out = new Float32Array(T * c.nMels);
  for(let t = 0; t < T; t++) {
    const o = t * c.hop;
    for(let i = 0; i < n; i++) buf[i] = o + i < x.length ? x[o + i] * win[i] : 0;
    rfftPower(buf, n, pw, re, im);
    for(let m = 0; m < c.nMels; m++) {
      let s = 0;
      const row = fb[m];
      for(let k = row.lo; k < row.hi; k++) s += row[k] * pw[k];
      out[t * c.nMels + m] = Math.max(c.floorDb, 10 * Math.log10(s + 1e-10));
    }
  }
  return { data: out, frames: T };
}
