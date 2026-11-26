/*
 * params.js - parameter spec handling shared by dataset generation,
 * training labels and inference. Pure JS.
 *
 * A label vector has one slot per spec parameter, in spec order: float
 * parameters hold the normalised value 0..1 (log scale where the spec
 * says so), choice parameters hold the class index.
 */

export function denormalize(spec, labels) {
  const v = {};
  spec.params.forEach((p, i) => {
    const u = labels[i];
    if(p.type == 'choice') v[p.name] = Math.round(u);
    else if(p.scale == 'log') v[p.name] = p.min * Math.pow(p.max / p.min, u);
    else v[p.name] = p.min + (p.max - p.min) * u;
  });
  return v;
}

export function normalize(spec, values) {
  return Float32Array.from(spec.params, p => {
    const x = values[p.name];
    if(p.type == 'choice') return x;
    return p.scale == 'log' ? Math.log(x / p.min) / Math.log(p.max / p.min) : (x - p.min) / (p.max - p.min);
  });
}

/* seeded PRNG (mulberry32), returns a function giving [0, 1) */
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/*
 * random label vector. Switch-like parameters get a share of exact zeros
 * (level 0 means the module is off), so the model learns "absent".
 */
export function sampleLabels(spec, r) {
  const off = { osc2Level: 0.3, subLevel: 0.4, noiseLevel: 0.5, echoMix: 0.5, reverbMix: 0.5, envAmount: 0.15 };
  return Float32Array.from(spec.params, p => {
    if(p.type == 'choice') return Math.floor(r() * p.choices.length);
    if(p.name in off && r() < off[p.name]) return 0;
    return r();
  });
}
