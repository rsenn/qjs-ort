/*
 * stkpatch.js - the offline STK patch that Case 4 learns to match.
 *
 * render(values, midi) -> Float32Array, deterministic: the same values
 * and note always give the same samples. values are in real units
 * (see stkpatch.json, params are denormalised by params.js); choice
 * parameters are given as an index.
 *
 * Needs the qjs 'stk' module, so it runs on qjsm only.
 */
import { Stk, StkFrames, BlitSaw, BlitSquare, SineWave, Noise, ADSR, BiQuad, Echo, JCRev } from 'stk';

const BLOCK = 32;
const mtof = m => 440 * Math.pow(2, (m - 69) / 12);

export function render(v, midi, spec) {
  const sr = spec.sampleRate, p = spec.protocol;
  const N = Math.round(p.duration * sr), gate = Math.round(p.gate * sr);
  const f0 = mtof(midi);

  Stk.setSampleRate(sr);

  const frames = () => new StkFrames(N, 1);
  const view = f => new Float64Array(f.buffer);
  const gen = (g, hz) => {
    g.setFrequency(hz);
    if(!(g instanceof SineWave)) g.setHarmonics(0);
    const f = frames();
    g.tick(f, 0);
    return view(f);
  };

  /* oscillators */
  const saw = gen(new BlitSaw(), f0), sq = gen(new BlitSquare(), f0);
  const saw2 = v.osc2Level > 0 ? gen(new BlitSaw(), f0 * Math.pow(2, v.osc2Detune / 1200)) : null;
  const sub = v.subLevel > 0 ? gen(new SineWave(), f0 / 2) : null;
  const noise = v.noiseLevel > 0 ? view((() => { const f = frames(); new Noise(1).tick(f, 0); return f; })()) : null;

  const mix = new Float64Array(N);
  for(let i = 0; i < N; i++) {
    let s = saw[i] * (1 - v.wave) + sq[i] * v.wave;
    if(saw2) s += saw2[i] * v.osc2Level;
    if(sub) s += sub[i] * v.subLevel;
    if(noise) s += noise[i] * v.noiseLevel;
    mix[i] = s * 0.4;
  }

  /* envelopes: amplitude and filter */
  const adsr = (a, d, s, r) => {
    const e = new ADSR(), out = new Float64Array(N);
    e.setAllTimes(a, d, s, r);
    e.keyOn();
    for(let i = 0; i < N; i++) {
      if(i == gate) e.keyOff();
      out[i] = e.tick();
    }
    return out;
  };
  const amp = adsr(v.attack, v.decay, v.sustain, v.release);
  const fenv = adsr(0.002, v.envDecay, 0, v.release);

  /* filter, cutoff updated once per block */
  const filt = new BiQuad(), blk = new StkFrames(BLOCK, 1), bv = new Float64Array(blk.buffer);
  const out = new Float64Array(N);
  const setters = ['setLowPass', 'setBandPass', 'setHighPass'];
  const nyq = sr * 0.45;
  for(let o = 0; o < N; o += BLOCK) {
    const fc = Math.min(nyq, v.cutoff * Math.pow(2, v.envAmount * fenv[o]));
    filt[setters[v.filterType]](fc, v.resonance);
    const n = Math.min(BLOCK, N - o);
    bv.fill(0);
    for(let i = 0; i < n; i++) bv[i] = mix[o + i];
    filt.tick(blk, 0);
    for(let i = 0; i < n; i++) out[o + i] = bv[i] * amp[o + i];
  }

  /* effects */
  const fx = new StkFrames(N, 1), fv = view(fx);
  fv.set(out);
  if(v.echoMix > 0) {
    const e = new Echo(Math.round(0.5 * sr));
    e.setDelay(Math.round(v.echoTime * sr));
    e.setEffectMix(v.echoMix);
    e.tick(fx, 0);
  }
  if(v.reverbMix > 0) {
    /* JCRev is stereo: tick(frames) on a mono buffer gives garbage, so tick per sample and fold to mono */
    const r = new JCRev(v.reverbTime);
    r.setEffectMix(v.reverbMix);
    for(let i = 0; i < N; i++) {
      const [l, rr] = r.tick(fv[i]);
      fv[i] = (l + rr) / 2;
    }
  }

  const res = new Float32Array(N);
  for(let i = 0; i < N; i++) {
    if(!Number.isFinite(fv[i])) throw new Error('stkpatch: non-finite sample');
    res[i] = fv[i];
  }
  return res;
}
