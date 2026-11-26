// nn-features-test.js - FFT, mel front end and parameter helpers of nn/
import { rfftPower, logMel, melFilterbank, defaultConfig } from '../nn/features.js';
import { sampleLabels, denormalize, normalize, rng } from '../nn/synths/params.js';
import * as std from 'std';

let fails = 0;
const ok = (name, cond, extra = '') => {
  if(!cond) fails++;
  console.log(`${cond ? 'ok  ' : 'FAIL'} ${name} ${extra}`);
};

// rfftPower against a direct DFT
{
  const n = 64, r = rng(3), x = Float64Array.from({ length: n }, () => r() - 0.5);
  const pw = new Float64Array(n / 2 + 1);
  rfftPower(x, n, pw, new Float64Array(n / 2), new Float64Array(n / 2));
  let d = 0;
  for(let k = 0; k <= n / 2; k++) {
    let re = 0, im = 0;
    for(let i = 0; i < n; i++) {
      re += x[i] * Math.cos((2 * Math.PI * k * i) / n);
      im -= x[i] * Math.sin((2 * Math.PI * k * i) / n);
    }
    d = Math.max(d, Math.abs(re * re + im * im - pw[k]));
  }
  ok('rfftPower vs DFT', d < 1e-9, d.toExponential(1));
}

// a sine at 1 kHz peaks in the mel band that contains 1 kHz
{
  const c = defaultConfig, fb = melFilterbank(c), x = new Float32Array(c.sampleRate / 2);
  for(let i = 0; i < x.length; i++) x[i] = 0.5 * Math.sin((2 * Math.PI * 1000 * i) / c.sampleRate);
  const { data, frames } = logMel(x, c, fb);
  let best = 0;
  for(let m = 1; m < c.nMels; m++) if(data[m] > data[best]) best = m;
  const bin = Math.round((1000 * c.nFft) / c.sampleRate);
  ok('mel peak holds 1 kHz', fb[best][bin] > 0, `band ${best}, ${frames} frames`);
  ok('mel size', data.length == frames * c.nMels);
}

// params: normalize(denormalize(x)) == x, seeded sampling repeats
{
  const spec = JSON.parse(std.loadFile(new URL('../nn/synths/stkpatch.json', import.meta.url).pathname));
  const a = sampleLabels(spec, rng(9)), b = sampleLabels(spec, rng(9)), back = normalize(spec, denormalize(spec, a));
  ok('sampleLabels seeded', a.every((v, i) => v === b[i]));
  ok('normalize roundtrip', a.every((v, i) => Math.abs(v - back[i]) < 1e-5));
}

console.log(fails ? `${fails} FAILED` : 'all passed');
if(fails) throw new Error('failures');
