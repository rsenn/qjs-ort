/*
 * render-dataset.js - renders random patches and writes their features
 * and labels as one .nnw file (read in Python with nn/nnw.py).
 *
 *   QUICKJS_MODULE_PATH=/usr/local/lib/x86_64-linux-gnu/quickjs \
 *     qjsm tools/render-dataset.js <out.nnw> <count> <seed> [synth.json]
 *
 * Example i uses its own seed (seed * 1000003 + i), so a shard can be
 * regenerated alone. Tensors: features [N, T, nMels], labels [N, P]
 * (see nn/synths/params.js), midi [N]. meta holds spec and feature config.
 */
import * as std from 'std';
import { render } from '../nn/synths/stkpatch.js';
import { sampleLabels, denormalize, rng } from '../nn/synths/params.js';
import { logMel, melFilterbank, defaultConfig } from '../nn/features.js';

const [out, countArg, seedArg, specPath = new URL('../nn/synths/stkpatch.json', import.meta.url).pathname] = scriptArgs.slice(1);
if(!out || !countArg) {
  console.log('usage: render-dataset.js <out.nnw> <count> <seed> [synth.json]');
  std.exit(1);
}
const count = +countArg, seed = +seedArg || 1;
const spec = JSON.parse(std.loadFile(specPath));
const cfg = { ...defaultConfig, sampleRate: spec.sampleRate };
const fb = melFilterbank(cfg);
const P = spec.params.length, { midiMin, midiMax } = spec.protocol;

let features = null, T = 0;
const labels = new Float32Array(count * P), midis = new Float32Array(count);

for(let i = 0; i < count; i++) {
  const r = rng(seed * 1000003 + i);
  const lab = sampleLabels(spec, r);
  const midi = midiMin + Math.floor(r() * (midiMax - midiMin + 1));
  const { data, frames } = logMel(render(denormalize(spec, lab), midi, spec), cfg, fb);
  if(!features) {
    T = frames;
    features = new Float32Array(count * T * cfg.nMels);
  }
  features.set(data, i * T * cfg.nMels);
  labels.set(lab, i * P);
  midis[i] = midi;
  if(i % 500 == 499) std.err.puts(`${out}: ${i + 1}/${count}\n`);
}

const tensors = { features: [[count, T, cfg.nMels], features], labels: [[count, P], labels], midi: [[count], midis] };
const index = {};
let off = 0;
for(const [k, [shape, a]] of Object.entries(tensors)) {
  index[k] = { shape, offset: off };
  off += a.byteLength;
}
let header = JSON.stringify({ meta: { synth: spec.name, spec, features: cfg, count, seed }, tensors: index });
header += ' '.repeat((4 - ((8 + header.length) % 4)) % 4);  // ASCII only, so length == bytes

const f = std.open(out, 'wb');
const head = new ArrayBuffer(8);
new Uint8Array(head).set([78, 78, 87, 49]);
new DataView(head).setUint32(4, header.length, true);
f.write(head, 0, 8);
f.puts(header);
for(const [, a] of Object.values(tensors)) f.write(a.buffer, a.byteOffset, a.byteLength);
f.close();
console.log(`wrote ${out}: ${count} examples, features [${count}, ${T}, ${cfg.nMels}]`);
