/*
 * eval-synth.js - how well does the synth-params model match sounds?
 *
 *   QUICKJS_MODULE_PATH=/usr/local/lib/x86_64-linux-gnu/quickjs \
 *     qjsm tools/eval-synth.js [model.nnw] [test.nnw] [count] [train.nnw]
 *
 * For each test note, four parameter sets are rendered with the same
 * STK patch and MIDI note, and the log-mel distance (mean absolute
 * difference, in dB) to the target is measured:
 *   truth   the real labels, must give 0 (determinism check)
 *   model   labels predicted by the model
 *   nn      labels of the nearest training example in feature space
 *   random  freshly drawn labels
 * Also prints the mean absolute label error of the model and of nn.
 */
import * as std from 'std';
import { loadWeights } from '../nn/nn.js';
import { SynthParams } from '../nn/models/synth-params.js';
import { render } from '../nn/synths/stkpatch.js';
import { sampleLabels, denormalize, rng } from '../nn/synths/params.js';
import { logMel, melFilterbank } from '../nn/features.js';

const dir = new URL('..', import.meta.url).pathname;
const [modelPath = dir + 'out/synth-params.nnw', testPath = dir + 'data/synth/test.nnw', countArg = '100', trainPath = dir + 'data/synth/train-1.nnw'] =
  scriptArgs.slice(1);

function load(path) {
  const f = std.open(path, 'rb');
  if(!f) throw new Error(`cannot open ${path}`);
  f.seek(0, std.SEEK_END);
  const n = f.tell();
  const buf = new ArrayBuffer(n);
  f.seek(0, std.SEEK_SET);
  f.read(buf, 0, n);
  f.close();
  return buf;
}

const model = SynthParams.load(load(modelPath));
const test = loadWeights(load(testPath));
const train = loadWeights(load(trainPath));
const { spec, features: cfg } = model;
const fb = melFilterbank(cfg);
const P = spec.params.length;
const [, T, M] = test.tensors.features.shape;
const count = Math.min(+countArg, test.tensors.features.shape[0]);
const nTrain = train.tensors.features.shape[0];

/* nearest neighbour on every 4th frame of the log-mel, L1 */
const STEP = 4, FR = Math.ceil(T / STEP);
function descriptor(data, off) {
  const d = new Float32Array(FR * M);
  for(let t = 0, k = 0; t < T; t += STEP, k++) for(let m = 0; m < M; m++) d[k * M + m] = data[off + t * M + m];
  return d;
}
const trainDesc = [];
for(let i = 0; i < nTrain; i++) trainDesc.push(descriptor(train.tensors.features.data, i * T * M));
function nearest(d) {
  let best = 0, bd = Infinity;
  for(let i = 0; i < nTrain; i++) {
    const c = trainDesc[i];
    let s = 0;
    for(let k = 0; k < c.length && s < bd; k++) s += Math.abs(c[k] - d[k]);
    if(s < bd) {
      bd = s;
      best = i;
    }
  }
  return best;
}

const distance = (a, b) => {
  let s = 0;
  for(let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s / a.length;
};
const rand = rng(12345);
const dists = { truth: [], model: [], nn: [], random: [] };
const labErr = { model: 0, nn: 0 };
const tf = test.tensors.features.data, tl = test.tensors.labels.data, tm = test.tensors.midi.data;

for(let i = 0; i < count; i++) {
  const target = tf.subarray(i * T * M, (i + 1) * T * M), midi = tm[i], truth = tl.subarray(i * P, (i + 1) * P);
  const j = nearest(descriptor(tf, i * T * M));
  const cand = {
    truth,
    model: model.predict(target, midi),
    nn: train.tensors.labels.data.subarray(j * P, (j + 1) * P),
    random: sampleLabels(spec, rand)
  };
  for(const [k, lab] of Object.entries(cand)) {
    const mel = logMel(render(denormalize(spec, lab), midi, spec), cfg, fb).data;
    dists[k].push(distance(target, mel));
  }
  for(const k of ['model', 'nn']) {
    let e = 0;
    for(let p = 0; p < P; p++) e += Math.abs(cand[k][p] - truth[p]);
    labErr[k] += e / P;
  }
  if(i % 10 == 9) std.err.puts(`${i + 1}/${count}\n`);
}

const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const median = a => [...a].sort((x, y) => x - y)[a.length >> 1];
console.log(`${count} test notes, ${nTrain} nn candidates, log-mel L1 distance`);
for(const k of Object.keys(dists)) console.log(`${k.padEnd(7)} mean ${mean(dists[k]).toFixed(3)}  median ${median(dists[k]).toFixed(3)}`);
console.log(`label L1 (not comparable across choice params): model ${(labErr.model / count).toFixed(4)}, nn ${(labErr.nn / count).toFixed(4)}`);
