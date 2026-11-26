// synth-params-test.js - JS inference vs the PyTorch outputs stored by train/synth-params.py
//   qjsm tests/synth-params-test.js out/synth-params.nnw   (reads <model>.check.nnw too)
import * as std from 'std';
import { SynthParams } from '../nn/models/synth-params.js';
import { loadWeights } from '../nn/nn.js';

function readBuf(path) {
  const f = std.open(path, 'rb');
  f.seek(0, std.SEEK_END);
  const n = f.tell();
  f.seek(0, std.SEEK_SET);
  const b = new ArrayBuffer(n);
  f.read(b, 0, n);
  f.close();
  return b;
}

const path = scriptArgs[1];
if(!path) {
  console.log('usage: synth-params-test.js <model.nnw>');
  std.exit(1);
}
const model = SynthParams.load(readBuf(path)), check = loadWeights(readBuf(path + '.check.nnw'));
const { features, midi, out } = check.tensors;
const per = features.shape[1] * features.shape[2], nOut = out.shape[1];
let worst = 0, t0 = Date.now();
for(let i = 0; i < features.shape[0]; i++) {
  const got = model.forward(features.data.subarray(i * per, (i + 1) * per), midi.data[i]);
  for(let k = 0; k < nOut; k++) worst = Math.max(worst, Math.abs(got[k] - out.data[i * nOut + k]));
}
console.log(`max diff ${worst.toExponential(2)}, ${((Date.now() - t0) / features.shape[0]).toFixed(1)} ms per inference`);
if(!(worst < 1e-3)) throw new Error('JS inference differs from PyTorch');
console.log('ok');
