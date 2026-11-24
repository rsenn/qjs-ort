/*
 * gru-loop.js - runs a sequence model one time step at a time.
 *
 * The model (see make-model.py) is a GRU cell: x [1, 1, 8] and the hidden
 * state h [1, 1, 16] go in, y and the new hidden state hn come out. The
 * script feeds hn back as h, the way a realtime loop runs a model block
 * by block, and compares every y with what ONNX Runtime for Python gave.
 *
 * from the project root:
 *   QUICKJS_MODULE_PATH=build/x86_64-linux-debug qjsm examples/gru/gru-loop.js
 */
import * as std from 'std';
import { Session, Tensor } from 'ort';

const modelPath = scriptArgs[1] ?? 'examples/gru/gru-step.onnx';
const expectedPath = scriptArgs[2] ?? 'examples/gru/gru-expected.json';

const expected = JSON.parse(std.loadFile(expectedPath));
const session = new Session(modelPath, { intraOpThreads: 1 });

console.log('inputs :', JSON.stringify(session.inputs));
console.log('outputs:', JSON.stringify(session.outputs));

/* the hidden state lives in one Float32Array; hn is copied back into it */
const hidden = new Float32Array(expected.hidden);
const x = new Float32Array(expected.in);
const xTensor = new Tensor('float32', x, [1, 1, expected.in]);
const hTensor = new Tensor('float32', hidden, [1, 1, expected.hidden]);

let maxDiff = 0;
const t0 = Date.now();

for(let step = 0; step < expected.steps; step++) {
  x.set(expected.x[step]);

  const out = session.run({ x: xTensor, h: hTensor });
  const y = out.y.data;

  /* a view of the output stays valid, but the next run makes a new one: copy what must last */
  hidden.set(out.hn.data);

  for(let i = 0; i < y.length; i++)
    maxDiff = Math.max(maxDiff, Math.abs(y[i] - expected.y[step][i]));

  console.log(`step ${step}: y[0..3] =`, Array.from(y.subarray(0, 4), v => v.toFixed(5)).join(' '));
}

console.log(`${expected.steps} steps in ${Date.now() - t0} ms, max difference to Python ${maxDiff}`);

if(!(maxDiff < 1e-5)) {
  console.log('FAIL');
  std.exit(1);
}

console.log('ok');
