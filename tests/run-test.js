import { Session, Tensor } from 'ort';

const modelPath = '/home/roman/Sources/quickjs/qjs-opencv/models/u2netp.onnx';

/* Test 1: basic run and check output */
const session = new Session(modelPath);
console.log('inputNames:', session.inputNames);
console.log('outputNames:', session.outputNames);

const input = new Tensor('float32', new Float32Array(1 * 3 * 320 * 320), [1, 3, 320, 320]);
const out = session.run({ [session.inputNames[0]]: input });

const outputNames = session.outputNames;
console.log('output keys:', Object.keys(out));
console.log('expected outputNames:', outputNames);
console.log('keys match:', JSON.stringify(Object.keys(out).sort()) === JSON.stringify(outputNames.sort()));

const firstOut = out[outputNames[0]];
console.log('first output type:', firstOut.type);
console.log('first output dims:', firstOut.dims);
console.log('first output data.length:', firstOut.data.length);
console.log('expected data.length:', 1 * 1 * 320 * 320);

let allFinite = true;
let allInRange = true;
for(let i = 0; i < firstOut.data.length; i++) {
  const v = firstOut.data[i];
  if(!Number.isFinite(v)) allFinite = false;
  if(v < 0 || v > 1) allInRange = false;
}
console.log('all values finite:', allFinite);
console.log('all values in [0,1]:', allInRange);

/* Test 2: running twice gives equal first outputs */
const out2 = session.run({ [session.inputNames[0]]: input });
const firstOut2 = out2[outputNames[0]];
let equal = true;
for(let i = 0; i < firstOut.data.length; i++) {
  if(firstOut.data[i] !== firstOut2.data[i]) {
    equal = false;
    break;
  }
}
console.log('two runs equal:', equal);

/* Test 3: run with outputNames ['1959'] returns only that key */
const out3 = session.run({ [session.inputNames[0]]: input }, ['1959']);
console.log('outputNames ["1959"] keys:', Object.keys(out3));
console.log('has only 1959:', Object.keys(out3).length === 1 && Object.keys(out3)[0] === '1959');

/* Test 4: data view stays valid after result and tensor dropped (gc not available in qjsm) */
{
  const out4 = session.run({ [session.inputNames[0]]: input });
  const t = out4[outputNames[0]];
  const view = t.data;
  const firstVal = view[0];
  // drop result and tensor - out4 and t go out of scope
  // view should still be accessible here
  console.log('data view valid immediately after scope:', view[0] === firstVal);
}
console.log('data view test completed (gc not available in qjsm)');

/* Test 5: error cases */
console.log('\n--- Error cases ---');

// run() with no argument
try {
  session.run();
  console.log('FAIL: run() with no argument should throw');
} catch(e) {
  console.log('run() no args throws:', e.name, e.message);
  if(e.name !== 'TypeError') console.log('FAIL: expected TypeError');
}

// run({})
try {
  session.run({});
  console.log('FAIL: run({}) should throw');
} catch(e) {
  console.log('run({}) throws:', e.name, e.message);
  if(e.name !== 'TypeError') console.log('FAIL: expected TypeError');
}

// run({ 'input.1': 42 })
try {
  session.run({ 'input.1': 42 });
  console.log('FAIL: run({input.1: 42}) should throw');
} catch(e) {
  console.log('run({input.1: 42}) throws:', e.name, e.message);
  if(e.name !== 'TypeError') console.log('FAIL: expected TypeError');
}

// unknown input name
try {
  session.run({ 'unknown': input });
  console.log('FAIL: run({unknown}) should throw');
} catch(e) {
  console.log('run({unknown}) throws:', e.name, e.message);
  if(e.name !== 'TypeError') console.log('FAIL: expected TypeError');
}

// Tensor with wrong dims [1,3,10,10] - must throw Error with numeric code, not crash
try {
  const badInput = new Tensor('float32', new Float32Array(1 * 3 * 10 * 10), [1, 3, 10, 10]);
  session.run({ [session.inputNames[0]]: badInput });
  console.log('FAIL: run with wrong dims should throw');
} catch(e) {
  console.log('run with wrong dims throws:', e.name, e.message, 'code:', e.code);
  if(e.name !== 'Error') console.log('FAIL: expected Error');
  if(typeof e.code !== 'number') console.log('FAIL: expected numeric code');
}

// unknown output name in second argument
try {
  session.run({ [session.inputNames[0]]: input }, ['nonexistent']);
  console.log('FAIL: run with unknown output name should throw');
} catch(e) {
  console.log('run with unknown output throws:', e.name, e.message);
  if(e.name !== 'TypeError') console.log('FAIL: expected TypeError');
}

// run after session.release() throws Error session released
session.release();
try {
  session.run({ [session.inputNames[0]]: input });
  console.log('FAIL: run after release should throw');
} catch(e) {
  console.log('run after release throws:', e.name, e.message);
  if(e.name !== 'Error') console.log('FAIL: expected Error');
}

console.log('\nAll tests completed');