// tensor-test.js - throw-away test for Tensor, float32 only

import { Tensor } from 'ort';

// basic construct
const arr = new Float32Array(6);
for(let i = 0; i < 6; i++) arr[i] = i + 1;
const t = new Tensor('float32', arr, [2, 3]);
console.log('type:', t.type);
console.log('dims:', t.dims);
console.log('data === arr:', t.data === arr);
console.log('data[0]:', t.data[0]);
console.log('data[5]:', t.data[5]);

// dims mismatch throws RangeError
try {
  new Tensor('float32', new Float32Array(4), [3, 3]);  // 9 elements vs 4
  console.log('FAIL: dims mismatch should throw');
} catch(e) {
  console.log('dims mismatch throws RangeError:', e instanceof RangeError);
}

// wrong typed array throws TypeError
try {
  new Tensor('float32', new Int32Array(4), [4]);
  console.log('FAIL: wrong typed array should throw');
} catch(e) {
  console.log('wrong typed array throws TypeError:', e instanceof TypeError);
}

// unknown type throws TypeError
try {
  new Tensor('int7', new Float32Array(4), [4]);
  console.log('FAIL: unknown type should throw');
} catch(e) {
  console.log('unknown type throws TypeError:', e instanceof TypeError);
}

// subclass works
class MyTensor extends Tensor {}
const mt = new MyTensor('float32', new Float32Array(2), [2]);
console.log('subclass works:', mt instanceof MyTensor && mt instanceof Tensor);
console.log('subclass type:', mt.type);
console.log('subclass dims:', mt.dims);
console.log('subclass data:', mt.data);

// toStringTag
console.log('toStringTag:', Object.prototype.toString.call(t));

// zero-length dims (scalar)
const scalar = new Tensor('float32', new Float32Array([42]), []);
console.log('scalar dims:', scalar.dims);
console.log('scalar data[0]:', scalar.data[0]);

// zero dim with data should throw RangeError
try {
  new Tensor('float32', new Float32Array(3), [0, 3]);
  console.log('FAIL: zero dim with data should throw');
} catch(e) {
  console.log('zero dim with data throws:', e instanceof RangeError);
}

// zero dim with zero data should work
try {
  const t = new Tensor('float32', new Float32Array(0), [0, 3]);
  console.log('zero dim zero data works:', t.dims[0] === 0 && t.dims[1] === 3);
} catch(e) {
  console.log('FAIL zero dim zero data:', e.message);
}

console.log('All checks passed');