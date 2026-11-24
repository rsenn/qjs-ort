import { Tensor } from 'ort';

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