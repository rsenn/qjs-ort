// tensor-leak-test.js - leak test for Tensor

import { Tensor } from 'ort';
import * as std from 'std';

const rounds = 1000;
for(let i = 0; i < rounds; i++) {
  const t = new Tensor('float32', new Float32Array(6), [2, 3]);
  // use it
  t.dims;
  t.type;
  t.data;
}
// force gc
std.gc();