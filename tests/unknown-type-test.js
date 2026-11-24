import { Tensor } from 'ort';

try {
  new Tensor('int7', new Float32Array(4), [4]);
} catch(e) {
  console.log('error message:', e.message);
}