/* tests/u2netp-poc.js
 * proof of concept: u2netp.onnx via ort module and opencv dnn, compare outputs
 *
 * u2netp input name: input.1
 * u2netp output names: 1959, 1960, 1961, 1962, 1963, 1964, 1965
 * first output (1959) is the probability map 320x320
 */

import * as std from 'std';
import * as cv from 'opencv';
import { Session, Tensor } from 'ort';
import { performance } from 'perf_hooks';
import process from 'process';

/* helper: elapsed ms since t */
function ms(t) {
  return Math.round(performance.now() - t);
}

/* helper: max absolute difference between two Float32Arrays of same length */
function maxAbsDiff(a, b) {
  let maxd = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > maxd) maxd = d;
  }
  return maxd;
}

/* helper: min/max of Float32Array */
function minMax(arr) {
  let min = arr[0];
  let max = arr[0];
  for (let i = 1; i < arr.length; i++) {
    const v = arr[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return { min, max };
}

/* helper: normalise ORT output the way runSalient does and write 8-bit matte */
function writeMatte(ortOut, outPath) {
  const { min, max } = minMax(ortOut);
  const range = Math.max(max - min, 1e-6);

  const norm = new Float32Array(ortOut.length);
  for (let i = 0; i < ortOut.length; i++) {
    norm[i] = (ortOut[i] - min) / range;
  }

  const mat = new cv.Mat(320, 320, cv.CV_32F, norm.slice().buffer);
  const out8 = new cv.Mat();
  mat.convertTo(out8, cv.CV_8U, 255);
  cv.imwrite(outPath, out8);
}

/* main */

const modelDefault = '/home/roman/Sources/quickjs/qjs-opencv/models/u2netp.onnx';
const pictureDefault = '/home/roman/Sources/quickjs/qjs-opencv/tests/test_linesegmentdetector.jpg';

const modelsDir = process.env.WB_MODELS_DIR;
const model = scriptArgs[2] ?? (modelsDir ? modelsDir + '/u2netp.onnx' : modelDefault);
const picture = scriptArgs[1] ?? pictureDefault;

console.log('picture:', picture);
console.log('model:', model);

/* 1. session, print names */
const session = new Session(model);
console.log('inputNames:', session.inputNames);
console.log('outputNames:', session.outputNames);

/* 2. image and blob */
const img = cv.imread(picture);
const blob = cv.blobFromImage(img, 0.019, new cv.Size(320, 320), new cv.Scalar(123.6, 116.2, 103.5), true, false);
console.log('blob.data32F.length:', blob.data32F.length);

/* 3. ORT path with timing */
const tOrt = performance.now();
const inputTensor = new Tensor('float32', blob.data32F, [1, 3, 320, 320]);
const ortOutMap = session.run({ [session.inputNames[0]]: inputTensor });
const ortOut = ortOutMap[session.outputNames[0]].data;
const ortTime = ms(tOrt);
console.log('ORT time:', ortTime, 'ms');
console.log('ORT output length:', ortOut.length);

/* 4. OpenCV dnn path with timing */
const tLoad = performance.now();
const net = cv.readNetFromONNX(model, cv.dnn.ENGINE_AUTO);
console.log('OpenCV load time:', ms(tLoad), 'ms');

const tCv = performance.now();
net.setInput(blob);
const cvOut = net.forward(net.getUnconnectedOutLayersNames())[0].data32F;
const cvTime = ms(tCv);
console.log('OpenCV time:', cvTime, 'ms');
console.log('OpenCV output length:', cvOut.length);

/* 5. compare */
const diff = maxAbsDiff(ortOut, cvOut);
const ortMinMax = minMax(ortOut);
const cvMinMax = minMax(cvOut);
console.log('max abs diff:', diff);
console.log('ORT min/max:', ortMinMax.min, ortMinMax.max);
console.log('OpenCV min/max:', cvMinMax.min, cvMinMax.max);
console.log('shape: 320x320 =', 320 * 320);

if (diff >= 1e-3) {
  console.log('FAIL: max abs diff', diff, '>= 1e-3');
  std.exit(1);
}

/* 6. write matte */
writeMatte(ortOut, '.tmp/u2netp-matte.png');
console.log('matte written to .tmp/u2netp-matte.png');

/* 7. loop 25 times, check determinism, print total time */
const tLoop = performance.now();
let firstOut = ortOut;
for (let i = 1; i < 25; i++) {
  const outMap = session.run({ [session.inputNames[0]]: inputTensor });
  const out = outMap[session.outputNames[0]].data;
  for (let j = 0; j < out.length; j++) {
    if (out[j] !== firstOut[j]) {
      console.log('FAIL: run', i, 'differs at index', j);
      std.exit(1);
    }
  }
}
const loopTime = ms(tLoop);
console.log('25 runs total time:', loopTime, 'ms');

/* 8. success */
console.log('PoC passed');
std.exit(0);