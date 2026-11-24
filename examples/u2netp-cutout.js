/*
 * u2netp-cutout.js - cuts the salient object out of a picture.
 *
 * u2netp gives a 320x320 probability map. The script scales it to the size
 * of the picture and stores it as the alpha channel of a PNG, so that
 * everything but the object turns transparent.
 *
 *   QUICKJS_MODULE_PATH="build/x86_64-linux-debug;/usr/local/lib/x86_64-linux-gnu/quickjs" \
 *     qjsm examples/u2netp-cutout.js [picture] [output.png] [model.onnx]
 *
 * The model is not part of this repository, it is the one qjs-opencv
 * uses (WB_MODELS_DIR names the directory that holds u2netp.onnx).
 */
import * as std from 'std';
import * as cv from 'opencv';
import { Session, Tensor } from 'ort';

const SIZE = 320;

const modelsDir = std.getenv('WB_MODELS_DIR') ?? '/home/roman/Sources/quickjs/qjs-opencv/models';
const picture = scriptArgs[1] ?? '/home/roman/Sources/quickjs/qjs-opencv/tests/test_linesegmentdetector.jpg';
const output = scriptArgs[2] ?? 'cutout.png';
const modelPath = scriptArgs[3] ?? `${modelsDir}/u2netp.onnx`;

const img = cv.imread(picture);

if(img.empty()) {
  console.log(`cannot read ${picture}`);
  std.exit(1);
}

/* the same preprocessing as qjs-opencv's runSalient: scale, mean, swap R and B */
const blob = cv.blobFromImage(img, 0.019, new cv.Size(SIZE, SIZE), new cv.Scalar(123.6, 116.2, 103.5), true, false);

const session = new Session(modelPath);
const input = new Tensor('float32', blob.data32F, [1, 3, SIZE, SIZE]);
const t0 = Date.now();
const out = session.run({ [session.inputNames[0]]: input });
const prob = out[session.outputNames[0]];

console.log(`${modelPath}: ${Date.now() - t0} ms, output ${prob.dims.join('x')}`);

/* stretch the map to 0..1, then to 0..255 and to the size of the picture */
const map = new cv.Mat(SIZE, SIZE, cv.CV_32F, prob.data.slice().buffer);
const { minVal, maxVal } = cv.minMaxLoc(map);
const range = Math.max(maxVal - minVal, 1e-6);
const stretched = new cv.Mat();
const matte = new cv.Mat();
const alpha = new cv.Mat();
const bgra = new cv.Mat();

map.convertTo(stretched, cv.CV_32F, 1 / range, -minVal / range);
stretched.convertTo(matte, cv.CV_8U, 255);
cv.resize(matte, alpha, new cv.Size(img.cols, img.rows), 0, 0, cv.INTER_LINEAR);

const [b, g, r] = cv.split(img);

cv.merge([b, g, r, alpha], bgra);

if(!cv.imwrite(output, bgra)) {
  console.log(`cannot write ${output}`);
  std.exit(1);
}

console.log(`${img.cols}x${img.rows} picture, cutout written to ${output}`);
