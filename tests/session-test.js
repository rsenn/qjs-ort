import { Session, Tensor } from 'ort';

const modelPath = '/home/roman/Sources/quickjs/qjs-opencv/models/u2netp.onnx';
const textFile = '/home/roman/Sources/quickjs/qjs-ort/LICENSE';

/* Test 1: load model and print names */
const session = new Session(modelPath);
console.log('inputNames:', session.inputNames);
console.log('outputNames:', session.outputNames);

/* Test 2: missing path throws Error with code 3 */
try {
    new Session('/nonexistent/path.onnx');
    console.log('FAIL: missing path should throw');
} catch (e) {
    console.log('missing path throws:', e.message, 'code:', e.code);
    if (e.code !== 3) console.log('FAIL: expected code 3, got', e.code);
}

/* Test 3: text file throws Error with numeric code */
try {
    new Session(textFile);
    console.log('FAIL: text file should throw');
} catch (e) {
    console.log('text file throws:', e.message, 'code:', e.code);
    if (typeof e.code !== 'number') console.log('FAIL: expected numeric code');
}

/* Test 4: release() twice */
session.release();
session.release();
console.log('release() twice: OK');

/* Test 5: names still work after release */
console.log('inputNames after release:', session.inputNames);
console.log('outputNames after release:', session.outputNames);

/* Test 6: new Session() throws TypeError */
try {
    new Session();
    console.log('FAIL: new Session() should throw');
} catch (e) {
    console.log('new Session() throws:', e.name, e.message);
    if (e.name !== 'TypeError') console.log('FAIL: expected TypeError');
}

/* Test 7: new Session(42) throws TypeError */
try {
    new Session(42);
    console.log('FAIL: new Session(42) should throw');
} catch (e) {
    console.log('new Session(42) throws:', e.name, e.message);
    if (e.name !== 'TypeError') console.log('FAIL: expected TypeError');
}

/* Test 8: Symbol.toStringTag */
const session2 = new Session(modelPath);
console.log('toStringTag:', Object.prototype.toString.call(session2));
session2.release();

console.log('All tests completed');