import { Session, Tensor } from 'ort';
import * as std from 'std';

/* protobuf helpers: all lengths < 128 so one-byte length */
function vint(field, n) {
    const tag = (field << 3) | 0;
    const out = new Uint8Array(2);
    out[0] = tag;
    out[1] = n;
    return out;
}

function str(field, text) {
    const tag = (field << 3) | 2;
    const bytes = new TextEncoder().encode(text);
    const out = new Uint8Array(2 + bytes.length);
    out[0] = tag;
    out[1] = bytes.length;
    out.set(bytes, 2);
    return out;
}

/* concatenate field encodings into raw message content (no outer tag/length) */
function rawMsg(...parts) {
    let total = 0;
    for (const p of parts) total += p.length;
    const out = new Uint8Array(total);
    let off = 0;
    for (const p of parts) {
        out.set(p, off);
        off += p.length;
    }
    return out;
}

/* wrap a complete message as a field (tag + length + message) */
function fieldMsg(field, message) {
    const tag = (field << 3) | 2;
    const out = new Uint8Array(2 + message.length);
    out[0] = tag;
    out[1] = message.length;
    out.set(message, 2);
    return out;
}

/* build the model */
const dim = vint(1, 3);                          /* Dimension { dim_value: 3 } */
const shape = rawMsg(fieldMsg(1, dim));          /* TensorShapeProto { dim: [dim] } */
const tensorType = rawMsg(vint(1, 1), fieldMsg(2, shape)); /* TypeProto.Tensor { elem_type: 1, shape } */
const typeProto = rawMsg(fieldMsg(1, tensorType)); /* TypeProto { tensor_type } */

/* ValueInfoProto: name (field 1) + type (field 2: TypeProto) - as raw message */
const valueInfoA = rawMsg(str(1, 'a'), fieldMsg(2, typeProto));
const valueInfoB = rawMsg(str(1, 'b'), fieldMsg(2, typeProto));
const valueInfoC = rawMsg(str(1, 'c'), fieldMsg(2, typeProto));

/* NodeProto: input (field 1 repeated), output (field 2), name (field 3), op_type (field 4) - as raw message */
const node = rawMsg(
    str(1, 'a'), str(1, 'b'),
    str(2, 'c'),
    str(3, 'add'),
    str(4, 'Add')
);

/* GraphProto: node (field 1), name (field 2), input (field 11 repeated), output (field 12) */
const graph = rawMsg(
    fieldMsg(1, node),
    fieldMsg(2, str(2, 'g')),
    fieldMsg(11, valueInfoA),
    fieldMsg(11, valueInfoB),
    fieldMsg(12, valueInfoC)
);

/* OperatorSetIdProto: version (field 2) - domain omitted */
const opset = rawMsg(vint(2, 13));

/* ModelProto: ir_version (field 1), graph (field 7), opset_import (field 8) */
const model = rawMsg(
    vint(1, 8),
    fieldMsg(7, graph),
    fieldMsg(8, opset)
);

/* write to .tmp/add3.onnx */
const path = '.tmp/add3.onnx';
const f = std.open(path, 'wb');
f.write(model.buffer, 0, model.length);
f.close();

/* test helpers */
let passed = 0;
let failed = 0;

function check(name, cond) {
    if (cond) {
        console.log('PASS:', name);
        passed++;
    } else {
        console.log('FAIL:', name);
        failed++;
    }
}

function same(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

/* 1. load session, check names */
const session = new Session(path);
check('session loads', true);
check('inputNames', JSON.stringify(session.inputNames) === JSON.stringify(['a', 'b']));
check('outputNames', JSON.stringify(session.outputNames) === JSON.stringify(['c']));

/* 2. run with basic inputs */
const a1 = new Tensor('float32', new Float32Array([1, 2, 3]), [3]);
const b1 = new Tensor('float32', new Float32Array([10, 20, 30]), [3]);
const out1 = session.run({ a: a1, b: b1 });
check('run returns object', out1 && typeof out1 === 'object');
check('output has key c', 'c' in out1);
const c1 = out1.c;
check('c.type === float32', c1.type === 'float32');
check('c.dims === [3]', JSON.stringify(c1.dims) === JSON.stringify([3]));
check('c.data === [11,22,33]', same(c1.data, new Float32Array([11, 22, 33])));

/* 3. second run, first result unchanged */
const a2 = new Tensor('float32', new Float32Array([5, 6, 7]), [3]);
const b2 = new Tensor('float32', new Float32Array([100, 200, 300]), [3]);
const out2 = session.run({ a: a2, b: b2 });
const c2 = out2.c;
check('second run c.data === [105,206,307]', same(c2.data, new Float32Array([105, 206, 307])));
check('first result data unchanged', same(c1.data, new Float32Array([11, 22, 33])));

/* 4. run with outputNames ['c'] */
const out3 = session.run({ a: a1, b: b1 }, ['c']);
check('outputNames ["c"] works', 'c' in out3 && Object.keys(out3).length === 1);
check('outputNames ["c"] data correct', same(out3.c.data, new Float32Array([11, 22, 33])));

/* 5. input tensors not modified */
const a3 = new Tensor('float32', new Float32Array([1, 2, 3]), [3]);
const b3 = new Tensor('float32', new Float32Array([10, 20, 30]), [3]);
const a3Before = a3.data.slice();
const b3Before = b3.data.slice();
session.run({ a: a3, b: b3 });
check('input a not modified', same(a3.data, a3Before));
check('input b not modified', same(b3.data, b3Before));

/* 6. error cases */
function throwsTypeError(fn, name) {
    try {
        fn();
        check(name, false);
    } catch (e) {
        check(name, e instanceof TypeError);
    }
}

function throwsErrorWithCode(fn, name) {
    try {
        fn();
        check(name, false);
    } catch (e) {
        check(name, e instanceof Error && typeof e.code === 'number');
    }
}

throwsTypeError(() => session.run(), 'run() no args throws TypeError');
throwsTypeError(() => session.run({ a: a1 }), 'run({a}) missing b throws TypeError');
throwsTypeError(() => session.run({ a: a1, b: b1, x: a1 }), 'run({a,b,x}) unknown input throws TypeError');
throwsTypeError(() => session.run({ a: a1, b: 42 }), 'run({a,b:42}) non-tensor throws TypeError');
throwsTypeError(() => session.run({ a: a1, b: b1 }, ['nope']), 'run outputNames ["nope"] throws TypeError');

const badTensor = new Tensor('float32', new Float32Array([1, 2, 3, 4]), [4]);
throwsErrorWithCode(() => session.run({ a: badTensor, b: b1 }), 'run with dims [4] throws Error with code');

/* 7. bad model files */
const badPath = '.tmp/bad.onnx';
const fbad = std.open(badPath, 'wb');
fbad.write(new TextEncoder().encode('not a model').buffer, 0, 11);
fbad.close();

try {
    new Session(badPath);
    check('bad model throws', false);
} catch (e) {
    check('bad model throws Error with code', e instanceof Error && typeof e.code === 'number');
}

try {
    new Session('.tmp/does-not-exist.onnx');
    check('missing file throws', false);
} catch (e) {
    check('missing file throws Error code 3', e instanceof Error && e.code === 3);
}

/* 8. 300 rounds + gc */
let rounds = 0;
for (let i = 0; i < 300; i++) {
    const a = new Tensor('float32', new Float32Array([i, i+1, i+2]), [3]);
    const b = new Tensor('float32', new Float32Array([i*10, i*10+10, i*10+20]), [3]);
    const out = session.run({ a, b });
    const c = out.c;
    const expected = new Float32Array([i + i*10, i+1 + i*10+10, i+2 + i*10+20]);
    if (!same(c.data, expected)) {
        check(`round ${i} data correct`, false);
        break;
    }
    rounds++;
}
check('300 rounds data correct', rounds === 300);
std.gc();
const aFinal = new Tensor('float32', new Float32Array([1, 2, 3]), [3]);
const bFinal = new Tensor('float32', new Float32Array([10, 20, 30]), [3]);
const outFinal = session.run({ a: aFinal, b: bFinal });
check('after gc run works', same(outFinal.c.data, new Float32Array([11, 22, 33])));

/* run after release throws Error with message session released */
session.release();
try {
    session.run({ a: a1, b: b1 });
    check('run after release throws Error', false);
} catch (e) {
    check('run after release throws Error', e instanceof Error && e.message === 'session released');
}
session.release();
check('second release() is fine', true);

/* summary */
console.log(`\nPassed: ${passed}, Failed: ${failed}`);
std.exit(failed > 0 ? 1 : 0);