/* types-test.js - test all tensor element types */

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

/* build a model for the given element type and operation */
function makeModel(elemType, opType, opsetVersion) {
    const dim = vint(1, 3);                           /* Dimension { dim_value: 3 } */
    const shape = rawMsg(fieldMsg(1, dim));           /* TensorShapeProto { dim: [dim] } */
    const tensorType = rawMsg(vint(1, elemType), fieldMsg(2, shape));
    const typeProto = rawMsg(fieldMsg(1, tensorType));

    const valueInfoA = rawMsg(str(1, 'a'), fieldMsg(2, typeProto));
    const valueInfoB = rawMsg(str(1, 'b'), fieldMsg(2, typeProto));
    const valueInfoC = rawMsg(str(1, 'c'), fieldMsg(2, typeProto));

    let node;
    let graph;
    if (opType === 'Identity') {
        node = rawMsg(
            str(1, 'a'),
            str(2, 'c'),
            str(3, 'id'),
            str(4, 'Identity')
        );
        graph = rawMsg(
            fieldMsg(1, node),
            fieldMsg(2, str(2, 'g')),
            fieldMsg(11, valueInfoA),
            fieldMsg(12, valueInfoC)
        );
    } else {
        node = rawMsg(
            str(1, 'a'), str(1, 'b'),
            str(2, 'c'),
            str(3, 'op'),
            str(4, opType)
        );
        graph = rawMsg(
            fieldMsg(1, node),
            fieldMsg(2, str(2, 'g')),
            fieldMsg(11, valueInfoA),
            fieldMsg(11, valueInfoB),
            fieldMsg(12, valueInfoC)
        );
    }

    const opset = rawMsg(vint(2, opsetVersion));

    const model = rawMsg(
        vint(1, 8),
        fieldMsg(7, graph),
        fieldMsg(8, opset)
    );

    return model;
}

/* element type numbers */
const ElemType = {
    FLOAT: 1,
    UINT8: 2,
    INT32: 6,
    INT64: 7,
    BOOL: 9,
    DOUBLE: 11
};

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

function sameBigInt(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

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

/* run a test case for one type */
function testType(name, elemType, opType, dataA, dataB, expectedData, expectedTypedArrayName) {
    console.log('\n=== Testing ' + name + ' ===');

    const model = makeModel(elemType, opType, 14);
    const path = '.tmp/types-' + name + '.onnx';
    const f = std.open(path, 'wb');
    f.write(model.buffer, 0, model.length);
    f.close();

    const session = new Session(path);
    check(name + ': session loads', true);

    /* input tensors */
    let tensorA, tensorB;
    if (opType === 'Identity') {
        tensorA = new Tensor(name, dataA, [3]);
    } else {
        tensorA = new Tensor(name, dataA, [3]);
        tensorB = new Tensor(name, dataB, [3]);
    }

    /* input tensor reads back the same typed array object */
    check(name + ': tensor.data === input array', tensorA.data === dataA);
    if (tensorB) {
        check(name + ': tensorB.data === input array', tensorB.data === dataB);
    }

    /* run */
    let out;
    if (opType === 'Identity') {
        out = session.run({ a: tensorA });
    } else {
        out = session.run({ a: tensorA, b: tensorB });
    }
    check(name + ': run returns object', out && typeof out === 'object');
    check(name + ': output has key c', 'c' in out);

    const c = out.c;
    check(name + ': c.type === ' + name, c.type === name);
    check(name + ': c.dims === [3]', JSON.stringify(c.dims) === JSON.stringify([3]));
    check(name + ': c.data.constructor.name === ' + expectedTypedArrayName, c.data.constructor.name === expectedTypedArrayName);

    /* check expected values */
    let dataMatch = false;
    if (name === 'int64') {
        dataMatch = sameBigInt(c.data, expectedData);
    } else {
        dataMatch = same(c.data, expectedData);
    }
    check(name + ': c.data values correct', dataMatch);

    /* test subarray with offset - use a larger buffer */
    let buffer, offset, subData, expectedSubData;
    if (name === 'float32') {
        buffer = new Float32Array([1, 2, 3, 4]).buffer;
        offset = 4;  // 1 element (4 bytes)
        subData = new Float32Array(buffer, offset, 3);  // elements at index 1,2,3
        expectedSubData = new Float32Array([12, 23, 34]);  // 2+10, 3+20, 4+30
    } else if (name === 'float64') {
        buffer = new Float64Array([1.5, 2.5, 3.5, 4.5]).buffer;
        offset = 8;  // 1 element (8 bytes)
        subData = new Float64Array(buffer, offset, 3);
        expectedSubData = new Float64Array([13.0, 24.0, 35.0]);  // 2.5+10.5, 3.5+20.5, 4.5+30.5
    } else if (name === 'int32') {
        buffer = new Int32Array([1, 2, 3, 4]).buffer;
        offset = 4;  // 1 element (4 bytes)
        subData = new Int32Array(buffer, offset, 3);
        expectedSubData = new Int32Array([12, 23, 34]);  // 2+10, 3+20, 4+30
    } else if (name === 'int64') {
        buffer = new BigInt64Array([9007199254740991n, 9007199254740992n, 9007199254740993n, 9007199254740994n]).buffer;
        offset = 8;  // 1 element (8 bytes)
        subData = new BigInt64Array(buffer, offset, 3);
        expectedSubData = new BigInt64Array([9007199254740993n, 9007199254740994n, 9007199254740995n]);
    } else if (name === 'uint8') {
        buffer = new Uint8Array([1, 2, 3, 4]).buffer;
        offset = 1;  // 1 element (1 byte)
        subData = new Uint8Array(buffer, offset, 3);
        expectedSubData = new Uint8Array([12, 23, 34]);  // 2+10, 3+20, 4+30
    } else if (name === 'bool') {
        buffer = new Uint8Array([0, 1, 0, 1]).buffer;
        offset = 1;  // 1 element (1 byte)
        subData = new Uint8Array(buffer, offset, 3);
        expectedSubData = new Uint8Array([1, 0, 1]);  // Identity, so output = input
    }

    if (subData) {
        const subTensor = new Tensor(name, subData, [3]);
        let outSub;
        if (opType === 'Identity') {
            outSub = session.run({ a: subTensor });
        } else {
            outSub = session.run({ a: subTensor, b: tensorB });
        }
        if (name === 'int64') {
            check(name + ': subarray offset works', sameBigInt(outSub.c.data, expectedSubData));
        } else {
            check(name + ': subarray offset works', same(outSub.c.data, expectedSubData));
        }
    }
}

/* run tests for each type */
try {
    /* float32 */
    testType('float32', ElemType.FLOAT, 'Add',
        new Float32Array([1, 2, 3]),
        new Float32Array([10, 20, 30]),
        new Float32Array([11, 22, 33]),
        'Float32Array');

    /* float64 */
    testType('float64', ElemType.DOUBLE, 'Add',
        new Float64Array([1.5, 2.5, 3.5]),
        new Float64Array([10.5, 20.5, 30.5]),
        new Float64Array([12.0, 23.0, 34.0]),
        'Float64Array');

    /* int32 */
    testType('int32', ElemType.INT32, 'Add',
        new Int32Array([1, 2, 3]),
        new Int32Array([10, 20, 30]),
        new Int32Array([11, 22, 33]),
        'Int32Array');

    /* int64 with BigInt values to prove no precision loss */
    testType('int64', ElemType.INT64, 'Add',
        new BigInt64Array([9007199254740991n, 9007199254740992n, 9007199254740993n]),
        new BigInt64Array([1n, 1n, 1n]),
        new BigInt64Array([9007199254740992n, 9007199254740993n, 9007199254740994n]),
        'BigInt64Array');

    /* uint8 - Add wraps around modulo 256 */
    const uint8Expected = new Uint8Array([(1 + 10) & 0xFF, (2 + 20) & 0xFF, (3 + 30) & 0xFF]);
    console.log('uint8 expected:', Array.from(uint8Expected));
    testType('uint8', ElemType.UINT8, 'Add',
        new Uint8Array([1, 2, 3]),
        new Uint8Array([10, 20, 30]),
        uint8Expected,
        'Uint8Array');

    /* bool - Identity (single input) */
    testType('bool', ElemType.BOOL, 'Identity',
        new Uint8Array([0, 1, 0]),
        new Uint8Array([0, 0, 0]),  /* dummy, not used by Identity */
        new Uint8Array([0, 1, 0]),
        'Uint8Array');

} catch (e) {
    console.log('Test error:', e);
    failed++;
}

/* error cases: wrong typed array for type */
console.log('\n=== Wrong typed array tests ===');
throwsTypeError(() => new Tensor('int64', new Int32Array(3), [3]), 'int64 with Int32Array throws TypeError');
throwsTypeError(() => new Tensor('uint8', new Int8Array(3), [3]), 'uint8 with Int8Array throws TypeError');
throwsTypeError(() => new Tensor('bool', new Uint8ClampedArray(3), [3]), 'bool with Uint8ClampedArray throws TypeError');
throwsTypeError(() => new Tensor('float32', new Float64Array(3), [3]), 'float32 with Float64Array throws TypeError');
throwsTypeError(() => new Tensor('float64', new Float32Array(3), [3]), 'float64 with Float32Array throws TypeError');
throwsTypeError(() => new Tensor('int32', new Int16Array(3), [3]), 'int32 with Int16Array throws TypeError');

/* feeding int64 model a float32 tensor makes run throw Error with numeric code */
console.log('\n=== Type mismatch tests ===');
const int64Model = makeModel(ElemType.INT64, 'Add', 14);
const int64Path = '.tmp/types-int64-mismatch.onnx';
const f2 = std.open(int64Path, 'wb');
f2.write(int64Model.buffer, 0, int64Model.length);
f2.close();
const int64Session = new Session(int64Path);
throwsErrorWithCode(() => {
    const a = new Tensor('float32', new Float32Array([1, 2, 3]), [3]);
    const b = new Tensor('float32', new Float32Array([10, 20, 30]), [3]);
    int64Session.run({ a, b });
}, 'int64 model with float32 input throws Error with code');

/* summary */
console.log('\nPassed: ' + passed + ', Failed: ' + failed);
std.exit(failed > 0 ? 1 : 0);