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
    const bytes = Uint8Array.from(text, (c) => c.charCodeAt(0));
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

/* make a Dimension message: field 1 = dim_value (varint), field 2 = dim_param (string) */
function makeDimension(dimsSpec) {
    /* dimsSpec can be a number (dim_value) or a string (dim_param) */
    if (typeof dimsSpec === 'number') {
        return vint(1, dimsSpec);
    } else {
        return str(2, dimsSpec);
    }
}

/* build a model with given element type and dimsSpec (array of numbers or strings) */
function makeModel(elemType, dimsSpec, opType) {
    const dims = dimsSpec.map(makeDimension);
    const shape = rawMsg(...dims.map((d) => fieldMsg(1, d)));
    const tensorType = rawMsg(vint(1, elemType), fieldMsg(2, shape));
    const typeProto = rawMsg(fieldMsg(1, tensorType));

    let valueInfoA, valueInfoB, valueInfoC;
    let node;
    let graph;

    if (opType === 'Identity') {
        valueInfoA = rawMsg(str(1, 'a'), fieldMsg(2, typeProto));
        valueInfoC = rawMsg(str(1, 'c'), fieldMsg(2, typeProto));

        node = rawMsg(
            str(1, 'a'),
            str(2, 'c'),
            str(3, 'id'),
            str(4, 'Identity')
        );
        graph = rawMsg(
            fieldMsg(1, node),
            str(2, 'g'),
            fieldMsg(11, valueInfoA),
            fieldMsg(12, valueInfoC)
        );
    } else {
        valueInfoA = rawMsg(str(1, 'a'), fieldMsg(2, typeProto));
        valueInfoB = rawMsg(str(1, 'b'), fieldMsg(2, typeProto));
        valueInfoC = rawMsg(str(1, 'c'), fieldMsg(2, typeProto));

        node = rawMsg(
            str(1, 'a'), str(1, 'b'),
            str(2, 'c'),
            str(3, 'op'),
            str(4, opType)
        );
        graph = rawMsg(
            fieldMsg(1, node),
            str(2, 'g'),
            fieldMsg(11, valueInfoA),
            fieldMsg(11, valueInfoB),
            fieldMsg(12, valueInfoC)
        );
    }

    const opset = rawMsg(vint(2, 14));

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
    INT64: 7,
    FLOAT16: 10,
    STRING: 8
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

function deepEqual(a, b) {
    return JSON.stringify(a) === JSON.stringify(b);
}

function throwsErrorWithMessage(fn, name, expectedMsg) {
    try {
        fn();
        check(name, false);
    } catch (e) {
        check(name, e instanceof Error && e.message === expectedMsg);
    }
}

/* 1. Add float32 with dims ['N', 3] */
console.log('\n=== 1. float32 with dynamic dim ["N", 3] ===');
{
    const model = makeModel(ElemType.FLOAT, ['N', 3], 'Add');
    const path = '.tmp/test1.onnx';
    const f = std.open(path, 'wb');
    f.write(model.buffer, 0, model.length);
    f.close();

    const session = new Session(path);

    const inputs = session.inputs;
    check('inputs is array', Array.isArray(inputs));
    check('inputs length 2', inputs.length === 2);
    check('inputs[0] name a', inputs[0].name === 'a');
    check('inputs[0] type float32', inputs[0].type === 'float32');
    check('inputs[0] dims [-1, 3]', deepEqual(inputs[0].dims, [-1, 3]));
    check('inputs[1] name b', inputs[1].name === 'b');
    check('inputs[1] type float32', inputs[1].type === 'float32');
    check('inputs[1] dims [-1, 3]', deepEqual(inputs[1].dims, [-1, 3]));

    const outputs = session.outputs;
    check('outputs is array', Array.isArray(outputs));
    check('outputs length 1', outputs.length === 1);
    check('outputs[0] name c', outputs[0].name === 'c');
    check('outputs[0] type float32', outputs[0].type === 'float32');
    check('outputs[0] dims [-1, 3]', deepEqual(outputs[0].dims, [-1, 3]));

    /* run with a Tensor of dims [2,3] */
    const a = new Tensor('float32', new Float32Array([0, 1, 2, 3, 4, 5]), [2, 3]);
    const b = new Tensor('float32', new Float32Array(2 * 3).fill(10), [2, 3]);
    const out = session.run({ a, b });
    check('run works on dynamic model', out && out.c);
    check('output dims [2, 3]', deepEqual(out.c.dims, [2, 3]));
    check('output values correct', same(out.c.data, new Float32Array([10, 11, 12, 13, 14, 15])));

    session.release();
}

/* 2. int64 model with dims [4] */
console.log('\n=== 2. int64 with dims [4] ===');
{
    const model = makeModel(ElemType.INT64, [4], 'Add');
    const path = '.tmp/test2.onnx';
    const f = std.open(path, 'wb');
    f.write(model.buffer, 0, model.length);
    f.close();

    const session = new Session(path);

    const inputs = session.inputs;
    check('inputs type int64', inputs[0].type === 'int64');
    check('inputs dims [4]', deepEqual(inputs[0].dims, [4]));

    const outputs = session.outputs;
    check('outputs type int64', outputs[0].type === 'int64');
    check('outputs dims [4]', deepEqual(outputs[0].dims, [4]));

    session.release();
}

/* 3. float16 (element type 10) - type should be undefined */
console.log('\n=== 3. float16 (type 10) ===');
{
    const model = makeModel(ElemType.FLOAT16, [2], 'Identity');
    const path = '.tmp/test3.onnx';
    const f = std.open(path, 'wb');
    f.write(model.buffer, 0, model.length);
    f.close();

    const session = new Session(path);

    const inputs = session.inputs;
    check('inputs type undefined (float16 not in table)', inputs[0].type === undefined);
    check('inputs dims [2]', deepEqual(inputs[0].dims, [2]));

    const outputs = session.outputs;
    check('outputs type undefined', outputs[0].type === undefined);
    check('outputs dims [2]', deepEqual(outputs[0].dims, [2]));

    session.release();
}

/* 4. string (element type 8) - type should be undefined */
console.log('\n=== 4. string (type 8) ===');
{
    const model = makeModel(ElemType.STRING, [2], 'Identity');
    const path = '.tmp/test4.onnx';
    const f = std.open(path, 'wb');
    f.write(model.buffer, 0, model.length);
    f.close();

    const session = new Session(path);

    const inputs = session.inputs;
    check('inputs type undefined (string not in table)', inputs[0].type === undefined);
    check('inputs dims [2]', deepEqual(inputs[0].dims, [2]));

    const outputs = session.outputs;
    check('outputs type undefined', outputs[0].type === undefined);
    check('outputs dims [2]', deepEqual(outputs[0].dims, [2]));

    session.release();
}

/* 5. scalar model (dimsSpec []) */
console.log('\n=== 5. scalar (empty dims) ===');
{
    const model = makeModel(ElemType.FLOAT, [], 'Add');
    const path = '.tmp/test5.onnx';
    const f = std.open(path, 'wb');
    f.write(model.buffer, 0, model.length);
    f.close();

    const session = new Session(path);

    const inputs = session.inputs;
    check('inputs dims [] (scalar)', deepEqual(inputs[0].dims, []));

    const outputs = session.outputs;
    check('outputs dims [] (scalar)', deepEqual(outputs[0].dims, []));

    session.release();
}

/* 6. reading inputs twice gives two different Array objects with equal content */
console.log('\n=== 6. inputs twice are different arrays ===');
{
    const model = makeModel(ElemType.FLOAT, [3], 'Add');
    const path = '.tmp/test6.onnx';
    const f = std.open(path, 'wb');
    f.write(model.buffer, 0, model.length);
    f.close();

    const session = new Session(path);

    const inputs1 = session.inputs;
    const inputs2 = session.inputs;

    check('inputs1 !== inputs2 (different objects)', inputs1 !== inputs2);
    check('inputs1[0] !== inputs2[0] (different objects)', inputs1[0] !== inputs2[0]);
    check('inputs1 content equals inputs2 content', deepEqual(inputs1, inputs2));

    /* mutating first does not change second */
    inputs1[0].name = 'mutated';
    check('inputs2 unchanged after mutating inputs1', inputs2[0].name === 'a');

    session.release();
}

/* 7. after session.release() reading inputs/outputs throws Error('session released') while inputNames still works */
console.log('\n=== 7. released session throws on inputs/outputs ===');
{
    const model = makeModel(ElemType.FLOAT, [3], 'Add');
    const path = '.tmp/test7.onnx';
    const f = std.open(path, 'wb');
    f.write(model.buffer, 0, model.length);
    f.close();

    const session = new Session(path);
    session.release();

    throwsErrorWithMessage(() => session.inputs, 'inputs throws after release', 'session released');
    throwsErrorWithMessage(() => session.outputs, 'outputs throws after release', 'session released');
    check('inputNames still works after release', deepEqual(session.inputNames, ['a', 'b']));
    check('outputNames still works after release', deepEqual(session.outputNames, ['c']));
}

/* 8. loop of 3000 reads of inputs and outputs without checks per round, then one final check */
console.log('\n=== 8. stress: 3000 reads ===');
{
    const model = makeModel(ElemType.FLOAT, ['N', 3], 'Add');
    const path = '.tmp/test8.onnx';
    const f = std.open(path, 'wb');
    f.write(model.buffer, 0, model.length);
    f.close();

    const session = new Session(path);

    for (let i = 0; i < 3000; i++) {
        const inputs = session.inputs;
        const outputs = session.outputs;
        /* no checks per round */
    }

    /* final check */
    const inputs = session.inputs;
    const outputs = session.outputs;
    check('inputs still correct after 3000 reads', deepEqual(inputs, [
        { name: 'a', type: 'float32', dims: [-1, 3] },
        { name: 'b', type: 'float32', dims: [-1, 3] }
    ]));
    check('outputs still correct after 3000 reads', deepEqual(outputs, [
        { name: 'c', type: 'float32', dims: [-1, 3] }
    ]));

    session.release();
}

/* summary */
console.log(`\nPassed: ${passed}, Failed: ${failed}`);
std.exit(failed > 0 ? 1 : 0);