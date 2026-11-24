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

/* build the Add model (same as ort-test.js) */
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

function throwsTypeError(fn, name) {
    try {
        fn();
        check(name, false);
    } catch (e) {
        check(name, e instanceof TypeError);
    }
}

function throwsRangeError(fn, name) {
    try {
        fn();
        check(name, false);
    } catch (e) {
        check(name, e instanceof RangeError);
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

/* write model to a file for path-based tests */
const modelPath = '.tmp/opts-add3.onnx';
const f = std.open(modelPath, 'wb');
f.write(model.buffer, 0, model.length);
f.close();

/* ============================================================
 * 1. Session from Uint8Array
 * ============================================================ */
console.log('\n--- Session from Uint8Array ---');
{
    const session = new Session(new Uint8Array(model));
    const a = new Tensor('float32', new Float32Array([1, 2, 3]), [3]);
    const b = new Tensor('float32', new Float32Array([10, 20, 30]), [3]);
    const out = session.run({ a, b });
    check('Uint8Array source runs correctly', same(out.c.data, new Float32Array([11, 22, 33])));
    session.release();
}

/* ============================================================
 * 2. Session from ArrayBuffer
 * ============================================================ */
console.log('\n--- Session from ArrayBuffer ---');
{
    const session = new Session(model.buffer);
    const a = new Tensor('float32', new Float32Array([1, 2, 3]), [3]);
    const b = new Tensor('float32', new Float32Array([10, 20, 30]), [3]);
    const out = session.run({ a, b });
    check('ArrayBuffer source runs correctly', same(out.c.data, new Float32Array([11, 22, 33])));
    session.release();
}

/* ============================================================
 * 3. Session from subarray (offset in larger buffer)
 * ============================================================ */
console.log('\n--- Session from subarray ---');
{
    const big = new Uint8Array(100);
    big.set(model, 7);  // place model at offset 7
    const sub = big.subarray(7, 7 + model.length);
    const session = new Session(sub);
    const a = new Tensor('float32', new Float32Array([1, 2, 3]), [3]);
    const b = new Tensor('float32', new Float32Array([10, 20, 30]), [3]);
    const out = session.run({ a, b });
    check('subarray source runs correctly', same(out.c.data, new Float32Array([11, 22, 33])));
    session.release();
}

/* ============================================================
 * 4. Overwriting the buffer after construction does not break run
 * ============================================================ */
console.log('\n--- Overwriting buffer after construction ---');
{
    const buf = new Uint8Array(model);
    const session = new Session(buf);
    buf.fill(0);  // overwrite the buffer
    const a = new Tensor('float32', new Float32Array([1, 2, 3]), [3]);
    const b = new Tensor('float32', new Float32Array([10, 20, 30]), [3]);
    const out = session.run({ a, b });
    check('run works after buffer overwritten', same(out.c.data, new Float32Array([11, 22, 33])));
    session.release();
}

/* ============================================================
 * 5. Empty Uint8Array throws Error with numeric code
 * ============================================================ */
console.log('\n--- Empty buffer throws Error ---');
{
    throwsErrorWithCode(() => new Session(new Uint8Array(0)), 'empty Uint8Array throws Error with code');
}

/* ============================================================
 * 6. Invalid first argument throws TypeError
 * ============================================================ */
console.log('\n--- Invalid first argument ---');
{
    throwsTypeError(() => new Session(), 'new Session() throws TypeError');
    throwsTypeError(() => new Session(42), 'new Session(42) throws TypeError');
    throwsTypeError(() => new Session({}), 'new Session({}) throws TypeError');
}

/* ============================================================
 * 7. Valid options: intraOpThreads, interOpThreads
 * ============================================================ */
console.log('\n--- Valid options: intraOpThreads, interOpThreads ---');
{
    const session = new Session(new Uint8Array(model), { intraOpThreads: 1, interOpThreads: 1 });
    const a = new Tensor('float32', new Float32Array([1, 2, 3]), [3]);
    const b = new Tensor('float32', new Float32Array([10, 20, 30]), [3]);
    const out = session.run({ a, b });
    check('intraOpThreads=1, interOpThreads=1 works', same(out.c.data, new Float32Array([11, 22, 33])));
    session.release();
}

/* ============================================================
 * 8. Valid options: graphOptimization values
 * ============================================================ */
console.log('\n--- Valid options: graphOptimization ---');
for (const opt of ['disable', 'basic', 'extended', 'all']) {
    const session = new Session(new Uint8Array(model), { graphOptimization: opt });
    const a = new Tensor('float32', new Float32Array([1, 2, 3]), [3]);
    const b = new Tensor('float32', new Float32Array([10, 20, 30]), [3]);
    const out = session.run({ a, b });
    check(`graphOptimization='${opt}' works`, same(out.c.data, new Float32Array([11, 22, 33])));
    session.release();
}

/* ============================================================
 * 9. Valid options: logLevel
 * ============================================================ */
console.log('\n--- Valid options: logLevel ---');
{
    const session = new Session(new Uint8Array(model), { logLevel: 'error' });
    const a = new Tensor('float32', new Float32Array([1, 2, 3]), [3]);
    const b = new Tensor('float32', new Float32Array([10, 20, 30]), [3]);
    const out = session.run({ a, b });
    check("logLevel='error' works", same(out.c.data, new Float32Array([11, 22, 33])));
    session.release();
}

/* ============================================================
 * 10. options undefined and {} work
 * ============================================================ */
console.log('\n--- options undefined and {} ---');
{
    let session = new Session(new Uint8Array(model), undefined);
    const a = new Tensor('float32', new Float32Array([1, 2, 3]), [3]);
    const b = new Tensor('float32', new Float32Array([10, 20, 30]), [3]);
    const out = session.run({ a, b });
    check('options undefined works', same(out.c.data, new Float32Array([11, 22, 33])));
    session.release();

    session = new Session(new Uint8Array(model), {});
    const out2 = session.run({ a, b });
    check('options {} works', same(out2.c.data, new Float32Array([11, 22, 33])));
    session.release();
}

/* ============================================================
 * 11. Invalid options: RangeError cases
 * ============================================================ */
console.log('\n--- Invalid options: RangeError ---');
{
    throwsRangeError(() => new Session(new Uint8Array(model), { intraOpThreads: -1 }), 'intraOpThreads: -1 throws RangeError');
    throwsRangeError(() => new Session(new Uint8Array(model), { intraOpThreads: 'two' }), 'intraOpThreads: "two" throws RangeError');
    throwsRangeError(() => new Session(new Uint8Array(model), { interOpThreads: 1.5 }), 'interOpThreads: 1.5 throws RangeError');
}

/* ============================================================
 * 12. Invalid options: TypeError for string options
 * ============================================================ */
console.log('\n--- Invalid options: TypeError ---');
{
    throwsTypeError(() => new Session(new Uint8Array(model), { graphOptimization: 'max' }), "graphOptimization: 'max' throws TypeError");
    throwsTypeError(() => new Session(new Uint8Array(model), { logLevel: 'loud' }), "logLevel: 'loud' throws TypeError");
}

/* ============================================================
 * 13. Options 42 throws TypeError
 * ============================================================ */
console.log('\n--- Options 42 throws TypeError ---');
{
    throwsTypeError(() => new Session(new Uint8Array(model), 42), 'options 42 throws TypeError');
}

/* ============================================================
 * 14. Same bad options throw with buffer source and path source
 * ============================================================ */
console.log('\n--- Bad options with buffer source ---');
{
    throwsRangeError(() => new Session(new Uint8Array(model), { intraOpThreads: -1 }), 'buffer: intraOpThreads -1 throws RangeError');
    throwsTypeError(() => new Session(new Uint8Array(model), { graphOptimization: 'max' }), 'buffer: graphOptimization max throws TypeError');
    throwsTypeError(() => new Session(new Uint8Array(model), 42), 'buffer: options 42 throws TypeError');
}

console.log('\n--- Bad options with path source ---');
{
    throwsRangeError(() => new Session(modelPath, { intraOpThreads: -1 }), 'path: intraOpThreads -1 throws RangeError');
    throwsTypeError(() => new Session(modelPath, { graphOptimization: 'max' }), 'path: graphOptimization max throws TypeError');
    throwsTypeError(() => new Session(modelPath, 42), 'path: options 42 throws TypeError');
}

/* ============================================================
 * 15. Stress test: 300 constructions with release(), half buffer, half path
 * ============================================================ */
console.log('\n--- Stress test: 300 constructions ---');
{
    let rounds = 0;
    for (let i = 0; i < 300; i++) {
        const useBuffer = (i % 2 === 0);
        const session = useBuffer ? new Session(new Uint8Array(model)) : new Session(modelPath);

        const a = new Tensor('float32', new Float32Array([i, i+1, i+2]), [3]);
        const b = new Tensor('float32', new Float32Array([i*10, i*10+10, i*10+20]), [3]);
        const out = session.run({ a, b });
        const expected = new Float32Array([i + i*10, i+1 + i*10+10, i+2 + i*10+20]);
        if (!same(out.c.data, expected)) {
            check(`round ${i} data correct`, false);
            break;
        }
        session.release();
        rounds++;
    }
    check('300 rounds data correct', rounds === 300);

    std.gc();
    const aFinal = new Tensor('float32', new Float32Array([1, 2, 3]), [3]);
    const bFinal = new Tensor('float32', new Float32Array([10, 20, 30]), [3]);
    const sessionFinal = new Session(new Uint8Array(model));
    const outFinal = sessionFinal.run({ a: aFinal, b: bFinal });
    check('after gc run works', same(outFinal.c.data, new Float32Array([11, 22, 33])));
    sessionFinal.release();
}

/* ============================================================
 * Summary
 * ============================================================ */
console.log(`\nPassed: ${passed}, Failed: ${failed}`);
std.exit(failed > 0 ? 1 : 0);