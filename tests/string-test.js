/* string-test.js - string tensors: construction, errors, round trip through an Identity model */

import { Session, Tensor } from 'ort';
import * as std from 'std';

/* protobuf helpers, every length < 128 */
const vint = (field, n) => Uint8Array.of(field << 3, n);
const cat = (...parts) => {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let off = 0;

  for(const p of parts) {
    out.set(p, off);
    off += p.length;
  }

  return out;
};
const field = (f, msg) => cat(Uint8Array.of((f << 3) | 2, msg.length), msg);
const str = (f, text) => field(f, Uint8Array.from(text, c => c.charCodeAt(0)));

/* Identity on a string tensor of shape [3]: a -> c */
function identityModel() {
  const shape = field(1, field(1, vint(1, 3)));
  const type = field(1, cat(vint(1, 8), field(2, shape)));  /* elem_type 8 is STRING */
  const info = name => cat(str(1, name), field(2, type));
  const node = cat(str(1, 'a'), str(2, 'c'), str(3, 'id'), str(4, 'Identity'));
  const graph = cat(field(1, node), field(2, str(2, 'g')), field(11, info('a')), field(12, info('c')));

  return cat(vint(1, 8), field(7, graph), field(8, vint(2, 13)));
}

let passed = 0, failed = 0;
const check = (name, cond) => {
  console.log(cond ? 'PASS:' : 'FAIL:', name);
  cond ? passed++ : failed++;
};
const throwsA = (name, ctor, fn) => {
  try {
    fn();
    check(name, false);
  } catch(e) {
    check(name + ' (' + e.name + ': ' + e.message + ')', e instanceof ctor);
  }
};
const same = (a, b) => JSON.stringify(a) == JSON.stringify(b);

const words = ['abc', 'héllo', '日本', ''];
const t = new Tensor('string', words, [2, 2]);

check('type is string', t.type == 'string');
check('dims', same(t.dims, [2, 2]));
check('data round trip', same(t.data, words));
check('data is an Array', Array.isArray(t.data));

const d = t.data;
d[0] = 'changed';
check('data is a copy', same(t.data, words));

const src = ['x', 'y'];
const t2 = new Tensor('string', src, [2]);
src[0] = 'z';
check('the constructor copied the strings', same(t2.data, ['x', 'y']));

check('scalar', same(new Tensor('string', ['one'], []).data, ['one']));
check('empty', same(new Tensor('string', [], [0]).data, []));

throwsA('not an Array', TypeError, () => new Tensor('string', 'abc', [1]));
throwsA('a typed array is not an Array', TypeError, () => new Tensor('string', new Uint8Array(1), [1]));
throwsA('length mismatch', RangeError, () => new Tensor('string', ['a'], [2]));
throwsA('non-string element', TypeError, () => new Tensor('string', ['a', 5], [2]));
throwsA('NUL byte', TypeError, () => new Tensor('string', ['a\0b'], [1]));

const session = new Session(identityModel());

check('input type string', session.inputs[0].type == 'string');
check('output type string', session.outputs[0].type == 'string');

const out = session.run({ a: new Tensor('string', ['one', 'zwei', '三'], [3]) });

check('run: type', out.c.type == 'string');
check('run: dims', same(out.c.dims, [3]));
check('run: data', same(out.c.data, ['one', 'zwei', '三']));
check('run: data twice', same(out.c.data, out.c.data));

session.release();
console.log('\nPassed: ' + passed + ', Failed: ' + failed);
std.exit(failed ? 1 : 0);
