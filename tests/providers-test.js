/* providers-test.js - ort.providers and the providers option of Session */

import * as ort from 'ort';
import { Session } from 'ort';
import * as std from 'std';

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
    return e;
  }
};

/* the smallest valid model, Identity on float32[1], built by hand */
const vint = (f, n) => Uint8Array.of(f << 3, n);
const cat = (...p) => {
  const o = new Uint8Array(p.reduce((n, x) => n + x.length, 0));
  let off = 0;
  for(const x of p) { o.set(x, off); off += x.length; }
  return o;
};
const field = (f, m) => cat(Uint8Array.of((f << 3) | 2, m.length), m);
const str = (f, t) => field(f, Uint8Array.from(t, c => c.charCodeAt(0)));
const type = field(1, cat(vint(1, 1), field(2, field(1, field(1, vint(1, 1))))));
const info = n => cat(str(1, n), field(2, type));
const node = cat(str(1, 'a'), str(2, 'c'), str(3, 'id'), str(4, 'Identity'));
const model = cat(vint(1, 8), field(7, cat(field(1, node), field(2, str(2, 'g')), field(11, info('a')), field(12, info('c')))), field(8, vint(2, 13)));

check('ort.providers is an Array', Array.isArray(ort.providers));
check('CPU is available', ort.providers.includes('CPUExecutionProvider'));
check('providers are strings', ort.providers.every(p => typeof p == 'string'));

const ok = (name, providers) => {
  try {
    const s = new Session(model, { providers });
    check(name, s.inputNames[0] == 'a');
    s.release();
  } catch(e) {
    check(name + ' threw ' + e.message, false);
  }
};

ok('no providers option', undefined);
ok('empty providers', []);
ok("['cpu']", ['cpu']);
ok("[{ name: 'cpu' }]", [{ name: 'cpu' }]);

throwsA('providers is not an Array', TypeError, () => new Session(model, { providers: 'cpu' }));
throwsA('unknown provider name', TypeError, () => new Session(model, { providers: ['nonexistent'] }));
throwsA('entry is a number', TypeError, () => new Session(model, { providers: [5] }));
throwsA('object without a name', TypeError, () => new Session(model, { providers: [{ deviceId: 0 }] }));
throwsA('provider option of a bad kind', TypeError, () => new Session(model, { providers: [{ name: 'cuda', device_id: {} }] }));

/* a provider this build does not have: ONNX Runtime's own error, with a code */
for(const name of ['cuda', 'xnnpack', 'nnapi']) {
  if(ort.providers.some(p => p.toLowerCase().startsWith(name))) {
    console.log('SKIP:', name, 'is available here');
    continue;
  }

  const e = throwsA(name + ' without the provider throws', Error, () => new Session(model, { providers: [name == 'cuda' ? { name, device_id: 0 } : name == 'nnapi' ? { name, flags: 1 } : name] }));
  check(name + ' error has a numeric code', e && typeof e.code == 'number');
}

throwsA('nnapi takes only flags', Error, () => new Session(model, { providers: [{ name: 'nnapi', device_id: 0 }] }));

console.log('\nPassed: ' + passed + ', Failed: ' + failed);
std.exit(failed ? 1 : 0);
