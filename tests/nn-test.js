// nn-test.js - nn/nn.js layers against numpy references (tests/nn/make-fixtures.py)
import * as std from 'std';
import * as nn from '../nn/nn.js';

const path = scriptArgs[1] ?? 'tests/nn/nn-test.nnw';
const file = std.open(path, 'rb');
file.seek(0, std.SEEK_END);
const size = file.tell();
file.seek(0, std.SEEK_SET);
const buf = new ArrayBuffer(size);
file.read(buf, 0, size);
file.close();

const w = nn.loadWeights(buf);
let fails = 0, checks = 0;

function check(name, got, want, tol = 1e-5) {
  let d = got.length == want.length ? 0 : Infinity;
  for(let i = 0; i < want.length && d < Infinity; i++) d = Math.max(d, Math.abs(got[i] - want[i]));
  checks++;
  if(!(d <= tol)) {
    fails++;
    console.log(`FAIL ${name}: max diff ${d}`);
  } else console.log(`ok   ${name} (${d.toExponential(1)})`);
}
const t = n => w.tensors[n].data;

check('linear', nn.Linear.from(w, 'fc').forward(t('fc.x')), t('fc.y'));
check('embedding', nn.Embedding.from(w, 'emb').forward(t('emb.ids')), t('emb.y'), 0);
check('layernorm', nn.LayerNorm.from(w, 'ln').forward(t('ln.x')), t('ln.y'));
check('conv1d', nn.Conv1d.from(w, 'conv', { stride: 2, padding: 2, dilation: 2 }).forward(t('conv.x')), t('conv.y'));

for(const a of ['relu', 'tanh', 'sigmoid', 'silu', 'gelu'])
  check(a, nn.apply(Float32Array.from(t('act.x')), a), t(`act.${a}`));
check('softmax', nn.softmax(t('act.x')), t('act.softmax'));
check('softmax temperature', nn.softmax(t('act.x'), undefined, 0.5), t('act.softmax_t'));

check('gru', nn.Recurrent.from(w, 'gru', 'gru').forward(t('gru.x')).y, t('gru.y'));
check('bigru 2 layers', nn.Recurrent.from(w, 'bigru', 'gru', { bidirectional: true }).forward(t('bigru.x')).y, t('bigru.y'));
check('lstm 2 layers', nn.Recurrent.from(w, 'lstm', 'lstm').forward(t('lstm.x')).y, t('lstm.y'));

// stepping one frame at a time with a carried state equals one pass
{
  const g = nn.Recurrent.from(w, 'gru', 'gru'), x = t('gru.x'), st = g.initialState(), out = [];
  for(let i = 0; i < x.length; i += g.inputSize) out.push(...g.forward(x.subarray(i, i + g.inputSize), st).y);
  check('gru stepwise', out, t('gru.y'));
}

// sampling: seeded rng, topK 1 is argmax, bad id throws
{
  const lg = t('act.x');
  checks++;
  if(nn.sample(lg, { topK: 1 }) != nn.argmax(lg)) { fails++; console.log('FAIL sample topK 1'); } else console.log('ok   sample topK 1');
  checks++;
  try { nn.Embedding.from(w, 'emb').forward([99]); fails++; console.log('FAIL embedding range'); } catch(e) { console.log('ok   embedding range', e instanceof RangeError); }
  checks++;
  try { nn.loadWeights(new ArrayBuffer(16)); fails++; console.log('FAIL bad magic'); } catch(e) { console.log('ok   bad magic'); }
}

console.log(`${checks - fails}/${checks} passed`);
if(fails) throw new Error('failures');
