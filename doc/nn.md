# nn/ - models trained in PyTorch, run in JS

The plan and the use cases are in [`AM-DNN-CASES.md`](AM-DNN-CASES.md). The rule: **train in PyTorch, export one
file, run it with no native code** (`nn/nn.js`, qjsm, Node, Bun, Deno) when the model is small, and with
[`ort.so`](native/ort.md) when it is not.

## Layout

| Path | What |
|---|---|
| `nn/nn.js` | The layers (below), the `.nnw` loader, `sample`, `argmax`. Pure JS, no runtime imports. |
| `nn/export.py` | `write_nnw(path, state_dict, meta)`. Needs numpy only (torch tensors are accepted). |
| `nn/nnw.py` | `read_nnw(path)`: `(meta, {name: ndarray})`, memory-mapped. |
| `nn/features.js` | Log-mel front end (radix-2 FFT, real FFT in one half-size transform, mel filterbank). The one definition used for training and inference, never ported to Python. |
| `nn/synths/stkpatch.js`, `stkpatch.json` | The STK patch of Case 4 (`render(values, midi, spec)`) and its parameter spec. qjsm only (needs the `stk` module). |
| `nn/synths/params.js` | `denormalize`, `normalize`, seeded `rng`, `sampleLabels` for a spec. |
| `nn/models/synth-params.js` | `SynthParams.load(buffer).predict(mel, midi)`: JS inference of the Case 4 model. |
| `tools/render-dataset.js` | Renders random patches and writes features, labels and pitch as `.nnw`. |
| `train/synth-params.py` | The PyTorch trainer of Case 4 (conv + MLP), writes the model and a `.check.nnw`. |
| `tools/eval-synth.js` | Renders truth, model, nearest-neighbour and random labels for the test notes and prints the log-mel L1 distance to the target. |
| `tests/nn-test.js` | Every layer against numpy references (`tests/nn/make-fixtures.py` writes `tests/nn/nn-test.nnw`). |
| `tests/nn-features-test.js` | FFT against a DFT, the mel band of a sine, the label roundtrip. |
| `tests/synth-params-test.js` | JS inference against the PyTorch outputs stored next to the model. |

## The `.nnw` file

All integers little endian. `"NNW1"`, `uint32` header length, a JSON header padded with spaces to a multiple of 4,
then float32 data with every tensor on a 4-byte boundary. The header is
`{ "meta": {...}, "tensors": { name: { "shape": [...], "offset": bytes } } }`, offsets from the start of the blob.
`meta` carries everything the JS side must not hard-code: the feature config, the parameter spec, normalisation
(means and stds are tensors), vocabularies. Datasets use the same format (`features`, `labels`, `midi`).

```python
from export import write_nnw
write_nnw('model.nnw', model.state_dict(), meta={'hidden': 64})
```

```js
import * as nn from './nn/nn.js';
const w = nn.loadWeights(buffer);                    // buffer: ArrayBuffer of the file
const gru = nn.Recurrent.from(w, 'rnn', 'gru', { bidirectional: true });
const { y, state } = gru.forward(frames);            // frames: Float32Array [T * in]
```

## Layers

All take the PyTorch parameter layout, so a `state_dict` loads without reshaping; batch is always 1 and a
sequence is `[T, features]` row-major.

| Layer | Notes |
|---|---|
| `Linear.from(w, prefix)` | `forward(x, rows)`, `weight` `[out, in]`. |
| `Embedding.from` | `forward(ids)`, throws `RangeError` for an id out of range. |
| `LayerNorm.from` | `eps` argument, default 1e-5. |
| `Conv1d.from(w, prefix, {stride, padding, dilation})` | input `[T, inCh]`, weight `[outCh, inCh, k]`. |
| `Recurrent.from(w, prefix, 'gru' \| 'lstm', {bidirectional})` | torch gate order (r,z,n and i,f,g,o), several layers, `forward(x, state)` continues from `state` (`initialState()`), which is how a realtime loop steps frame by frame. |
| `apply(x, 'relu' \| 'tanh' \| 'sigmoid' \| 'silu' \| 'gelu')`, `softmax(x, out, temperature)`, `sample(logits, {temperature, topK, rng})`, `argmax` | |

Not there yet: attention. Case 5 needs it, and runs on `ort.so` instead.

## Speed (measured 2026-11-26)

This `qjsm` runs about 7 M simple loop iterations per second, so a model costs roughly what its multiply-adds say
at a few M per second: the 360 k-parameter conv model of Case 4 (about 2.3 M MAC) took 1.7 s per inference while the
machine was busy rendering. Pure JS is fine for features, small MLPs and a single GRU step; a transformer or a conv net
on images belongs on `ort.so`.

## Case 4 workflow

```sh
export QUICKJS_MODULE_PATH=/usr/local/lib/x86_64-linux-gnu/quickjs
for s in 1 2 3 4 5 6 7 8; do qjsm tools/render-dataset.js data/synth/train-$s.nnw 6000 $s & done   # 6000 examples each
qjsm tools/render-dataset.js data/synth/val.nnw 2000 100
qjsm tools/render-dataset.js data/synth/test.nnw 300 101
~/.venvs/audio-ml/bin/python train/synth-params.py data/synth out/synth-params.nnw --epochs 60
qjsm tests/synth-params-test.js out/synth-params.nnw
qjsm tools/eval-synth.js out/synth-params.nnw data/synth/test.nnw 300   # about 15 min, 10 notes take 30 s
```

Measured on the 300 test notes: model 2.88 dB, nearest neighbour 5.69, random 21.65 (truth 0.000), see
`AM-DNN-CASES.md` Case 4.

`data/` and `out/` are git-ignored. The venv `~/.venvs/audio-ml` has PyTorch with CUDA and numpy.

## Quirks found

- `stk` reverbs: `FreeVerb` renders differently on every call (uninitialised state) and `JCRev.tick(frames)` on a mono
  `StkFrames` gives garbage (NaN, hundreds of dB). The patch ticks `JCRev` per sample (it returns `[left, right]`)
  and `render()` throws on a non-finite sample. Both are qjs-sound matters, not fixed here.
- `qjsm`'s `os.seek(fd, 0, os.SEEK_END)` returned 0 here; the tests read files with `std.open`, `seek`, `tell`.
