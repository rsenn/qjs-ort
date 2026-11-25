# qjs-ort

[ONNX Runtime](https://onnxruntime.ai/) for [QuickJS](https://bellard.org/quickjs/): load an `.onnx` model
and run it from JavaScript, with typed arrays in and out.

```js
import { Session, Tensor } from 'ort';

const session = new Session('bassline.onnx');
console.log(session.inputs, session.outputs);   // [{ name, type, dims }, ...]

const input = new Tensor('float32', new Float32Array(1 * 16), [1, 16]);
const { next } = session.run({ context: input });
console.log(next.dims, next.data);       // a Float32Array view of the result
```

> **Status: works, early.** `Tensor` (float32, float64, int32, int64, uint8, bool), `Session` (from a file or
> from bytes, thread and optimisation options, input and output metadata) and a synchronous `session.run()` are
> implemented and tested; `u2netp.onnx` gives the same result as OpenCV `dnn`. Only the CPU execution provider
> is used so far. See [TODO.md](TODO.md) for what comes next and [doc/native/ort.md](doc/native/ort.md) for the
> full API page.

## Why

OpenCV's `dnn` module runs a subset of ONNX only. Models you train yourself (sequence models for music
generation, synth-parameter models, small accompaniment networks) use operators it does not have, so this
binding talks to ONNX Runtime directly. The workflow it is built for:

1. train on the workstation (PyTorch or anything that exports ONNX),
2. export to `.onnx`,
3. run the same file from a QuickJS script on the workstation, the Raspberry Pi (aarch64 glibc) and the phone
   (Termux, aarch64 bionic), next to the other `qjs-*` modules such as [qjs-sound](https://github.com/rsenn/qjs-sound).

Through the N-API adapter of [js-engines](https://github.com/rsenn/js-engines) the same prebuilt `ort.so`
can also be imported from Bun, Node and Deno.

## API

| | |
|---|---|
| `new Tensor(type, data, dims)` | `type` is `'float32'`, `'float64'`, `'int32'`, `'int64'`, `'uint8'` or `'bool'` (`'string'` and `'float16'` are planned). `data` is the matching typed array (`BigInt64Array` for `int64`), used without copying. `.type`, `.data`, `.dims`. |
| `new Session(pathOrBuffer, options?)` | A file path, or an `ArrayBuffer` or typed array that holds the model. `options`: `intraOpThreads`, `interOpThreads`, `graphOptimization` (`'disable'`, `'basic'`, `'extended'`, `'all'`), `logLevel`. `providers` is planned. |
| `session.inputNames`, `.outputNames` | Arrays of strings. |
| `session.inputs`, `.outputs` | Arrays of `{ name, type, dims }`; a dynamic dimension is `-1`. |
| `session.run(feeds, outputNames?)` | `feeds` maps an input name to a `Tensor`; returns an object that maps an output name to a `Tensor`. Synchronous. The `data` of an output is a view of the memory ONNX Runtime allocated, `.slice()` it to keep it. |
| `session.release()` | Frees the native session before the garbage collector does. |
| `ort.version` | The ONNX Runtime version. `ort.providers` is planned. |

Errors are thrown as exceptions: ONNX Runtime failures as `Error` with its message and a numeric `code`, wrong
arguments as `TypeError` or `RangeError`. The naming follows the other ONNX Runtime bindings (onnxruntime-node,
Python, Rust, Go), which agree on this shape.

## Examples

| | |
|---|---|
| [`examples/gru/`](examples/gru) | A GRU cell run one step at a time with the hidden state fed back, the shape of a realtime loop. The 5 KB model is in the repository and `make-model.py` exports it; the script checks every step against ONNX Runtime for Python. |
| [`examples/u2netp-cutout.js`](examples/u2netp-cutout.js) | Cuts the salient object out of a picture with `u2netp.onnx` (not in the repository) and writes a PNG with the matte as its alpha channel. Needs [qjs-opencv](https://github.com/rsenn/qjs-opencv). |

## Requirements

| Needed | For |
|---|---|
| QuickJS with `qjsm` and its headers | building and running the module |
| ONNX Runtime 1.x (`onnxruntime_c_api.h`, `libonnxruntime`) | linking `ort.so`; a copy can be fetched from the NuGet package |
| CMake 3.5+ | building |

## Build

```sh
tools/fetch-onnxruntime.sh                 # ONNX Runtime into third_party/, or pass -DORT_PREFIX=...
cmake -B build/x86_64-linux-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/x86_64-linux-debug
QUICKJS_MODULE_PATH=build/x86_64-linux-debug qjsm tests/ort-test.js
cmake --install build/x86_64-linux-debug   # ort.so and libonnxruntime.so.1 into the QuickJS module directory
```

Use `qjsm` for the scripts in `examples/`; the tests also run on a plain `qjs`. The other tests are in `tests/`, run each the
same way.

### Raspberry Pi (aarch64)

Cross build with the host clang and lld, no gcc cross package. It needs a sysroot copied from the Pi and the linux-arm64
ONNX Runtime (`tools/fetch-onnxruntime.sh 1.32.0 linux-arm64`):

```sh
AARCH64_SYSROOT=~/sysroots/pi cmake -S . -B build/aarch64-linux-gnu -DCMAKE_BUILD_TYPE=Release -DBUILD_QUICKJS=ON \
  -DCMAKE_TOOLCHAIN_FILE=tools/aarch64-clang.toolchain.cmake -DORT_PREFIX=third_party/onnxruntime-linux-arm64-1.32.0
cmake --build build/aarch64-linux-gnu
DESTDIR=/tmp/stage cmake --install build/aarch64-linux-gnu    # qjs, libquickjs, ort.so, libonnxruntime.so.1
```

On a Raspberry Pi 4 the tests pass and `u2netp` runs in about 2.3 s per 320x320 image.

## Related

[qjs-opencv](https://github.com/rsenn/qjs-opencv) (image models through OpenCV `dnn`),
[qjs-sound](https://github.com/rsenn/qjs-sound) (the audio side),
[js-engines](https://github.com/rsenn/js-engines) (the stack that builds all of them, and the N-API adapter).

## License

MIT, see [LICENSE](LICENSE).
