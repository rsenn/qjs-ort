# qjs-ort

[ONNX Runtime](https://onnxruntime.ai/) for [QuickJS](https://bellard.org/quickjs/): load an `.onnx` model
and run it from JavaScript, with typed arrays in and out.

```js
import { Session, Tensor } from 'ort';

const session = new Session('bassline.onnx');
console.log(session.inputNames, session.outputNames);

const input = new Tensor('float32', new Float32Array(1 * 16), [1, 16]);
const { next } = session.run({ context: input });
console.log(next.dims, next.data);       // a Float32Array view of the result
```

> **Status: design stage.** There is no code yet; this page describes what the module is for and what its API
> is going to look like. See [TODO.md](TODO.md) for the order of work.

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

## Planned API

| | |
|---|---|
| `new Tensor(type, data, dims)` | `type` is `'float32'`, `'float64'`, `'int32'`, `'int64'`, `'uint8'`, `'bool'`, `'string'`. `data` is the matching typed array, used without copying. `.type`, `.data`, `.dims`. |
| `new Session(pathOrBuffer, options)` | `options`: `providers`, `intraOpThreads`, `interOpThreads`, `graphOptimization`, `logLevel`. |
| `session.inputNames`, `.outputNames` | Arrays of strings. `.inputs` and `.outputs` also carry type and shape. |
| `session.run(feeds, outputNames?)` | `feeds` maps an input name to a `Tensor`; returns an object that maps an output name to a `Tensor`. Synchronous. |
| `session.release()` | Frees the native session before the garbage collector does. |
| `ort.version`, `ort.providers` | The ONNX Runtime version and the execution providers it was built with. |

Errors are thrown as exceptions. The naming follows the other ONNX Runtime bindings (onnxruntime-node, Python,
Rust, Go), which agree on this shape.

## Requirements

| Needed | For |
|---|---|
| QuickJS with `qjsm` and its headers | building and running the module |
| ONNX Runtime 1.x (`onnxruntime_c_api.h`, `libonnxruntime`) | linking `ort.so`; a copy can be fetched from the NuGet package |
| CMake 3.5+ | building |

## Build

Not wired up yet. It will follow the other `qjs-*` projects:

```sh
cmake -B build -DORT_PREFIX=/path/to/onnxruntime
cmake --build build
cmake --install build
```

## Related

[qjs-opencv](https://github.com/rsenn/qjs-opencv) (image models through OpenCV `dnn`),
[qjs-sound](https://github.com/rsenn/qjs-sound) (the audio side),
[js-engines](https://github.com/rsenn/js-engines) (the stack that builds all of them, and the N-API adapter).

## License

MIT, see [LICENSE](LICENSE).
