# ort - ONNX Runtime for QuickJS

> **Rough sketch.** This page is the API as designed, written before the code. Entries marked **(PoC)** are
> what the first milestone implements (see `TODO.md`); the rest is planned and may change. Update this page
> with every API change, and drop the marks when the code exists.

```js
import * as ort from 'ort';
import { Session, Tensor } from 'ort';
```

Native module `ort` (`ort.so`), built on the ONNX Runtime C API (`onnxruntime_c_api.h`). Names follow the other
ONNX Runtime bindings (onnxruntime-node, Python, Rust, Go).

## Module

| Export | |
|---|---|
| `ort.version` **(PoC, done)** | The ONNX Runtime version string, `OrtGetApiBase()->GetVersionString()`, e.g. `'1.32.0'`. |
| `ort.providers` | Array of the execution providers this build of ONNX Runtime has, `['CPUExecutionProvider']` at least. |
| `Tensor`, `Session` **(PoC)** | The classes below. |

## Tensor **(PoC)**

A typed, shaped block of numbers. Inputs wrap a typed array without copying; outputs of `Session.run()` own the
memory ONNX Runtime allocated.

```js
new Tensor(type, data, dims)
```

| Argument | |
|---|---|
| `type` | Element type name. Supported: `'float32'`, `'float64'`, `'int32'`, `'int64'`, `'uint8'`, `'bool'`. Planned: `'string'`, `'float16'`. |
| `data` | The matching typed array (`Float32Array` for `'float32'`; `Float64Array` for `'float64'`; `Int32Array` for `'int32'`; `BigInt64Array` for `'int64'`; `Uint8Array` for `'uint8'` and `'bool'`). It is **not copied**: writing to it changes the tensor, and the tensor keeps it alive. For `'bool'` the data is a `Uint8Array` with values 0 or 1. For `'int64'` the data is a `BigInt64Array`. |
| `dims` | Array of non-negative integers. The product must equal `data.length`, else `RangeError`. `[]` is a scalar (one element). |

| Member | |
|---|---|
| `tensor.type` | The element type name, e.g. `'float32'`. |
| `tensor.dims` | A new plain `Array` of numbers each time. |
| `tensor.data` | A typed array over the tensor's memory, no copy. For an input tensor it is the array passed to the constructor. For an output tensor a new view each time, and it keeps the tensor alive, so it stays valid after the tensor object is gone. Copy with `.slice()` when the result must outlive the next `run`. |
| `tensor[Symbol.toStringTag]` | `'Tensor'`. |

Errors: wrong `type` name or a `data` that is not the typed array the type needs: `TypeError`. Dims that do not
match `data.length`, or a negative dim: `RangeError`.

## Session **(PoC)**

A loaded model. CPU execution provider, default options.

```js
const session = new Session('u2netp.onnx');
```

| Member | |
|---|---|
| `new Session(pathOrBuffer, options?)` | Loads the model from a file path (string) or from an `ArrayBuffer` or typed array holding the model bytes (not copied, the array may change afterwards). Throws `Error` (with `.code`, see Errors) when the file is missing or the bytes are not a model, `TypeError` for any other first argument. |
| `session.inputNames` | `Array` of strings, in model order. |
| `session.outputNames` | `Array` of strings, in model order. |
| `session.run(feeds, outputNames?)` | Runs the model, synchronously. `feeds` is an object, input name to `Tensor`; every model input must be present, no others. `outputNames` is an array of output names, default all. Returns an object, output name to `Tensor`. |
| `session.release()` | Frees the native session now. Later `run()` throws `Error('session released')`; calling `release()` twice is fine. Without it the garbage collector frees the session. |
| `session[Symbol.toStringTag]` | `'Session'`. |

```js
const input = new Tensor('float32', new Float32Array(1 * 3 * 320 * 320), [1, 3, 320, 320]);
const out = session.run({ [session.inputNames[0]]: input });
const map = out[session.outputNames[0]];
console.log(map.dims, map.data.length);
```

Options (second argument, an object, all optional; unknown names are ignored):

| Option | |
|---|---|
| `intraOpThreads`, `interOpThreads` | Thread counts, a non-negative integer, `0` is the ONNX Runtime default. `RangeError` otherwise. |
| `graphOptimization` | `'disable'`, `'basic'`, `'extended'` or `'all'` (default). `TypeError` otherwise. |
| `logLevel` | `'verbose'`, `'info'`, `'warning'`, `'error'` or `'fatal'`. `TypeError` otherwise. |

Planned (not PoC):

| Member | |
|---|---|
| `session.inputs`, `session.outputs` | Arrays of `{ name, type, dims }`; a dynamic dimension is `-1`, an unnamed element type `undefined`. |
| `options.providers` | Array of provider names or `{ name, ...providerOptions }`, tried in order: `'cpu'`, `'xnnpack'`, `'nnapi'`, `'cuda'`. |
| `session.runAsync(feeds, outputNames?)` | `Promise` version, the model runs on a worker thread; ORT never calls into JS. |

## Errors

A failed ONNX Runtime call throws an `Error` whose message is ONNX Runtime's own message and which has a numeric
`code` property, the numeric `OrtErrorCode` of `onnxruntime_error_code.h` (`2` is `ORT_INVALID_ARGUMENT`, `3`
`ORT_NO_SUCHFILE`, `7` `ORT_INVALID_PROTOBUF`, ...; the module does not export the names yet). Mistakes the
binding can see itself are `TypeError` (wrong kind of argument, unknown or missing input name) or `RangeError`
(dims). A call never crashes the process on bad input.

For a tensor of type `'bool'` the data is a `Uint8Array` (one byte per element, 0 or 1) in both
directions; no conversion is done. For a tensor of type `'int64'` the data is a `BigInt64Array`.

## Threads

ONNX Runtime is not used from other threads than the one running the `JSContext`; it runs its own worker
threads internally for a single `run()` but never calls back into JavaScript.

## Example: salient-object matte with u2netp

```js
import { Session, Tensor } from 'ort';
import * as cv from 'opencv';

const session = new Session('models/u2netp.onnx');
const img = cv.imread('photo.jpg');
const blob = cv.blobFromImage(img, 0.019, new cv.Size(320, 320), new cv.Scalar(123.6, 116.2, 103.5), true, false);
const input = new Tensor('float32', blob.data32F, [1, 3, 320, 320]);
const out = session.run({ [session.inputNames[0]]: input });
const prob = out[session.outputNames[0]].data;      // 320 x 320 Float32Array
```

This is what `tests/u2netp-poc.js` does, with the comparison against OpenCV `dnn` added.
