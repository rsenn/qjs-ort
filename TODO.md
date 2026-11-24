# TODO

## In progress

- (nothing)

## First tasks, in this order

2. **FindONNXRuntime, rest**: `cmake/FindONNXRuntime.cmake` and `tools/fetch-onnxruntime.sh` exist (ORT_PREFIX, pkg-config,
   `third_party/onnxruntime-*`). Missing: let CMake call the fetch script itself when nothing is found (`BUILD_ORT`
   style option), the aarch64 / Termux variants (`fetch-onnxruntime.sh VERSION linux-arm64`, android needs the aar),
   an `INSTALL_RPATH`, and a cross build against `/mnt/data/sysroots/ort-1.32.0-aarch64`.
3. **Tensor class**: `new Tensor(type, data, dims)` over an `OrtValue` created with
   `CreateTensorWithDataAsOrtValue` (zero copy, keep a reference to the typed array so it outlives the value),
   getters `type`, `dims`, `data`; element types float32/float64/int32/int64/uint8/bool first, string later.
4. **Session class**: `new Session(pathOrBuffer, options)`, `inputNames`, `outputNames`, `inputs`/`outputs`
   (name, element type, dims), `release()`; CPU provider only.
5. **`session.run(feeds, outputNames?)`**: synchronous; outputs are copied into fresh typed arrays first
   (`JS_NewArrayBufferCopy`), zero copy with a release callback afterwards.
6. **Tests without a download**: `tests/ort-test.js` builds a tiny ONNX model (an Add or MatMul graph written as
   protobuf bytes by hand in the test) and checks `run` against known numbers; also a bad-input test (wrong
   dtype, wrong dims, missing feed) that must throw, not crash.
7. **Docs and example**: `doc/native/ort.md`, `examples/` with a small sequence model run in a loop.

## Later

- `runAsync()` through a worker thread and a promise (no JS calls from ORT threads).
- Other execution providers (XNNPACK, NNAPI on the phone, CUDA on the workstation) through `options.providers`.
- Float16 tensors; `IoBinding` for reusing output buffers in a realtime loop (audio blocks).
- js-engines: add `PROJ_ort` to its menuconfig (`Config.qjs-projects.in`, `configs/*_defconfig`) and to
  `tools/kconfig/check-options.sh`; run through the N-API adapter on Bun/Deno.
- Cross builds: Raspberry Pi (`cfg-aarch64`, the ORT 1.32.0 sysroot) and Termux (the Termux ORT package).
