# TODO

## Goal: minimal proof of concept, run u2netp.onnx from JavaScript

The first milestone is not the full API of README.md. It is one real model running end to end with as little of
qjs-ort as needed: `u2netp.onnx` (salient-object matte, 4.4 MB, float32 only, one image input, several outputs),
the same model `~/Sources/quickjs/qjs-opencv/tools/opencv-imgproc.js` runs through `cv.readNetFromONNX`
(`MODELS.u2netp`, `runSalient()`: input 320x320, mean `[123.6, 116.2, 103.5]`, scale `0.019`, swapRB, first output
is the probability map). The weights are `~/Sources/quickjs/qjs-opencv/models/u2netp.onnx`; never copy them into
this repo (the models directory is outside git, see `modelsDir()` there).

Done when `tests/u2netp-poc.js` loads the model, feeds a picture, takes the first output and the saliency map
matches the OpenCV `dnn` one (same script path, compare max abs difference, expect below 1e-3), and the script
runs twice in a loop without growing memory.

Cut line, everything not listed in tasks 1 to 4 stays out of the PoC: no `Session` options, no buffers as model
source, no `inputs`/`outputs` metadata, no string tensors, no async, no other execution providers, no docs,
no example directory.

Done so far (see ChangeLog): CMake skeleton, module `ort` that exports `version`, `FindONNXRuntime.cmake`,
`tools/fetch-onnxruntime.sh` with the x86_64 copy in `third_party/`.

## In progress

- (nothing)

## PoC tasks, in this order

1. **`Tensor`, float32 only**: `new Tensor('float32', data, dims)` over an `OrtValue` made with
   `CreateTensorWithDataAsOrtValue` (zero copy; keep the typed array referenced from the object so it outlives
   the value, with a `gc_mark`), getters `type`, `dims`, `data` (a `Float32Array` view). `dims` are checked
   against `data.length` and throw `RangeError` on a mismatch; a non-`Float32Array` throws `TypeError`. Other
   element types are not part of the PoC, but make the type a table lookup so adding them later is one row.
   Class registered with `js_class_define`/`js_object_from_target` (qjs-modules `utils.h`), `Symbol.toStringTag`,
   finalizer tolerates NULL. The `OrtEnv` is created once at module init (lazily, on first use) and shared.
2. **`Session`, minimal**: `new Session(path)` with the default CPU provider and default options,
   `inputNames`, `outputNames` (arrays of strings, via `SessionGetInputName`/`SessionGetOutputName` with the
   default allocator, freed after copying into JS strings), `release()`. A path that does not exist or a file that
   is not a model throws `Error` with the ORT message; nothing crashes.
3. **`session.run(feeds, outputNames?)`, synchronous**: `feeds` maps input name to `Tensor`, `outputNames`
   defaults to all outputs. Returns an object mapping output name to `Tensor`. Outputs own their `OrtValue`
   and `data` is a view onto it, no copy; copying out with `.slice()` is the caller's choice. Unknown or missing
   feed names, a wrong dims or wrong element type throw (`TypeError`/`RangeError`), never crash: check the
   bad-input cases before trusting the happy path (a silent exit 139 in a test run is the symptom).
4. **The PoC script `tests/u2netp-poc.js`** (qjsm, imports `ort` and `opencv`): read a picture with OpenCV,
   build the blob with `cv.blobFromImage` exactly as `runSalient()`, wrap `blob.data32F` in a `Tensor` with dims
   `[1, 3, 320, 320]`, run, take the first output (names come from `session.outputNames`; do not hard-code them,
   print them once), min/max normalise as `runSalient()` does, write the matte as an image, and compare with the
   `dnn` result on the same input. Print timings of both. Loop twice and check with `qjsm -d` that the memory
   counts of a run with N and 25*N rounds are equal. Test picture: a small JPEG from
   `~/Sources/quickjs/qjs-opencv/tests/` or the one from `opencv-imgproc.js`' own tests.

## After the PoC works (do not start before)

- **Tests without a download**: `tests/ort-test.js` builds a tiny ONNX model (an Add or MatMul graph written as
  protobuf bytes by hand) and checks `run` against known numbers, plus the bad-input cases; tinytest style if the
  tree has `tinytest.js` (ask before adding it).
- **Rest of the Tensor/Session surface** from README.md: element types float64/int32/int64/uint8/bool, string later;
  `new Session(pathOrBuffer, options)` with `providers`, `intraOpThreads`, `interOpThreads`, `graphOptimization`,
  `logLevel`; `inputs`/`outputs` (name, element type, dims); `ort.providers`.
- **Outputs without a copy surprises**: decide whether `Tensor.data` of an output stays valid after the tensor is
  garbage collected (it must not dangle: the view keeps the tensor alive, test it).
- **FindONNXRuntime, rest**: let CMake call `tools/fetch-onnxruntime.sh` itself when nothing is found
  (`BUILD_ORT`-style option), the aarch64 and Termux variants (`fetch-onnxruntime.sh VERSION linux-arm64`,
  Android needs the aar), `INSTALL_RPATH`, and a cross build against `/mnt/data/sysroots/ort-1.32.0-aarch64`.
- **Docs and example**: `doc/native/ort.md`, `examples/` with the u2netp cutout and a small sequence model in a
  loop; README status line and Build section updated to the real state.
- **opencv-imgproc.js**: an `ort` backend for `--dnn-model` next to `dnn` (qjs-opencv's file, ask first).
- `runAsync()` through a worker thread and a promise (no JS calls from ORT threads).
- Other execution providers (XNNPACK, NNAPI on the phone, CUDA on the workstation) through `options.providers`.
- Float16 tensors; `IoBinding` for reusing output buffers in a realtime loop (audio blocks).
- Register in the stacks: js-engines `PROJ_ort` in menuconfig (`Config.qjs-projects.in`, `configs/*_defconfig`,
  `tools/kconfig/check-options.sh`, requested from that session), through the N-API adapter on Bun/Deno
  (`qjs-api-wrap/napi-qjs-wrap.cpp`; the other adapters are `sm-`, `v8-`, `jsc-qjs-wrap.cpp` there). The quickjs
  repository wiring (`ORT:OFF` in `add_quickjs_modules_subdirs()`, `.gitmodules`) is done but uncommitted there.
- Cross builds: Raspberry Pi (`cfg-aarch64`, the ORT 1.32.0 sysroot) and Termux (the Termux ORT package).
