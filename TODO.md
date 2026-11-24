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
source, no `inputs`/`outputs` metadata, no string tensors, no async, no other execution providers, no example
directory. The API is sketched in `doc/native/ort.md`; keep that page in step with the code (the PoC entries are
marked there).

Done so far (see ChangeLog): CMake skeleton, module `ort` that exports `version`, `FindONNXRuntime.cmake`,
`tools/fetch-onnxruntime.sh` with the x86_64 copy in `third_party/`, `js-utils.[ch]`, the float32 `Tensor` class, the minimal `Session` class, `session.run()` with output tensors.

## In progress

- (nothing)

## How to work on the tasks (applies to 1 to 4)

Read first: `CLAUDE.md` of this project, `doc/native/ort.md`, the `quickjs-native-bindings` skill (sections 3, 4, 7,
16, 17, 19, 23), `comment-style` (context `C | quickjs | ...`). Everything goes into `quickjs-ort.c` for now (one
file; split into `ort-*.c` only when it passes about 800 lines, and then ask). Build and run exactly as in
`CLAUDE.md`: `cmake --build build`, `QUICKJS_MODULE_PATH=build qjsm tests/x.js`. `qjsm -e` cannot `import`, write
a script file. Check the build output for warnings; there must be none.

Facts verified against this tree, so you do not have to rediscover them:

- ONNX Runtime headers: `third_party/onnxruntime-linux-x64-1.32.0/include/onnxruntime_c_api.h`
  (`ORT_API_VERSION` is 32). Every call goes through a `const OrtApi*` table, `ort->CreateEnv(...)`, there are
  no free functions. Get it once with `OrtGetApiBase()->GetApi(ORT_API_VERSION)`; it is NULL when the loaded
  library is older than the header, throw `Error` then.
- Almost every `OrtApi` function returns an `OrtStatus*`: NULL is success, else it must be released with
  `ort->ReleaseStatus(status)` after reading `ort->GetErrorMessage(status)` and `ort->GetErrorCode(status)`.
  Never ignore a returned status; that is how bad input becomes a crash later.
- `OrtValue`, `OrtSession`, `OrtEnv`, `OrtMemoryInfo`, `OrtSessionOptions`, `OrtTensorTypeAndShapeInfo` are freed
  with `ort->ReleaseValue`, `ReleaseSession`, ... (all tolerate NULL).
- Strings from `SessionGetInputName`/`SessionGetOutputName` are allocated with the allocator passed in; use
  `ort->GetAllocatorWithDefaultOptions(&alloc)` (do not release it, it is a process singleton) and free each name
  with `alloc->Free(alloc, name)` after copying it.
- Helpers live in `js-utils.[ch]`, read `js-utils.h` first, every function has its contract there. They are
  tested, do not reimplement them: `js_class_register` (class, prototype, constructor in one call),
  `js_object_from_target` (the constructor tail), `js_typedarray_kind`/`js_typedarray_get`/`js_typedarray_expect`
  (exact typed array kind, bytes without copy, offset applied), `js_typedarray_external` (view over native memory
  with a free callback), `js_typedarray_copy`, `js_array_toint64` (dims), `js_array_length`, `js_global_get_str`.
  Kinds are the `JSTypedArrayEnum` of `quickjs.h`. qjs-modules `utils.h` is not installed, so it is not used.
- QuickJS facts, tested; `js-utils.c` already handles them, they explain why it looks as it does:
  `JS_NewArrayBuffer` does not call its free function when it fails, `js_typedarray_external` calls it exactly
  once in every case; `JS_NewTypedArray(ctx, 1, &buf, kind)` gives a **length 0** array for an ArrayBuffer, the
  view needs `(buf, 0, count)`; `JS_GetTypedArrayBuffer` rejects a `DataView` here; the class id of a typed array
  gives its exact kind (an `Int16Array` is not a `Uint16Array`), the `constructor.name` trick of qjs-sound does not.
- Pointer rules from the skill: `JS_GetOpaque2` in every method, `JS_GetOpaque` only in finalizers; finalizers get a
  `JSRuntime*`, so free with `js_free_rt`, `JS_FreeValueRT`; read `argv[i]` only after checking `argc > i`; a
  `JSValue` kept in a native struct needs `JS_DupValue` and a `gc_mark` that calls `JS_MarkValue` on it.
- Code style: no braces around one-statement bodies, declarations initialised where the value is known, group
  declarations as the skill says; `JSValueConst argv[]` in signatures; names `js_ort_<class>_<what>`, class ids
  `js_ort_tensor_class_id`, `js_ort_session_class_id`; comments per `comment-style` (print the ruleset with
  `comment-style/scripts/rules-for 'C | quickjs | js-facing'` and `'... | prototype'`).
- Every task ends with: ChangeLog entry (format in `CLAUDE.md`), this TODO item deleted, `doc/native/ort.md`
  adjusted if the behaviour differs from the sketch, bad-input cases tried by hand (a crash shows up as a silent
  exit 139), and a leak check as in skill section 16 (run with N and 25*N rounds under `qjsm -d`, diff the dumps).
  Do not commit unless asked.

## PoC tasks, in this order

(Tasks 1 to 3, the float32 `Tensor`, the minimal `Session` and `session.run()`, are done: see the ChangeLog. The u2netp input name is `input.1`, its outputs are `1959` to `1965`; the first output is the probability map.)

### 4. The PoC script `tests/u2netp-poc.js`

Files: `tests/u2netp-poc.js` only (plus the ChangeLog and TODO updates). Runs under `qjsm`:
`QUICKJS_MODULE_PATH=build:<opencv build dir> qjsm tests/u2netp-poc.js [picture] [model]` (use `;` as separator
if `:` fails, skill section 19); defaults: model `~/Sources/quickjs/qjs-opencv/models/u2netp.onnx` (from the
`WB_MODELS_DIR` environment variable if set, as `modelsDir()` does), picture `~/Sources/quickjs/qjs-opencv/tests/
test_linesegmentdetector.jpg` or any JPEG given. Find a working qjs-opencv build with `ls ~/Sources/quickjs/qjs-opencv/build*`
or ask; do not build OpenCV.

Steps, in this order, printing one line each:

1. `session = new Session(model)`; print `inputNames` and `outputNames` (this is where the real names are learned;
   write them as a comment at the top of the script).
2. `img = cv.imread(picture)`; blob with exactly the call of `runSalient()` in opencv-imgproc.js (copy its
   arguments, `work = img` unscaled is fine for the PoC): `cv.blobFromImage(img, 0.019, new cv.Size(320, 320),
   new cv.Scalar(123.6, 116.2, 103.5), true, false)`.
3. ORT path: `new Tensor('float32', blob.data32F, [1, 3, 320, 320])` (check `blob.data32F` is a `Float32Array`
   view; if it is not, build one from the blob's buffer and note it in BUGS of qjs-opencv, not here), run with the
   first input name, take the first output name's tensor, `.data` is the 320x320 map. Time it.
4. OpenCV path: `cv.readNetFromONNX(model, cv.dnn.ENGINE_AUTO)`, `net.setInput(blob)`,
   `net.forward(net.getUnconnectedOutLayersNames())[0].data32F`. Time it.
5. Compare: max absolute difference over all 320*320 values, print it; fail (`std.exit(1)`) when it is 1e-3
   or more. Print the shape and min/max of both.
6. Min/max normalise the ORT map as `runSalient()` does, build a `cv.Mat` (`CV_32F`, 320x320), convert to 8 bit
   and write `u2netp-matte.png` next to the script (in the scratch dir, not the repo; accept `--out`).
7. Loop: run the ORT path 25 times in a row, check each result equals the first one exactly (same input, same
   model: deterministic on CPU; if not, relax to 1e-6 and say why).
8. Leak check, by hand, recorded in the ChangeLog entry: `qjsm -d tests/u2netp-poc.js` with the loop count 5 and
   125 (take the count from `argv`), `diff` the memory dumps: object and byte counts equal.

The script uses plain top-level code and `console.log`; no `tinytest.js` yet (adding it is a separate decision,
ask first). Exit status 0 on success, 1 on any failed check; any exception is printed with its stack. When it
passes, the PoC milestone is reached: update the "Done so far" paragraph above, move what is next from "After
the PoC works" into the numbered list, and tell the user.

## After the PoC works (do not start before)

- **Tests without a download**: `tests/ort-test.js` builds a tiny ONNX model (an Add or MatMul graph written as
  protobuf bytes by hand) and checks `run` against known numbers, plus the bad-input cases of tasks 1 to 3 as
  permanent tests; tinytest style if the tree has `tinytest.js` (ask before adding it).
- **Rest of the Tensor/Session surface** from `doc/native/ort.md`: element types float64/int32/int64/uint8/bool,
  string later; `new Session(pathOrBuffer, options)` with `providers`, `intraOpThreads`, `interOpThreads`,
  `graphOptimization`, `logLevel`; `inputs`/`outputs` (name, element type, dims); `ort.providers`; wide paths on
  Windows.
- **FindONNXRuntime, rest**: let CMake call `tools/fetch-onnxruntime.sh` itself when nothing is found
  (`BUILD_ORT`-style option), the aarch64 and Termux variants (`fetch-onnxruntime.sh VERSION linux-arm64`,
  Android needs the aar), `INSTALL_RPATH`, and a cross build against `/mnt/data/sysroots/ort-1.32.0-aarch64`.
- **Examples and README**: `examples/` with the u2netp cutout and a small sequence model in a loop; README status
  line and Build section updated to the real state.
- **opencv-imgproc.js**: an `ort` backend for `--dnn-model` next to `dnn` (qjs-opencv's file, ask first).
- `runAsync()` through a worker thread and a promise (no JS calls from ORT threads).
- Other execution providers (XNNPACK, NNAPI on the phone, CUDA on the workstation) through `options.providers`.
- Float16 tensors; `IoBinding` for reusing output buffers in a realtime loop (audio blocks).
- Register in the stacks: js-engines `PROJ_ort` in menuconfig (`Config.qjs-projects.in`, `configs/*_defconfig`,
  `tools/kconfig/check-options.sh`, requested from that session), through the N-API adapter on Bun/Deno
  (`qjs-api-wrap/napi-qjs-wrap.cpp`; the other adapters are `sm-`, `v8-`, `jsc-qjs-wrap.cpp` there). The quickjs
  repository wiring (`ORT:OFF` in `add_quickjs_modules_subdirs()`, `.gitmodules`) is done but uncommitted there.
- Cross builds: Raspberry Pi (`cfg-aarch64`, the ORT 1.32.0 sysroot) and Termux (the Termux ORT package).
