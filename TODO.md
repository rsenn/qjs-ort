# TODO

## Status: the proof of concept works (2026-11-24)

`tests/u2netp-poc.js` runs `u2netp.onnx` (the salient-object model of `qjs-opencv/tools/opencv-imgproc.js`) through
`Session`, `Tensor` and `session.run()` and compares the first output with OpenCV `dnn`: max abs difference 7.6e-5,
ORT 290 ms against OpenCV 294 ms for one 320x320 run, 25 runs in a loop give identical results. Run it with the
command in `CLAUDE.md`. The weights stay in `~/Sources/quickjs/qjs-opencv/models/`, never copy them into this repo.

The API is sketched in `doc/native/ort.md`, keep that page in step with the code (entries marked PoC exist).
Done so far, see the ChangeLog: CMake skeleton, `FindONNXRuntime.cmake`, `tools/fetch-onnxruntime.sh`,
`js-utils.[ch]`, the float32 `Tensor`, the minimal `Session`, `session.run()` with output tensors, the PoC script.

## In progress

- (nothing)

## How to work on C tasks here

Read first: `CLAUDE.md` of this project, `doc/native/ort.md`, the `quickjs-native-bindings` skill (sections 3, 4, 7,
16, 17, 19, 23), `comment-style` (context `C | quickjs | ...`). Everything goes into `quickjs-ort.c` for now (one
file; split into `ort-*.c` only when it passes about 800 lines, and then ask). Build and run exactly as in
`CLAUDE.md` (`cmake --build build/x86_64-linux-debug`, `QUICKJS_MODULE_PATH=build/x86_64-linux-debug qjsm tests/x.js`). `qjsm -e` cannot `import`, write
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

## Next, in this order

1. **Rest of the Tensor/Session surface** from `doc/native/ort.md`: element type string, float16 later;
  `new Session(pathOrBuffer, options)` with `providers`, `intraOpThreads`, `interOpThreads`,
  `graphOptimization`, `logLevel`; `inputs`/`outputs` (name, element type, dims); `ort.providers`; wide paths on
  Windows.
2. **FindONNXRuntime, rest**: let CMake call `tools/fetch-onnxruntime.sh` itself when nothing is found
  (`BUILD_ORT`-style option), the aarch64 and Termux variants (`fetch-onnxruntime.sh VERSION linux-arm64`,
  Android needs the aar), and a cross build against `/mnt/data/sysroots/ort-1.32.0-aarch64`.
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
