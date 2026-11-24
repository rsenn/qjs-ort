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
`tools/fetch-onnxruntime.sh` with the x86_64 copy in `third_party/`.

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
- No qjs-modules `utils.h`: it is not installed (only `libmodules.a` of the qjs-modules tree has it), so the PoC
  does not use `js_class_define`/`js_object_from_target`. Use the plain registration of skill section 3 (`JS_NewClassID`
  once, `JS_NewClass`, prototype object with `JS_SetPropertyFunctionList`, `JS_NewCFunction2(..., JS_CFUNC_constructor, 0)`,
  `JS_SetConstructor`, `JS_SetClassProto`, `JS_SetModuleExport`) and the constructor tail "read `new_target.prototype`,
  `JS_NewObjectProtoClass`" with a fallback to `JS_GetClassProto`. Write both once as two small static helpers
  (`js_ort_class_register`, `js_ort_object_new(ctx, new_target, class_id)`) used by `Tensor` and `Session`, not
  repeated per class. Whether to depend on `libmodules.a` later is a decision for the user.
- QuickJS: `JS_NewArrayBuffer(ctx, ptr, len, free_func, opaque, FALSE)` does **not** call `free_func` when it fails
  (checked in `quickjs.c`), so free what the opaque owns yourself on a failure. `JS_NewTypedArray(ctx, 1, &buf,
  JS_TYPED_ARRAY_FLOAT32)` makes the view. `JS_GetTypedArrayBuffer(ctx, v, &off, &bytes, &bpe)` returns a new
  reference to the backing `ArrayBuffer` (free it) and fails by throwing when `v` is not a typed array or
  `DataView`. There is no public "which typed array is this", test with `JS_IsInstanceOf(ctx, v, ctor)` where
  `ctor` is `Float32Array` fetched once from the global object.
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

### 1. `Tensor`, float32 only

Files: `quickjs-ort.c`. Test by hand with a throw-away script; the permanent test is task 4.

Shared state, all `static` and created on first use by `js_ort_init_api(JSContext*)`, which returns 0 or throws:

```c
static const OrtApi* ort_api;      /* OrtGetApiBase()->GetApi(ORT_API_VERSION), NULL: library too old */
static OrtEnv* ort_env;            /* CreateEnv(ORT_LOGGING_LEVEL_WARNING, "qjs-ort", &ort_env) */
static OrtMemoryInfo* ort_cpu;     /* CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ort_cpu) */
```

The env and memory info live until the process ends (a module has no unload hook); say so in a comment. Call
`js_ort_init_api` at the top of the `Tensor` and `Session` constructors, not at module init, so that
`import 'ort'` keeps working to print `version` on a machine where the runtime is unusable.

Error helper used by every later task:

```c
static JSValue js_ort_throw_status(JSContext* ctx, OrtStatus* status);
```

It reads message and code, builds the exception with `JS_NewError`, sets the message with
`JS_DefinePropertyValueStr(ctx, err, "message", JS_NewString(ctx, msg), JS_PROP_C_W_E)` (so it is a plain own
property) and `code` with `JS_NewInt32`, releases the status, and returns `JS_Throw(ctx, err)`. Callers write
`if((status = ort_api->X(...))) return js_ort_throw_status(ctx, status);`, so the helper may assume `status` is non-NULL.

Element type table, the only place that knows about types (one row per type; the PoC enables one row, the other
rows are written as comments so adding them is uncommenting):

```c
static const struct { const char* name; ONNXTensorElementDataType onnx; size_t size; JSTypedArrayEnum array; }
js_ort_types[] = {
  { "float32", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 4, JS_TYPED_ARRAY_FLOAT32 },
  /* { "float64", ..._DOUBLE, 8, JS_TYPED_ARRAY_FLOAT64 }, int32, int64 (BigInt64Array), uint8, bool (Uint8Array) */
};
```

with two lookups, by name and by `ONNXTensorElementDataType`. The `Float32Array` constructor needed by
`JS_IsInstanceOf` is looked up from the global object at first use and kept in a static `JSValue` (a global
class constructor, never freed; acceptable, note it in a comment). Per-type constructors for later types are a
second table column, not new code.

Native struct, in `JS_SetOpaque` of every Tensor object:

```c
typedef struct {
  OrtValue* value;     /* owned, released in the finalizer */
  int type;            /* index into js_ort_types */
  JSValue data;        /* input tensor: the typed array, owned (JS_DupValue); output tensor: JS_UNDEFINED */
} JSOrtTensor;
```

Allocate with `js_mallocz`, free with `js_free_rt`.

Constructor `js_ort_tensor_ctor(ctx, new_target, argc, argv)`:

1. `argc < 3` is `TypeError("Tensor needs type, data and dims")`.
2. `type`: `JS_ToCString`, look it up, free the C string, unknown name is `TypeError("unknown tensor type 'x'")`.
3. `data`: must be a `Float32Array` for this type (`JS_IsInstanceOf` against the stored constructor), else
   `TypeError("float32 tensor needs a Float32Array")`. Get buffer pointer and window with
   `JS_GetTypedArrayBuffer` + `JS_GetArrayBuffer`, add the byte offset (skill section 7, "the +offset is
   required"), free the buffer reference. A zero-length array is allowed (dims containing 0).
4. `dims`: `JS_IsArray`, else `TypeError`. Read `length` and each element with `JS_ToInt64`/`JS_ToIndex`-style
   checks: negative or non-integer is `RangeError("bad dimension")`. Keep them in an `int64_t[]` (`js_malloc`
   sized `length`, a `dims` of length 0 is a scalar: allocate at least 1 element). Multiply with overflow
   check, the product must equal `data.length` (`bytes / size`), else `RangeError("dims do not match data
   length: <prod> vs <len>")`.
5. `ort_api->CreateTensorWithDataAsOrtValue(ort_cpu, ptr, bytes, dims, ndims, onnx_type, &value)`; failure goes
   through `js_ort_throw_status`.
6. `obj = js_ort_object_new(ctx, new_target, js_ort_tensor_class_id)`, `s->data = JS_DupValue(ctx, argv[1])`,
   `JS_SetOpaque(obj, s)`.
7. Single `fail:` label that frees dims, releases `value`, frees `s`; written once, every error path jumps there
   (skill section 3).

The typed array's memory must not move while the tensor exists: QuickJS ArrayBuffers do not move, but they can
be **detached** or resized; a detached buffer would leave a dangling pointer. State the limit in the doc page
("do not transfer or resize the buffer"), and check `JS_GetArrayBuffer` returning NULL at construction. Do not
try to guard against later detaching in the PoC.

Finalizer `js_ort_tensor_finalizer(JSRuntime*, JSValue)`: `JS_GetOpaque`, NULL tolerated, `ReleaseValue`,
`JS_FreeValueRT(rt, s->data)`, `js_free_rt`. Order matters for an input tensor: the OrtValue does not own the
data, so release the value first, then drop the array. `gc_mark` marks `s->data` with `JS_MarkValue`.

Getters (`JS_CGETSET_MAGIC_DEF` or plain `JS_CGETSET_DEF`, all read-only):

- `type`: `JS_NewString(js_ort_types[s->type].name)`.
- `dims`: `ort_api->GetTensorTypeAndShape(value, &info)`, `GetDimensionsCount`, `GetDimensions` into a stack or
  `js_malloc`ed `int64_t[]`, build a plain `Array` of `JS_NewInt64`, always `ReleaseTensorTypeAndShapeInfo(info)`
  (also on the error path; one `goto done`).
- `data`: input tensor: `JS_DupValue(ctx, s->data)`. Output tensor (`s->data` undefined): see task 3, it is
  written there; in task 1 return `JS_UNDEFINED` for that case with a `/* task 3 */` comment, not a guess.
- `Symbol.toStringTag` `"Tensor"` (`JS_PROP_STRING_DEF("[Symbol.toStringTag]", "Tensor", JS_PROP_CONFIGURABLE)`).

Add `Tensor` to the module exports (`JS_AddModuleExport(ctx, m, "Tensor")` in the init function, and
`JS_SetModuleExport` of the constructor in `js_ort_init`; both names must match). Class registration happens in
`js_ort_init`, not in the exported-names function (see `js_ort_class_register` above).

Also export a second helper for task 3, `static JSValue js_ort_tensor_wrap(JSContext*, OrtValue*)`: makes an
output tensor object that **takes ownership** of an `OrtValue` (type read from `GetTensorTypeAndShape` +
`GetTensorElementType`; a type that is not in the table throws `TypeError("unsupported element type N")` and
releases the value), uses `JS_GetClassProto` for the prototype (`JS_NewObjectProtoClass`, free the proto reference). Write it now, it shares the struct with the
constructor.

Verify: construct with a `Float32Array(6)` and `[2, 3]`, check `type`, `dims`, `data === the same array`; dims
`[2, 2]` throws `RangeError`; `new Tensor('float32', new Int32Array(4), [4])` and `new Tensor('int7', ...)` throw
`TypeError`; a subclass `class T extends Tensor {}` works; 25000 rounds of constructing and dropping keep the
`qjsm -d` counts equal; `Object.prototype.toString.call(t)` is `[object Tensor]`.

### 2. `Session`, minimal

Files: `quickjs-ort.c`.

Native struct:

```c
typedef struct {
  OrtSession* session;                  /* owned; NULL after release() */
  size_t n_inputs, n_outputs;
  char** input_names;                   /* js_malloc'd array of js_strdup'd names, n_inputs long */
  char** output_names;
} JSOrtSession;
```

Names are copied out once at construction and kept as C strings, because `Run` needs `const char* const*`
arrays and the getters need them again. They are freed in the finalizer with `js_free_rt` (one function
`js_ort_session_free_names(JSRuntime*, char**, size_t)`; `js_strdup` is context-bound, so use
`js_malloc` + `memcpy` or `js_strdup` at construction and `js_free_rt` in the finalizer, both are the same
allocator).

Constructor `js_ort_session_ctor`:

1. `js_ort_init_api`; `argc < 1` is `TypeError("Session needs a model path")`; the argument must be a string
   (`JS_IsString`) else `TypeError` (buffers are not in the PoC; say so in the message: "model path expected").
2. `path = JS_ToCString`; `ort_api->CreateSessionOptions(&opts)`; `CreateSession(ort_env, path, opts, &session)`;
   `ReleaseSessionOptions(opts)` and `JS_FreeCString` on every path. Linux/macOS only: under `#ifdef _WIN32`
   the path type is `wchar_t`, write `#error "wide path conversion not implemented"` rather than a wrong
   `char*` call (a task for the later list).
3. `SessionGetInputCount`/`SessionGetOutputCount`, then per index `SessionGetInputName(session, i, alloc,
   &name)` / `SessionGetOutputName`, copy into the arrays, `alloc->Free(alloc, name)`.
4. Object via `js_ort_object_new`, `JS_SetOpaque`. One `fail:` label releasing the session and the names
   collected so far.

Missing file: ORT answers with code `ORT_NO_SUCHFILE`; not a model: `ORT_INVALID_PROTOBUF` or `ORT_NO_MODEL`;
both surface through `js_ort_throw_status`, the message is ORT's, do not rewrite it.

Members:

- `inputNames`, `outputNames`: getters that build a new `Array` of `JS_NewString` each call (`JS_NewArray`,
  `JS_SetPropertyUint32`). After `release()` the cached names are still valid, so they keep working.
- `release()`: `JS_GetOpaque2`, `ReleaseSession`, set `session = NULL`, return `undefined`; second call is a no-op.
- `Symbol.toStringTag` `"Session"`.
- `run` is task 3; register its slot only then.

Finalizer: release the session if not NULL, free names and struct; NULL opaque tolerated.

Export `Session` next to `Tensor`.

Verify (by hand, the model file is `~/Sources/quickjs/qjs-opencv/models/u2netp.onnx`): construct it, print
`inputNames` and `outputNames`, write them down in the TODO item of task 4 (they are not known yet); a missing
path throws `Error` with `code === 3`; a text file as model throws `Error` with `code` 7 or 4; `release()` twice;
`new Session()` and `new Session(42)` throw `TypeError`; leak check with 100 rounds of load and drop (the model
is 4.4 MB, so also watch the process RSS once, it must not climb by 4.4 MB per round).

### 3. `session.run(feeds, outputNames?)`

Files: `quickjs-ort.c`.

Method `js_ort_session_run(ctx, this_val, argc, argv)`, registered as `JS_CFUNC_DEF("run", 1, ...)`:

1. `s = JS_GetOpaque2(...)`; `s->session == NULL` is `Error("session released")`.
2. `argc < 1` or `feeds` not an object is `TypeError("run needs an object of input tensors")`.
3. Feeds: `JS_GetOwnPropertyNames(ctx, &tab, &len, feeds, JS_GPN_STRING_MASK | JS_GPN_ENUM_ONLY)`. Allocate
   `const char** in_names` and `const OrtValue** in_values` (both `js_mallocz` of `n_inputs`; a `len` different
   from `n_inputs` is already an error: `TypeError` naming the first missing or unknown input, found by the two
   loops below). For each key: `JS_AtomToCString`, find it in `s->input_names` with `strcmp`, not found is
   `TypeError("unknown input 'name'")`; the value `JS_GetProperty(ctx, feeds, atom)`, `JS_GetOpaque2` with
   the tensor class id (a non-Tensor is `TypeError("input 'name' is not a Tensor")`); store the **cached** name
   pointer and `tensor->value` in the position of the model's input order (so `in_names[i]`/`in_values[i]`
   follow `s->input_names[i]`). Free the atom string, the value reference and, at the end, the property table
   (`JS_FreePropertyEnum(ctx, tab, len)`; it also frees the atoms in the table). After the loop every slot must be filled, a hole is `TypeError("missing input 'name'")`.
   The `Tensor` JS objects stay alive for the whole call because `feeds` holds them; do not release their
   values.
4. Output names: when `argc > 1` and not undefined, an array of strings, each checked against
   `s->output_names` (`TypeError("unknown output 'name'")`), the cached pointer stored; else all outputs.
   `n_out` is their count (zero outputs requested is `TypeError`).
5. `OrtValue** out_values = js_mallocz(n_out * sizeof(void*))` (all NULL), then
   `ort_api->Run(s->session, NULL, in_names, in_values, n_in, out_names, n_out, out_values)`. Run is blocking and
   must not call into JS; do nothing else between the arrays and the call.
6. On a failed status: release every non-NULL `out_values[i]`, free all arrays, `js_ort_throw_status`. Shape and
   element-type mismatches of an input are reported by ORT as `ORT_INVALID_ARGUMENT` / `ORT_FAIL`; do not
   duplicate the checks in the binding, but make sure the bad-input test below really gets a status and no
   crash.
7. On success: `result = JS_NewObject(ctx)`, then per output `t = js_ort_tensor_wrap(ctx, out_values[i])` (it
   takes ownership; on its failure release the values from `i + 1` on, free the result, rethrow),
   `JS_DefinePropertyValueStr(ctx, result, out_names[i], t, JS_PROP_C_W_E)` (that consumes `t`). Return it.
8. Free the three arrays on every exit path (use one `done:` label, no early `return` after the first
   allocation).

Output tensor `data` (completes the getter from task 1; `s->data` is undefined for an output):

```c
typedef struct { JSValue tensor; } JSOrtView;   /* keeps the Tensor object alive as long as the view exists */
static void js_ort_view_free(JSRuntime* rt, void* opaque, void* ptr);   /* JS_FreeValueRT(tensor), js_free_rt(opaque) */
```

`data` getter for an output: `ort_api->GetTensorMutableData(value, &ptr)`; element count from
`GetTensorTypeAndShape` + `GetTensorShapeElementCount` (release the info); `bytes = count * size`;
`view = js_malloc(sizeof *view)`, `view->tensor = JS_DupValue(ctx, this_val)`;
`buf = JS_NewArrayBuffer(ctx, ptr, bytes, js_ort_view_free, view, FALSE)`; on failure free the dup and the
view yourself (the callback is not called then); `JS_NewTypedArray(ctx, 1, &buf, type.array)` and free `buf`.
**Do not cache the view in the tensor**: a cached view whose buffer holds the tensor is a reference cycle that
QuickJS cannot see, and it leaks. A fresh view per read is the rule; document it (it is in `doc/native/ort.md`).
If creating the view per read turns out to cost real time in the PoC loop, measure first, then consider
`JS_NewArrayBufferCopy` for outputs (a copy, no lifetime coupling) as the fallback and say so in the ChangeLog.

Verify with a model that needs no download: none exists until the hand-built test model (later list), so this
task is verified through task 4; until then, check only the error paths without a model: a released session,
`run()` with no arguments, with `{}` against a session that needs an input (use u2netp from task 2),
with `{ [name]: 42 }`, with an unknown key, a `Tensor` with wrong dims for the input (must throw `Error`
with a code, not crash), `outputNames` with an unknown name. Each of those throws; none crashes.

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
