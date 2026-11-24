/*
 * quickjs-ort.c - ONNX Runtime for QuickJS, through its C API.
 */
#include <math.h>
#include <quickjs.h>
#include <cutils.h>
#include <onnxruntime_c_api.h>
#include "js-utils.h"

static const OrtApi* ort_api;      /* OrtGetApiBase()->GetApi(ORT_API_VERSION), NULL: library too old */
static OrtEnv* ort_env;            /* CreateEnv(ORT_LOGGING_LEVEL_WARNING, "qjs-ort", &ort_env) */
static OrtMemoryInfo* ort_cpu;     /* CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ort_cpu) */

static const struct { const char* name; ONNXTensorElementDataType onnx; size_t size; JSTypedArrayEnum array; }
js_ort_types[] = {
  { "float32", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 4, JS_TYPED_ARRAY_FLOAT32 },
  { "float64", ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, 8, JS_TYPED_ARRAY_FLOAT64 },
  { "int32", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, 4, JS_TYPED_ARRAY_INT32 },
  { "int64", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 8, JS_TYPED_ARRAY_BIG_INT64 },
  { "uint8", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, 1, JS_TYPED_ARRAY_UINT8 },
  { "bool", ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, 1, JS_TYPED_ARRAY_UINT8 },
};

static int
js_ort_type_by_name(const char* name) {
  for(size_t i = 0; i < countof(js_ort_types); i++)
    if(!strcmp(name, js_ort_types[i].name))
      return i;
  return -1;
}

static int
js_ort_type_by_onnx(ONNXTensorElementDataType onnx) {
  for(size_t i = 0; i < countof(js_ort_types); i++)
    if(js_ort_types[i].onnx == onnx)
      return i;
  return -1;
}

typedef struct {
  OrtValue* value;     /* owned, released in the finalizer */
  int type;            /* index into js_ort_types */
  JSValue data;        /* input tensor: the typed array, owned (JS_DupValue); output tensor: JS_UNDEFINED */
} JSOrtTensor;

typedef struct {
  OrtSession* session;                  /* owned; NULL after release() */
  size_t n_inputs, n_outputs;
  char** input_names;                   /* js_malloc'd array of js_strdup'd names, n_inputs long */
  char** output_names;                  /* js_malloc'd array of js_strdup'd names, n_outputs long */
} JSOrtSession;

/* view over an output tensor's memory, keeps the tensor object alive */
typedef struct {
  JSValue tensor;                       /* JS_DupValue of the Tensor object */
} JSOrtView;

static JSClassID js_ort_tensor_class_id;
static JSClassID js_ort_session_class_id;

static JSValue js_ort_throw_status(JSContext* ctx, OrtStatus* status);

/* the env and memory info live until the process ends, a module has
 * no unload hook */
static int
js_ort_init_api(JSContext* ctx) {
  OrtStatus* status;

  if(ort_api)
    return 0;

  if(!(ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION))) {
    JS_ThrowTypeError(ctx, "onnxruntime library too old, need API version %d", ORT_API_VERSION);
    return -1;
  }

  if((status = ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "qjs-ort", &ort_env))) {
    js_ort_throw_status(ctx, status);
    return -1;
  }

  if((status = ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ort_cpu))) {
    js_ort_throw_status(ctx, status);
    return -1;
  }

  return 0;
}

static JSValue
js_ort_throw_status(JSContext* ctx, OrtStatus* status) {
  const char* msg = ort_api->GetErrorMessage(status);
  OrtErrorCode code = ort_api->GetErrorCode(status);
  JSValue err = JS_NewError(ctx);

  JS_DefinePropertyValueStr(ctx, err, "message", JS_NewString(ctx, msg), JS_PROP_C_W_E);
  JS_DefinePropertyValueStr(ctx, err, "code", JS_NewInt32(ctx, code), JS_PROP_C_W_E);
  ort_api->ReleaseStatus(status);
  return JS_Throw(ctx, err);
}

static void
js_ort_tensor_finalizer(JSRuntime* rt, JSValue val) {
  JSOrtTensor* s;

  if(!(s = JS_GetOpaque(val, js_ort_tensor_class_id)))
    return;

  if(s->value)
    ort_api->ReleaseValue(s->value);

  if(!JS_IsUndefined(s->data))
    JS_FreeValueRT(rt, s->data);

  js_free_rt(rt, s);
}

static void
js_ort_session_free_names(JSRuntime* rt, char** names, size_t n) {
  for(size_t i = 0; i < n; i++)
    js_free_rt(rt, names[i]);
  js_free_rt(rt, names);
}

static void
js_ort_session_finalizer(JSRuntime* rt, JSValue val) {
  JSOrtSession* s;

  if(!(s = JS_GetOpaque(val, js_ort_session_class_id)))
    return;

  if(s->session)
    ort_api->ReleaseSession(s->session);

  if(s->input_names)
    js_ort_session_free_names(rt, s->input_names, s->n_inputs);

  if(s->output_names)
    js_ort_session_free_names(rt, s->output_names, s->n_outputs);

  js_free_rt(rt, s);
}

static void
js_ort_view_free(JSRuntime* rt, void* opaque, void* ptr) {
  JSOrtView* view = opaque;
  JS_FreeValueRT(rt, view->tensor);
  js_free_rt(rt, view);
}

static void
js_ort_tensor_gc_mark(JSRuntime* rt, JSValueConst val, JS_MarkFunc* mark_func) {
  JSOrtTensor* s;

  if((s = JS_GetOpaque(val, js_ort_tensor_class_id)) && !JS_IsUndefined(s->data))
    JS_MarkValue(rt, s->data, mark_func);
}

static JSValue
js_ort_tensor_ctor(JSContext* ctx, JSValueConst new_target, int argc, JSValueConst argv[]) {
  const char* type_name = NULL;
  int type;
  BufferView view;
  size_t ndims;
  int64_t* dims = NULL;
  size_t elem_size;
  size_t total = 1;
  OrtValue* value = NULL;
  OrtStatus* status = NULL;
  JSValue obj;
  JSValue err;
  JSOrtTensor* s;

  if(js_ort_init_api(ctx))
    return JS_EXCEPTION;

  if(argc < 3)
    return JS_ThrowTypeError(ctx, "Tensor needs type, data and dims");

  if(!(type_name = JS_ToCString(ctx, argv[0])))
    return JS_EXCEPTION;

  if((type = js_ort_type_by_name(type_name)) < 0) {
    err = JS_ThrowTypeError(ctx, "unknown tensor type '%s'", type_name);
    JS_FreeCString(ctx, type_name);
    return err;
  }

  JS_FreeCString(ctx, type_name);

  if(js_typedarray_expect(ctx, argv[1], js_ort_types[type].array, &view) < 0)
    return JS_EXCEPTION;

  if(!(dims = js_array_toint64(ctx, argv[2], &ndims)))
    return JS_EXCEPTION;

  elem_size = js_ort_types[type].size;

  for(size_t i = 0; i < ndims; i++) {
    if(dims[i] == 0) {
      total = 0;
    } else if(total > SIZE_MAX / (size_t)dims[i]) {
      js_free(ctx, dims);
      return JS_ThrowRangeError(ctx, "dims overflow");
    } else {
      total *= (size_t)dims[i];
    }
  }

  if(total * elem_size != view.len) {
    js_free(ctx, dims);
    return JS_ThrowRangeError(ctx, "dims do not match data length: %zu vs %zu", total * elem_size, view.len);
  }

  if((status = ort_api->CreateTensorWithDataAsOrtValue(ort_cpu, view.ptr, view.len, dims, ndims, js_ort_types[type].onnx, &value))) {
    js_free(ctx, dims);
    return js_ort_throw_status(ctx, status);
  }

  js_free(ctx, dims);

  if(JS_IsException(obj = js_object_from_target(ctx, new_target, js_ort_tensor_class_id))) {
    ort_api->ReleaseValue(value);
    return JS_EXCEPTION;
  }

  if(!(s = js_mallocz(ctx, sizeof(*s)))) {
    ort_api->ReleaseValue(value);
    return JS_ThrowOutOfMemory(ctx);
  }

  s->value = value;
  s->type = type;
  s->data = JS_DupValue(ctx, argv[1]);
  JS_SetOpaque(obj, s);
  return obj;
}

static JSValue
js_ort_tensor_get_type(JSContext* ctx, JSValueConst this_val) {
  JSOrtTensor* s;

  if(!(s = JS_GetOpaque2(ctx, this_val, js_ort_tensor_class_id)))
    return JS_EXCEPTION;

  return JS_NewString(ctx, js_ort_types[s->type].name);
}

static JSValue
js_ort_tensor_get_dims(JSContext* ctx, JSValueConst this_val) {
  OrtTensorTypeAndShapeInfo* info = NULL;
  OrtStatus* status = NULL;
  size_t count = 0;
  JSOrtTensor* s;
  JSValue arr;
  int64_t* dims = NULL;

  if(!(s = JS_GetOpaque2(ctx, this_val, js_ort_tensor_class_id)))
    return JS_EXCEPTION;

  if((status = ort_api->GetTensorTypeAndShape(s->value, &info)))
    return js_ort_throw_status(ctx, status);

  if((status = ort_api->GetDimensionsCount(info, &count))) {
    ort_api->ReleaseTensorTypeAndShapeInfo(info);
    return js_ort_throw_status(ctx, status);
  }

  if(!JS_IsException(arr = JS_NewArray(ctx))) {
    if((dims = js_malloc(ctx, count * sizeof(int64_t)))) {
      if((status = ort_api->GetDimensions(info, dims, count))) {
        JS_FreeValue(ctx, arr);
        arr = js_ort_throw_status(ctx, status);
      } else {
        for(int64_t i = 0; i < count; i++)
          JS_SetPropertyUint32(ctx, arr, i, JS_NewInt64(ctx, dims[i]));
      }

      js_free(ctx, dims);
    } else {
      JS_FreeValue(ctx, arr);
      arr = JS_ThrowOutOfMemory(ctx);
    }
  }

  ort_api->ReleaseTensorTypeAndShapeInfo(info);
  return arr;
}

static JSValue
js_ort_tensor_get_data(JSContext* ctx, JSValueConst this_val) {
  JSOrtTensor* s;
  OrtTensorTypeAndShapeInfo* info = NULL;
  OrtStatus* status = NULL;
  size_t count = 0;
  void* ptr = NULL;
  JSOrtView* view = NULL;
  JSValue arr;

  if(!(s = JS_GetOpaque2(ctx, this_val, js_ort_tensor_class_id)))
    return JS_EXCEPTION;

  if(!JS_IsUndefined(s->data))
    return JS_DupValue(ctx, s->data);

  if((status = ort_api->GetTensorTypeAndShape(s->value, &info)))
    return js_ort_throw_status(ctx, status);

  if((status = ort_api->GetTensorShapeElementCount(info, &count))) {
    ort_api->ReleaseTensorTypeAndShapeInfo(info);
    return js_ort_throw_status(ctx, status);
  }

  if((status = ort_api->GetTensorMutableData(s->value, &ptr))) {
    ort_api->ReleaseTensorTypeAndShapeInfo(info);
    return js_ort_throw_status(ctx, status);
  }

  ort_api->ReleaseTensorTypeAndShapeInfo(info);

  if(!(view = js_malloc(ctx, sizeof(*view))))
    return JS_ThrowOutOfMemory(ctx);

  view->tensor = JS_DupValue(ctx, this_val);
  arr = js_typedarray_external(ctx, ptr, count, js_ort_types[s->type].array, js_ort_view_free, view);

  /* on failure js_typedarray_external has already called js_ort_view_free */
  return arr;
}

static const JSCFunctionListEntry js_ort_tensor_proto_funcs[] = {
  JS_CGETSET_DEF("type", js_ort_tensor_get_type, NULL),
  JS_CGETSET_DEF("dims", js_ort_tensor_get_dims, NULL),
  JS_CGETSET_DEF("data", js_ort_tensor_get_data, NULL),
  JS_PROP_STRING_DEF("[Symbol.toStringTag]", "Tensor", JS_PROP_CONFIGURABLE),
};

static JSClassDef js_ort_tensor_class = {
  .class_name = "Tensor",
  .finalizer = js_ort_tensor_finalizer,
  .gc_mark = js_ort_tensor_gc_mark,
};

static JSValue
js_ort_tensor_wrap(JSContext* ctx, OrtValue* value) {
  OrtTensorTypeAndShapeInfo* info = NULL;
  OrtStatus* status = NULL;
  ONNXTensorElementDataType onnx_type;
  int type;
  JSValue proto;
  JSValue obj;
  JSOrtTensor* s;

  if((status = ort_api->GetTensorTypeAndShape(value, &info))) {
    ort_api->ReleaseValue(value);
    return js_ort_throw_status(ctx, status);
  }

  if((status = ort_api->GetTensorElementType(info, &onnx_type))) {
    ort_api->ReleaseTensorTypeAndShapeInfo(info);
    ort_api->ReleaseValue(value);
    return js_ort_throw_status(ctx, status);
  }

  if((type = js_ort_type_by_onnx(onnx_type)) < 0) {
    ort_api->ReleaseTensorTypeAndShapeInfo(info);
    ort_api->ReleaseValue(value);
    return JS_ThrowTypeError(ctx, "unsupported element type %d", onnx_type);
  }

  proto = JS_GetClassProto(ctx, js_ort_tensor_class_id);
  obj = JS_NewObjectProtoClass(ctx, proto, js_ort_tensor_class_id);
  JS_FreeValue(ctx, proto);

  if(JS_IsException(obj)) {
    ort_api->ReleaseTensorTypeAndShapeInfo(info);
    ort_api->ReleaseValue(value);
    return JS_EXCEPTION;
  }

  if(!(s = js_mallocz(ctx, sizeof(*s)))) {
    ort_api->ReleaseTensorTypeAndShapeInfo(info);
    ort_api->ReleaseValue(value);
    return JS_ThrowOutOfMemory(ctx);
  }

  s->value = value;
  s->type = type;
  s->data = JS_UNDEFINED;
  JS_SetOpaque(obj, s);
  ort_api->ReleaseTensorTypeAndShapeInfo(info);
  return obj;
}

/* name and value of a string option */
typedef struct {
  const char* name;
  int value;
} js_ort_choice;

static const js_ort_choice js_ort_graph_opt_levels[] = {
  { "disable", ORT_DISABLE_ALL },
  { "basic", ORT_ENABLE_BASIC },
  { "extended", ORT_ENABLE_EXTENDED },
  { "all", ORT_ENABLE_ALL },
};

static const js_ort_choice js_ort_log_levels[] = {
  { "verbose", ORT_LOGGING_LEVEL_VERBOSE },
  { "info", ORT_LOGGING_LEVEL_INFO },
  { "warning", ORT_LOGGING_LEVEL_WARNING },
  { "error", ORT_LOGGING_LEVEL_ERROR },
  { "fatal", ORT_LOGGING_LEVEL_FATAL },
};

/*
 * reads a non-negative integer option.
 *
 * returns 1 and sets *out, 0 when the option is undefined, -1 with a
 * RangeError pending.
 */
static int
js_ort_option_count(JSContext* ctx, JSValueConst options, const char* name, int* out) {
  JSValue val = JS_GetPropertyStr(ctx, options, name);
  double d;
  int ret = 1;

  if(JS_IsUndefined(val))
    return 0;

  if(!JS_IsNumber(val) || JS_ToFloat64(ctx, &d, val) < 0 || d != floor(d) || d < 0 || d > INT32_MAX) {
    JS_ThrowRangeError(ctx, "%s must be a non-negative integer", name);
    ret = -1;
  } else
    *out = (int)d;

  JS_FreeValue(ctx, val);
  return ret;
}

/*
 * reads an option that is one of the names of a table.
 *
 * returns 1 and sets *out, 0 when the option is undefined, -1 with a
 * TypeError pending that says what accepted lists.
 */
static int
js_ort_option_choice(JSContext* ctx, JSValueConst options, const char* name, const js_ort_choice* table, size_t count, const char* accepted, int* out) {
  JSValue val = JS_GetPropertyStr(ctx, options, name);
  const char* str;
  int ret = -1;

  if(JS_IsUndefined(val))
    return 0;

  if(JS_IsString(val) && (str = JS_ToCString(ctx, val))) {
    for(size_t i = 0; i < count; i++)
      if(!strcmp(table[i].name, str)) {
        *out = table[i].value;
        ret = 1;
        break;
      }

    JS_FreeCString(ctx, str);
  }

  if(ret < 0 && !JS_HasException(ctx))
    JS_ThrowTypeError(ctx, "%s must be %s", name, accepted);

  JS_FreeValue(ctx, val);
  return ret;
}

/*
 * applies the options object to an OrtSessionOptions.
 *
 * returns 0, or -1 with the exception pending. Owns nothing, the caller
 * releases opts.
 */
static int
js_ort_session_options(JSContext* ctx, OrtSessionOptions* opts, JSValueConst options) {
  OrtStatus* status;
  int n, value;

  if(JS_IsUndefined(options))
    return 0;

  if(!JS_IsObject(options)) {
    JS_ThrowTypeError(ctx, "Session options must be an object");
    return -1;
  }

  if((n = js_ort_option_count(ctx, options, "intraOpThreads", &value)) < 0)
    return -1;

  if(n && (status = ort_api->SetIntraOpNumThreads(opts, value))) {
    js_ort_throw_status(ctx, status);
    return -1;
  }

  if((n = js_ort_option_count(ctx, options, "interOpThreads", &value)) < 0)
    return -1;

  if(n && (status = ort_api->SetInterOpNumThreads(opts, value))) {
    js_ort_throw_status(ctx, status);
    return -1;
  }

  if((n = js_ort_option_choice(ctx, options, "graphOptimization", js_ort_graph_opt_levels, countof(js_ort_graph_opt_levels), "'disable', 'basic', 'extended' or 'all'", &value)) < 0)
    return -1;

  if(n && (status = ort_api->SetSessionGraphOptimizationLevel(opts, value))) {
    js_ort_throw_status(ctx, status);
    return -1;
  }

  if((n = js_ort_option_choice(ctx, options, "logLevel", js_ort_log_levels, countof(js_ort_log_levels), "'verbose', 'info', 'warning', 'error' or 'fatal'", &value)) < 0)
    return -1;

  if(n && (status = ort_api->SetSessionLogSeverityLevel(opts, value))) {
    js_ort_throw_status(ctx, status);
    return -1;
  }

  return 0;
}

static JSValue
js_ort_session_ctor(JSContext* ctx, JSValueConst new_target, int argc, JSValueConst argv[]) {
  const char* path = NULL;
  OrtSessionOptions* opts = NULL;
  OrtSession* session = NULL;
  OrtStatus* status = NULL;
  OrtAllocator* alloc = NULL;
  JSValue obj;
  JSOrtSession* s = NULL;
  size_t n_inputs = 0;
  size_t n_outputs = 0;
  char** input_names = NULL;
  char** output_names = NULL;
  BufferView view;
  JSValueConst options = argc > 1 ? argv[1] : JS_UNDEFINED;
  int has_buffer = 0;

  if(js_ort_init_api(ctx))
    return JS_EXCEPTION;

  if(argc < 1)
    return JS_ThrowTypeError(ctx, "Session needs a model path or a buffer");

  if(JS_IsString(argv[0])) {
    if(!(path = JS_ToCString(ctx, argv[0])))
      return JS_EXCEPTION;
  } else if(js_typedarray_get(ctx, argv[0], &view) < 0) {
    /* js_typedarray_get left its own TypeError, say what a Session takes */
    JS_FreeValue(ctx, JS_GetException(ctx));
    return JS_ThrowTypeError(ctx, "Session needs a model path or a buffer");
  } else
    has_buffer = 1;

#ifdef _WIN32
#error "wide path conversion not implemented"
#else
  if((status = ort_api->CreateSessionOptions(&opts)))
    goto fail;

  if(js_ort_session_options(ctx, opts, options) < 0)
    goto cleanup;  /* exception already pending */

  if(has_buffer) {
    if((status = ort_api->CreateSessionFromArray(ort_env, view.ptr, view.len, opts, &session)))
      goto fail;
  } else {
    if((status = ort_api->CreateSession(ort_env, path, opts, &session)))
      goto fail;
  }

  ort_api->ReleaseSessionOptions(opts);
  opts = NULL;
#endif

  if(path)
    JS_FreeCString(ctx, path);

  path = NULL;

  if((status = ort_api->GetAllocatorWithDefaultOptions(&alloc)))
    goto fail;

  if((status = ort_api->SessionGetInputCount(session, &n_inputs)))
    goto fail;

  if((status = ort_api->SessionGetOutputCount(session, &n_outputs)))
    goto fail;

  if(!(input_names = js_mallocz(ctx, n_inputs * sizeof(char*)))) {
    JS_ThrowOutOfMemory(ctx);
    goto cleanup;
  }

  if(!(output_names = js_mallocz(ctx, n_outputs * sizeof(char*)))) {
    JS_ThrowOutOfMemory(ctx);
    goto cleanup;
  }

  for(size_t i = 0; i < n_inputs; i++) {
    char* name = NULL;

    if((status = ort_api->SessionGetInputName(session, i, alloc, &name)))
      goto fail;

    input_names[i] = js_strdup(ctx, name);
    alloc->Free(alloc, name);

    if(!input_names[i]) {
      JS_ThrowOutOfMemory(ctx);
      goto cleanup;
    }
  }

  for(size_t i = 0; i < n_outputs; i++) {
    char* name = NULL;

    if((status = ort_api->SessionGetOutputName(session, i, alloc, &name)))
      goto fail;

    output_names[i] = js_strdup(ctx, name);
    alloc->Free(alloc, name);

    if(!output_names[i]) {
      JS_ThrowOutOfMemory(ctx);
      goto cleanup;
    }
  }

  if(JS_IsException(obj = js_object_from_target(ctx, new_target, js_ort_session_class_id)))
    goto cleanup;

  if(!(s = js_mallocz(ctx, sizeof(*s)))) {
    JS_FreeValue(ctx, obj);
    JS_ThrowOutOfMemory(ctx);
    goto cleanup;
  }

  s->session = session;
  s->n_inputs = n_inputs;
  s->n_outputs = n_outputs;
  s->input_names = input_names;
  s->output_names = output_names;
  JS_SetOpaque(obj, s);
  return obj;

fail:
  js_ort_throw_status(ctx, status);

cleanup:
  if(session)
    ort_api->ReleaseSession(session);
  if(opts)
    ort_api->ReleaseSessionOptions(opts);
  if(input_names)
    js_ort_session_free_names(JS_GetRuntime(ctx), input_names, n_inputs);
  if(output_names)
    js_ort_session_free_names(JS_GetRuntime(ctx), output_names, n_outputs);
  if(path)
    JS_FreeCString(ctx, path);

  return JS_EXCEPTION;
}

static JSValue
js_ort_session_get_input_names(JSContext* ctx, JSValueConst this_val) {
  JSOrtSession* s;
  JSValue arr;

  if(!(s = JS_GetOpaque2(ctx, this_val, js_ort_session_class_id)))
    return JS_EXCEPTION;

  if(JS_IsException(arr = JS_NewArray(ctx)))
    return arr;

  for(size_t i = 0; i < s->n_inputs; i++) {
    JSValue name;

    if(JS_IsException(name = JS_NewString(ctx, s->input_names[i]))) {
      JS_FreeValue(ctx, arr);
      return JS_EXCEPTION;
    }

    JS_SetPropertyUint32(ctx, arr, i, name);
  }

  return arr;
}

static JSValue
js_ort_session_get_output_names(JSContext* ctx, JSValueConst this_val) {
  JSOrtSession* s;
  JSValue arr;

  if(!(s = JS_GetOpaque2(ctx, this_val, js_ort_session_class_id)))
    return JS_EXCEPTION;

  if(JS_IsException(arr = JS_NewArray(ctx)))
    return arr;

  for(size_t i = 0; i < s->n_outputs; i++) {
    JSValue name;

    if(JS_IsException(name = JS_NewString(ctx, s->output_names[i]))) {
      JS_FreeValue(ctx, arr);
      return JS_EXCEPTION;
    }

    JS_SetPropertyUint32(ctx, arr, i, name);
  }

  return arr;
}

static JSValue
js_ort_session_release(JSContext* ctx, JSValueConst this_val, int argc, JSValueConst argv[]) {
  JSOrtSession* s;

  if(!(s = JS_GetOpaque2(ctx, this_val, js_ort_session_class_id)))
    return JS_EXCEPTION;

  if(s->session) {
    ort_api->ReleaseSession(s->session);
    s->session = NULL;
  }

  return JS_UNDEFINED;
}

static JSValue
js_ort_session_run(JSContext* ctx, JSValueConst this_val, int argc, JSValueConst argv[]) {
  JSOrtSession* s;
  JSPropertyEnum* tab = NULL;
  uint32_t len = 0;
  const char** in_names = NULL;
  const OrtValue** in_values = NULL;
  const char** out_names = NULL;
  OrtValue** out_values = NULL;
  JSValue obj = JS_UNDEFINED, result = JS_EXCEPTION;
  OrtStatus* status;
  size_t n_in, n_out, i, j;

  if(!(s = JS_GetOpaque2(ctx, this_val, js_ort_session_class_id)))
    return JS_EXCEPTION;

  if(!s->session) {
    JSValue err = JS_NewError(ctx);

    JS_DefinePropertyValueStr(ctx, err, "message", JS_NewString(ctx, "session released"), JS_PROP_C_W_E);
    return JS_Throw(ctx, err);
  }

  if(argc < 1 || !JS_IsObject(argv[0]))
    return JS_ThrowTypeError(ctx, "run needs an object of input tensors");

  if(JS_GetOwnPropertyNames(ctx, &tab, &len, argv[0], JS_GPN_STRING_MASK | JS_GPN_ENUM_ONLY) < 0)
    return JS_EXCEPTION;

  n_in = s->n_inputs;
  n_out = s->n_outputs;

  /* one element more, so that a model without inputs still gets a pointer */
  if(!(in_names = js_mallocz(ctx, (n_in + 1) * sizeof(char*))) || !(in_values = js_mallocz(ctx, (n_in + 1) * sizeof(OrtValue*)))) {
    JS_ThrowOutOfMemory(ctx);
    goto done;
  }

  /* every key of feeds names an input of the model, the cached name goes to Run */
  for(i = 0; i < len; i++) {
    JSOrtTensor* tensor;
    const char* name;
    JSValue val;

    if(!(name = JS_AtomToCString(ctx, tab[i].atom)))
      goto done;

    for(j = 0; j < n_in; j++)
      if(!strcmp(s->input_names[j], name))
        break;

    if(j == n_in) {
      JS_ThrowTypeError(ctx, "unknown input '%s'", name);
      JS_FreeCString(ctx, name);
      goto done;
    }

    if(JS_IsException(val = JS_GetProperty(ctx, argv[0], tab[i].atom))) {
      JS_FreeCString(ctx, name);
      goto done;
    }

    if(!(tensor = JS_GetOpaque(val, js_ort_tensor_class_id))) {
      JS_ThrowTypeError(ctx, "input '%s' is not a Tensor", name);
      JS_FreeValue(ctx, val);
      JS_FreeCString(ctx, name);
      goto done;
    }

    in_names[j] = s->input_names[j];
    in_values[j] = tensor->value;
    JS_FreeValue(ctx, val);
    JS_FreeCString(ctx, name);
  }

  for(j = 0; j < n_in; j++) {
    if(!in_names[j]) {
      JS_ThrowTypeError(ctx, "missing input '%s'", s->input_names[j]);
      goto done;
    }
  }

  if(argc > 1 && !JS_IsUndefined(argv[1])) {
    int64_t count;

    if(JS_IsArray(ctx, argv[1]) != 1 || (count = js_array_length(ctx, argv[1])) <= 0) {
      JS_ThrowTypeError(ctx, "outputNames must be a non-empty array of strings");
      goto done;
    }

    n_out = count;
  }

  if(!(out_names = js_mallocz(ctx, n_out * sizeof(char*))) || !(out_values = js_mallocz(ctx, n_out * sizeof(OrtValue*)))) {
    JS_ThrowOutOfMemory(ctx);
    goto done;
  }

  if(argc > 1 && !JS_IsUndefined(argv[1])) {
    for(i = 0; i < n_out; i++) {
      const char* name;
      JSValue item;

      if(JS_IsException(item = JS_GetPropertyUint32(ctx, argv[1], i)))
        goto done;

      if(!JS_IsString(item)) {
        JS_ThrowTypeError(ctx, "outputNames must be a non-empty array of strings");
        JS_FreeValue(ctx, item);
        goto done;
      }

      name = JS_ToCString(ctx, item);
      JS_FreeValue(ctx, item);

      if(!name)
        goto done;

      for(j = 0; j < s->n_outputs; j++)
        if(!strcmp(s->output_names[j], name))
          break;

      if(j == s->n_outputs) {
        JS_ThrowTypeError(ctx, "unknown output '%s'", name);
        JS_FreeCString(ctx, name);
        goto done;
      }

      out_names[i] = s->output_names[j];
      JS_FreeCString(ctx, name);
    }
  } else {
    for(i = 0; i < n_out; i++)
      out_names[i] = s->output_names[i];
  }

  /* nothing here calls back into JS, ONNX Runtime blocks until the model is done */
  if((status = ort_api->Run(s->session, NULL, in_names, in_values, n_in, out_names, n_out, out_values))) {
    js_ort_throw_status(ctx, status);
    goto done;
  }

  if(JS_IsException(obj = JS_NewObject(ctx)))
    goto done;

  for(i = 0; i < n_out; i++) {
    JSValue tensor = js_ort_tensor_wrap(ctx, out_values[i]);

    /* wrap owns the value, also when it fails */
    out_values[i] = NULL;

    if(JS_IsException(tensor))
      goto done;

    if(JS_DefinePropertyValueStr(ctx, obj, out_names[i], tensor, JS_PROP_C_W_E) < 0)
      goto done;
  }

  result = obj;
  obj = JS_UNDEFINED;

done:
  if(out_values)
    for(i = 0; i < n_out; i++)
      if(out_values[i])
        ort_api->ReleaseValue(out_values[i]);

  JS_FreeValue(ctx, obj);

  if(tab)
    JS_FreePropertyEnum(ctx, tab, len);

  js_free(ctx, in_names);
  js_free(ctx, in_values);
  js_free(ctx, out_names);
  js_free(ctx, out_values);
  return result;
}

static const JSCFunctionListEntry js_ort_session_proto_funcs[] = {
  JS_CGETSET_DEF("inputNames", js_ort_session_get_input_names, NULL),
  JS_CGETSET_DEF("outputNames", js_ort_session_get_output_names, NULL),
  JS_CFUNC_DEF("run", 1, js_ort_session_run),
  JS_CFUNC_DEF("release", 0, js_ort_session_release),
  JS_PROP_STRING_DEF("[Symbol.toStringTag]", "Session", JS_PROP_CONFIGURABLE),
};

static JSClassDef js_ort_session_class = {
  .class_name = "Session",
  .finalizer = js_ort_session_finalizer,
};

#ifdef JS_SHARED_LIBRARY
#if defined(_WIN32) || defined(__MINGW32__)
#define VISIBLE __declspec(dllexport)
#else
#define VISIBLE __attribute__((visibility("default")))
#endif
#define JS_INIT_MODULE js_init_module
#else
#define VISIBLE
#define JS_INIT_MODULE js_init_module_ort
#endif

static JSValue
js_ort_version(JSContext* ctx) {
  return JS_NewString(ctx, OrtGetApiBase()->GetVersionString());
}

static int
js_ort_init(JSContext* ctx, JSModuleDef* m) {
  JSValue ctor;

  if(JS_IsException(ctor = js_class_register(ctx, &js_ort_tensor_class_id, &js_ort_tensor_class, js_ort_tensor_ctor, 3, js_ort_tensor_proto_funcs, countof(js_ort_tensor_proto_funcs))))
    return -1;

  if(JS_SetModuleExport(ctx, m, "Tensor", ctor) < 0)
    return -1;

  if(JS_IsException(ctor = js_class_register(ctx, &js_ort_session_class_id, &js_ort_session_class, js_ort_session_ctor, 1, js_ort_session_proto_funcs, countof(js_ort_session_proto_funcs))))
    return -1;

  if(JS_SetModuleExport(ctx, m, "Session", ctor) < 0)
    return -1;

  return JS_SetModuleExport(ctx, m, "version", js_ort_version(ctx));
}

VISIBLE JSModuleDef*
JS_INIT_MODULE(JSContext* ctx, const char* module_name) {
  JSModuleDef* m;

  if(!(m = JS_NewCModule(ctx, module_name, js_ort_init)))
    return NULL;

  JS_AddModuleExport(ctx, m, "version");
  JS_AddModuleExport(ctx, m, "Tensor");
  JS_AddModuleExport(ctx, m, "Session");

  return m;
}
