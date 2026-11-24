#include "js-utils.h"
#include <inttypes.h>

JSValue
js_global_get_str(JSContext* ctx, const char* name) {
  JSValue global = JS_GetGlobalObject(ctx);
  JSValue ret = JS_GetPropertyStr(ctx, global, name);

  JS_FreeValue(ctx, global);
  return ret;
}

int64_t
js_array_length(JSContext* ctx, JSValueConst array) {
  int64_t len = -1;
  JSValue length = JS_GetPropertyStr(ctx, array, "length");

  if(JS_IsNumber(length))
    JS_ToInt64(ctx, &len, length);

  JS_FreeValue(ctx, length);
  return len;
}

int64_t*
js_array_toint64(JSContext* ctx, JSValueConst array, size_t* plen) {
  int64_t len, *ret;

  if(JS_IsArray(ctx, array) != 1 || (len = js_array_length(ctx, array)) < 0) {
    JS_ThrowTypeError(ctx, "expected an Array");
    return NULL;
  }

  if(len > (1 << 24)) {
    JS_ThrowRangeError(ctx, "array too long");
    return NULL;
  }

  if(!(ret = js_malloc(ctx, sizeof(int64_t) * (len ? len : 1))))
    return NULL;

  for(int64_t i = 0; i < len; i++) {
    JSValue item = JS_GetPropertyUint32(ctx, array, i);
    double d;

    if(JS_IsException(item))
      goto fail;

    /* 2^53 - 1 is the largest integer a double holds exactly */
    if(!JS_IsNumber(item) || JS_ToFloat64(ctx, &d, item) || !(d >= 0 && d <= 9007199254740991.0) || (double)(int64_t)d != d) {
      if(!JS_HasException(ctx))
        JS_ThrowRangeError(ctx, "element %" PRId64 " is not a non-negative integer", i);
      JS_FreeValue(ctx, item);
      goto fail;
    }

    JS_FreeValue(ctx, item);
    ret[i] = (int64_t)d;
  }

  *plen = len;
  return ret;

fail:
  js_free(ctx, ret);
  return NULL;
}

JSValue
js_object_from_target(JSContext* ctx, JSValueConst new_target, JSClassID id) {
  JSValue obj, proto = JS_GetPropertyStr(ctx, new_target, "prototype");

  if(JS_IsException(proto))
    return proto;

  if(!JS_IsObject(proto)) {
    JS_FreeValue(ctx, proto);
    proto = JS_GetClassProto(ctx, id);
  }

  obj = JS_NewObjectProtoClass(ctx, proto, id);
  JS_FreeValue(ctx, proto);
  return obj;
}

JSValue
js_class_register(JSContext* ctx, JSClassID* id, JSClassDef* def, JSCFunction* ctor, int length, const JSCFunctionListEntry* funcs, size_t nfuncs) {
  JSValue proto, func;

  JS_NewClassID(id);

  if(JS_NewClass(JS_GetRuntime(ctx), *id, def) < 0)
    return JS_EXCEPTION;

  proto = JS_NewObject(ctx);

  if(JS_IsException(proto))
    return proto;

  if(funcs)
    JS_SetPropertyFunctionList(ctx, proto, funcs, nfuncs);

  func = JS_NewCFunction2(ctx, ctor, def->class_name, length, JS_CFUNC_constructor, 0);

  if(JS_IsException(func)) {
    JS_FreeValue(ctx, proto);
    return func;
  }

  JS_SetConstructor(ctx, func, proto);
  JS_SetClassProto(ctx, *id, proto);  /* takes the reference to proto */
  return func;
}

const char*
js_typedarray_name(int kind) {
  switch(kind) {
    case JS_TYPED_ARRAY_UINT8C: return "Uint8ClampedArray";
    case JS_TYPED_ARRAY_INT8: return "Int8Array";
    case JS_TYPED_ARRAY_UINT8: return "Uint8Array";
    case JS_TYPED_ARRAY_INT16: return "Int16Array";
    case JS_TYPED_ARRAY_UINT16: return "Uint16Array";
    case JS_TYPED_ARRAY_INT32: return "Int32Array";
    case JS_TYPED_ARRAY_UINT32: return "Uint32Array";
    case JS_TYPED_ARRAY_BIG_INT64: return "BigInt64Array";
    case JS_TYPED_ARRAY_BIG_UINT64: return "BigUint64Array";
    case JS_TYPED_ARRAY_FLOAT16: return "Float16Array";
    case JS_TYPED_ARRAY_FLOAT32: return "Float32Array";
    case JS_TYPED_ARRAY_FLOAT64: return "Float64Array";
  }

  return "ArrayBuffer";
}

size_t
js_typedarray_size(int kind) {
  switch(kind) {
    case JS_TYPED_ARRAY_INT16:
    case JS_TYPED_ARRAY_UINT16:
    case JS_TYPED_ARRAY_FLOAT16: return 2;
    case JS_TYPED_ARRAY_INT32:
    case JS_TYPED_ARRAY_UINT32:
    case JS_TYPED_ARRAY_FLOAT32: return 4;
    case JS_TYPED_ARRAY_BIG_INT64:
    case JS_TYPED_ARRAY_BIG_UINT64:
    case JS_TYPED_ARRAY_FLOAT64: return 8;
  }

  return 1;
}

int
js_typedarray_kind(JSContext* ctx, JSValueConst obj) {
  /* the class id of a builtin class is the same in every runtime, so a
   * table filled once by probing with empty arrays is safe to share */
  static JSClassID ids[JS_TYPED_ARRAY_FLOAT64 + 1];
  JSClassID id;

  if(!JS_IsObject(obj))
    return JS_TYPEDARRAY_NONE;

  if(!ids[JS_TYPED_ARRAY_UINT8]) {
    JSValue zero = JS_NewInt32(ctx, 0);

    for(int k = 0; k <= JS_TYPED_ARRAY_FLOAT64; k++) {
      JSValue probe = JS_NewTypedArray(ctx, 1, &zero, k);

      if(JS_IsException(probe))
        JS_FreeValue(ctx, JS_GetException(ctx));
      else
        ids[k] = JS_GetClassID(probe);

      JS_FreeValue(ctx, probe);
    }
  }

  if(!(id = JS_GetClassID(obj)))
    return JS_TYPEDARRAY_NONE;

  for(int k = 0; k <= JS_TYPED_ARRAY_FLOAT64; k++)
    if(ids[k] == id)
      return k;

  return JS_TYPEDARRAY_NONE;
}

int
js_typedarray_get(JSContext* ctx, JSValueConst obj, BufferView* out) {
  size_t offset = 0, length = 0, size = 0, bpe = 0;
  JSValue buf = JS_GetTypedArrayBuffer(ctx, obj, &offset, &length, &bpe);
  uint8_t* data;

  if(!JS_IsException(buf)) {
    data = JS_GetArrayBuffer(ctx, &size, buf);
    JS_FreeValue(ctx, buf);

    if(!data && JS_HasException(ctx))
      return -1;

    if(offset > size || length > size - offset) {
      JS_ThrowRangeError(ctx, "view lies outside its ArrayBuffer");
      return -1;
    }

    out->ptr = data + offset;
    out->len = length;
    out->kind = js_typedarray_kind(ctx, obj);
    return 0;
  }

  /* not a TypedArray: drop the probe exception, try an ArrayBuffer */
  JS_FreeValue(ctx, JS_GetException(ctx));

  data = JS_GetArrayBuffer(ctx, &size, obj);

  if(!data && JS_HasException(ctx)) {
    JS_FreeValue(ctx, JS_GetException(ctx));
    JS_ThrowTypeError(ctx, "expected a TypedArray or ArrayBuffer");
    return -1;
  }

  out->ptr = data;
  out->len = size;
  out->kind = JS_TYPEDARRAY_NONE;
  return 0;
}

int
js_typedarray_expect(JSContext* ctx, JSValueConst obj, int kind, BufferView* out) {
  if(js_typedarray_get(ctx, obj, out))
    return -1;

  if(out->kind != kind) {
    JS_ThrowTypeError(ctx, "expected a %s", js_typedarray_name(kind));
    return -1;
  }

  return 0;
}

/* a typed array of count elements over the whole of buf. The one-argument
 * form of JS_NewTypedArray gives a length 0 array for an ArrayBuffer. */
static JSValue
js_typedarray_over(JSContext* ctx, JSValueConst buf, size_t count, int kind) {
  JSValue args[3] = { JS_DupValue(ctx, buf), JS_NewInt32(ctx, 0), JS_NewInt64(ctx, count) };
  JSValue ret = JS_NewTypedArray(ctx, 3, args, kind);

  JS_FreeValue(ctx, args[0]);
  return ret;
}

JSValue
js_typedarray_external(JSContext* ctx, void* ptr, size_t count, int kind, JSFreeArrayBufferDataFunc* free_func, void* opaque) {
  size_t size = js_typedarray_size(kind);
  JSValue buf, ret;

  if(count > (size_t)-1 / size)
    buf = JS_ThrowRangeError(ctx, "too many elements");
  else
    buf = JS_NewArrayBuffer(ctx, ptr, count * size, free_func, opaque, FALSE);

  if(JS_IsException(buf)) {
    /* the ArrayBuffer does not exist, so nothing else will release opaque */
    if(free_func)
      free_func(JS_GetRuntime(ctx), opaque, ptr);

    return buf;
  }

  ret = js_typedarray_over(ctx, buf, count, kind);
  JS_FreeValue(ctx, buf);
  return ret;
}

JSValue
js_typedarray_copy(JSContext* ctx, const void* ptr, size_t count, int kind) {
  size_t size = js_typedarray_size(kind);
  JSValue buf, ret;

  if(count > (size_t)-1 / size)
    return JS_ThrowRangeError(ctx, "too many elements");

  buf = JS_NewArrayBufferCopy(ctx, ptr, count * size);

  if(JS_IsException(buf))
    return buf;

  ret = js_typedarray_over(ctx, buf, count, kind);
  JS_FreeValue(ctx, buf);
  return ret;
}
