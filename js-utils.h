/*
 * js-utils.h - QuickJS helpers of qjs-ort: class registration, typed
 * array access without copying, views over memory the library owns.
 *
 * Taken from qjs-sound js-utils.[ch] and js-typedarray.[ch] and from
 * qjs-nanovg nvgjs-utils.[ch], cut down to what ort needs. Element kinds
 * are the JSTypedArrayEnum of quickjs.h, no private enum.
 */
#ifndef ORT_JS_UTILS_H
#define ORT_JS_UTILS_H

#include <quickjs.h>
#include <cutils.h>

/* kind of a value that is a plain ArrayBuffer */
#define JS_TYPEDARRAY_NONE (-1)

typedef struct {
  void* ptr;   /* already offset into the backing ArrayBuffer */
  size_t len;  /* bytes, not elements */
  int kind;    /* JSTypedArrayEnum, or JS_TYPEDARRAY_NONE */
} BufferView;

/*
 * Gets a global variable by name.
 *
 * refcount: returns a new reference, undefined when there is none.
 */
JSValue js_global_get_str(JSContext*, const char* name);

/*
 * Reads the length property of an array or array-like.
 *
 * returns the length, or -1 when length is not a number.
 */
int64_t js_array_length(JSContext*, JSValueConst array);

/*
 * Reads a JS Array of integers into a new C array.
 *
 *   JSValueConst array  Array of numbers, every one a non-negative integer
 *   size_t* plen        receives the element count
 *
 * returns js_malloc memory with at least one element so that a scalar
 * shape [] gives a valid pointer, NULL with an exception pending
 * (TypeError not an Array, RangeError bad element).
 *
 * refcount: the caller frees the result with js_free.
 */
int64_t* js_array_toint64(JSContext*, JSValueConst array, size_t* plen);

/*
 * Creates an object of class id for a constructor.
 *
 * The prototype is new_target.prototype, so that a subclass works, else
 * the prototype registered for the class.
 *
 * returns the object, or JS_EXCEPTION. exception pending: the caller
 * frees what it owns.
 */
JSValue js_object_from_target(JSContext*, JSValueConst new_target, JSClassID id);

/*
 * Registers a native class and makes its constructor.
 *
 *   JSClassID* id        receives a new class id
 *   JSClassDef* def      the class, def->class_name names the constructor
 *   JSCFunction* ctor    the constructor, called with new_target
 *   int length           the length property of the constructor
 *   funcs, nfuncs        members of the prototype, funcs may be NULL
 *
 * returns the constructor, or JS_EXCEPTION. The prototype is registered
 * with JS_SetClassProto, so subclasses and js_object_from_target find it.
 * Same job as js_class_define of qjs-modules, without the dependency.
 *
 * refcount: returns a new reference to the constructor.
 */
JSValue js_class_register(JSContext*, JSClassID* id, JSClassDef* def, JSCFunction* ctor, int length, const JSCFunctionListEntry* funcs, size_t nfuncs);

/*
 * Names of a typed array kind.
 *
 * returns Float32Array and so on, ArrayBuffer for JS_TYPEDARRAY_NONE.
 */
const char* js_typedarray_name(int kind);

/*
 * Size of one element in bytes.
 *
 * returns 1, 2, 4 or 8, 1 for JS_TYPEDARRAY_NONE.
 */
size_t js_typedarray_size(int kind);

/*
 * Tells which typed array a value is, by class id.
 *
 * An Int16Array is not a Uint16Array, a subclass of Float32Array is a
 * Float32Array.
 *
 * returns a JSTypedArrayEnum, or JS_TYPEDARRAY_NONE for anything else.
 */
int js_typedarray_kind(JSContext*, JSValueConst);

/*
 * Resolves a TypedArray or ArrayBuffer to its bytes, no copy.
 *
 * A DataView is rejected, JS_GetTypedArrayBuffer does not take one.
 *
 * borrowed: out->ptr is valid until the buffer is detached or resized,
 * take it right before the native call and never keep it across a call
 * back into JS.
 *
 * returns 0, or -1 with a TypeError pending when it is none of those, or
 * a RangeError when the view lies outside its buffer.
 */
int js_typedarray_get(JSContext*, JSValueConst, BufferView* out);

/*
 * Like js_typedarray_get, but only one kind is accepted.
 *
 * returns 0, or -1 with a TypeError such as expected a Float32Array.
 */
int js_typedarray_expect(JSContext*, JSValueConst, int kind, BufferView* out);

/*
 * Makes a typed array of kind over native memory, no copy.
 *
 *   void* ptr        the memory, count elements of kind
 *   free_func        called with opaque and ptr when the ArrayBuffer dies,
 *                    may be NULL for memory owned elsewhere
 *   opaque           whatever free_func needs, for example a reference
 *                    that keeps the owner of ptr alive
 *
 * refcount: free_func runs exactly once, also when this fails, so the
 * caller never releases opaque itself after the call.
 *
 * returns the typed array, or JS_EXCEPTION.
 */
JSValue js_typedarray_external(JSContext*, void* ptr, size_t count, int kind, JSFreeArrayBufferDataFunc* free_func, void* opaque);

/*
 * Copies count elements of kind into a new typed array.
 *
 * returns the typed array, or JS_EXCEPTION.
 */
JSValue js_typedarray_copy(JSContext*, const void* ptr, size_t count, int kind);

#endif /* defined(ORT_JS_UTILS_H) */
