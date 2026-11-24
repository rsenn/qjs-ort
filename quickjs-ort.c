/*
 * quickjs-ort.c - ONNX Runtime for QuickJS, through its C API.
 */
#include <quickjs.h>
#include <cutils.h>
#include <onnxruntime_c_api.h>

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
  return JS_SetModuleExport(ctx, m, "version", js_ort_version(ctx));
}

VISIBLE JSModuleDef*
JS_INIT_MODULE(JSContext* ctx, const char* module_name) {
  JSModuleDef* m;

  if(!(m = JS_NewCModule(ctx, module_name, js_ort_init)))
    return NULL;

  JS_AddModuleExport(ctx, m, "version");
  return m;
}
