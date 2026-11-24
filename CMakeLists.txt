cmake_minimum_required(VERSION 3.5)

project(qjs-ort VERSION 0.1 LANGUAGES C)

message("Building project [1;31m${PROJECT_NAME}[0m")

include(CheckCCompilerFlag)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/UseMultiArch.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Functions.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Compat.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindQuickJS.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/QuickJSModule.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindONNXRuntime.cmake)
quickjs_module_options(SHARED_DEFAULT ON STATIC_DEFAULT OFF)

check_c_compiler_flag("-fvisibility=hidden" VISIBILITY_HIDDEN)
if(VISIBILITY_HIDDEN)
  set(MODULE_COMPILE_FLAGS ${MODULE_COMPILE_FLAGS} "-fvisibility=hidden")
endif(VISIBILITY_HIDDEN)

find_onnxruntime()
find_quickjs()
configure_quickjs()

include_directories(${ORT_INCLUDE_DIR} ${QUICKJS_INCLUDE_DIR})
link_directories(${ORT_LIBRARY_DIR} ${QUICKJS_LIBRARY_DIR})

set(ort_LIBRARIES ${ORT_LIBRARY})

make_module(ort quickjs-ort.c js-utils.c js-utils.h)

foreach(_t qjs-ort qjs-ort-static)
  if(TARGET ${_t})
    # find libonnxruntime.so.1 next to a relocated install and in the build tree
    set_property(TARGET ${_t} APPEND PROPERTY BUILD_RPATH "${ORT_LIBRARY_DIR}")
  endif(TARGET ${_t})
endforeach(_t)
