cmake_minimum_required(VERSION 3.5)

project(qjs-ort VERSION 0.1 LANGUAGES C)

# clangd (Sublime LSP) finds build/compile_commands.json by itself
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

set(ort_LIBRARIES ${ORT_LIBRARY} ${CMAKE_DL_LIBS})

make_module(ort quickjs-ort.c js-utils.c js-utils.h)

# A vendored or prefixed ONNX Runtime is not on the loader's path. ort.so has the module directory in
# its RUNPATH, so the libraries go next to it. A system copy needs nothing.
if(ORT_PREFIX OR ORT_LIBRARY_DIR MATCHES "^${CMAKE_CURRENT_SOURCE_DIR}/third_party/")
  set(_ort_install_default ON)
else()
  set(_ort_install_default OFF)
endif()
option(INSTALL_ORT_LIBRARY "Install the ONNX Runtime libraries next to ort.so" ${_ort_install_default})

if(INSTALL_ORT_LIBRARY AND BUILD_SHARED)
  # ort.so asks for the SONAME of the library, and the files of a package do not always have a link of that name
  # (a sysroot copy has libonnxruntime.so and libonnxruntime.so.<version>), so install the real file under the SONAME
  find_program(_ort_readelf NAMES readelf "${CMAKE_READELF}")
  set(_ort_soname "libonnxruntime.so.1")

  if(_ort_readelf)
    execute_process(COMMAND "${_ort_readelf}" -d "${ORT_LIBRARY}" OUTPUT_VARIABLE _ort_dynamic ERROR_QUIET)

    if(_ort_dynamic MATCHES "SONAME[^\n]*\\[([^]]+)\\]")
      set(_ort_soname "${CMAKE_MATCH_1}")
    endif(_ort_dynamic MATCHES "SONAME[^\n]*\\[([^]]+)\\]")
  endif(_ort_readelf)

  get_filename_component(_ort_real "${ORT_LIBRARY}" REALPATH)
  install(FILES "${_ort_real}" DESTINATION "${QUICKJS_C_MODULE_DIR}" RENAME "${_ort_soname}")

  if(EXISTS "${ORT_LIBRARY_DIR}/libonnxruntime_providers_shared.so")
    install(FILES "${ORT_LIBRARY_DIR}/libonnxruntime_providers_shared.so" DESTINATION "${QUICKJS_C_MODULE_DIR}")
  endif(EXISTS "${ORT_LIBRARY_DIR}/libonnxruntime_providers_shared.so")
endif(INSTALL_ORT_LIBRARY AND BUILD_SHARED)

foreach(_t qjs-ort qjs-ort-static)
  if(TARGET ${_t})
    # find libonnxruntime.so.1 next to a relocated install and in the build tree
    set_property(TARGET ${_t} APPEND PROPERTY BUILD_RPATH "${ORT_LIBRARY_DIR}")
  endif(TARGET ${_t})
endforeach(_t)
