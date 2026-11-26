# find_onnxruntime()
#
# Locates ONNX Runtime (C API only) and sets ORT_INCLUDE_DIR, ORT_LIBRARY,
# ORT_LIBRARY_DIR and ORT_FOUND. Order: ORT_PREFIX (or ORT_INCLUDE_DIR /
# ORT_LIBRARY_DIR), the copy of this target's runtime that
# tools/fetch-onnxruntime.sh unpacked into third_party/, pkg-config,
# find_path/find_library. When nothing is found and -DBUILD_ORT=ON, the
# script is run (network needed) and the search repeated; ORT_VERSION
# picks the version. The runtime is linux-x64, linux-arm64 or
# android-<arch> (also on Termux, which uses the Android AAR); other
# targets need ORT_PREFIX.
#
option(BUILD_ORT "Fetch a prebuilt ONNX Runtime with tools/fetch-onnxruntime.sh when none is found" OFF)
set(ORT_VERSION "1.32.0" CACHE STRING "ONNX Runtime version for tools/fetch-onnxruntime.sh")

# ort_runtime_name(OUT): the RUNTIME argument of tools/fetch-onnxruntime.sh for the target, "" when there is none
function(ort_runtime_name out)
  set(_os "")
  set(_cpu "")
  set(_proc "${CMAKE_SYSTEM_PROCESSOR}")

  if(ANDROID OR CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(_os android)
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # a cmake on Termux may say Linux; the machine tells
    if(NOT CMAKE_CROSSCOMPILING AND (DEFINED ENV{TERMUX_VERSION} OR EXISTS /system/bin/app_process))
      set(_os android)
    else()
      set(_os linux)
    endif()
  endif()

  if(_os STREQUAL "android")
    if(_proc MATCHES "^(aarch64|arm64|ARM64)$")
      set(_cpu arm64)
    elseif(_proc MATCHES "^(x86_64|AMD64)$")
      set(_cpu x86_64)
    elseif(_proc MATCHES "^(i[3-6]86|x86)$")
      set(_cpu x86)
    elseif(_proc MATCHES "^arm")
      set(_cpu arm)
    endif()
  elseif(_os STREQUAL "linux")
    if(_proc MATCHES "^(x86_64|AMD64)$")
      set(_cpu x64)
    elseif(_proc MATCHES "^(aarch64|arm64)$")
      set(_cpu arm64)
    endif()
  endif()

  if(_os AND _cpu)
    set(${out} "${_os}-${_cpu}" PARENT_SCOPE)
  else()
    set(${out} "" PARENT_SCOPE)
  endif()
endfunction(ort_runtime_name)

# the unpacked copy in third_party/ for the runtime, the newest version; any runtime when the target is unknown
macro(_ort_find_vendored)
  file(GLOB _ort_vendored "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-*")
  list(SORT _ort_vendored)
  foreach(_d ${_ort_vendored})
    if(EXISTS "${_d}/include/onnxruntime_c_api.h" AND EXISTS "${_d}/lib/libonnxruntime.so")
      if(NOT _ort_rt OR _d MATCHES "/onnxruntime-${_ort_rt}-[^/]*$")
        set(_ort_hint "${_d}")
      endif()
    endif()
  endforeach(_d)
endmacro(_ort_find_vendored)

macro(_ort_search)
  find_path(ORT_INCLUDE_DIR onnxruntime_c_api.h
            HINTS ${_ort_hint}/include ${ORT_INCLUDE_DIR}
            PATH_SUFFIXES onnxruntime)
  find_library(ORT_LIBRARY NAMES onnxruntime
               HINTS ${_ort_hint}/lib ${ORT_LIBRARY_DIR})

  if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
      pkg_check_modules(_ORT_PC QUIET libonnxruntime)
      if(_ORT_PC_FOUND)
        set(ORT_INCLUDE_DIR "${_ORT_PC_INCLUDE_DIRS}" CACHE PATH "" FORCE)
        find_library(ORT_LIBRARY NAMES onnxruntime HINTS ${_ORT_PC_LIBRARY_DIRS})
      endif(_ORT_PC_FOUND)
    endif(PKG_CONFIG_FOUND)
  endif(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
endmacro(_ort_search)

macro(find_onnxruntime)
  set(ORT_PREFIX "" CACHE PATH "ONNX Runtime install prefix (include/, lib/)")
  set(ORT_FOUND FALSE)
  set(_ort_hint "")
  ort_runtime_name(_ort_rt)

  # a vendored copy next to the sources counts as a prefix
  if(NOT ORT_PREFIX AND NOT ORT_INCLUDE_DIR)
    _ort_find_vendored()
  endif()
  if(ORT_PREFIX)
    set(_ort_hint "${ORT_PREFIX}")
  endif(ORT_PREFIX)

  _ort_search()

  if((NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY) AND BUILD_ORT)
    if(NOT _ort_rt)
      message(FATAL_ERROR "BUILD_ORT: no prebuilt ONNX Runtime for ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}, pass -DORT_PREFIX=<dir>")
    endif()

    message(STATUS "ONNX Runtime: fetching ${ORT_VERSION} for ${_ort_rt}")
    execute_process(COMMAND sh "${CMAKE_CURRENT_SOURCE_DIR}/tools/fetch-onnxruntime.sh" "${ORT_VERSION}" "${_ort_rt}"
                    OUTPUT_VARIABLE _ort_fetched ERROR_VARIABLE _ort_fetch_err RESULT_VARIABLE _ort_fetch_rc
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(_ort_fetch_rc)
      message(FATAL_ERROR "tools/fetch-onnxruntime.sh ${ORT_VERSION} ${_ort_rt} failed (${_ort_fetch_rc}):\n${_ort_fetch_err}")
    endif(_ort_fetch_rc)

    set(_ort_hint "${_ort_fetched}")
    # a failed search left NOTFOUND in the cache, which find_path and find_library search again for
    _ort_search()
  endif()

  if(ORT_INCLUDE_DIR AND ORT_LIBRARY)
    set(ORT_FOUND TRUE)
    get_filename_component(ORT_LIBRARY_DIR "${ORT_LIBRARY}" DIRECTORY)
    message(STATUS "ONNX Runtime: ${ORT_LIBRARY} (${ORT_INCLUDE_DIR})")
  else()
    message(FATAL_ERROR "ONNX Runtime not found: pass -DORT_PREFIX=<dir>, run tools/fetch-onnxruntime.sh or configure with -DBUILD_ORT=ON")
  endif()
endmacro(find_onnxruntime)
