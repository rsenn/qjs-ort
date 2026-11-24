# find_onnxruntime()
#
# Locates ONNX Runtime (C API only) and sets ORT_INCLUDE_DIR, ORT_LIBRARY,
# ORT_LIBRARY_DIR and ORT_FOUND. Order: ORT_PREFIX (or ORT_INCLUDE_DIR /
# ORT_LIBRARY_DIR), pkg-config, find_path/find_library, then the copy
# that tools/fetch-onnxruntime.sh unpacks into third_party/.
#
macro(find_onnxruntime)
  set(ORT_PREFIX "" CACHE PATH "ONNX Runtime install prefix (include/, lib/)")
  set(ORT_FOUND FALSE)

  # a vendored NuGet copy next to the sources counts as a prefix
  if(NOT ORT_PREFIX AND NOT ORT_INCLUDE_DIR)
    file(GLOB _ort_vendored "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-*")
    foreach(_d ${_ort_vendored})
      if(EXISTS "${_d}/include/onnxruntime_c_api.h" AND EXISTS "${_d}/lib/libonnxruntime.so")
        set(_ort_hint "${_d}")
      endif()
    endforeach(_d)
  endif()
  if(ORT_PREFIX)
    set(_ort_hint "${ORT_PREFIX}")
  endif(ORT_PREFIX)

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

  if(ORT_INCLUDE_DIR AND ORT_LIBRARY)
    set(ORT_FOUND TRUE)
    get_filename_component(ORT_LIBRARY_DIR "${ORT_LIBRARY}" DIRECTORY)
    message(STATUS "ONNX Runtime: ${ORT_LIBRARY} (${ORT_INCLUDE_DIR})")
  else()
    message(FATAL_ERROR "ONNX Runtime not found: pass -DORT_PREFIX=<dir> or run tools/fetch-onnxruntime.sh")
  endif()
endmacro(find_onnxruntime)
