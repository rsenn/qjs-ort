# Cross build for aarch64 Linux (Raspberry Pi) with the host clang and lld, no gcc cross package needed.
#
#   cmake -S . -B build/aarch64-linux-gnu -DCMAKE_BUILD_TYPE=Release -DBUILD_QUICKJS=ON \
#     -DCMAKE_TOOLCHAIN_FILE=tools/aarch64-clang.toolchain.cmake -DORT_PREFIX=<aarch64 ONNX Runtime>
#
# AARCH64_SYSROOT (environment or cache variable) is a copy of the board's /lib, /usr/include and /usr/lib; its glibc
# must not be newer than the board's. ORT_PREFIX needs include/ and lib/ of the linux-arm64 ONNX Runtime
# (tools/fetch-onnxruntime.sh VERSION linux-arm64).
if(NOT AARCH64_SYSROOT AND DEFINED ENV{AARCH64_SYSROOT})
  set(AARCH64_SYSROOT "$ENV{AARCH64_SYSROOT}")
endif(NOT AARCH64_SYSROOT AND DEFINED ENV{AARCH64_SYSROOT})

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER clang)
set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)

if(AARCH64_SYSROOT)
  set(CMAKE_SYSROOT "${AARCH64_SYSROOT}")
endif(AARCH64_SYSROOT)

foreach(_kind EXE SHARED MODULE)
  set(CMAKE_${_kind}_LINKER_FLAGS_INIT "-fuse-ld=lld")
endforeach(_kind EXE SHARED MODULE)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)

# UseMultiArch asks the compiler for -dumpmachine without --target and would name the host triplet
set(CMAKE_LIBRARY_ARCHITECTURE aarch64-linux-gnu)
set(CMAKE_ARCH_LIBDIR lib/aarch64-linux-gnu)
