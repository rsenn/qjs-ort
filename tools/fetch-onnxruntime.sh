#!/bin/sh
# fetch-onnxruntime.sh [VERSION] [RUNTIME] - unpack a prebuilt ONNX Runtime
# into third_party/onnxruntime-<RUNTIME>-<VERSION> (include/ and lib/) and
# print that directory.
#
# RUNTIME             source                       what is taken
#   linux-x64         NuGet package                runtimes/<rt>/native/*.so
#   linux-arm64       NuGet package                same
#   android-arm64     Maven Central AAR            jni/arm64-v8a/libonnxruntime.so
#   android-arm       Maven Central AAR            jni/armeabi-v7a/...
#   android-x86_64    Maven Central AAR            jni/x86_64/...
#   android-x86       Maven Central AAR            jni/x86/...
#
# Without RUNTIME the one of this machine is used (uname: Linux x86_64 is
# linux-x64, aarch64 is linux-arm64, Android, which is what Termux reports,
# is android-<arch>). GitHub release downloads 404 from this network, hence
# the package repositories. An Android library is the one that runs on
# Termux (bionic libc); it is about 35 MB.
set -eu
ver=${1:-${ORT_VERSION:-1.32.0}}
rt=${2:-}

if [ -z "$rt" ]; then
  arch=$(uname -m)
  case $(uname -o 2>/dev/null || uname -s) in
    Android) os=android ;;
    *Linux*) os=linux ;;
    *) echo "fetch-onnxruntime.sh: no prebuilt ONNX Runtime for $(uname -s) $arch, give RUNTIME" >&2; exit 1 ;;
  esac
  case $arch in
    x86_64|amd64) cpu=x64; [ $os = android ] && cpu=x86_64 ;;
    aarch64|arm64|armv8*) cpu=arm64 ;;
    armv7*|armv8l|arm) cpu=arm ;;
    i?86) cpu=x86 ;;
    *) echo "fetch-onnxruntime.sh: unknown machine $arch, give RUNTIME" >&2; exit 1 ;;
  esac
  rt=$os-$cpu
fi

here=$(cd "$(dirname "$0")/.." && pwd)
out=$here/third_party/onnxruntime-$rt-$ver
if [ -f "$out/include/onnxruntime_c_api.h" ] && [ -f "$out/lib/libonnxruntime.so" ]; then
  echo "$out"
  exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

case $rt in
  linux-x64|linux-arm64)
    curl -fL --retry 2 -o "$tmp/ort.nupkg" \
      "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/$ver/microsoft.ml.onnxruntime.$ver.nupkg"
    unzip -q -o "$tmp/ort.nupkg" 'build/native/include/*' "runtimes/$rt/native/*" -d "$tmp/x"
    inc=$tmp/x/build/native/include
    lib=$tmp/x/runtimes/$rt/native
    ;;
  android-*)
    case $rt in
      android-arm64) abi=arm64-v8a ;;
      android-arm) abi=armeabi-v7a ;;
      android-x86_64) abi=x86_64 ;;
      android-x86) abi=x86 ;;
      *) echo "fetch-onnxruntime.sh: unknown RUNTIME $rt" >&2; exit 1 ;;
    esac
    curl -fL --retry 2 -o "$tmp/ort.aar" \
      "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/$ver/onnxruntime-android-$ver.aar"
    unzip -q -o "$tmp/ort.aar" 'headers/*' "jni/$abi/libonnxruntime.so" -d "$tmp/x"
    inc=$tmp/x/headers
    lib=$tmp/x/jni/$abi
    ;;
  *)
    echo "fetch-onnxruntime.sh: unknown RUNTIME $rt (linux-x64, linux-arm64, android-arm64, android-arm, android-x86_64, android-x86)" >&2
    exit 1
    ;;
esac

mkdir -p "$out/include" "$out/lib"
cp -a "$inc"/. "$out/include/"
cp -a "$lib"/libonnxruntime*.so "$out/lib/"

# a package has the unversioned name only: add the SONAME link when the
# SONAME differs (libonnxruntime.so.1 on Linux, none on Android)
if command -v readelf >/dev/null 2>&1; then
  so=$(readelf -d "$out/lib/libonnxruntime.so" | sed -n 's/.*SONAME.*\[\(.*\)\]/\1/p')
  [ -n "$so" ] && [ "$so" != libonnxruntime.so ] && ln -sf libonnxruntime.so "$out/lib/$so"
fi
echo "$out"
