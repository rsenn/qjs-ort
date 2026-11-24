#!/bin/sh
# fetch-onnxruntime.sh [VERSION] [RUNTIME] - unpack the NuGet package of
# ONNX Runtime into third_party/onnxruntime-<RUNTIME>-<VERSION> (GitHub
# release downloads 404 from this network). RUNTIME: linux-x64, linux-arm64.
set -eu
ver=${1:-${ORT_VERSION:-1.32.0}}
rt=${2:-linux-x64}
here=$(cd "$(dirname "$0")/.." && pwd)
out=$here/third_party/onnxruntime-$rt-$ver
[ -f "$out/lib/libonnxruntime.so.1" ] && { echo "$out"; exit 0; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fL --retry 2 -o "$tmp/ort.nupkg" \
  "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/$ver/microsoft.ml.onnxruntime.$ver.nupkg"
unzip -q -o "$tmp/ort.nupkg" 'build/native/include/*' "runtimes/$rt/native/*" -d "$tmp/x"
mkdir -p "$out/include" "$out/lib"
cp -a "$tmp"/x/build/native/include/. "$out/include/"
cp -a "$tmp"/x/runtimes/$rt/native/*.so "$out/lib/"
# the package has the unversioned name only: add the SONAME links
so=$(readelf -d "$out/lib/libonnxruntime.so" | sed -n 's/.*SONAME.*\[\(.*\)\]/\1/p')
[ -n "$so" ] && [ "$so" != libonnxruntime.so ] && ln -sf libonnxruntime.so "$out/lib/$so"
echo "$out"
