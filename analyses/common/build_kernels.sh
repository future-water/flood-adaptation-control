#!/usr/bin/env bash
# Compile every C++ kernel under analyses/ into analyses/common/build/, one shared library per source file.
#
#   bash analyses/common/build_kernels.sh
#   CXX=g++ CBLAS_LIBS="-lopenblas" bash analyses/common/build_kernels.sh     (Linux, other BLAS)
#
# Library names: build/<stem>.dylib on macOS, build/<stem>.so on Linux, where <stem> is the source file name
# without .cpp. Scripts load them from analyses/common/build/ with ctypes:
#   control_kernel.dylib / control_kernel.so   from common/control_kernel.cpp, loaded by common/kernel.py
#   <stem>.dylib / <stem>.so                   from analyses/<analysis>/<stem>.cpp, loaded with
#                                              common.native.load_library("<stem>")
# Two sources with the same stem must be identical (the second is skipped); otherwise the build stops.
#
# macOS: clang++ from the Xcode Command Line Tools and the Accelerate framework for CBLAS, the same flags as the
#   original runs (clang++ -O3 -std=c++17 -DACCELERATE_NEW_LAPACK -dynamiclib ... -framework Accelerate).
# Linux: g++ and a CBLAS with its header, e.g. `conda install -c conda-forge cxx-compiler blas-devel` (default
#   link -lcblas) or `apt install g++ libopenblas-dev` with CBLAS_LIBS=-lopenblas. Sources that include
#   <Accelerate/Accelerate.h> use only its CBLAS interface; compat/ maps that include to <cblas.h>. With a BLAS
#   other than Accelerate, results agree with the original runs to round-off, not bit for bit.
# No -ffast-math: control_kernel reproduces the Python recurrence of the pipeline.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSES="$(dirname "$HERE")"
BUILD="$HERE/build"
FLAGS=(-O3 -std=c++17)

case "$(uname -s)" in
  Darwin)
    CXX="${CXX:-clang++}"; EXT=dylib
    FLAGS+=(-dynamiclib -DACCELERATE_NEW_LAPACK -Wno-deprecated-declarations)
    BLAS=(-framework Accelerate)
    ;;
  Linux)
    CXX="${CXX:-g++}"; EXT=so
    FLAGS+=(-shared -fPIC -I"$HERE/compat")
    read -r -a BLAS <<< "${CBLAS_LIBS:--lcblas}"
    if [ -n "${CONDA_PREFIX:-}" ]; then
      FLAGS+=(-I"$CONDA_PREFIX/include")
      BLAS=(-L"$CONDA_PREFIX/lib" -Wl,-rpath,"$CONDA_PREFIX/lib" "${BLAS[@]}")
    fi
    ;;
  *)
    echo "unsupported platform: $(uname -s)" >&2; exit 1
    ;;
esac

sources="$(find "$ANALYSES" -name '*.cpp' -not -path "$BUILD/*" | LC_ALL=C sort)"
[ -n "$sources" ] || { echo "no .cpp files under $ANALYSES" >&2; exit 1; }

# Check names before compiling anything, so a conflict never leaves a wrong library in build/.
unique=""
seen=""
while IFS= read -r src; do
  stem="$(basename "$src" .cpp)"
  previous="$(printf '%s\n' "$seen" | awk -F'\t' -v s="$stem" '$1 == s { print $2; exit }')"
  if [ -n "$previous" ]; then
    if cmp -s "$previous" "$src"; then
      echo "skip   ${src#"$ANALYSES"/} (identical to ${previous#"$ANALYSES"/})"; continue
    fi
    echo "error: ${src#"$ANALYSES"/} and ${previous#"$ANALYSES"/} share the name $stem but differ" >&2; exit 1
  fi
  seen="$seen$stem"$'\t'"$src"$'\n'
  unique="$unique$src"$'\n'
done <<< "$sources"

mkdir -p "$BUILD"
built=0
while IFS= read -r src; do
  [ -n "$src" ] || continue
  stem="$(basename "$src" .cpp)"
  link=()
  if grep -q -E '#include[[:space:]]*<(Accelerate/Accelerate|cblas)\.h>' "$src"; then link=("${BLAS[@]}"); fi
  echo "build  ${src#"$ANALYSES"/} -> common/build/$stem.$EXT"
  "$CXX" "${FLAGS[@]}" "$src" ${link[@]+"${link[@]}"} -o "$BUILD/$stem.$EXT"
  built=$((built + 1))
done <<< "$unique"

echo "$built librar$([ "$built" = 1 ] && echo y || echo ies) in $BUILD"
