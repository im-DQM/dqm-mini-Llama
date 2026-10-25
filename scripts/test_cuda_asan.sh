#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build-cuda-asan}"
model_dir="${MODEL_DIR:-${repo_dir}/models/chat}"
n_predict="${N_PREDICT:-32}"

jobs="${JOBS:-4}"

# NVIDIA CUDA libraries require an accessible ASan shadow gap. Non-PIE host
# binaries keep the CUDA and ASan virtual-address layout stable under ASLR.
export ASAN_OPTIONS="detect_leaks=0:protect_shadow_gap=0:abort_on_error=1:halt_on_error=1${MINI_LLAMA_ASAN_OPTIONS:+:${MINI_LLAMA_ASAN_OPTIONS}}"

cmake -S "${repo_dir}" -B "${build_dir}" \
  -DMINI_LLAMA_CUDA=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g -fno-pie" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address -no-pie"
cmake --build "${build_dir}" -j"${jobs}"
ctest --test-dir "${build_dir}" --output-on-failure --timeout 600

if [[ ! -d "${model_dir}" ]]; then
  echo "CUDA ASan CTest passed; model directory unavailable for benchmark: ${model_dir}"
  exit 0
fi

"${build_dir}/dqm-llama" bench "${model_dir}" \
  --backend cuda \
  -p hello \
  -n "${n_predict}"
