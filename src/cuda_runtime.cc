// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "dqm_llama/cuda_runtime.h"

#include <sstream>
#include <stdexcept>
#include <string>

#ifdef DQM_LLAMA_USE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace dqm {

namespace {

#ifndef DQM_LLAMA_USE_CUDA
std::runtime_error CudaNotBuiltError() {
  return std::runtime_error(
      "CUDA backend was not built. Reconfigure with -DDQM_LLAMA_CUDA=ON on a "
      "NVIDIA CUDA machine.");
}
#endif

#ifdef DQM_LLAMA_USE_CUDA
void CheckCuda(cudaError_t err, const char* expr) {
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA runtime error in " + std::string(expr) +
                             ": " + cudaGetErrorString(err));
  }
}

cudaMemcpyKind ToCudaMemcpyKind(CudaMemcpyKind kind) {
  switch (kind) {
    case CudaMemcpyKind::kHostToDevice:
      return cudaMemcpyHostToDevice;
    case CudaMemcpyKind::kDeviceToHost:
      return cudaMemcpyDeviceToHost;
    case CudaMemcpyKind::kDeviceToDevice:
      return cudaMemcpyDeviceToDevice;
  }
  throw std::runtime_error("unknown CUDA memcpy kind");
}
#endif

}  // namespace

bool CudaRuntimeBuilt() {
#ifdef DQM_LLAMA_USE_CUDA
  return true;
#else
  return false;
#endif
}

int CudaDeviceCount() {
#ifdef DQM_LLAMA_USE_CUDA
  int count = 0;
  CheckCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  return count;
#else
  throw CudaNotBuiltError();
#endif
}

void CudaSetDevice(int device_id) {
#ifdef DQM_LLAMA_USE_CUDA
  CheckCuda(cudaSetDevice(device_id), "cudaSetDevice");
#else
  (void)device_id;
  throw CudaNotBuiltError();
#endif
}

CudaDeviceInfo CudaGetDeviceInfo(int device_id) {
#ifdef DQM_LLAMA_USE_CUDA
  int count = CudaDeviceCount();
  if (device_id < 0 || device_id >= count) {
    throw std::runtime_error("CUDA device " + std::to_string(device_id) +
                             " is not available; detected " +
                             std::to_string(count) + " device(s)");
  }

  cudaDeviceProp prop{};
  CheckCuda(cudaGetDeviceProperties(&prop, device_id),
            "cudaGetDeviceProperties");

  CudaDeviceInfo info;
  info.id = device_id;
  info.name = prop.name;
  info.compute_major = prop.major;
  info.compute_minor = prop.minor;
  info.total_memory_bytes = prop.totalGlobalMem;
  CheckCuda(cudaDriverGetVersion(&info.driver_version), "cudaDriverGetVersion");
  CheckCuda(cudaRuntimeGetVersion(&info.runtime_version),
            "cudaRuntimeGetVersion");
  return info;
#else
  (void)device_id;
  throw CudaNotBuiltError();
#endif
}

std::string CudaFormatDeviceInfo(const CudaDeviceInfo& info) {
  double total_gb =
      static_cast<double>(info.total_memory_bytes) / (1024.0 * 1024.0 * 1024.0);
  std::ostringstream out;
  out << "device " << info.id << ": " << info.name << ", compute capability "
      << info.compute_major << "." << info.compute_minor << ", total memory "
      << total_gb << " GB"
      << ", cuda runtime " << info.runtime_version << ", driver "
      << info.driver_version;
  return out.str();
}

void* CudaMallocBytes(size_t bytes) {
#ifdef DQM_LLAMA_USE_CUDA
  if (bytes == 0) {
    return nullptr;
  }
  void* ptr = nullptr;
  CheckCuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
  return ptr;
#else
  (void)bytes;
  throw CudaNotBuiltError();
#endif
}

void CudaFreeBytes(void* ptr) {
#ifdef DQM_LLAMA_USE_CUDA
  if (ptr == nullptr) {
    return;
  }
  CheckCuda(cudaFree(ptr), "cudaFree");
#else
  (void)ptr;
  throw CudaNotBuiltError();
#endif
}

void CudaMemcpyBytes(void* dst, const void* src, size_t bytes,
                     CudaMemcpyKind kind) {
#ifdef DQM_LLAMA_USE_CUDA
  if (bytes == 0) {
    return;
  }
  if (dst == nullptr || src == nullptr) {
    throw std::runtime_error(
        "cudaMemcpy requires non-null src and dst for non-empty copies");
  }
  CheckCuda(cudaMemcpy(dst, src, bytes, ToCudaMemcpyKind(kind)), "cudaMemcpy");
#else
  (void)dst;
  (void)src;
  (void)bytes;
  (void)kind;
  throw CudaNotBuiltError();
#endif
}

CudaDeviceBuffer::CudaDeviceBuffer(size_t bytes, int device_id) {
  Reset(bytes, device_id);
}

CudaDeviceBuffer::~CudaDeviceBuffer() { ReleaseNoexcept(); }

CudaDeviceBuffer::CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept
    : data_(other.data_), bytes_(other.bytes_), device_id_(other.device_id_) {
  other.data_ = nullptr;
  other.bytes_ = 0;
  other.device_id_ = 0;
}

CudaDeviceBuffer& CudaDeviceBuffer::operator=(
    CudaDeviceBuffer&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  ReleaseNoexcept();
  data_ = other.data_;
  bytes_ = other.bytes_;
  device_id_ = other.device_id_;
  other.data_ = nullptr;
  other.bytes_ = 0;
  other.device_id_ = 0;
  return *this;
}

void CudaDeviceBuffer::Reset() {
  if (data_ == nullptr) {
    return;
  }
  CudaSetDevice(device_id_);
  CudaFreeBytes(data_);
  data_ = nullptr;
  bytes_ = 0;
  device_id_ = 0;
}

void CudaDeviceBuffer::Reset(size_t bytes, int device_id) {
  Reset();
  if (bytes == 0) {
    device_id_ = device_id;
    return;
  }
  CudaSetDevice(device_id);
  data_ = CudaMallocBytes(bytes);
  bytes_ = bytes;
  device_id_ = device_id;
}

void CudaDeviceBuffer::Upload(const void* src, size_t bytes) {
  if (bytes > bytes_) {
    throw std::runtime_error("CUDA upload exceeds device buffer size");
  }
  CudaSetDevice(device_id_);
  CudaMemcpyBytes(data_, src, bytes, CudaMemcpyKind::kHostToDevice);
}

void CudaDeviceBuffer::Download(void* dst, size_t bytes) const {
  if (bytes > bytes_) {
    throw std::runtime_error("CUDA download exceeds device buffer size");
  }
  CudaSetDevice(device_id_);
  CudaMemcpyBytes(dst, data_, bytes, CudaMemcpyKind::kDeviceToHost);
}

void CudaDeviceBuffer::ReleaseNoexcept() noexcept {
  try {
    Reset();
  } catch (...) {
  }
}

}  // namespace dqm
