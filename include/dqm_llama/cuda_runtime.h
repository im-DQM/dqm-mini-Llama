// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_DQM_LLAMA_CUDA_RUNTIME_H_
#define INCLUDE_DQM_LLAMA_CUDA_RUNTIME_H_

#include <cstddef>
#include <string>

namespace dqm {

struct CudaDeviceInfo {
  int id = 0;
  std::string name;
  int compute_major = 0;
  int compute_minor = 0;
  size_t total_memory_bytes = 0;
  int driver_version = 0;
  int runtime_version = 0;
};

enum class CudaMemcpyKind {
  kHostToDevice,
  kDeviceToHost,
  kDeviceToDevice,
};

bool CudaRuntimeBuilt();
int CudaDeviceCount();
void CudaSetDevice(int device_id);
CudaDeviceInfo CudaGetDeviceInfo(int device_id);
std::string CudaFormatDeviceInfo(const CudaDeviceInfo& info);

void* CudaMallocBytes(size_t bytes);
void CudaFreeBytes(void* ptr);
void CudaMemcpyBytes(void* dst, const void* src, size_t bytes,
                     CudaMemcpyKind kind);

class CudaDeviceBuffer {
 public:
  CudaDeviceBuffer() = default;
  explicit CudaDeviceBuffer(size_t bytes, int device_id = 0);
  ~CudaDeviceBuffer();

  CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
  CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

  CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept;
  CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept;

  void Reset();
  void Reset(size_t bytes, int device_id = 0);
  void Upload(const void* src, size_t bytes);
  void Download(void* dst, size_t bytes) const;

  void* data() { return data_; }

  const void* data() const { return data_; }

  size_t bytes() const { return bytes_; }

  int device_id() const { return device_id_; }

  bool empty() const { return data_ == nullptr; }

 private:
  void ReleaseNoexcept() noexcept;

  void* data_ = nullptr;
  size_t bytes_ = 0;
  int device_id_ = 0;
};

}  // namespace dqm

#endif  // INCLUDE_DQM_LLAMA_CUDA_RUNTIME_H_
