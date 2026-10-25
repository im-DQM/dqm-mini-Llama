// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_DQM_LLAMA_CUDA_TENSOR_H_
#define INCLUDE_DQM_LLAMA_CUDA_TENSOR_H_

#include <cstddef>
#include <string>
#include <vector>

#include "dqm_llama/cuda_runtime.h"
#include "dqm_llama/tensor.h"

namespace dqm {

class CudaTensor {
 public:
  CudaTensor() = default;
  explicit CudaTensor(const std::vector<int>& shape, int device_id = 0);

  CudaTensor(const CudaTensor&) = delete;
  CudaTensor& operator=(const CudaTensor&) = delete;

  CudaTensor(CudaTensor&&) noexcept = default;
  CudaTensor& operator=(CudaTensor&&) noexcept = default;

  void Reset();
  void Reset(const std::vector<int>& shape, int device_id = 0);
  void UploadFrom(const Tensor& src);
  Tensor Download() const;
  void DownloadTo(Tensor& dst) const;

  void* data();
  const void* data() const;

  const std::vector<int>& shape() const { return shape_; }

  int num_dims() const { return static_cast<int>(shape_.size()); }

  size_t size() const { return numel_; }

  size_t bytes() const { return buffer_.bytes(); }

  int device_id() const { return device_id_; }

  bool empty() const { return buffer_.empty(); }

  std::string ShapeStringShort() const;

 private:
  std::vector<int> shape_;
  size_t numel_ = 0;
  int device_id_ = 0;
  CudaDeviceBuffer buffer_;
};

CudaTensor CudaTensorFromHost(const Tensor& src, int device_id = 0);

}  // namespace dqm

#endif  // INCLUDE_DQM_LLAMA_CUDA_TENSOR_H_
