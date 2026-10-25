// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "dqm_llama/cuda_tensor.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace dqm {

namespace {

size_t CheckedNumel(const std::vector<int>& shape, const char* caller) {
  if (shape.empty()) {
    return 0;
  }
  size_t total = 1;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    int dim = shape[axis];
    if (dim <= 0) {
      throw std::runtime_error(std::string(caller) + ": dimension at axis " +
                               std::to_string(axis) +
                               " must be positive, got " + std::to_string(dim));
    }
    size_t dim_size = static_cast<size_t>(dim);
    if (total > std::numeric_limits<size_t>::max() / dim_size) {
      throw std::runtime_error(std::string(caller) +
                               ": shape element count overflow");
    }
    total *= dim_size;
  }
  return total;
}

std::string ShapeToString(const std::vector<int>& shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      out += ", ";
    }
    out += std::to_string(shape[i]);
  }
  out += "]";
  return out;
}

}  // namespace

CudaTensor::CudaTensor(const std::vector<int>& shape, int device_id) {
  Reset(shape, device_id);
}

void CudaTensor::Reset() {
  buffer_.Reset();
  shape_.clear();
  numel_ = 0;
  device_id_ = 0;
}

void CudaTensor::Reset(const std::vector<int>& shape, int device_id) {
  size_t num_elements = CheckedNumel(shape, "CudaTensor");
  shape_ = shape;
  numel_ = num_elements;
  device_id_ = device_id;
  buffer_.Reset(num_elements * sizeof(float), device_id);
}

void CudaTensor::UploadFrom(const Tensor& src) {
  if (src.shape != shape_) {
    throw std::runtime_error(
        "CudaTensor::UploadFrom shape mismatch: device=" + ShapeStringShort() +
        " host=" + src.ShapeStringShort());
  }
  buffer_.Upload(src.data.data(), src.size() * sizeof(float));
}

Tensor CudaTensor::Download() const {
  if (shape_.empty()) {
    return Tensor();
  }
  Tensor dst(shape_, 0.0f);
  DownloadTo(dst);
  return dst;
}

void CudaTensor::DownloadTo(Tensor& dst) const {
  if (dst.shape != shape_) {
    throw std::runtime_error(
        "CudaTensor::DownloadTo shape mismatch: device=" + ShapeStringShort() +
        " host=" + dst.ShapeStringShort());
  }
  buffer_.Download(dst.data.data(), dst.size() * sizeof(float));
}

void* CudaTensor::data() { return buffer_.data(); }

const void* CudaTensor::data() const { return buffer_.data(); }

std::string CudaTensor::ShapeStringShort() const {
  return ShapeToString(shape_);
}

CudaTensor CudaTensorFromHost(const Tensor& src, int device_id) {
  CudaTensor dst(src.shape, device_id);
  dst.UploadFrom(src);
  return dst;
}

}  // namespace dqm
