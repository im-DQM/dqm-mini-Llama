// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "dqm_llama/cuda_kv_cache.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace dqm {

namespace {

#ifndef DQM_LLAMA_USE_CUDA
std::runtime_error CudaKvCacheNotBuiltError() {
  return std::runtime_error(
      "CUDA KV cache was not built. Reconfigure with -DDQM_LLAMA_CUDA=ON on a "
      "NVIDIA CUDA machine.");
}
#endif

void RequirePositive(int value, const char* name) {
  if (value <= 0) {
    throw std::runtime_error(std::string("CudaKvCache: ") + name +
                             " must be positive");
  }
}

size_t CheckedMul(size_t a, size_t b, const char* label) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
    throw std::runtime_error(std::string("CudaKvCache: size overflow for ") +
                             label);
  }
  return a * b;
}

}  // namespace

bool CudaKvCacheBuilt() {
#ifdef DQM_LLAMA_USE_CUDA
  return true;
#else
  return false;
#endif
}

CudaKvCache::CudaKvCache(int n_layers, int max_seq_len, int n_kv_heads,
                         int head_dim, int device_id) {
  Reset(n_layers, max_seq_len, n_kv_heads, head_dim, device_id);
}

void CudaKvCache::Reset() {
  keys_.Reset();
  values_.Reset();
  n_layers_ = 0;
  max_seq_len_ = 0;
  n_kv_heads_ = 0;
  head_dim_ = 0;
  device_id_ = 0;
}

void CudaKvCache::Reset(int n_layers, int max_seq_len, int n_kv_heads,
                        int head_dim, int device_id) {
#ifdef DQM_LLAMA_USE_CUDA
  RequirePositive(n_layers, "n_layers");
  RequirePositive(max_seq_len, "max_seq_len");
  RequirePositive(n_kv_heads, "n_kv_heads");
  RequirePositive(head_dim, "head_dim");

  size_t elements = static_cast<size_t>(n_layers);
  elements =
      CheckedMul(elements, static_cast<size_t>(max_seq_len), "layers * seq");
  elements = CheckedMul(elements, static_cast<size_t>(n_kv_heads),
                        "layers * seq * heads");
  elements = CheckedMul(elements, static_cast<size_t>(head_dim),
                        "layers * seq * heads * dim");
  size_t bytes_per_cache = CheckedMul(elements, sizeof(float), "cache bytes");

  n_layers_ = n_layers;
  max_seq_len_ = max_seq_len;
  n_kv_heads_ = n_kv_heads;
  head_dim_ = head_dim;
  device_id_ = device_id;
  keys_.Reset(bytes_per_cache, device_id);
  values_.Reset(bytes_per_cache, device_id);
  Clear();
#else
  (void)n_layers;
  (void)max_seq_len;
  (void)n_kv_heads;
  (void)head_dim;
  (void)device_id;
  throw CudaKvCacheNotBuiltError();
#endif
}

void CudaKvCache::Clear() {
#ifdef DQM_LLAMA_USE_CUDA
  if (empty()) {
    return;
  }
  CudaSetDevice(device_id_);
  std::vector<float> zeros(keys_.bytes() / sizeof(float), 0.0f);
  keys_.Upload(zeros.data(), keys_.bytes());
  values_.Upload(zeros.data(), values_.bytes());
#else
  throw CudaKvCacheNotBuiltError();
#endif
}

void CudaKvCache::Write(int layer, int pos, const Tensor& k, const Tensor& v) {
#ifdef DQM_LLAMA_USE_CUDA
  ValidateIndices(layer, pos);
  ValidateWriteTensors(k, v);
  size_t offset = SlotOffsetBytes(layer, pos);
  size_t bytes_to_copy = k.size() * sizeof(float);
  char* key_dst = static_cast<char*>(keys_.data()) + offset;
  char* value_dst = static_cast<char*>(values_.data()) + offset;
  CudaSetDevice(device_id_);
  CudaMemcpyBytes(key_dst, k.data.data(), bytes_to_copy,
                  CudaMemcpyKind::kHostToDevice);
  CudaMemcpyBytes(value_dst, v.data.data(), bytes_to_copy,
                  CudaMemcpyKind::kHostToDevice);
#else
  (void)layer;
  (void)pos;
  (void)k;
  (void)v;
  throw CudaKvCacheNotBuiltError();
#endif
}

void CudaKvCache::WriteDevice(int layer, int pos, const CudaTensor& k,
                              const CudaTensor& v) {
#ifdef DQM_LLAMA_USE_CUDA
  ValidateIndices(layer, pos);
  if (k.device_id() != device_id_ || v.device_id() != device_id_) {
    throw std::runtime_error(
        "CudaKvCache::WriteDevice tensor is on a different CUDA device");
  }
  const size_t expected =
      static_cast<size_t>(n_kv_heads_) * static_cast<size_t>(head_dim_);
  if (k.size() != expected || v.size() != expected) {
    throw std::out_of_range(
        "CudaKvCache::WriteDevice tensor shape does not match cache shape");
  }
  size_t offset = SlotOffsetBytes(layer, pos);
  size_t bytes_to_copy = expected * sizeof(float);
  char* key_dst = static_cast<char*>(keys_.data()) + offset;
  char* value_dst = static_cast<char*>(values_.data()) + offset;
  CudaSetDevice(device_id_);
  CudaMemcpyBytes(key_dst, k.data(), bytes_to_copy,
                  CudaMemcpyKind::kDeviceToDevice);
  CudaMemcpyBytes(value_dst, v.data(), bytes_to_copy,
                  CudaMemcpyKind::kDeviceToDevice);
#else
  (void)layer;
  (void)pos;
  (void)k;
  (void)v;
  throw CudaKvCacheNotBuiltError();
#endif
}

Tensor CudaKvCache::ReadKey(int layer, int pos) const {
#ifdef DQM_LLAMA_USE_CUDA
  ValidateIndices(layer, pos);
  Tensor out({n_kv_heads_, head_dim_}, 0.0f);
  const char* src =
      static_cast<const char*>(keys_.data()) + SlotOffsetBytes(layer, pos);
  CudaSetDevice(device_id_);
  CudaMemcpyBytes(out.data.data(), src, out.size() * sizeof(float),
                  CudaMemcpyKind::kDeviceToHost);
  return out;
#else
  (void)layer;
  (void)pos;
  throw CudaKvCacheNotBuiltError();
#endif
}

Tensor CudaKvCache::ReadValue(int layer, int pos) const {
#ifdef DQM_LLAMA_USE_CUDA
  ValidateIndices(layer, pos);
  Tensor out({n_kv_heads_, head_dim_}, 0.0f);
  const char* src =
      static_cast<const char*>(values_.data()) + SlotOffsetBytes(layer, pos);
  CudaSetDevice(device_id_);
  CudaMemcpyBytes(out.data.data(), src, out.size() * sizeof(float),
                  CudaMemcpyKind::kDeviceToHost);
  return out;
#else
  (void)layer;
  (void)pos;
  throw CudaKvCacheNotBuiltError();
#endif
}

Tensor CudaKvCache::ReadKeyHead(int layer, int pos, int kv_head) const {
#ifdef DQM_LLAMA_USE_CUDA
  ValidateIndices(layer, pos);
  ValidateHeadIndex(kv_head);
  Tensor out({head_dim_}, 0.0f);
  const char* src = static_cast<const char*>(keys_.data()) +
                    HeadOffsetBytes(layer, pos, kv_head);
  CudaSetDevice(device_id_);
  CudaMemcpyBytes(out.data.data(), src, out.size() * sizeof(float),
                  CudaMemcpyKind::kDeviceToHost);
  return out;
#else
  (void)layer;
  (void)pos;
  (void)kv_head;
  throw CudaKvCacheNotBuiltError();
#endif
}

Tensor CudaKvCache::ReadValueHead(int layer, int pos, int kv_head) const {
#ifdef DQM_LLAMA_USE_CUDA
  ValidateIndices(layer, pos);
  ValidateHeadIndex(kv_head);
  Tensor out({head_dim_}, 0.0f);
  const char* src = static_cast<const char*>(values_.data()) +
                    HeadOffsetBytes(layer, pos, kv_head);
  CudaSetDevice(device_id_);
  CudaMemcpyBytes(out.data.data(), src, out.size() * sizeof(float),
                  CudaMemcpyKind::kDeviceToHost);
  return out;
#else
  (void)layer;
  (void)pos;
  (void)kv_head;
  throw CudaKvCacheNotBuiltError();
#endif
}

const void* CudaKvCache::keys_data() const {
  if (empty()) {
    throw std::runtime_error("CudaKvCache: cache is empty");
  }
  return keys_.data();
}

const void* CudaKvCache::values_data() const {
  if (empty()) {
    throw std::runtime_error("CudaKvCache: cache is empty");
  }
  return values_.data();
}

bool CudaKvCache::empty() const { return keys_.empty() || values_.empty(); }

size_t CudaKvCache::bytes() const { return keys_.bytes() + values_.bytes(); }

void CudaKvCache::ValidateIndices(int layer, int pos) const {
  if (empty()) {
    throw std::runtime_error("CudaKvCache: cache is empty");
  }
  if (layer < 0 || layer >= n_layers_) {
    throw std::out_of_range("CudaKvCache layer out of range");
  }
  if (pos < 0 || pos >= max_seq_len_) {
    throw std::out_of_range("CudaKvCache position out of range");
  }
}

void CudaKvCache::ValidateHeadIndex(int kv_head) const {
  if (kv_head < 0 || kv_head >= n_kv_heads_) {
    throw std::out_of_range("CudaKvCache head out of range");
  }
}

void CudaKvCache::ValidateWriteTensors(const Tensor& k, const Tensor& v) const {
  if (k.num_dims() != 2 || v.num_dims() != 2 || k.shape != v.shape) {
    throw std::out_of_range(
        "CudaKvCache::Write expected matching [n_kv_heads, head_dim] tensors");
  }
  if (k.shape[0] != n_kv_heads_ || k.shape[1] != head_dim_) {
    throw std::out_of_range(
        "CudaKvCache::Write tensor shape does not match cache shape");
  }
}

size_t CudaKvCache::SlotOffsetBytes(int layer, int pos) const {
  size_t slot = static_cast<size_t>(layer);
  slot = slot * static_cast<size_t>(max_seq_len_) + static_cast<size_t>(pos);
  slot =
      slot * static_cast<size_t>(n_kv_heads_) * static_cast<size_t>(head_dim_);
  return slot * sizeof(float);
}

size_t CudaKvCache::HeadOffsetBytes(int layer, int pos, int kv_head) const {
  size_t offset = SlotOffsetBytes(layer, pos);
  offset += static_cast<size_t>(kv_head) * static_cast<size_t>(head_dim_) *
            sizeof(float);
  return offset;
}

}  // namespace dqm
