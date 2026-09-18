#include "dqm_llama/kv_cache.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace dqm { 

namespace {

void Require4DCacheShape(const Tensor& t, const char* name) {
  if (t.num_dims() != 4)
 {
   throw std::runtime_error(std::string(name) + " need 4D Tensor");
 } 
}

}  // namespace

KvCache::KvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim) {
  keys = MakeTensor4D(n_layers, max_seq_len, n_kv_heads, head_dim, 0.0f);
  values = MakeTensor4D(n_layers, max_seq_len, n_kv_heads, head_dim, 0.0f);
}

void KvCache::Write(int layer, int pos, const Tensor& k, const Tensor& v) {
  Require4DCacheShape(keys, "keys");
  Require4DCacheShape(values, "values");
  int n_layers = keys.shape[0];
  int max_seq_len = keys.shape[1];
  int n_kv_heads = keys.shape[2];
  int head_dim = keys.shape[3];

  if (k.num_dims() != 2 || v.num_dims() != 2 || k.shape != v.shape) {
    throw std::out_of_range("kv_cache: Write : k v not matched");
  }
  if (layer < 0 || layer >= n_layers) {
    throw std::out_of_range("kv_cache: Write : layer out of range");
  }
  if (pos < 0 || pos >= max_seq_len) {
    throw std::out_of_range("kv_cache: Write : pos out of range");
  }
  if (k.shape[0] != n_kv_heads || k.shape[1] != head_dim) {
    throw std::out_of_range("kv_cache: Write : k v not match cache");
  }

  size_t layer_stride = static_cast<size_t>(max_seq_len) * n_kv_heads * head_dim;
  size_t pos_stride = static_cast<size_t>(n_kv_heads) * head_dim;
  size_t base = static_cast<size_t>(layer) * layer_stride +
                static_cast<size_t>(pos) * pos_stride;
  
  memcpy(&keys.data[base], k.data.data(), pos_stride * sizeof(float));
  memcpy(&values.data[base], v.data.data(), pos_stride * sizeof(float));
}

const float* KvCache::KeyPtr(int layer, int pos, int kv_head) const {
  Require4DCacheShape(keys, "keys");
  if (layer < 0 || layer >= keys.shape[0]) {
    throw std::out_of_range("kv_cache: KeyPtr : layer out of range");
  }
  if (pos < 0 || pos >= keys.shape[1]) {
    throw std::out_of_range("kv_cache: KeyPtr : pos out of range");
  }
  if (kv_head < 0 || kv_head >= keys.shape[2]) {
    throw std::out_of_range("kv_cache: KeyPtr : kv_head out of range");
  }
  size_t layer_stride = static_cast<size_t>(keys.shape[1] * keys.shape[2] * keys.shape[3]);
  size_t pos_stride = static_cast<size_t>(keys.shape[2] * keys.shape[3]);
  size_t head_stride = static_cast<size_t>(keys.shape[3]);
  size_t base = layer * layer_stride + pos * pos_stride + kv_head * head_stride;

  return &keys.data[base];
}

const float* KvCache::ValuePtr(int layer, int pos, int kv_head) const {
  Require4DCacheShape(values, "values");
  if (layer < 0 || layer >= values.shape[0]) {
    throw std::out_of_range("kv_cache: ValuePtr : layer out of range");
  }
  if (pos < 0 || pos >= values.shape[1]) {
    throw std::out_of_range("kv_cache: ValuePtr : pos out of range");
  }
  if (kv_head < 0 || kv_head >= values.shape[2]) {
    throw std::out_of_range("kv_cache: ValuePtr : kv_head out of range");
  }
  size_t layer_stride = static_cast<size_t>(values.shape[1] * values.shape[2] * values.shape[3]);
  size_t pos_stride = static_cast<size_t>(values.shape[2] * values.shape[3]);
  size_t head_stride = static_cast<size_t>(values.shape[3]);
  size_t base = layer * layer_stride + pos * pos_stride + kv_head * head_stride;

  return &values.data[base];
}

}  // namespace dqm
