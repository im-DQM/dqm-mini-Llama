#pragma once

#include "dqm_llama/tensor.h"

namespace dqm {

// KV 缓存：所有层、所有位置的 keys / values
// keys:   [n_layers, max_seq_len, n_kv_heads, head_dim]
// values: [n_layers, max_seq_len, n_kv_heads, head_dim]
struct KvCache {
  Tensor keys;
  Tensor values;

  KvCache() = default;
  KvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim);

  // 写入某层、某位置的 k / v
  // k: [n_kv_heads, head_dim]
  // v: [n_kv_heads, head_dim]
  void Write(int layer, int pos, const Tensor& k, const Tensor& v);

  // 返回某层、某位置、某个 kv_head 的 K/V 首地址
  // 调用方自行保证后续读取的 shape 为 [head_dim]
  const float* KeyPtr(int layer, int pos, int kv_head) const;
  const float* ValuePtr(int layer, int pos, int kv_head) const;
};

}  // namespace dqm
