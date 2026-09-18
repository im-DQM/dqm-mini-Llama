#pragma once

// 测试共用的 tiny 模型夹具 —— 手工造，不经过 loader。
//
// 放在共享头里（对应参考项目 tests/gguf_loader_fixture.h 的做法）：
// batch_test.cc 和 forward_test.cc 都需要它，而匿名 namespace 是内部链接、
// 另一个 TU 看不见，所以只能共享或者复制。选了共享 —— 免得两份副本静默分叉。
//
// 全部 inline：被多个 TU include 时不会产生重复符号。
// inline 函数也不会有 -Wunused-function（未使用的 inline 函数不报警）。

#include <cmath>
#include <cstddef>
#include <vector>

#include "dqm_llama/model.h"
#include "dqm_llama/quantized_tensor.h"
#include "dqm_llama/tensor.h"

namespace dqm {
namespace test {

// 确定性伪随机数（LCG）—— 权重必须非零。
// 全 0 权重会让所有 logits 都是 0，「不同 token 结果不同」这类断言根本测不了。
inline float NextWeight(unsigned int& state) {
  state = state * 1664525u + 1013904223u;
  return static_cast<float>((state >> 9) % 2001u) / 1000.0f - 1.0f;  // [-1, 1]
}

inline Tensor MakeF32(const std::vector<int>& shape, unsigned int& state) {
  Tensor t(shape, 0.0f);
  for (size_t i = 0; i < t.data.size(); ++i) {
    t.data[i] = NextWeight(state);
  }
  return t;
}

inline QuantizedTensor MakeF32Container(const std::vector<int>& shape,
                                        unsigned int& state) {
  return ToQuantizedTensor(MakeF32(shape, state));
}

// 手工造一个 tiny 形状、权重确定性的模型 —— 不经过 loader。
// ⚠️ 每个权重的形状必须和 LayerWeights 的约定一致（wq 是 n_heads*head_dim，
//    wk/wv 是 n_kv_heads*head_dim）。改 model.h 时记得回来对一遍。
inline MiniLlamaModel MakeTinyModel(unsigned int seed = 12345u) {
  MiniLlamaModel m;
  m.loaded = true;
  const ModelConfig& c = m.config;
  unsigned int s = seed;

  m.token_embedding = MakeF32({c.vocab_size, c.dim}, s);
  m.final_norm = MakeF32({c.dim}, s);
  m.lm_head = MakeF32Container({c.vocab_size, c.dim}, s);

  m.layers.resize(static_cast<size_t>(c.n_layers));
  for (LayerWeights& lw : m.layers) {
    lw.attention_norm = MakeF32({c.dim}, s);
    lw.wq = MakeF32Container({c.n_heads * c.head_dim, c.dim}, s);
    lw.wk = MakeF32Container({c.n_kv_heads * c.head_dim, c.dim}, s);
    lw.wv = MakeF32Container({c.n_kv_heads * c.head_dim, c.dim}, s);
    lw.wo = MakeF32Container({c.dim, c.n_heads * c.head_dim}, s);
    lw.ffn_norm = MakeF32({c.dim}, s);
    lw.w_gate = MakeF32Container({c.hidden_dim, c.dim}, s);
    lw.w_up = MakeF32Container({c.hidden_dim, c.dim}, s);
    lw.w_down = MakeF32Container({c.dim, c.hidden_dim}, s);
    // bq / bk / bv 留空 —— tiny 模型没有 attention bias
  }
  return m;
}

inline bool NearlyEqual(const Tensor& a, const Tensor& b, double tol = 1e-6) {
  if (a.data.size() != b.data.size()) {
    return false;
  }
  for (size_t i = 0; i < a.data.size(); ++i) {
    if (std::fabs(static_cast<double>(a.data[i]) -
                  static_cast<double>(b.data[i])) > tol) {
      return false;
    }
  }
  return true;
}

}  // namespace test
}  // namespace dqm
