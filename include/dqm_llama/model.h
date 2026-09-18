#pragma once

#include <string>
#include <vector>

#include "dqm_llama/quantized_tensor.h"
#include "dqm_llama/tensor.h"

namespace dqm {

// RoPE 的两套配对约定（契约见 ops-spec.md §1）：
//   kNormal —— 相邻配对：(x0,x1) (x2,x3) ...
//   kNeoX   —— 前后半配对：(x0,x_{d/2}) (x1,x_{1+d/2}) ...
enum class RopeType {
  kNormal,
  kNeoX,
};

// 模型超参数：形状描述 + 算法配置。
// 默认值对齐 models/tiny/model.json。
struct ModelConfig {
  int vocab_size = 128;    // 词表大小
  int dim = 32;            // 模型宽度（残差流维度）
  int hidden_dim = 86;     // FFN 中间维度
  int n_layers = 2;        // 层数
  int n_heads = 4;         // q 的注意力头数
  int n_kv_heads = 4;      // k/v 的注意力头数（GQA 时 < n_heads）
  int head_dim = 8;        // 单个头的维度 = dim / n_heads（manifest 里没有，加载时推导）
  int max_seq_len = 128;   // KV cache 最大长度
  float rope_theta = 10000.0f;
  float rms_norm_eps = 1e-5f;
  RopeType rope_type = RopeType::kNormal;
};

// 一层的全部权重。会做矩阵乘的走 QuantizedTensor，其余走 Tensor。
struct LayerWeights {
  Tensor attention_norm;   // [dim]
  QuantizedTensor wq;      // [n_heads * head_dim, dim]
  QuantizedTensor wk;      // [n_kv_heads * head_dim, dim]
  QuantizedTensor wv;      // [n_kv_heads * head_dim, dim]
  Tensor bq;               // 可选 [n_heads * head_dim]，模型没有 bias 时留空
  Tensor bk;               // 可选 [n_kv_heads * head_dim]
  Tensor bv;               // 可选 [n_kv_heads * head_dim]
  QuantizedTensor wo;      // [dim, n_heads * head_dim]
  Tensor ffn_norm;         // [dim]
  QuantizedTensor w_gate;  // [hidden_dim, dim]
  QuantizedTensor w_up;    // [hidden_dim, dim]
  QuantizedTensor w_down;  // [dim, hidden_dim]
};

// 一个已加载的模型的全部内容。
struct MiniLlamaModel {
  ModelConfig config;
  bool loaded = false;
  std::string load_error;

  Tensor token_embedding;            // [vocab_size, dim] 永远是 F32
  std::vector<LayerWeights> layers;  // [n_layers]
  Tensor final_norm;                 // [dim]
  QuantizedTensor lm_head;           // [vocab_size, dim]
};

// 按当前格式，模型权重的实际字节数。
size_t ModelWeightBytes(const MiniLlamaModel& model);

// 同样的权重若全部存成 F32 会占的字节数。
size_t ModelWeightBytesF32(const MiniLlamaModel& model);

}  // namespace dqm
