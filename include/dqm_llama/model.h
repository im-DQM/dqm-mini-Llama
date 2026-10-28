#pragma once

#include <memory>
#include <string>
#include <vector>

#include "dqm_llama/cuda_runtime.h"
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

// 一份已经复制到 GPU 显存里的权重（按名字查表用）。
struct CudaUploadedWeight {
  std::string name;
  QuantType type = QuantType::kF32;
  bool linear = true;  // true = 会参与 Linear，false = norm / bias / embedding
  std::vector<int> shape;
  size_t block_count = 0;  // 量化权重才有
  CudaDeviceBuffer buffer;
};

// 一个模型的全部 GPU 侧权重 + 运行统计。
struct CudaModelWeights {
  int device_id = 0;
  size_t uploaded_weight_count = 0;
  size_t uploaded_bytes = 0;
  size_t linear_calls = 0;
  size_t activation_calls = 0;
  size_t attention_calls = 0;
  size_t attention_cpu_fallbacks = 0;
  size_t kv_cache_write_bytes = 0;
  size_t kv_cache_read_bytes = 0;
  size_t host_to_device_copies = 0;
  size_t device_to_host_copies = 0;
  size_t host_to_device_bytes = 0;
  size_t device_to_host_bytes = 0;
  std::vector<CudaUploadedWeight> weights;
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

  // 非空 = 走 CUDA 后端；CPU 构建 / 未上传时保持 nullptr。
  std::shared_ptr<CudaModelWeights> cuda_weights;
};

// 按当前格式，模型权重的实际字节数。
size_t ModelWeightBytes(const MiniLlamaModel& model);

// 同样的权重若全部存成 F32 会占的字节数。
size_t ModelWeightBytesF32(const MiniLlamaModel& model);

// 把 CUDA 支持的常驻权重复制到设备；CPU 权重原样保留。
void UploadModelWeightsToCuda(MiniLlamaModel& model, int device_id = 0);

void ClearModelCudaWeights(MiniLlamaModel& model);

bool ModelHasCudaWeights(const MiniLlamaModel& model);

size_t ModelCudaUploadedWeightCount(const MiniLlamaModel& model);

size_t ModelCudaUploadedF32WeightCount(const MiniLlamaModel& model);

size_t ModelCudaUploadedQ80WeightCount(const MiniLlamaModel& model);

size_t ModelCudaUploadedQ40WeightCount(const MiniLlamaModel& model);

size_t ModelCudaUploadedQ41WeightCount(const MiniLlamaModel& model);

size_t ModelCudaMemoryBytes(const MiniLlamaModel& model);

void ResetModelCudaRuntimeStats(MiniLlamaModel& model);

size_t ModelCudaLinearCalls(const MiniLlamaModel& model);

size_t ModelCudaActivationCalls(const MiniLlamaModel& model);

size_t ModelCudaAttentionCalls(const MiniLlamaModel& model);

size_t ModelCudaAttentionCpuFallbacks(const MiniLlamaModel& model);

size_t ModelCudaKvCacheWriteBytes(const MiniLlamaModel& model);

size_t ModelCudaKvCacheReadBytes(const MiniLlamaModel& model);

size_t ModelCudaHostToDeviceCopies(const MiniLlamaModel& model);

size_t ModelCudaDeviceToHostCopies(const MiniLlamaModel& model);

size_t ModelCudaHostToDeviceBytes(const MiniLlamaModel& model);

size_t ModelCudaDeviceToHostBytes(const MiniLlamaModel& model);

// 把所有 Linear 权重就地转成 Q8_0 / Q4_0；embedding / norm / bias 不动。
void QuantizeModelToQ80(MiniLlamaModel& model);

void QuantizeModelToQ40(MiniLlamaModel& model);

// 反向操作：把所有 Linear 权重就地展开成 F32（测量四档 matmul 用）。
void DequantizeModelToF32(MiniLlamaModel& model);

}  // namespace dqm
