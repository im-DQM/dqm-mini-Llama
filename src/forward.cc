#include "dqm_llama/forward.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dqm_llama/kv_cache.h"
#include "dqm_llama/cuda_attention.h"
#include "dqm_llama/cuda_matmul.h"
#include "dqm_llama/cuda_ops.h"
#include "dqm_llama/cuda_quant.h"
#include "dqm_llama/ops.h"
#include "dqm_llama/tensor.h"
#include "dqm_llama/thread_pool.h"

namespace dqm {

namespace {

// ---------------------------------------------------------------------------
// 入参校验
// ---------------------------------------------------------------------------

void ValidateForwardInputs(const MiniLlamaContext& ctx,
                           const MiniLlamaModel& model, int token) {
  const ModelConfig& c = model.config;

  // 模型自己不自洽 —— 用 runtime_error
  if (!model.loaded) {
    throw std::runtime_error("ForwardToken called with an unloaded model");
  }
  if (model.layers.size() != static_cast<size_t>(c.n_layers)) {
    throw std::runtime_error(
        "ForwardToken layer count does not match model config");
  }
  if (model.token_embedding.data.size() <
      static_cast<size_t>(c.vocab_size * c.dim)) {
    throw std::runtime_error(
        "ForwardToken token embedding tensor is smaller than config");
  }

  // 调用方传的值越界 —— 用 out_of_range
  if (token < 0 || token >= c.vocab_size) {
    throw std::out_of_range("ForwardToken token id out of range");
  }
  if (ctx.pos < 0 || ctx.pos >= c.max_seq_len) {
    throw std::out_of_range("ForwardToken position out of range");
  }
}

// 形状断言：把 Tensor::AssertShape 包一层，参数顺序对齐本文件的调用习惯。
void RequireShape(const Tensor& t, const std::vector<int>& expected,
                  const char* caller) {
  t.AssertShape(expected, caller);
}

// bias 为空张量时直接返回 x（tiny 模型没有 attention bias）。
Tensor AddOptionalBias(const Tensor& x, const Tensor& bias,
                       const char* caller) {
  if (bias.data.empty()) {
    return x;
  }
  if (x.num_dims() != 1 || bias.num_dims() != 1 || x.size() != bias.size()) {
    throw std::runtime_error(std::string(caller) + x.ShapeStringShort() + " not matched " + bias.ShapeStringShort());
  }
  Tensor t(x.shape, 0.0f);
  for (size_t i = 0; i < x.size(); i ++) {
    t.data[i] = x.data[i] + bias.data[i];
  }
  return t;
}

// ---------------------------------------------------------------------------
// CUDA 运行时统计 / 权重查表
// ---------------------------------------------------------------------------

int CudaDeviceId(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->device_id : 0;
}

const CudaUploadedWeight* FindCudaWeight(const MiniLlamaModel& model,
                                         const std::string& name,
                                         QuantType type) {
  if (!model.cuda_weights) {
    return nullptr;
  }
  for (const auto& weight : model.cuda_weights->weights) {
    if (weight.name == name && weight.type == type) {
      return &weight;
    }
  }
  return nullptr;
}

void RecordCudaLinearCopy(const MiniLlamaModel& model, const Tensor& x,
                          const Tensor& y) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->linear_calls += 1;
  model.cuda_weights->host_to_device_copies += 1;
  model.cuda_weights->device_to_host_copies += 1;
  model.cuda_weights->host_to_device_bytes += x.size() * sizeof(float);
  model.cuda_weights->device_to_host_bytes += y.size() * sizeof(float);
}

void RecordCudaLinearCall(const MiniLlamaModel& model) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->linear_calls += 1;
}

void RecordCudaHostToDeviceCopy(const MiniLlamaModel& model, size_t bytes) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->host_to_device_copies += 1;
  model.cuda_weights->host_to_device_bytes += bytes;
}

void RecordCudaDeviceToHostCopy(const MiniLlamaModel& model, size_t bytes) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->device_to_host_copies += 1;
  model.cuda_weights->device_to_host_bytes += bytes;
}

void RecordCudaActivationCall(const MiniLlamaModel& model, size_t calls = 1) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->activation_calls += calls;
}

void RecordCudaAttentionCall(const MiniLlamaModel& model) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->attention_calls += 1;
}

void RecordCudaAttentionCpuFallback(const MiniLlamaModel& model) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->attention_cpu_fallbacks += 1;
}

void RecordCudaKvCacheWrite(const MiniLlamaModel& model, size_t bytes) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->kv_cache_write_bytes += bytes;
}

void RecordCudaKvCacheRead(const MiniLlamaModel& model, size_t bytes) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->kv_cache_read_bytes += bytes;
}

bool CudaLinearSupported(const QuantizedTensor& weight) {
  return weight.type == QuantType::kF32 || weight.type == QuantType::kQ80 ||
         weight.type == QuantType::kQ40 || weight.type == QuantType::kQ41;
}

bool CudaLinearReady(const MiniLlamaModel& model, const std::string& name,
                     const QuantizedTensor& weight) {
  return model.cuda_weights && CudaLinearSupported(weight) &&
         FindCudaWeight(model, name, weight.type) != nullptr;
}

CudaTensor UploadCudaTensor(const MiniLlamaModel& model, const Tensor& x) {
  CudaTensor x_dev = CudaTensorFromHost(x, CudaDeviceId(model));
  RecordCudaHostToDeviceCopy(model, x.size() * sizeof(float));
  return x_dev;
}

Tensor DownloadCudaTensor(const MiniLlamaModel& model, const CudaTensor& x) {
  Tensor y = x.Download();
  RecordCudaDeviceToHostCopy(model, y.size() * sizeof(float));
  return y;
}

CudaTensor ForwardLinearDevice(const MiniLlamaModel& model,
                               const std::string& name, const CudaTensor& x,
                               const QuantizedTensor& weight) {
  if (!CudaLinearSupported(weight)) {
    throw std::runtime_error("CUDA forward Linear unsupported weight type: " +
                             name);
  }

  const CudaUploadedWeight* cuda_weight =
      FindCudaWeight(model, name, weight.type);
  if (cuda_weight == nullptr) {
    throw std::runtime_error("CUDA forward Linear missing uploaded weight: " +
                             name);
  }

  CudaTensor y;
  switch (weight.type) {
    case QuantType::kF32:
      y = CudaLinearDeviceInput(x, cuda_weight->buffer.data(),
                                cuda_weight->shape, CudaDeviceId(model));
      break;
    case QuantType::kQ80:
      y = CudaQ80LinearDeviceInput(x, cuda_weight->buffer.data(),
                                   cuda_weight->block_count, cuda_weight->shape,
                                   CudaDeviceId(model));
      break;
    case QuantType::kQ40:
      y = CudaQ40LinearDeviceInput(x, cuda_weight->buffer.data(),
                                   cuda_weight->block_count, cuda_weight->shape,
                                   CudaDeviceId(model));
      break;
    case QuantType::kQ41:
      y = CudaQ41LinearDeviceInput(x, cuda_weight->buffer.data(),
                                   cuda_weight->block_count, cuda_weight->shape,
                                   CudaDeviceId(model));
      break;
  }
  RecordCudaLinearCall(model);
  return y;
}

// 线性投影。模型权重已上传时走 CUDA，否则走 CPU 内核。
Tensor ForwardLinear(const MiniLlamaModel& model, const std::string& name,
                     const Tensor& x, const QuantizedTensor& weight) {
  if (!model.cuda_weights || !CudaLinearSupported(weight)) {
    return Linear(x, weight);
  }

  const CudaUploadedWeight* cuda_weight =
      FindCudaWeight(model, name, weight.type);
  if (cuda_weight == nullptr) {
    throw std::runtime_error("CUDA forward Linear missing uploaded weight: " +
                             name);
  }
  Tensor y;
  switch (weight.type) {
    case QuantType::kF32:
      y = CudaLinearDeviceWeight(x, cuda_weight->buffer.data(),
                                 cuda_weight->shape, nullptr,
                                 model.cuda_weights->device_id);
      break;
    case QuantType::kQ80:
      y = CudaQ80LinearDeviceWeight(
          x, cuda_weight->buffer.data(), cuda_weight->block_count,
          cuda_weight->shape, nullptr, model.cuda_weights->device_id);
      break;
    case QuantType::kQ40:
      y = CudaQ40LinearDeviceWeight(
          x, cuda_weight->buffer.data(), cuda_weight->block_count,
          cuda_weight->shape, nullptr, model.cuda_weights->device_id);
      break;
    case QuantType::kQ41:
      y = CudaQ41LinearDeviceWeight(
          x, cuda_weight->buffer.data(), cuda_weight->block_count,
          cuda_weight->shape, nullptr, model.cuda_weights->device_id);
      break;
  }
  RecordCudaLinearCopy(model, x, y);
  return y;
}

// q / k / v 三个投影一次出。
void ForwardQkvProjection(const MiniLlamaModel& model,
                          const std::string& layer_prefix, const Tensor& h,
                          const LayerWeights& lw, Tensor& q_flat,
                          Tensor& k_flat, Tensor& v_flat) {
  const std::string q_name = layer_prefix + "wq";
  const std::string k_name = layer_prefix + "wk";
  const std::string v_name = layer_prefix + "wv";

  // 三个线性都已在 GPU 上时，把输入搬一次、三个投影连着算、再各搬回来。
  if (CudaLinearReady(model, q_name, lw.wq) &&
      CudaLinearReady(model, k_name, lw.wk) &&
      CudaLinearReady(model, v_name, lw.wv)) {
    CudaTensor h_dev = UploadCudaTensor(model, h);
    CudaTensor q_dev = ForwardLinearDevice(model, q_name, h_dev, lw.wq);
    CudaTensor k_dev = ForwardLinearDevice(model, k_name, h_dev, lw.wk);
    CudaTensor v_dev = ForwardLinearDevice(model, v_name, h_dev, lw.wv);
    q_flat = AddOptionalBias(DownloadCudaTensor(model, q_dev), lw.bq, "forward_layer q");
    k_flat = AddOptionalBias(DownloadCudaTensor(model, k_dev), lw.bk, "forward_layer k");
    v_flat = AddOptionalBias(DownloadCudaTensor(model, v_dev), lw.bv, "forward_layer v");
    return;
  }

  q_flat = AddOptionalBias(ForwardLinear(model, q_name, h, lw.wq), lw.bq, "forward_layer q");
  k_flat = AddOptionalBias(ForwardLinear(model, k_name, h, lw.wk), lw.bk, "forward_layer k");
  v_flat = AddOptionalBias(ForwardLinear(model, v_name, h, lw.wv), lw.bv, "forward_layer v");
}

// ---------------------------------------------------------------------------
// 基础算子包装
// ---------------------------------------------------------------------------

Tensor ForwardRmsNorm(const MiniLlamaModel& model, const Tensor& x,
                      const Tensor& weight, float eps) {
  if (!model.cuda_weights) {
    return RmsNorm(x, weight, eps);
  }
  Tensor y = CudaRmsNorm(x, weight, eps, CudaDeviceId(model));
  RecordCudaActivationCall(model);
  return y;
}

CudaTensor ForwardRmsNormDevice(const MiniLlamaModel& model,
                                const std::string& name, const CudaTensor& x,
                                const Tensor& weight, float eps) {
  const CudaUploadedWeight* cuda_weight =
      FindCudaWeight(model, name, QuantType::kF32);
  if (cuda_weight == nullptr || cuda_weight->linear) {
    throw std::runtime_error("CUDA forward RMSNorm missing uploaded weight: " +
                             name);
  }
  if (cuda_weight->shape != weight.shape) {
    throw std::runtime_error(
        "CUDA forward RMSNorm uploaded weight shape mismatch: " + name);
  }
  CudaTensor y = CudaRmsNormDeviceWeight(
      x, cuda_weight->buffer.data(), cuda_weight->shape, eps,
      CudaDeviceId(model));
  RecordCudaActivationCall(model);
  return y;
}

Tensor ForwardSwiGlu(const MiniLlamaModel& model, const Tensor& gate,
                     const Tensor& up) {
  if (!model.cuda_weights) {
    return SwiGlu(gate, up);
  }
  Tensor silu_gate = CudaSilu(gate, CudaDeviceId(model));
  Tensor y = CudaElementwiseMul(silu_gate, up, CudaDeviceId(model));
  RecordCudaActivationCall(model, 2);
  return y;
}

CudaTensor ForwardAddDevice(const MiniLlamaModel& model, const CudaTensor& a,
                            const CudaTensor& b) {
  CudaTensor y = CudaElementwiseAddDeviceInput(a, b, CudaDeviceId(model));
  RecordCudaActivationCall(model);
  return y;
}

Tensor ForwardAdd(const MiniLlamaModel& model, const Tensor& a,
                  const Tensor& b) {
  if (model.cuda_weights) {
    Tensor y = CudaElementwiseAdd(a, b, CudaDeviceId(model));
    RecordCudaActivationCall(model);
    return y;
  }
  if (a.num_dims() != b.num_dims()) {
    throw std::runtime_error("ForwardAdd: a_shape not matched b_shape");
  }
  Tensor t(a.shape, 0.0f);
  for (size_t i = 0; i < a.size(); i ++) {
    t.data[i] = a.data[i] + b.data[i];
  }
  return t;
}

Tensor ForwardSoftmax(const MiniLlamaModel& model, const Tensor& x) {
  if (!model.cuda_weights) {
    return Softmax(x);
  }
  Tensor y = CudaSoftmax(x, CudaDeviceId(model));
  RecordCudaActivationCall(model);
  return y;
}

void ForwardRope(const MiniLlamaModel& model, Tensor& q, Tensor& k, int pos,
                 float theta, RopeType rope_type) {
  if (!model.cuda_weights) {
    Rope(q, k, pos, theta, rope_type);
    return;
  }
  CudaRope(q, k, pos, theta, rope_type, CudaDeviceId(model));
  RecordCudaActivationCall(model);
}

void ForwardRopeDevice(const MiniLlamaModel& model, CudaTensor& q,
                       CudaTensor& k, int n_heads, int n_kv_heads, int head_dim,
                       int pos, float theta, RopeType rope_type) {
  CudaRopeDeviceInput(q, k, n_heads, n_kv_heads, head_dim, pos, theta, rope_type,
                      CudaDeviceId(model));
  RecordCudaActivationCall(model);
}

// ---------------------------------------------------------------------------
// 前向的主干
// ---------------------------------------------------------------------------

// 词嵌入查表：token id -> x[dim]
Tensor EmbedToken(const MiniLlamaModel& model, int token_id) {
  int dim = model.config.dim;
  Tensor t({dim}, 0.0f);
  for (int i = 0; i < dim; i ++) {
    t.data[i] = model.token_embedding[token_id * dim + i];
  }
  return t;
}

// 设备侧查表：embedding 整张量已常驻显存，只取一行。
CudaTensor EmbedTokenDevice(const MiniLlamaModel& model, int token_id) {
  const CudaUploadedWeight* cuda_weight =
      FindCudaWeight(model, "token_embedding", QuantType::kF32);
  if (cuda_weight == nullptr || cuda_weight->linear) {
    throw std::runtime_error(
        "CUDA forward embedding missing uploaded weight: token_embedding");
  }
  if (cuda_weight->shape != model.token_embedding.shape) {
    throw std::runtime_error(
        "CUDA forward embedding uploaded weight shape mismatch");
  }
  return CudaEmbeddingLookupDeviceWeight(cuda_weight->buffer.data(),
                                         cuda_weight->shape, token_id,
                                         CudaDeviceId(model));
}

// GQA：给定一个 query head，返回它对应的 kv head
int MapQHeadToKvHead(int q_head, int n_heads, int n_kv_heads) {
  //q编号 / q的头数 / kv的头数（1个kv头有多个q头）
  return q_head / (n_heads / n_kv_heads);
}

// 单层的 attention：写 KV cache、从 cache 读全历史、加权求和
//   q: [n_heads, head_dim] / k: [n_kv_heads, head_dim] / v 同
//   返回: [n_heads, head_dim]
Tensor AttentionForward(const MiniLlamaModel& model, const Tensor& q,
                        const Tensor& k, const Tensor& v, int pos, int layer,
                        KvCache& kv_cache, CudaKvCache* cuda_kv_cache,
                        int n_heads, int n_kv_heads, int head_dim) {
  //先把当前kv存一下
  kv_cache.Write(layer, pos, k, v);

  // CUDA 后端：GPU KV cache 有两份（CPU 那份也照写，供回退用）。
  if (cuda_kv_cache != nullptr && !cuda_kv_cache->empty()) {
    cuda_kv_cache->Write(layer, pos, k, v);
    RecordCudaKvCacheWrite(model, (k.size() + v.size()) * sizeof(float));
    Tensor attn_out =
        CudaAttentionDecode(q, *cuda_kv_cache, layer, pos, n_heads, n_kv_heads,
                            head_dim, CudaDeviceId(model));
    RecordCudaAttentionCall(model);
    RecordCudaKvCacheRead(
        model, static_cast<size_t>(pos + 1) * static_cast<size_t>(n_heads) *
                   static_cast<size_t>(head_dim) * sizeof(float) * 2);
    RecordCudaHostToDeviceCopy(model, q.size() * sizeof(float));
    RecordCudaDeviceToHostCopy(model, attn_out.size() * sizeof(float));
    return attn_out;
  }
  if (model.cuda_weights) {
    RecordCudaAttentionCpuFallback(model);
  }

  //构建结果，结果会是一个权重
  Tensor atten_out({n_heads, head_dim}, 0.0f);
  //scale用来给数据降温
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // 各 head 互不依赖（只写 atten_out 自己那一片 [h*head_dim, +head_dim)），可并行。
  auto compute_heads = [&](int begin, int end) {
    for (int h = begin; h < end; ++h) {
      // GQA：多个 q head 可能共享同一个 kv head，读的是同一份 K/V
      const int kv_head = MapQHeadToKvHead(h, n_heads, n_kv_heads);

      // ① 打分：这个 head 的 q 和 0..pos 每个位置的 K 做点积
      Tensor scores({pos + 1}, 0.0f);
      for (int t = 0; t <= pos; ++t) {
        const float* k_ptr = kv_cache.KeyPtr(layer, t, kv_head);
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          dot += q[h * head_dim + d] * k_ptr[d];
        }
        scores.data[t] = dot * scale;
      }

      // ② 归一化：分数 -> 和为 1 的权重
      const Tensor probs = ForwardSoftmax(model, scores);

      // ③ 加权求和：按关注度把 0..pos 的 V 混合成一个 [head_dim]
      for (int d = 0; d < head_dim; ++d) {
        float attn = 0.0f;
        for (int t = 0; t <= pos; ++t) {
          const float* v_ptr = kv_cache.ValuePtr(layer, t, kv_head);
          attn += probs[t] * v_ptr[d];
        }
        atten_out[h * head_dim + d] = attn;
      }
    }
  };

  // CUDA 后端时 compute_heads 内部会调 Cuda*（串行，避免多线程同时打 CUDA）；
  // 纯 CPU 时按 head 并行。
  if (model.cuda_weights) {
    compute_heads(0, n_heads);
  } else {
    ParallelFor(n_heads, compute_heads);
  }

  return atten_out;
}

// 设备侧 attention：q/k/v 全在显存里，结果也留在显存。
CudaTensor AttentionForwardDevice(const MiniLlamaModel& model,
                                  const CudaTensor& q, const CudaTensor& k,
                                  const CudaTensor& v, int pos, int layer,
                                  CudaKvCache& cuda_kv_cache, int n_heads,
                                  int n_kv_heads, int head_dim) {
  cuda_kv_cache.WriteDevice(layer, pos, k, v);
  RecordCudaKvCacheWrite(model, (k.size() + v.size()) * sizeof(float));
  CudaTensor attn_out =
      CudaAttentionDecodeDeviceInput(q, cuda_kv_cache, layer, pos, n_heads,
                                     n_kv_heads, head_dim, CudaDeviceId(model));
  RecordCudaAttentionCall(model);
  RecordCudaKvCacheRead(
      model, static_cast<size_t>(pos + 1) * static_cast<size_t>(n_heads) *
                 static_cast<size_t>(head_dim) * sizeof(float) * 2);
  return attn_out;
}

// 单层的 FFN（SwiGLU）
Tensor FfnForward(const MiniLlamaModel& model, const std::string& layer_prefix,
                  const Tensor& h, const LayerWeights& lw) {
  const std::string gate_name = layer_prefix + "w_gate";
  const std::string up_name = layer_prefix + "w_up";
  const std::string down_name = layer_prefix + "w_down";

  // 三个线性都在 GPU 上时：h 搬一次上去，gate/up/silu/mul/down 全在显存里连着算。
  if (CudaLinearReady(model, gate_name, lw.w_gate) &&
      CudaLinearReady(model, up_name, lw.w_up) &&
      CudaLinearReady(model, down_name, lw.w_down)) {
    CudaTensor h_dev = UploadCudaTensor(model, h);
    CudaTensor gate_dev = ForwardLinearDevice(model, gate_name, h_dev, lw.w_gate);
    CudaTensor up_dev = ForwardLinearDevice(model, up_name, h_dev, lw.w_up);
    CudaTensor silu_gate_dev =
        CudaSiluDeviceInput(gate_dev, CudaDeviceId(model));
    CudaTensor ff_dev = CudaElementwiseMulDeviceInput(silu_gate_dev, up_dev,
                                                      CudaDeviceId(model));
    RecordCudaActivationCall(model, 2);
    CudaTensor out_dev =
        ForwardLinearDevice(model, down_name, ff_dev, lw.w_down);
    return DownloadCudaTensor(model, out_dev);
  }

  // 两个并行分支：都从 h 放大到 hidden_dim
  const Tensor gate = ForwardLinear(model, gate_name, h, lw.w_gate);
  const Tensor up = ForwardLinear(model, up_name, h, lw.w_up);

  // SwiGLU 门控：silu(gate) 逐元素乘 up，再压回 dim
  return ForwardLinear(model, down_name, ForwardSwiGlu(model, gate, up), lw.w_down);
}

// 全设备 FFN：h 已在显存，整段不落回主机。
CudaTensor FfnForwardDevice(const MiniLlamaModel& model,
                            const std::string& layer_prefix,
                            const CudaTensor& h, const LayerWeights& lw) {
  CudaTensor gate =
      ForwardLinearDevice(model, layer_prefix + "w_gate", h, lw.w_gate);
  CudaTensor up = ForwardLinearDevice(model, layer_prefix + "w_up", h, lw.w_up);
  CudaTensor silu_gate = CudaSiluDeviceInput(gate, CudaDeviceId(model));
  CudaTensor ff =
      CudaElementwiseMulDeviceInput(silu_gate, up, CudaDeviceId(model));
  RecordCudaActivationCall(model, 2);
  return ForwardLinearDevice(model, layer_prefix + "w_down", ff, lw.w_down);
}

// 单层：attention 子层 + FFN 子层，各带一个残差
Tensor ForwardLayer(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    const Tensor& x, int layer, const LayerWeights& lw,
                    const ModelConfig& c) {
  const int dim = c.dim;
  const int n_heads = c.n_heads;
  const int n_kv_heads = c.n_kv_heads;
  const int head_dim = c.head_dim;
  const int pos = ctx.pos;
  const std::string layer_prefix = "layers." + std::to_string(layer) + ".";

  // ---- attention 子层 ----
  const Tensor h = ForwardRmsNorm(model, x, lw.attention_norm, c.rms_norm_eps);

  Tensor q_flat;
  Tensor k_flat;
  Tensor v_flat;
  //根据权重文件，在张量中提取出对应的q,k,v(此时flat维度为1)
  ForwardQkvProjection(model, layer_prefix, h, lw, q_flat, k_flat, v_flat);

  //把q,k,v升维，仅改变shape，不改变任何数据和顺序
  Tensor q = q_flat.ReshapeChecked({n_heads, head_dim}, "forward_layer q");
  Tensor k = k_flat.ReshapeChecked({n_kv_heads, head_dim}, "forward_layer k");
  Tensor v = v_flat.ReshapeChecked({n_kv_heads, head_dim}, "forward_layer v");

  //把位置信息嵌入q和k中
  ForwardRope(model, q, k, pos, c.rope_theta, c.rope_type);

  //AttentionForward把各个token的信息按照权重(q和k的匹配程度)融合到这个token里(有多个head，这些head之间信息尚未交互)
  const Tensor attn_out =
      AttentionForward(model, q, k, v, pos, layer, ctx.kv_cache,
                       model.cuda_weights ? &ctx.cuda_kv_cache : nullptr,
                       n_heads, n_kv_heads, head_dim);

  //把attn_out窄化，不改变任何数据和顺序，只改变shape(但仍然是2D)
  const Tensor attn_out_flat = attn_out.ReshapeChecked(
      {1, n_heads * head_dim}, "forward_layer attn_out");
  //通过wo，让一个token中的各个head实现信息交互（利用权重文件加权求和）
  Tensor attn_proj =
      ForwardLinear(model, layer_prefix + "wo", attn_out_flat, lw.wo);
  //断言
  RequireShape(attn_proj, {1, dim}, "forward_layer attn_proj");
  //降维，可以去残差链接计算
  Tensor attn_proj_1d =
      attn_proj.ReshapeChecked({dim}, "forward_layer attn_proj 1d");

  //残差连接
  const Tensor x_attn = ForwardAdd(model, x, attn_proj_1d);

  // ---- FFN 子层 ----
  const Tensor h2 = ForwardRmsNorm(model, x_attn, lw.ffn_norm, c.rms_norm_eps);
  const Tensor ff = FfnForward(model, layer_prefix, h2, lw);
  RequireShape(ff, {dim}, "forward_layer ffn");

  return ForwardAdd(model, x_attn, ff);
}

// bias 的设备版：bias 整张量已常驻显存，用 ElementwiseAdd(权重) 加。
CudaTensor AddOptionalBiasDevice(const MiniLlamaModel& model,
                                 const std::string& name, CudaTensor x,
                                 const Tensor& bias) {
  if (bias.data.empty()) {
    return x;
  }
  const CudaUploadedWeight* cuda_weight =
      FindCudaWeight(model, name, QuantType::kF32);
  if (cuda_weight == nullptr || cuda_weight->linear) {
    throw std::runtime_error("CUDA forward bias missing uploaded weight: " +
                             name);
  }
  if (cuda_weight->shape != bias.shape) {
    throw std::runtime_error(
        "CUDA forward bias uploaded weight shape mismatch: " + name);
  }
  CudaTensor y = CudaElementwiseAddDeviceWeight(
      x, cuda_weight->buffer.data(), cuda_weight->shape, CudaDeviceId(model));
  RecordCudaActivationCall(model);
  return y;
}

// 全设备 q/k/v：h 已在显存，三个投影 + bias 都不落回主机。
void ForwardQkvProjectionDevice(const MiniLlamaModel& model,
                                const std::string& layer_prefix,
                                const CudaTensor& h, const LayerWeights& lw,
                                CudaTensor& q, CudaTensor& k, CudaTensor& v) {
  q = ForwardLinearDevice(model, layer_prefix + "wq", h, lw.wq);
  k = ForwardLinearDevice(model, layer_prefix + "wk", h, lw.wk);
  v = ForwardLinearDevice(model, layer_prefix + "wv", h, lw.wv);
  q = AddOptionalBiasDevice(model, layer_prefix + "bq", std::move(q), lw.bq);
  k = AddOptionalBiasDevice(model, layer_prefix + "bk", std::move(k), lw.bk);
  v = AddOptionalBiasDevice(model, layer_prefix + "bv", std::move(v), lw.bv);
}

// 全设备单层：整层只在显存里流动，不产生 H2D/D2H。
CudaTensor ForwardLayerDevice(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                              const CudaTensor& x, int layer,
                              const LayerWeights& lw, const ModelConfig& c) {
  const int dim = c.dim;
  const int n_heads = c.n_heads;
  const int n_kv_heads = c.n_kv_heads;
  const int head_dim = c.head_dim;
  const int pos = ctx.pos;
  const std::string layer_prefix = "layers." + std::to_string(layer) + ".";

  CudaTensor h = ForwardRmsNormDevice(model, layer_prefix + "attention_norm", x,
                                      lw.attention_norm, c.rms_norm_eps);

  CudaTensor q;
  CudaTensor k;
  CudaTensor v;
  ForwardQkvProjectionDevice(model, layer_prefix, h, lw, q, k, v);
  ForwardRopeDevice(model, q, k, n_heads, n_kv_heads, head_dim, pos,
                    c.rope_theta, c.rope_type);

  CudaTensor attn_out =
      AttentionForwardDevice(model, q, k, v, pos, layer, ctx.cuda_kv_cache,
                             n_heads, n_kv_heads, head_dim);

  CudaTensor attn_proj =
      ForwardLinearDevice(model, layer_prefix + "wo", attn_out, lw.wo);
  if (attn_proj.size() != static_cast<size_t>(dim)) {
    throw std::runtime_error(
        "forward_layer_device: attention projection shape mismatch");
  }
  CudaTensor x_attn = ForwardAddDevice(model, x, attn_proj);

  CudaTensor h2 = ForwardRmsNormDevice(model, layer_prefix + "ffn_norm", x_attn,
                                       lw.ffn_norm, c.rms_norm_eps);
  CudaTensor ff = FfnForwardDevice(model, layer_prefix, h2, lw);
  if (ff.size() != static_cast<size_t>(dim)) {
    throw std::runtime_error("forward_layer_device: FFN output shape mismatch");
  }
  return ForwardAddDevice(model, x_attn, ff);
}

// final norm -> lm_head
Tensor ComputeLogits(const Tensor& x, const MiniLlamaModel& model,
                     const ModelConfig& c) {
  const Tensor normed = ForwardRmsNorm(model, x, model.final_norm, c.rms_norm_eps);
  const Tensor logits_flat = ForwardLinear(model, "lm_head", normed, model.lm_head);
  return logits_flat.ReshapeChecked({c.vocab_size}, "compute_logits");
}

// 全设备 logits：最后只在 logits 这一步落回主机（sampler 还在 CPU）。
Tensor ComputeLogitsDevice(const CudaTensor& x, const MiniLlamaModel& model,
                           const ModelConfig& c) {
  CudaTensor normed = ForwardRmsNormDevice(model, "final_norm", x,
                                           model.final_norm, c.rms_norm_eps);
  CudaTensor logits_dev =
      ForwardLinearDevice(model, "lm_head", normed, model.lm_head);
  Tensor logits = DownloadCudaTensor(model, logits_dev);
  return logits.ReshapeChecked({c.vocab_size}, "compute_logits_device");
}

}  // namespace

Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    int token) {
  ValidateForwardInputs(ctx, model, token);
  auto c = model.config;
  int n_layers = c.n_layers;

  // CUDA 后端：整条解码链留在显存里，只有 logits 回主机。
  if (model.cuda_weights) {
    if (ctx.cuda_kv_cache.empty()) {
      ctx.cuda_kv_cache.Reset(c.n_layers, c.max_seq_len, c.n_kv_heads,
                              c.head_dim, model.cuda_weights->device_id);
    }
    CudaTensor x_dev = EmbedTokenDevice(model, token);
    for (int i = 0; i < n_layers; i ++) {
      x_dev = ForwardLayerDevice(ctx, model, x_dev, i, model.layers[i], c);
    }
    return ComputeLogitsDevice(x_dev, model, c);
  }

  Tensor t = EmbedToken(model, token);
  //逐层计算
  for (int i = 0; i < n_layers; i ++) {
    t = ForwardLayer(ctx, model, t, i, model.layers[i], c);
  }
  return ComputeLogits(t, model, c);
}

}  // namespace dqm
