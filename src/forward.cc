#include "dqm_llama/forward.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "dqm_llama/kv_cache.h"
#include "dqm_llama/ops.h"
#include "dqm_llama/tensor.h"

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

// 线性投影。权重是 QuantizedTensor —— 量化路径已砍，只会走 kF32 状态。
Tensor ForwardLinear(const MiniLlamaModel& model, const std::string& name,
                     const Tensor& x, const QuantizedTensor& weight) {
  return Linear(x, weight);
}

// q / k / v 三个投影一次出。
void ForwardQkvProjection(const MiniLlamaModel& model,
                          const std::string& layer_prefix, const Tensor& h,
                          const LayerWeights& lw, Tensor& q_flat,
                          Tensor& k_flat, Tensor& v_flat) {
  const std::string q_name = layer_prefix + "wq";
  const std::string k_name = layer_prefix + "wk";
  const std::string v_name = layer_prefix + "wv";

  q_flat = AddOptionalBias(ForwardLinear(model, q_name, h, lw.wq), lw.bq, "forward_layer q");
  k_flat = AddOptionalBias(ForwardLinear(model, k_name, h, lw.wk), lw.bk, "forward_layer k");
  v_flat = AddOptionalBias(ForwardLinear(model, v_name, h, lw.wv), lw.bv, "forward_layer v");
}

// ---------------------------------------------------------------------------
// 基础算子包装
// ---------------------------------------------------------------------------

Tensor ForwardRmsNorm(const MiniLlamaModel& model, const Tensor& x,
                      const Tensor& weight, float eps) {
  return RmsNorm(x, weight, eps);
}

Tensor ForwardSwiGlu(const MiniLlamaModel& model, const Tensor& gate,
                     const Tensor& up) {
  return SwiGlu(gate, up);
}

Tensor ForwardAdd(const MiniLlamaModel& model, const Tensor& a,
                  const Tensor& b) {
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
  return Softmax(x);
}

void ForwardRope(const MiniLlamaModel& model, Tensor& q, Tensor& k, int pos,
                 float theta, RopeType rope_type) {
  Rope(q, k, pos, theta, rope_type);
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
                        KvCache& kv_cache, int n_heads, int n_kv_heads,
                        int head_dim) {
  //先把当前kv存一下
  kv_cache.Write(layer, pos, k, v);
  //构建结果，结果会是一个权重
  Tensor atten_out({n_heads, head_dim}, 0.0f);
  //scale用来给数据降温
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // ⚠️ 本轮串行。各 head 互不依赖（只写 atten_out 自己那一片 [h*head_dim, +head_dim)），
  // 后续有了 thread_pool 可换成 ParallelFor(n_heads, ...)；详见 forward-spec.md §4「待定」。
  for (int h = 0; h < n_heads; ++h) {
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

  return atten_out;
  

}

// 单层的 FFN（SwiGLU）
Tensor FfnForward(const MiniLlamaModel& model, const std::string& layer_prefix,
                  const Tensor& h, const LayerWeights& lw) {
  // 两个并行分支：都从 h 放大到 hidden_dim
  const Tensor gate = ForwardLinear(model, layer_prefix + "w_gate", h, lw.w_gate);
  const Tensor up = ForwardLinear(model, layer_prefix + "w_up", h, lw.w_up);

  // SwiGLU 门控：silu(gate) 逐元素乘 up，再压回 dim
  return ForwardLinear(model, layer_prefix + "w_down", ForwardSwiGlu(model, gate, up), lw.w_down);
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
      AttentionForward(model, q, k, v, pos, layer, ctx.kv_cache, n_heads,
                       n_kv_heads, head_dim);

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

// final norm -> lm_head
Tensor ComputeLogits(const Tensor& x, const MiniLlamaModel& model,
                     const ModelConfig& c) {
  const Tensor normed = ForwardRmsNorm(model, x, model.final_norm, c.rms_norm_eps);
  const Tensor logits_flat = ForwardLinear(model, "lm_head", normed, model.lm_head);
  return logits_flat.ReshapeChecked({c.vocab_size}, "compute_logits");
}

}  // namespace

Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    int token) {
  ValidateForwardInputs(ctx, model, token);
  auto c = model.config;
  int n_layers = c.n_layers;
  Tensor t = EmbedToken(model, token);
  //逐层计算
  for (int i = 0; i < n_layers; i ++) {
    t = ForwardLayer(ctx, model, t, i, model.layers[i], c);
  }
  return ComputeLogits(t, model, c);
}

}  // namespace dqm
