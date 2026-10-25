// gguf_loader.cc —— 把 GGUF 的解析产物装配成 MiniLlamaModel。

#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "dqm_llama/gguf_loader.h"
#include "dqm_llama/gguf.h"
#include "dqm_llama/model.h"
#include "dqm_llama/quant.h"

namespace dqm {

// ---------------------------------------------------------------------------
// 名字映射：GGUF（llama.cpp 风格）-> 本项目命名
// ---------------------------------------------------------------------------

static std::string MapTensorName(const std::string& gguf_name) {
  if (gguf_name == "token_embd.weight") return "token_embedding";
  if (gguf_name == "output_norm.weight") return "final_norm";
  if (gguf_name == "output.weight") return "lm_head";

  if (gguf_name.rfind("blk.", 0) != 0) return "";

  size_t dot1 = gguf_name.find('.', 4);
  if (dot1 == std::string::npos) return "";
  std::string layer_str = gguf_name.substr(4, dot1 - 4);

  size_t dot2 = gguf_name.find('.', dot1 + 1);
  if (dot2 == std::string::npos) return "";
  std::string op = gguf_name.substr(dot1 + 1, dot2 - dot1 - 1);
  std::string suffix = gguf_name.substr(dot2 + 1);

  std::string mapped;
  if (suffix == "weight") {
    if (op == "attn_norm") mapped = "attention_norm";
    else if (op == "attn_q") mapped = "wq";
    else if (op == "attn_k") mapped = "wk";
    else if (op == "attn_v") mapped = "wv";
    else if (op == "attn_output") mapped = "wo";
    else if (op == "ffn_norm") mapped = "ffn_norm";
    else if (op == "ffn_gate") mapped = "w_gate";
    else if (op == "ffn_up") mapped = "w_up";
    else if (op == "ffn_down") mapped = "w_down";
    else return "";
  } else if (suffix == "bias") {
    if (op == "attn_q") mapped = "bq";
    else if (op == "attn_k") mapped = "bk";
    else if (op == "attn_v") mapped = "bv";
    else return "";
  } else {
    return "";
  }

  return "layers." + layer_str + "." + mapped;
}

// ---------------------------------------------------------------------------
// shape 映射：2D 张量对调两维（GGUF 的 [in,out] -> 本项目的 [out,in]）
// ---------------------------------------------------------------------------

static std::vector<int> MapShape(const std::vector<uint64_t>& gguf_shape) {
  std::vector<int> shape;
  if (gguf_shape.size() == 2) {
    shape.push_back(static_cast<int>(gguf_shape[1]));
    shape.push_back(static_cast<int>(gguf_shape[0]));
    return shape;
  }
  for (uint64_t dim : gguf_shape) shape.push_back(static_cast<int>(dim));
  return shape;
}

// ---------------------------------------------------------------------------
// 把不透明字节按 GGML 类型装箱成 QuantizedTensor
// ---------------------------------------------------------------------------

static QuantizedTensor LoadQuantizedTensor(const std::vector<uint8_t>& raw,
                                           const std::vector<int>& shape,
                                           int ggml_type) {
  QuantizedTensor q;
  q.shape = shape;

  uint64_t num_elements = 1;
  for (int dim : shape) num_elements *= static_cast<uint64_t>(dim);

  switch (ggml_type) {
    case kGgmlTypeF32: {
      q.type = QuantType::kF32;
      size_t expected = static_cast<size_t>(num_elements) * sizeof(float);
      if (raw.size() != expected) {
        throw std::runtime_error("F32 tensor size mismatch");
      }
      q.f32_data.resize(static_cast<size_t>(num_elements));
      std::memcpy(q.f32_data.data(), raw.data(), expected);
      return q;
    }
    case kGgmlTypeQ80: {
      q.type = QuantType::kQ80;
      size_t n_blocks = static_cast<size_t>(
          (num_elements + static_cast<uint64_t>(kQ80BlockSize) - 1) /
          static_cast<uint64_t>(kQ80BlockSize));
      size_t expected = n_blocks * sizeof(BlockQ80);
      if (raw.size() != expected) {
        throw std::runtime_error("Q8_0 tensor size mismatch");
      }
      q.q8_0_data.resize(n_blocks);
      std::memcpy(q.q8_0_data.data(), raw.data(), expected);
      return q;
    }
    case kGgmlTypeQ40: {
      q.type = QuantType::kQ40;
      size_t n_blocks = static_cast<size_t>(
          (num_elements + static_cast<uint64_t>(kQ40BlockSize) - 1) /
          static_cast<uint64_t>(kQ40BlockSize));
      size_t expected = n_blocks * sizeof(BlockQ40);
      if (raw.size() != expected) {
        throw std::runtime_error("Q4_0 tensor size mismatch");
      }
      q.q4_0_data.resize(n_blocks);
      std::memcpy(q.q4_0_data.data(), raw.data(), expected);
      return q;
    }
    case kGgmlTypeQ41: {
      q.type = QuantType::kQ41;
      size_t n_blocks = static_cast<size_t>(
          (num_elements + static_cast<uint64_t>(kQ41BlockSize) - 1) /
          static_cast<uint64_t>(kQ41BlockSize));
      size_t expected = n_blocks * sizeof(BlockQ41);
      if (raw.size() != expected) {
        throw std::runtime_error("Q4_1 tensor size mismatch");
      }
      q.q4_1_data.resize(n_blocks);
      std::memcpy(q.q4_1_data.data(), raw.data(), expected);
      return q;
    }
    default:
      throw std::runtime_error("unsupported GGML tensor type");
  }
}

// ---------------------------------------------------------------------------
// 形状校验
// ---------------------------------------------------------------------------

static void RequireShape(const Tensor& tensor, const std::vector<int>& expected,
                         const std::string& name) {
  tensor.AssertShape(expected, name.c_str());
}

static void RequireShape(const QuantizedTensor& tensor,
                         const std::vector<int>& expected,
                         const std::string& name) {
  tensor.AssertShape(expected, name.c_str());
}

static void RequireOptionalShape(const Tensor& tensor,
                                 const std::vector<int>& expected,
                                 const std::string& name) {
  if (tensor.data.empty()) return;
  RequireShape(tensor, expected, name);
}

static void ValidateLoadedShapes(const MiniLlamaModel& model) {
  const ModelConfig& c = model.config;
  const int q_dim = c.n_heads * c.head_dim;
  const int kv_dim = c.n_kv_heads * c.head_dim;

  RequireShape(model.token_embedding, {c.vocab_size, c.dim},
               "GGUF token_embedding");
  RequireShape(model.final_norm, {c.dim}, "GGUF final_norm");
  RequireShape(model.lm_head, {c.vocab_size, c.dim}, "GGUF lm_head");

  if (static_cast<int>(model.layers.size()) != c.n_layers) {
    throw std::runtime_error("GGUF layer count mismatch");
  }

  for (int i = 0; i < c.n_layers; ++i) {
    const LayerWeights& lw = model.layers[static_cast<size_t>(i)];
    const std::string prefix = "GGUF layers." + std::to_string(i) + ".";
    RequireShape(lw.attention_norm, {c.dim}, prefix + "attention_norm");
    RequireShape(lw.wq, {q_dim, c.dim}, prefix + "wq");
    RequireShape(lw.wk, {kv_dim, c.dim}, prefix + "wk");
    RequireShape(lw.wv, {kv_dim, c.dim}, prefix + "wv");
    RequireOptionalShape(lw.bq, {q_dim}, prefix + "bq");
    RequireOptionalShape(lw.bk, {kv_dim}, prefix + "bk");
    RequireOptionalShape(lw.bv, {kv_dim}, prefix + "bv");
    RequireShape(lw.wo, {c.dim, q_dim}, prefix + "wo");
    RequireShape(lw.ffn_norm, {c.dim}, prefix + "ffn_norm");
    RequireShape(lw.w_gate, {c.hidden_dim, c.dim}, prefix + "w_gate");
    RequireShape(lw.w_up, {c.hidden_dim, c.dim}, prefix + "w_up");
    RequireShape(lw.w_down, {c.dim, c.hidden_dim}, prefix + "w_down");
  }
}

// ---------------------------------------------------------------------------
// 从 GGUF metadata 组 ModelConfig
// ---------------------------------------------------------------------------

static ModelConfig BuildConfig(const std::string& path) {
  ModelConfig config;

  auto get_u32 = [&](const std::string& key, uint32_t fallback) -> uint32_t {
    int64_t value = 0;
    if (GgufGetMetadataInt(path, key, value)) {
      return static_cast<uint32_t>(value);
    }
    return fallback;
  };
  auto get_f32 = [&](const std::string& key, float fallback) -> float {
    double value = 0.0;
    if (GgufGetMetadataFloat(path, key, value)) {
      return static_cast<float>(value);
    }
    return fallback;
  };

  std::string arch;
  if (GgufGetMetadataString(path, "general.architecture", arch) &&
      arch == "qwen2") {
    config.rope_type = RopeType::kNeoX;
  }

  config.n_layers = static_cast<int>(
      get_u32("qwen2.block_count", get_u32("llama.block_count", 0)));
  config.dim = static_cast<int>(
      get_u32("qwen2.embedding_length", get_u32("llama.embedding_length", 0)));
  config.hidden_dim =
      static_cast<int>(get_u32("qwen2.feed_forward_length",
                               get_u32("llama.feed_forward_length", 0)));
  config.n_heads = static_cast<int>(get_u32(
      "qwen2.attention.head_count", get_u32("llama.attention.head_count", 0)));
  config.n_kv_heads =
      static_cast<int>(get_u32("qwen2.attention.head_count_kv",
                               get_u32("llama.attention.head_count_kv", 0)));
  config.max_seq_len = static_cast<int>(
      get_u32("qwen2.context_length", get_u32("llama.context_length", 0)));
  config.rope_theta = get_f32("qwen2.rope.freq_base",
                              get_f32("llama.rope.freq_base", 10000.0f));
  config.rms_norm_eps =
      get_f32("qwen2.attention.layer_norm_rms_epsilon",
              get_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f));

  std::vector<std::string> tokens;
  config.vocab_size =
      GgufGetMetadataStringArray(path, "tokenizer.ggml.tokens", tokens)
          ? static_cast<int>(tokens.size())
          : 0;

  if (config.n_layers <= 0 || config.dim <= 0 || config.hidden_dim <= 0 ||
      config.n_heads <= 0 || config.n_kv_heads <= 0 ||
      config.max_seq_len <= 0 || config.vocab_size <= 0) {
    throw std::runtime_error("incomplete GGUF model config");
  }
  if (config.dim % config.n_heads != 0) {
    throw std::runtime_error("GGUF dim not divisible by n_heads");
  }
  if (config.n_heads % config.n_kv_heads != 0) {
    throw std::runtime_error("GGUF n_heads not divisible by n_kv_heads");
  }

  config.head_dim = config.dim / config.n_heads;

  std::cout << "GGUF config: vocab=" << config.vocab_size << " dim=" << config.dim
            << " layers=" << config.n_layers << " heads=" << config.n_heads
            << "/" << config.n_kv_heads << " head_dim=" << config.head_dim
            << " hidden=" << config.hidden_dim
            << " max_seq=" << config.max_seq_len << "\n";

  return config;
}

// ---------------------------------------------------------------------------
// LoadGgufModel
// ---------------------------------------------------------------------------

MiniLlamaModel LoadGgufModel(const std::string& gguf_path) {
  MiniLlamaModel model;

  GgufReader reader;
  if (!reader.Load(gguf_path)) {
    model.load_error = reader.load_error;
    return model;
  }

  try {
    ModelConfig config = BuildConfig(gguf_path);

    // 建表：项目名 -> 该张量的目录条目
    std::map<std::string, const GgufTensorInfo*> tensor_map;
    for (const auto& info : reader.tensors) {
      std::string mapped = MapTensorName(info.name);
      if (!mapped.empty()) tensor_map[mapped] = &info;
    }

    auto load_weight = [&](const std::string& name) -> QuantizedTensor {
      auto it = tensor_map.find(name);
      if (it == tensor_map.end()) {
        throw std::runtime_error("Missing tensor: " + name);
      }
      const GgufTensorInfo* info = it->second;
      std::vector<uint8_t> raw =
          ReadGgufTensorRaw(gguf_path, *info, reader.data_offset);
      if (raw.empty() && info->type != kGgmlTypeF32) {
        throw std::runtime_error("Failed to read tensor data: " + name);
      }
      return LoadQuantizedTensor(raw, MapShape(info->shape),
                                 static_cast<int>(info->type));
    };

    auto load_f32 = [&](const std::string& name) -> Tensor {
      QuantizedTensor q = load_weight(name);
      switch (q.type) {
        case QuantType::kF32:
          return ToTensor(q);
        case QuantType::kQ80:
          return DequantizeFromQ80(q.q8_0_data, q.shape);
        case QuantType::kQ40:
          return DequantizeFromQ40(q.q4_0_data, q.shape);
        case QuantType::kQ41:
          return DequantizeFromQ41(q.q4_1_data, q.shape);
      }
      throw std::runtime_error("Cannot convert tensor to F32: " + name);
    };

    auto load_optional_f32 = [&](const std::string& name) -> Tensor {
      if (tensor_map.find(name) == tensor_map.end()) return Tensor{};
      return load_f32(name);
    };

    model.config = config;
    model.token_embedding = load_f32("token_embedding");

    model.layers.resize(static_cast<size_t>(config.n_layers));
    for (int i = 0; i < config.n_layers; ++i) {
      LayerWeights& lw = model.layers[static_cast<size_t>(i)];
      const std::string prefix = "layers." + std::to_string(i) + ".";
      lw.attention_norm = load_f32(prefix + "attention_norm");
      lw.wq = load_weight(prefix + "wq");
      lw.wk = load_weight(prefix + "wk");
      lw.wv = load_weight(prefix + "wv");
      lw.bq = load_optional_f32(prefix + "bq");
      lw.bk = load_optional_f32(prefix + "bk");
      lw.bv = load_optional_f32(prefix + "bv");
      lw.wo = load_weight(prefix + "wo");
      lw.ffn_norm = load_f32(prefix + "ffn_norm");
      lw.w_gate = load_weight(prefix + "w_gate");
      lw.w_up = load_weight(prefix + "w_up");
      lw.w_down = load_weight(prefix + "w_down");
    }

    model.final_norm = load_f32("final_norm");

    if (tensor_map.find("lm_head") != tensor_map.end()) {
      model.lm_head = load_weight("lm_head");
    } else {
      model.lm_head = ToQuantizedTensor(model.token_embedding);
    }

    ValidateLoadedShapes(model);

    model.loaded = true;
  } catch (const std::exception& e) {
    model.loaded = false;
    model.load_error = e.what();
    return model;
  }

  std::cout << "GGUF model loaded successfully from " << gguf_path << std::endl;
  return model;
}

}  // namespace dqm
