#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "dqm_llama/model.h"
#include "dqm_llama/quantized_tensor.h"
#include "dqm_llama/tensor.h"

using dqm::LayerWeights;
using dqm::MiniLlamaModel;
using dqm::ModelConfig;
using dqm::ModelWeightBytes;
using dqm::ModelWeightBytesF32;
using dqm::QuantizedTensor;
using dqm::QuantType;
using dqm::RopeType;
using dqm::Tensor;
using dqm::ToQuantizedTensor;

namespace {

Tensor MakeF32(const std::vector<int>& shape) {
  return Tensor(shape, 0.0f);
}

QuantizedTensor MakeF32Container(const std::vector<int>& shape) {
  return ToQuantizedTensor(Tensor(shape, 0.0f));
}

// 造一个「形状和 tiny 模型一模一样」的模型（权重全 0，懒得读 .bin）。
// 期望的字节数可以拿 models/tiny/model.json 里的 offset + byte_size 对。
MiniLlamaModel MakeTinyShapedModel() {
  MiniLlamaModel m;
  const ModelConfig& c = m.config;

  m.token_embedding = MakeF32({c.vocab_size, c.dim});
  m.final_norm = MakeF32({c.dim});
  m.lm_head = MakeF32Container({c.vocab_size, c.dim});

  m.layers.resize(static_cast<size_t>(c.n_layers));
  for (LayerWeights& lw : m.layers) {
    lw.attention_norm = MakeF32({c.dim});
    lw.wq = MakeF32Container({c.n_heads * c.head_dim, c.dim});
    lw.wk = MakeF32Container({c.n_kv_heads * c.head_dim, c.dim});
    lw.wv = MakeF32Container({c.n_kv_heads * c.head_dim, c.dim});
    lw.wo = MakeF32Container({c.dim, c.n_heads * c.head_dim});
    lw.ffn_norm = MakeF32({c.dim});
    lw.w_gate = MakeF32Container({c.hidden_dim, c.dim});
    lw.w_up = MakeF32Container({c.hidden_dim, c.dim});
    lw.w_down = MakeF32Container({c.dim, c.hidden_dim});
    // bq / bk / bv 留空 —— tiny 模型没有 attention bias
  }
  return m;
}

}  // namespace

// ===========================================================================
// ModelConfig
// ===========================================================================

TEST(ModelConfig, DefaultsMatchTinyModel) {
  const ModelConfig c;
  EXPECT_EQ(c.vocab_size, 128);
  EXPECT_EQ(c.dim, 32);
  EXPECT_EQ(c.hidden_dim, 86);
  EXPECT_EQ(c.n_layers, 2);
  EXPECT_EQ(c.n_heads, 4);
  EXPECT_EQ(c.n_kv_heads, 4);
  EXPECT_EQ(c.head_dim, 8);
  EXPECT_EQ(c.max_seq_len, 128);
  EXPECT_NEAR(c.rope_theta, 10000.0f, 1e-6f);
  EXPECT_NEAR(c.rms_norm_eps, 1e-5f, 1e-12f);
  EXPECT_EQ(c.rope_type, RopeType::kNormal);
}

// 约定：dim = n_heads * head_dim（tiny 是 4 * 8 = 32）
TEST(ModelConfig, HeadCountTimesHeadDimEqualsDim) {
  const ModelConfig c;
  EXPECT_EQ(c.n_heads * c.head_dim, c.dim);
}

// ===========================================================================
// MiniLlamaModel 默认状态
// ===========================================================================

TEST(MiniLlamaModel, DefaultConstructedIsUnloaded) {
  const MiniLlamaModel m;
  EXPECT_FALSE(m.loaded);
  EXPECT_TRUE(m.load_error.empty());
  EXPECT_TRUE(m.layers.empty());
  EXPECT_TRUE(m.token_embedding.data.empty());
  EXPECT_TRUE(m.final_norm.data.empty());
  EXPECT_EQ(m.lm_head.type, QuantType::kF32);
}

// ===========================================================================
// LayerWeights 的类型分工
// ===========================================================================

// 会做矩阵乘的 7 个权重是 QuantizedTensor（将来要能量化）；
// norm 增益和 bias 是 Tensor（永远 F32）。
// 类型不对的话这几行根本编译不过 —— 这条测试的价值在「编译期钉住分工」。
TEST(LayerWeights, LinearWeightsAreQuantizableOthersArePlainTensors) {
  LayerWeights lw;

  // 这 7 个必须是 QuantizedTensor：能读到 .type / 四个 data vector
  lw.wq.type = QuantType::kQ80;
  lw.wk.type = QuantType::kQ80;
  lw.wv.type = QuantType::kQ80;
  lw.wo.type = QuantType::kQ80;
  lw.w_gate.type = QuantType::kQ80;
  lw.w_up.type = QuantType::kQ80;
  lw.w_down.type = QuantType::kQ80;
  EXPECT_EQ(lw.wq.type, QuantType::kQ80);
  EXPECT_EQ(lw.w_down.type, QuantType::kQ80);

  // 这 5 个必须是普通 Tensor：直接读 .data
  lw.attention_norm = MakeF32({32});
  lw.ffn_norm = MakeF32({32});
  lw.bq = MakeF32({32});
  lw.bk = MakeF32({32});
  lw.bv = MakeF32({32});
  EXPECT_EQ(lw.attention_norm.data.size(), 32u);
  EXPECT_EQ(lw.bv.data.size(), 32u);
}

// 可选 bias 用「空 Tensor」表示「没有」，不是 nullptr、不是 optional
TEST(LayerWeights, MissingBiasIsAnEmptyTensor) {
  const LayerWeights lw;
  EXPECT_TRUE(lw.bq.data.empty());
  EXPECT_TRUE(lw.bk.data.empty());
  EXPECT_TRUE(lw.bv.data.empty());
  EXPECT_EQ(lw.bq.shape.size(), 0u);
}

// ===========================================================================
// ModelWeightBytes / ModelWeightBytesF32
// ===========================================================================

TEST(ModelWeightBytes, EmptyModelIsZero) {
  EXPECT_EQ(ModelWeightBytes(MiniLlamaModel{}), 0u);
}

// ⚠️ 空 shape 的 QuantizedTensor，num_elements() 按乘法单位元算 = 1。
// 于是「若全存成 F32 会占多少」会给出 1 * 4 = 4 字节，而不是 0。
// 这不是笔误 —— 它是 num_elements() 对空形状的定义在边界上的体现。
// 真模型不会出现空 shape，所以不影响实际使用；钉在这里是为了
// 让这个行为显式可见，而不是等人某天发现「空模型的 F32 占用是 4 字节」。
TEST(ModelWeightBytes, EmptyModelF32CountsOnePhantomElement) {
  EXPECT_EQ(ModelWeightBytesF32(MiniLlamaModel{}), 4u);
}

// 全 F32 时两个函数必须相等
TEST(ModelWeightBytes, F32ModelMatchesBothMeasurements) {
  const MiniLlamaModel m = MakeTinyShapedModel();
  EXPECT_EQ(ModelWeightBytes(m), ModelWeightBytesF32(m));
}

// 黄金值：tiny 模型所有权重加起来 = model.json 里最后一个 tensor 的
// offset + byte_size = 115840 + 16384 = 132224
TEST(ModelWeightBytes, TinyShapedModelIs132224Bytes) {
  const MiniLlamaModel m = MakeTinyShapedModel();
  EXPECT_EQ(ModelWeightBytes(m), 132224u);
  EXPECT_EQ(ModelWeightBytesF32(m), 132224u);
}

// 不含 bias 的层（tiny）与含 bias 的层，bytes 差 = bias 的大小
TEST(ModelWeightBytes, OptionalBiasAddsItsBytes) {
  const MiniLlamaModel without_bias = MakeTinyShapedModel();
  MiniLlamaModel with_bias = MakeTinyShapedModel();

  const ModelConfig& c = with_bias.config;
  for (LayerWeights& lw : with_bias.layers) {
    lw.bq = MakeF32({c.n_heads * c.head_dim});         // 32 floats = 128 字节
    lw.bk = MakeF32({c.n_kv_heads * c.head_dim});      // 32 floats = 128 字节
    lw.bv = MakeF32({c.n_kv_heads * c.head_dim});      // 32 floats = 128 字节
  }

  const size_t per_layer = 3u * 32u * sizeof(float);
  const size_t extra =
      per_layer * static_cast<size_t>(c.n_layers);
  EXPECT_EQ(ModelWeightBytes(with_bias),
            ModelWeightBytes(without_bias) + extra);
}

// 核心：把一路权重压成 Q8_0 之后 ——
//   ModelWeightBytes 变小（实际占的堆内存少了）
//   ModelWeightBytesF32 不变（逻辑元素数没变）
// 这就是两个函数存在的理由：它们量的不是一个东西。
TEST(ModelWeightBytes, QuantizingShrinksActualButNotF32) {
  MiniLlamaModel m = MakeTinyShapedModel();

  const size_t before_actual = ModelWeightBytes(m);
  const size_t before_f32 = ModelWeightBytesF32(m);

  // layers[0].wq 是 [32, 32] = 1024 个元素；全 F32 占 4096 字节
  QuantizedTensor& wq = m.layers[0].wq;
  ASSERT_EQ(wq.num_elements(), 1024u);

  // 手工换成 Q8_0：1024 / 32 = 32 个 block，每块 34 字节 = 1088 字节
  wq.type = QuantType::kQ80;
  wq.q8_0_data.resize(wq.num_elements() / 32u);
  wq.f32_data.clear();
  wq.f32_data.shrink_to_fit();

  const size_t q80_bytes = 32u * sizeof(dqm::BlockQ80);
  EXPECT_EQ(q80_bytes, 1088u);

  EXPECT_EQ(ModelWeightBytes(m), before_actual - 4096u + q80_bytes);
  EXPECT_EQ(ModelWeightBytesF32(m), before_f32);
}

// 每个 type 走各自的分支，不能串
TEST(ModelWeightBytes, EachQuantTypeUsesItsOwnBlockSize) {
  MiniLlamaModel m;
  m.lm_head.shape = {32};  // 32 个元素 = 一个 block
  m.lm_head.type = QuantType::kQ40;
  m.lm_head.q4_0_data.resize(1);

  EXPECT_EQ(ModelWeightBytes(m), sizeof(dqm::BlockQ40));
  EXPECT_EQ(ModelWeightBytesF32(m), 32u * sizeof(float));

  m.lm_head.type = QuantType::kQ41;
  m.lm_head.q4_1_data.resize(1);
  EXPECT_EQ(ModelWeightBytes(m), sizeof(dqm::BlockQ41));
}
