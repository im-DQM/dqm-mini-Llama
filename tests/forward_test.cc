#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "dqm_llama/context.h"
#include "dqm_llama/forward.h"
#include "dqm_llama/model.h"
#include "dqm_llama/tensor.h"
#include "test_tiny_model.h"

using dqm::ForwardToken;
using dqm::MiniLlamaContext;
using dqm::MiniLlamaModel;
using dqm::Tensor;
using dqm::test::MakeTinyModel;
using dqm::test::NearlyEqual;

// ===========================================================================
// 入参校验 —— 全部跑在真正计算之前
// ===========================================================================

TEST(ForwardToken, RejectsUnloadedModel) {
  MiniLlamaModel model = MakeTinyModel();
  model.loaded = false;
  MiniLlamaContext ctx(&model);

  EXPECT_THROW(ForwardToken(ctx, model, 0), std::runtime_error);
}

TEST(ForwardToken, RejectsTokenAtVocabSize) {
  MiniLlamaModel model = MakeTinyModel();
  MiniLlamaContext ctx(&model);

  EXPECT_THROW(ForwardToken(ctx, model, model.config.vocab_size),
               std::out_of_range);
}

TEST(ForwardToken, RejectsNegativeToken) {
  MiniLlamaModel model = MakeTinyModel();
  MiniLlamaContext ctx(&model);

  EXPECT_THROW(ForwardToken(ctx, model, -1), std::out_of_range);
}

TEST(ForwardToken, RejectsPositionAtMaxSeqLen) {
  MiniLlamaModel model = MakeTinyModel();
  MiniLlamaContext ctx(&model);
  ctx.pos = model.config.max_seq_len;

  EXPECT_THROW(ForwardToken(ctx, model, 0), std::out_of_range);
}

TEST(ForwardToken, RejectsNegativePosition) {
  MiniLlamaModel model = MakeTinyModel();
  MiniLlamaContext ctx(&model);
  ctx.pos = -1;

  EXPECT_THROW(ForwardToken(ctx, model, 0), std::out_of_range);
}

TEST(ForwardToken, RejectsLayerCountMismatch) {
  MiniLlamaModel model = MakeTinyModel();
  model.layers.clear();
  MiniLlamaContext ctx(&model);

  EXPECT_THROW(ForwardToken(ctx, model, 0), std::runtime_error);
}

TEST(ForwardToken, RejectsUndersizedTokenEmbedding) {
  MiniLlamaModel model = MakeTinyModel();
  model.token_embedding = Tensor({1}, 0.0f);
  MiniLlamaContext ctx(&model);

  EXPECT_THROW(ForwardToken(ctx, model, 0), std::runtime_error);
}

// ===========================================================================
// 输出形状与确定性
// ===========================================================================

TEST(ForwardToken, ProducesLogitsOfVocabSize) {
  MiniLlamaModel model = MakeTinyModel();
  MiniLlamaContext ctx(&model);

  const Tensor logits = ForwardToken(ctx, model, 3);

  EXPECT_EQ(logits.shape, (std::vector<int>{model.config.vocab_size}));
  EXPECT_EQ(logits.size(), static_cast<size_t>(model.config.vocab_size));
}

// 同样入参、同样的起点 → 必须逐位相同（没有随机、没有并发）。
TEST(ForwardToken, IsDeterministicForSameInput) {
  MiniLlamaModel model = MakeTinyModel();
  MiniLlamaContext first(&model);
  MiniLlamaContext second(&model);

  const Tensor a = ForwardToken(first, model, 11);
  const Tensor b = ForwardToken(second, model, 11);

  EXPECT_EQ(a.shape, b.shape);
  EXPECT_TRUE(NearlyEqual(a, b));
}

TEST(ForwardToken, DifferentTokensGiveDifferentLogits) {
  MiniLlamaModel model = MakeTinyModel();
  MiniLlamaContext first(&model);
  MiniLlamaContext second(&model);

  const Tensor a = ForwardToken(first, model, 1);
  const Tensor b = ForwardToken(second, model, 2);

  EXPECT_FALSE(NearlyEqual(a, b));
}

// ===========================================================================
// KV cache
// ===========================================================================

TEST(ForwardToken, WritesKvCacheAtCurrentPosition) {
  MiniLlamaModel model = MakeTinyModel();
  MiniLlamaContext ctx(&model);
  ctx.pos = 0;

  ForwardToken(ctx, model, 3);

  const float* k_ptr = ctx.kv_cache.KeyPtr(0, 0, 0);
  const float* v_ptr = ctx.kv_cache.ValuePtr(0, 0, 0);

  bool key_written = false;
  bool value_written = false;
  for (int d = 0; d < model.config.head_dim; ++d) {
    if (k_ptr[d] != 0.0f) {
      key_written = true;
    }
    if (v_ptr[d] != 0.0f) {
      value_written = true;
    }
  }
  EXPECT_TRUE(key_written);
  EXPECT_TRUE(value_written);
}

// 第二步必须看得到第一步写进 cache 的 K/V —— 否则 attention 根本没读历史。
TEST(ForwardToken, LaterStepAttendsToWrittenHistory) {
  MiniLlamaModel model = MakeTinyModel();

  // 对照：直接算 pos=1，cache 里 pos=0 还是空的
  MiniLlamaContext alone(&model);
  alone.pos = 1;
  const Tensor without_history = ForwardToken(alone, model, 7);

  // 实验：先算 pos=0，再算 pos=1
  MiniLlamaContext with_history(&model);
  with_history.pos = 0;
  ForwardToken(with_history, model, 4);
  with_history.pos = 1;
  const Tensor with = ForwardToken(with_history, model, 7);

  EXPECT_FALSE(NearlyEqual(without_history, with));
}

// ===========================================================================
// 位置相关（RoPE）
// ===========================================================================

// RoPE 必须真正参与 forward：改变 rope_theta 会改变输出。
//
// 注意必须用「不同 token」的序列。若各位置 token 相同，则每个位置的
// v = wv·rmsnorm(embed(tok)) 全相等，attention 输出 Σ probs[t]·v_t 坍缩为
// v·Σprobs = v，与 probs 无关；而 RoPE 只改 q/k（只影响 probs），于是
// 位置对输出毫无影响 —— 这正是「同一 token 不同位置」测不出 RoPE 的原因。
TEST(ForwardToken, RopeInfluencesOutput) {
  auto run = [](float theta, int first, int second) {
    MiniLlamaModel model = MakeTinyModel();
    model.config.rope_theta = theta;
    MiniLlamaContext ctx(&model);
    ctx.pos = 0;
    ForwardToken(ctx, model, first);
    ctx.pos = 1;
    return ForwardToken(ctx, model, second);
  };

  const Tensor baseline = run(10000.0f, 5, 7);
  const Tensor shifted = run(1.0f, 5, 7);

  EXPECT_FALSE(NearlyEqual(baseline, shifted));
}

// ===========================================================================
// 职责边界
// ===========================================================================

// token_history 由 ForwardBatch 维护 —— ForwardToken 只管算，不记账。
TEST(ForwardToken, DoesNotAppendToTokenHistory) {
  MiniLlamaModel model = MakeTinyModel();
  MiniLlamaContext ctx(&model);

  ForwardToken(ctx, model, 9);

  EXPECT_TRUE(ctx.token_history.empty());
}
